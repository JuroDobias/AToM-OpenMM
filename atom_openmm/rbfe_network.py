from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr

from atom_openmm.neqti import analyze_two_leg_work


class RBFENetworkError(ValueError):
    pass


def _write_yaml_atomic(path, payload):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False))
    os.replace(temporary, path)


def _edge_result(root, edge):
    directory = root / edge["directory"]
    candidates = sorted((directory / "run").rglob("result.yaml"))
    if len(candidates) != 1:
        return None, f"expected one result.yaml below {directory / 'run'}"
    payload = yaml.safe_load(candidates[0].read_text()) or {}
    result = payload.get("result") or {}
    value = result.get("ddg_kcal_per_mol")
    error = result.get("ddg_error_kcal_per_mol")
    try:
        value = float(value)
        error = float(error)
    except (TypeError, ValueError):
        return None, "result lacks finite DDG and uncertainty"
    if not math.isfinite(value) or not math.isfinite(error) or error <= 0.0:
        return None, "result lacks finite positive uncertainty"
    return {
        "ligand_a": str(edge["ligand_a"]),
        "ligand_b": str(edge["ligand_b"]),
        "directory": edge["directory"],
        "result_file": str(candidates[0].resolve()),
        "workflow_status": payload.get("status"),
        "ddg_kcal_per_mol": value,
        "ddg_error_kcal_per_mol": error,
        "alchemy_model": payload.get("alchemy_model"),
        "components": result.get("components"),
        "_workdir": str(candidates[0].parent),
    }, None


def _separated_work_rows(edge):
    workdir = Path(edge["_workdir"])
    rows = {}
    for environment in ("complex", "solvent"):
        for direction in ("forward", "reverse"):
            path = workdir / f"{environment}_{direction}.csv"
            if not path.is_file():
                raise RBFENetworkError(f"missing separated work file: {path}")
            with path.open(newline="") as handle:
                values = list(csv.DictReader(handle))
            if not values or any(
                not row.get("active_snapshot_id")
                or not row.get("active_node")
                or not row.get("inactive_vacuum_snapshot_id")
                or not row.get("inactive_node")
                for row in values
            ):
                raise RBFENetworkError(
                    f"separated work file lacks node snapshot identifiers: {path}"
                )
            rows[(environment, direction)] = values
    return rows


def _node_bootstrap_cycles(observed, cycles, samples, temperature, seed):
    separated = {
        edge["directory"]: edge for edge in observed
        if edge.get("alchemy_model") == "separated_topology"
    }
    required = {
        term["edge"] for cycle in cycles for term in cycle["terms"]
    }
    if not required or not required <= set(separated):
        return {}
    work = {name: _separated_work_rows(separated[name]) for name in required}
    pools = {}
    for edge_rows in work.values():
        for (environment, _), rows in edge_rows.items():
            node = rows[0]["active_node"]
            ids = tuple(row["active_snapshot_id"] for row in rows)
            key = (node, environment)
            if key in pools and set(pools[key]) != set(ids):
                raise RBFENetworkError(
                    f"shared node {node} uses different {environment} snapshot sets"
                )
            pools[key] = ids
            inactive_node = rows[0]["inactive_node"]
            inactive_ids = tuple(
                row["inactive_vacuum_snapshot_id"] for row in rows
            )
            inactive_key = (inactive_node, "vacuum")
            if inactive_key in pools and set(pools[inactive_key]) != set(inactive_ids):
                raise RBFENetworkError(
                    f"shared node {inactive_node} uses different vacuum snapshot sets"
                )
            pools[inactive_key] = inactive_ids
    rng = np.random.default_rng(int(seed))
    estimates = {cycle["id"]: [] for cycle in cycles}
    complex_estimates = {cycle["id"]: [] for cycle in cycles}
    solvent_estimates = {cycle["id"]: [] for cycle in cycles}
    accepted_replicates = 0
    attempted_replicates = 0
    max_attempts = max(int(samples), 20 * int(samples))
    while accepted_replicates < int(samples) and attempted_replicates < max_attempts:
        attempted_replicates += 1
        multiplicities = {}
        for key, values in pools.items():
            sampled = rng.choice(values, size=len(values), replace=True)
            multiplicities[key] = {
                value: int(np.count_nonzero(sampled == value))
                for value in values
            }
        edge_values = {}
        failed = False
        for name, edge_rows in work.items():
            sampled_work = {}
            for environment, leg in (("complex", "leg_a"), ("solvent", "leg_b")):
                for direction in ("forward", "reverse"):
                    rows = edge_rows[(environment, direction)]
                    node = rows[0]["active_node"]
                    inactive_node = rows[0]["inactive_node"]
                    values = []
                    for row in rows:
                        count = (
                            multiplicities[(node, environment)][
                                row["active_snapshot_id"]
                            ]
                            * multiplicities[(inactive_node, "vacuum")][
                                row["inactive_vacuum_snapshot_id"]
                            ]
                        )
                        values.extend(
                            [float(row["work_kcal_per_mol"])] * count
                        )
                    sampled_work[f"{leg}_{direction}"] = values
            analysis = analyze_two_leg_work(
                sampled_work, temperature, bootstrap_samples=0, random_seed=seed
            )
            if analysis is None:
                failed = True
                break
            edge_values[name] = {
                "ddg": analysis["bar_dg_kcal_per_mol"],
                "complex": analysis["components"]["leg_a"]["dg_kcal_per_mol"],
                "solvent": analysis["components"]["leg_b"]["dg_kcal_per_mol"],
            }
        if failed:
            continue
        accepted_replicates += 1
        for cycle in cycles:
            terms = cycle["terms"]
            estimates[cycle["id"]].append(sum(
                float(term["coefficient"]) * edge_values[term["edge"]]["ddg"]
                for term in terms
            ))
            complex_estimates[cycle["id"]].append(sum(
                float(term["coefficient"]) * edge_values[term["edge"]]["complex"]
                for term in terms
            ))
            solvent_estimates[cycle["id"]].append(sum(
                float(term["coefficient"]) * edge_values[term["edge"]]["solvent"]
                for term in terms
            ))

    def summary(values):
        if len(values) < 2:
            return None
        return {
            "samples": len(values),
            "std_kcal_per_mol": float(np.std(values, ddof=1)),
            "ci95_kcal_per_mol": [
                float(value) for value in np.quantile(values, [0.025, 0.975])
            ],
        }

    return {
        cycle["id"]: {
            "binding": summary(estimates[cycle["id"]]),
            "complex": summary(complex_estimates[cycle["id"]]),
            "solvent": summary(solvent_estimates[cycle["id"]]),
        }
        for cycle in cycles
    }


def _edge_id(edge):
    return str(edge.get("id", edge["directory"]))


def _random_effects(records):
    values = np.asarray([row["ddg_kcal_per_mol"] for row in records], dtype=float)
    errors = np.asarray([row["ddg_error_kcal_per_mol"] for row in records], dtype=float)
    fixed_weights = 1.0 / errors**2
    fixed_mean = float(np.sum(fixed_weights * values) / np.sum(fixed_weights))
    fixed_error = float(math.sqrt(1.0 / np.sum(fixed_weights)))
    q = float(np.sum(fixed_weights * (values - fixed_mean) ** 2))
    degrees_of_freedom = max(0, len(records) - 1)
    denominator = float(
        np.sum(fixed_weights)
        - np.sum(fixed_weights**2) / np.sum(fixed_weights)
    )
    tau_squared = (
        max(0.0, (q - degrees_of_freedom) / denominator)
        if degrees_of_freedom and denominator > 0.0
        else 0.0
    )
    random_weights = 1.0 / (errors**2 + tau_squared)
    random_mean = float(np.sum(random_weights * values) / np.sum(random_weights))
    random_error = float(math.sqrt(1.0 / np.sum(random_weights)))
    i_squared = (
        max(0.0, (q - degrees_of_freedom) / q) if q > 0.0 else 0.0
    )
    return {
        "ddg_kcal_per_mol": random_mean,
        "ddg_error_kcal_per_mol": random_error,
        "fixed_effect_ddg_kcal_per_mol": fixed_mean,
        "fixed_effect_error_kcal_per_mol": fixed_error,
        "tau_squared_kcal2_per_mol2": float(tau_squared),
        "cochran_q": q,
        "heterogeneity_degrees_of_freedom": degrees_of_freedom,
        "i_squared": float(i_squared),
        "replicate_count": len(records),
    }


def _graph_fit(nodes, reference, observed, inflate_uncertainty=False):
    fitted_nodes = [node for node in nodes if node != reference]
    node_column = {node: index for index, node in enumerate(fitted_nodes)}
    matrix = np.zeros((len(observed), len(fitted_nodes)), dtype=float)
    values = np.zeros(len(observed), dtype=float)
    errors = np.zeros(len(observed), dtype=float)
    for row, edge in enumerate(observed):
        try:
            if edge["ligand_a"] != reference:
                matrix[row, node_column[edge["ligand_a"]]] -= 1.0
            if edge["ligand_b"] != reference:
                matrix[row, node_column[edge["ligand_b"]]] += 1.0
        except KeyError as exc:
            raise RBFENetworkError(f"edge references unknown node {exc.args[0]}") from exc
        values[row] = edge["ddg_kcal_per_mol"]
        errors[row] = edge["ddg_error_kcal_per_mol"]

    rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
    solved = rank == len(fitted_nodes)
    result = {
        "solved": solved,
        "matrix_rank": rank,
        "required_rank": len(fitted_nodes),
        "node_values": {reference: 0.0},
        "node_model_errors": {reference: 0.0},
        "node_errors": {reference: 0.0},
        "covariance": None,
        "scaled_covariance": None,
        "chi_squared": None,
        "degrees_of_freedom": None,
        "reduced_chi_squared": None,
        "uncertainty_scale": 1.0,
        "node_column": node_column,
    }
    if not solved:
        return result

    weights = 1.0 / errors**2
    normal = matrix.T @ (weights[:, None] * matrix)
    covariance = np.linalg.inv(normal)
    estimates = covariance @ matrix.T @ (weights * values)
    predicted = matrix @ estimates
    residuals = values - predicted
    chi_squared = float(np.sum((residuals / errors) ** 2))
    degrees_of_freedom = max(0, len(observed) - len(fitted_nodes))
    reduced = chi_squared / degrees_of_freedom if degrees_of_freedom else None
    scale = (
        math.sqrt(max(1.0, reduced))
        if inflate_uncertainty and reduced is not None
        else 1.0
    )
    scaled_covariance = covariance * scale**2
    result.update({
        "covariance": covariance,
        "scaled_covariance": scaled_covariance,
        "chi_squared": chi_squared,
        "degrees_of_freedom": degrees_of_freedom,
        "reduced_chi_squared": reduced,
        "uncertainty_scale": float(scale),
    })
    for node, index in node_column.items():
        result["node_values"][node] = float(estimates[index])
        result["node_model_errors"][node] = float(
            math.sqrt(max(0.0, covariance[index, index]))
        )
        result["node_errors"][node] = float(
            math.sqrt(max(0.0, scaled_covariance[index, index]))
        )
    for edge, prediction, residual in zip(observed, predicted, residuals):
        edge["fitted_ddg_kcal_per_mol"] = float(prediction)
        edge["residual_kcal_per_mol"] = float(residual)
        edge["standardized_residual"] = float(
            residual / edge["ddg_error_kcal_per_mol"]
        )
    return result


def _targets(config):
    raw = config.get("targets")
    if raw is None:
        raw = [config["target"]]
    targets = []
    for index, item in enumerate(raw, 1):
        if isinstance(item, dict):
            ligand_a = str(item["ligand_a"])
            ligand_b = str(item["ligand_b"])
            identifier = str(item.get("id", f"{ligand_a}--{ligand_b}"))
            references = item.get("references", {})
        else:
            ligand_a, ligand_b = map(str, item)
            identifier = f"{ligand_a}--{ligand_b}"
            references = config.get("target_references", {}) if index == 1 else {}
        targets.append({
            "id": identifier,
            "ligand_a": ligand_a,
            "ligand_b": ligand_b,
            "references": references,
        })
    return targets


def _target_result(target, fit):
    result = dict(target)
    result.update({
        "ddg_kcal_per_mol": None,
        "ddg_error_kcal_per_mol": None,
        "ddg_model_error_kcal_per_mol": None,
    })
    if not fit["solved"]:
        return result
    node_column = fit["node_column"]
    vector = np.zeros(len(node_column), dtype=float)
    if target["ligand_a"] in node_column:
        vector[node_column[target["ligand_a"]]] -= 1.0
    if target["ligand_b"] in node_column:
        vector[node_column[target["ligand_b"]]] += 1.0
    result["ddg_kcal_per_mol"] = float(
        fit["node_values"][target["ligand_b"]]
        - fit["node_values"][target["ligand_a"]]
    )
    result["ddg_model_error_kcal_per_mol"] = float(
        math.sqrt(max(0.0, vector @ fit["covariance"] @ vector))
    )
    result["ddg_error_kcal_per_mol"] = float(
        math.sqrt(max(0.0, vector @ fit["scaled_covariance"] @ vector))
    )
    return result


def _reference_comparisons(config, nodes, primary_fit):
    comparisons = []
    generated = {
        str(item["id"])
        for item in config.get("nodes", [])
        if isinstance(item, dict) and item.get("generated")
    }
    for name, dataset in (config.get("reference_datasets") or {}).items():
        rows = []
        for edge in dataset.get("edges", []):
            error = float(edge.get("ddg_error_kcal_per_mol", 1.0))
            rows.append({
                "ligand_a": str(edge["ligand_a"]),
                "ligand_b": str(edge["ligand_b"]),
                "ddg_kcal_per_mol": float(edge["ddg_kcal_per_mol"]),
                "ddg_error_kcal_per_mol": error,
            })
        dataset_nodes = [str(node) for node in dataset.get("nodes", [])]
        if not dataset_nodes:
            dataset_nodes = sorted({
                ligand for row in rows for ligand in (row["ligand_a"], row["ligand_b"])
            })
        dataset_reference = str(dataset.get("reference_node", config["reference_node"]))
        if dataset_reference not in dataset_nodes:
            dataset_nodes.append(dataset_reference)
        fit = _graph_fit(dataset_nodes, dataset_reference, rows)
        shared = [
            node for node in nodes
            if node not in generated
            and node in fit["node_values"]
            and node in primary_fit["node_values"]
        ] if fit["solved"] and primary_fit["solved"] else []
        nonreference = [node for node in shared if node != config["reference_node"]]

        def metrics(selected):
            if len(selected) < 2:
                return None
            predicted = np.asarray([primary_fit["node_values"][node] for node in selected])
            reference_values = np.asarray([fit["node_values"][node] for node in selected])
            difference = predicted - reference_values
            pearson = float(pearsonr(predicted, reference_values).statistic)
            spearman = float(spearmanr(predicted, reference_values).statistic)
            return {
                "node_count": len(selected),
                "nodes": selected,
                "pearson_r": pearson if math.isfinite(pearson) else None,
                "spearman_rho": spearman if math.isfinite(spearman) else None,
                "rmse_kcal_per_mol": float(np.sqrt(np.mean(difference**2))),
                "mae_kcal_per_mol": float(np.mean(np.abs(difference))),
            }

        comparisons.append({
            "id": str(name),
            "fit_status": "completed" if fit["solved"] else "partial",
            "reference_fit": {
                "chi_squared": fit["chi_squared"],
                "degrees_of_freedom": fit["degrees_of_freedom"],
                "reduced_chi_squared": fit["reduced_chi_squared"],
            },
            "node_free_energies_kcal_per_mol": {
                node: fit["node_values"].get(node) for node in shared
            },
            "metrics_including_anchor": metrics(shared),
            "metrics_excluding_anchor": metrics(nonreference),
        })
    return comparisons


def _write_csv_outputs(root, payload):
    tables = {
        "edge_replicates.csv": (
            ["edge", "replicate", "ddg_kcal_per_mol", "ddg_error_kcal_per_mol", "status", "result_file"],
            [
                {
                    "edge": edge["id"],
                    "replicate": replicate["replicate_id"],
                    "ddg_kcal_per_mol": replicate["ddg_kcal_per_mol"],
                    "ddg_error_kcal_per_mol": replicate["ddg_error_kcal_per_mol"],
                    "status": replicate.get("workflow_status"),
                    "result_file": replicate.get("result_file"),
                }
                for edge in payload.get("edges", [])
                for replicate in edge.get("replicates", [])
            ],
        ),
        "edge_aggregates.csv": (
            ["edge", "ligand_a", "ligand_b", "ddg_kcal_per_mol", "ddg_error_kcal_per_mol", "replicate_count", "tau_squared_kcal2_per_mol2", "i_squared", "residual_kcal_per_mol"],
            payload.get("edges", []),
        ),
        "node_free_energies.csv": (
            ["node", "value_kcal_per_mol", "error_kcal_per_mol", "model_error_kcal_per_mol"],
            [
                {"node": node, **values}
                for node, values in payload["node_free_energies_kcal_per_mol"].items()
            ],
        ),
        "targets.csv": (
            ["id", "ligand_a", "ligand_b", "ddg_kcal_per_mol", "ddg_error_kcal_per_mol", "ddg_model_error_kcal_per_mol"],
            payload.get("targets", []),
        ),
        "cycles.csv": (
            ["id", "status", "closure_kcal_per_mol", "closure_error_kcal_per_mol", "complex_closure_kcal_per_mol", "solvent_closure_kcal_per_mol"],
            payload.get("cycles", []),
        ),
        "reference_comparisons.csv": (
            ["dataset", "anchor", "node_count", "pearson_r", "spearman_rho", "rmse_kcal_per_mol", "mae_kcal_per_mol"],
            [
                {
                    "dataset": comparison["id"],
                    "anchor": anchor,
                    **metrics,
                }
                for comparison in payload.get("reference_comparisons", [])
                for anchor, metrics in (
                    ("included", comparison.get("metrics_including_anchor")),
                    ("excluded", comparison.get("metrics_excluding_anchor")),
                )
                if metrics is not None
            ],
        ),
    }
    for filename, (fields, rows) in tables.items():
        if not rows:
            continue
        with (root / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def _automatic_cycles(edges):
    parent = {}
    tree = {}
    chords = []

    def find(node):
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return False
        parent[root_b] = root_a
        return True

    for edge in edges:
        a, b = edge["ligand_a"], edge["ligand_b"]
        if union(a, b):
            tree.setdefault(a, []).append((b, edge))
            tree.setdefault(b, []).append((a, edge))
        else:
            chords.append(edge)

    cycles = []
    for index, chord in enumerate(chords, 1):
        start, end = chord["ligand_b"], chord["ligand_a"]
        pending = [(start, [])]
        visited = set()
        path = None
        while pending:
            node, terms = pending.pop()
            if node == end:
                path = terms
                break
            if node in visited:
                continue
            visited.add(node)
            for neighbor, edge in tree.get(node, []):
                if neighbor in visited:
                    continue
                coefficient = 1 if (
                    edge["ligand_a"] == node and edge["ligand_b"] == neighbor
                ) else -1
                pending.append((neighbor, terms + [{
                    "edge": edge["id"], "coefficient": coefficient,
                }]))
        if path is None:
            continue
        cycles.append({
            "id": f"automatic_cycle_{index}",
            "terms": [{"edge": chord["id"], "coefficient": 1}] + path,
        })
    return cycles


def _analyze_observed(config, observed, missing, root, *, repeated=False):
    nodes = [str(node["id"] if isinstance(node, dict) else node) for node in config["nodes"]]
    reference = str(config["reference_node"])
    if reference not in nodes or len(set(nodes)) != len(nodes):
        raise RBFENetworkError("network nodes must be unique and contain reference_node")
    fit = _graph_fit(nodes, reference, observed, inflate_uncertainty=repeated)
    edge_by_id = {edge["id"]: edge for edge in observed}
    cycle_definitions = list(config.get("cycles", []))
    if config.get("auto_cycles"):
        cycle_definitions.extend(_automatic_cycles([
            {
                "id": _edge_id(edge),
                "ligand_a": str(edge["ligand_a"]),
                "ligand_b": str(edge["ligand_b"]),
            }
            for edge in config["edges"]
        ]))
    cycles = []
    for cycle in cycle_definitions:
        terms = cycle["terms"]
        records = [edge_by_id.get(str(term["edge"])) for term in terms]
        if any(record is None for record in records):
            cycles.append({"id": cycle["id"], "status": "partial"})
            continue
        closure = sum(
            float(term["coefficient"]) * record["ddg_kcal_per_mol"]
            for term, record in zip(terms, records)
        )
        variance = sum(
            float(term["coefficient"]) ** 2
            * record["ddg_error_kcal_per_mol"] ** 2
            for term, record in zip(terms, records)
        )
        complex_closure = None
        solvent_closure = None
        if all(record.get("components") for record in records):
            complex_closure = sum(
                float(term["coefficient"])
                * record["components"]["complex"]["dg_kcal_per_mol"]
                for term, record in zip(terms, records)
            )
            solvent_closure = sum(
                float(term["coefficient"])
                * record["components"]["solvent"]["dg_kcal_per_mol"]
                for term, record in zip(terms, records)
            )
        cycles.append(
            {
                "id": cycle["id"],
                "status": "complete",
                "closure_kcal_per_mol": float(closure),
                "closure_error_kcal_per_mol": float(math.sqrt(variance)),
                "complex_closure_kcal_per_mol": complex_closure,
                "solvent_closure_kcal_per_mol": solvent_closure,
            }
        )

    bootstrap = {}
    if not repeated and int(config.get("node_bootstrap_samples", 0)) > 0:
        bootstrap = _node_bootstrap_cycles(
            observed,
            cycle_definitions,
            int(config.get("node_bootstrap_samples", 0)),
            float(config.get("temperature_k", 300.0)),
            int(config.get("random_seed", 2026)),
        )
    for cycle in cycles:
        if cycle["id"] in bootstrap:
            cycle["node_correlated_bootstrap"] = bootstrap[cycle["id"]]
    targets = [_target_result(target, fit) for target in _targets(config)]
    payload = {
        "schema_version": 2 if repeated else 1,
        "status": "completed" if fit["solved"] and not missing else "partial",
        "reference_node": reference,
        "observed_edge_count": len(observed),
        "expected_edge_count": len(config["edges"]),
        "matrix_rank": fit["matrix_rank"],
        "required_rank": fit["required_rank"],
        "graph_fit": {
            "chi_squared": fit["chi_squared"],
            "degrees_of_freedom": fit["degrees_of_freedom"],
            "reduced_chi_squared": fit["reduced_chi_squared"],
            "uncertainty_scale": fit["uncertainty_scale"],
            "primary_uncertainty": (
                "chi_squared_inflated" if repeated else "model_covariance"
            ),
        },
        "node_free_energies_kcal_per_mol": {
            node: {
                "value": fit["node_values"].get(node),
                "error": fit["node_errors"].get(node),
                "value_kcal_per_mol": fit["node_values"].get(node),
                "error_kcal_per_mol": fit["node_errors"].get(node),
                "model_error_kcal_per_mol": fit["node_model_errors"].get(node),
            }
            for node in nodes
        },
        "edges": [
            {key: value for key, value in edge.items() if not key.startswith("_")}
            for edge in observed
        ],
        "missing_edges": missing,
        "cycles": cycles,
        "targets": targets,
        "target": targets[0],
    }
    payload["reference_comparisons"] = _reference_comparisons(config, nodes, fit)
    return payload


def _single_network_records(config, root, replicate_id=None):
    observed = []
    missing = []
    for edge in config["edges"]:
        record, warning = _edge_result(root, edge)
        identifier = _edge_id(edge)
        if record is None:
            missing.append({
                "id": identifier,
                "directory": edge["directory"],
                "replicate_id": replicate_id,
                "warning": warning,
            })
        else:
            record["id"] = identifier
            record["replicate_id"] = replicate_id
            observed.append(record)
    return observed, missing


def _repeat_network(config, root):
    entries = config["replicate_networks"]
    if not entries:
        raise RBFENetworkError("replicate_networks must not be empty")
    base = None
    records = {}
    missing = []
    replicate_ids = []
    for index, entry in enumerate(entries, 1):
        if isinstance(entry, str):
            replicate_id = f"replicate_{index}"
            path = root / entry
        else:
            replicate_id = str(entry.get("id", f"replicate_{index}"))
            path = root / entry["path"]
        if replicate_id in replicate_ids:
            raise RBFENetworkError(f"duplicate replicate id {replicate_id}")
        replicate_ids.append(replicate_id)
        path = path.resolve()
        child = yaml.safe_load(path.read_text()) or {}
        if base is None:
            base = child
        else:
            for key in ("reference_node", "nodes", "edges"):
                if child.get(key) != base.get(key):
                    raise RBFENetworkError(
                        f"replicate {replicate_id} has inconsistent {key}"
                    )
        observed, absent = _single_network_records(child, path.parent, replicate_id)
        missing.extend(absent)
        for record in observed:
            records.setdefault(record["id"], []).append(record)

    merged = dict(base)
    for key in (
        "targets", "target", "target_references", "reference_datasets",
        "cycles", "auto_cycles", "node_bootstrap_samples", "temperature_k", "random_seed",
    ):
        if key in config:
            merged[key] = config[key]
    aggregate = []
    for edge in merged["edges"]:
        identifier = _edge_id(edge)
        replicates = records.get(identifier, [])
        if not replicates:
            continue
        summary = _random_effects(replicates)
        aggregate.append({
            "id": identifier,
            "ligand_a": str(edge["ligand_a"]),
            "ligand_b": str(edge["ligand_b"]),
            "directory": edge["directory"],
            **summary,
            "replicates": [
                {key: value for key, value in record.items() if not key.startswith("_")}
                for record in replicates
            ],
            "completed_replicate_count": sum(
                record.get("workflow_status") == "completed" for record in replicates
            ),
        })
    payload = _analyze_observed(merged, aggregate, missing, root, repeated=True)
    payload["replicate_networks"] = replicate_ids
    payload["expected_replicates_per_edge"] = len(replicate_ids)
    payload["replicate_result_count"] = sum(len(rows) for rows in records.values())
    payload["quality"] = {
        "edges_with_all_replicates": sum(
            edge["replicate_count"] == len(replicate_ids) for edge in aggregate
        ),
        "edges_with_partial_workflows": [
            edge["id"] for edge in aggregate
            if edge["completed_replicate_count"] < edge["replicate_count"]
        ],
        "warnings": [
            "Finite partial workflow results were retained as repeat estimates."
        ] if any(
            edge["completed_replicate_count"] < edge["replicate_count"]
            for edge in aggregate
        ) else [],
    }
    return payload


def analyze_network(config_path, output_path=None):
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = yaml.safe_load(config_path.read_text()) or {}
    if "replicate_networks" in config:
        payload = _repeat_network(config, root)
    else:
        observed, missing = _single_network_records(config, root)
        payload = _analyze_observed(config, observed, missing, root)
    output = Path(output_path).resolve() if output_path else root / "network_result.yaml"
    _write_yaml_atomic(output, payload)
    if "replicate_networks" in config:
        _write_csv_outputs(output.parent, payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit an RBFE graph from pair result.yaml files")
    parser.add_argument("network_yaml")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = analyze_network(args.network_yaml, args.output)
    print(
        f"Network {result['status']}: {result['observed_edge_count']}/"
        f"{result['expected_edge_count']} edges"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
