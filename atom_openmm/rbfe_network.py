from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import yaml

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


def analyze_network(config_path, output_path=None):
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = yaml.safe_load(config_path.read_text()) or {}
    nodes = [str(node["id"] if isinstance(node, dict) else node) for node in config["nodes"]]
    reference = str(config["reference_node"])
    if reference not in nodes or len(set(nodes)) != len(nodes):
        raise RBFENetworkError("network nodes must be unique and contain reference_node")

    observed = []
    missing = []
    for edge in config["edges"]:
        record, warning = _edge_result(root, edge)
        if record is None:
            missing.append({"directory": edge["directory"], "warning": warning})
        else:
            observed.append(record)

    fitted_nodes = [node for node in nodes if node != reference]
    node_column = {node: index for index, node in enumerate(fitted_nodes)}
    matrix = np.zeros((len(observed), len(fitted_nodes)), dtype=float)
    values = np.zeros(len(observed), dtype=float)
    errors = np.zeros(len(observed), dtype=float)
    for row, edge in enumerate(observed):
        if edge["ligand_a"] != reference:
            matrix[row, node_column[edge["ligand_a"]]] -= 1.0
        if edge["ligand_b"] != reference:
            matrix[row, node_column[edge["ligand_b"]]] += 1.0
        values[row] = edge["ddg_kcal_per_mol"]
        errors[row] = edge["ddg_error_kcal_per_mol"]

    rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
    solved = rank == len(fitted_nodes)
    node_values = {reference: 0.0}
    node_errors = {reference: 0.0}
    covariance = None
    if solved:
        weights = 1.0 / errors**2
        normal = matrix.T @ (weights[:, None] * matrix)
        covariance = np.linalg.inv(normal)
        estimates = covariance @ matrix.T @ (weights * values)
        for node, index in node_column.items():
            node_values[node] = float(estimates[index])
            node_errors[node] = float(math.sqrt(max(0.0, covariance[index, index])))
        predicted = matrix @ estimates
        for edge, prediction in zip(observed, predicted):
            edge["fitted_ddg_kcal_per_mol"] = float(prediction)
            edge["residual_kcal_per_mol"] = float(
                edge["ddg_kcal_per_mol"] - prediction
            )
            edge["standardized_residual"] = float(
                edge["residual_kcal_per_mol"] / edge["ddg_error_kcal_per_mol"]
            )

    edge_by_directory = {edge["directory"]: edge for edge in observed}
    cycles = []
    for cycle in config.get("cycles", []):
        terms = cycle["terms"]
        records = [edge_by_directory.get(term["edge"]) for term in terms]
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

    bootstrap = _node_bootstrap_cycles(
        observed,
        config.get("cycles", []),
        int(config.get("node_bootstrap_samples", 0)),
        float(config.get("temperature_k", 300.0)),
        int(config.get("random_seed", 2026)),
    ) if int(config.get("node_bootstrap_samples", 0)) > 0 else {}
    for cycle in cycles:
        if cycle["id"] in bootstrap:
            cycle["node_correlated_bootstrap"] = bootstrap[cycle["id"]]

    target_a, target_b = map(str, config["target"])
    target = {
        "ligand_a": target_a,
        "ligand_b": target_b,
        "ddg_kcal_per_mol": None,
        "ddg_error_kcal_per_mol": None,
    }
    if solved:
        target["ddg_kcal_per_mol"] = node_values[target_b] - node_values[target_a]
        vector = np.zeros(len(fitted_nodes), dtype=float)
        if target_a != reference:
            vector[node_column[target_a]] -= 1.0
        if target_b != reference:
            vector[node_column[target_b]] += 1.0
        target["ddg_error_kcal_per_mol"] = float(
            math.sqrt(max(0.0, vector @ covariance @ vector))
        )
    target["references"] = config.get("target_references", {})

    payload = {
        "schema_version": 1,
        "status": "completed" if solved and not missing else "partial",
        "reference_node": reference,
        "observed_edge_count": len(observed),
        "expected_edge_count": len(config["edges"]),
        "matrix_rank": rank,
        "required_rank": len(fitted_nodes),
        "node_free_energies_kcal_per_mol": {
            node: {
                "value": node_values.get(node),
                "error": node_errors.get(node),
            }
            for node in nodes
        },
        "edges": [
            {key: value for key, value in edge.items() if not key.startswith("_")}
            for edge in observed
        ],
        "missing_edges": missing,
        "cycles": cycles,
        "target": target,
    }
    output = Path(output_path).resolve() if output_path else root / "network_result.yaml"
    _write_yaml_atomic(output, payload)
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
