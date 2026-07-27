"""Artifact-based diagnostics for ATM AWH and endpoint REST2 sampling."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


DEFAULT_THRESHOLDS = {
    "min_adjacent_overlap": 0.03,
    "min_endpoint_effective_samples": 50.0,
    "min_rest2_hot_returns": 5,
    "min_uniform_occupancy_overlap": 0.8,
}


def read_state_trace(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_reduced_energies(path):
    path = Path(path)
    if not path.exists():
        return [], []
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return [], []
    return (
        [int(row[0]) for row in rows[1:]],
        [[float(value) for value in row[1:]] for row in rows[1:]],
    )


def read_bias_history(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mbar_weights(reduced_energies, sampled_states, free_energies, nstates):
    matrix = np.asarray(reduced_energies, dtype=float)
    states = np.asarray(sampled_states, dtype=int)
    free = np.asarray(free_energies, dtype=float)
    counts = np.bincount(states, minlength=nstates).astype(float)
    if (
        matrix.ndim != 2
        or matrix.shape != (len(states), nstates)
        or len(free) != nstates
        or np.any(counts == 0)
    ):
        return None
    log_denominator = logsumexp(
        np.log(counts)[None, :] + free[None, :] - matrix,
        axis=1,
    )
    weights = np.exp(free[None, :] - matrix - log_denominator[:, None])
    return weights, counts


def _stage_sequences(rows, nstates):
    sequences = {"adaptive": [], "production": []}
    for row in rows:
        stage = row.get("stage", "adaptive")
        if stage not in sequences:
            continue
        value = row.get("state")
        try:
            state = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= state < nstates:
            sequences[stage].append(state)
    return sequences


def _transition_counts(rows, nstates, fallback=None):
    by_stage = {
        stage: np.zeros((nstates, nstates), dtype=np.int64)
        for stage in ("adaptive", "production")
    }
    has_explicit_previous = False
    for row in rows:
        try:
            previous = int(row["previous_state"])
            selected = int(row["state"])
        except (KeyError, TypeError, ValueError):
            continue
        stage = row.get("stage", "adaptive")
        if stage in by_stage and 0 <= previous < nstates and 0 <= selected < nstates:
            by_stage[stage][previous, selected] += 1
            has_explicit_previous = True
    if not has_explicit_previous and fallback is not None:
        matrix = np.asarray(fallback, dtype=np.int64)
        if matrix.shape == (nstates, nstates):
            by_stage["aggregate"] = matrix
    return by_stage


def _selection_statistics(rows, nstates):
    records = [
        {
            "attempts": 0,
            "stays": 0,
            "expected_stay_sum": 0.0,
            "expected_stay_count": 0,
            "expected_left_sum": 0.0,
            "expected_left_count": 0,
            "expected_right_sum": 0.0,
            "expected_right_count": 0,
        }
        for _ in range(nstates)
    ]
    for row in rows:
        try:
            previous = int(row["previous_state"])
            selected = int(row["state"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= previous < nstates:
            continue
        record = records[previous]
        record["attempts"] += 1
        record["stays"] += int(previous == selected)
        for source, total_key, count_key in (
            ("stay_probability", "expected_stay_sum", "expected_stay_count"),
            ("left_probability", "expected_left_sum", "expected_left_count"),
            ("right_probability", "expected_right_sum", "expected_right_count"),
        ):
            try:
                value = float(row.get(source, ""))
            except (TypeError, ValueError):
                continue
            record[total_key] += value
            record[count_key] += 1
    return records


def _state_records(graph, sequences, selection_statistics, target=None):
    nstates = len(graph)
    records = []
    target = (
        np.full(nstates, 1.0 / nstates)
        if target is None
        else np.asarray(target, dtype=float)
    )
    totals = {stage: len(values) for stage, values in sequences.items()}
    for index, node in enumerate(graph):
        record = {
            "index": index,
            "name": node["name"],
            "kind": node["kind"],
            "atm_state": int(node["atm_state"]),
            "rest2": {key: float(value) for key, value in node["rest2"].items()},
            "adaptive_visits": int(sequences["adaptive"].count(index)),
            "production_visits": int(sequences["production"].count(index)),
            "target_probability": float(target[index]),
        }
        for stage in ("adaptive", "production"):
            record[f"{stage}_occupancy"] = (
                record[f"{stage}_visits"] / totals[stage] if totals[stage] else None
            )
        selection = selection_statistics[index]
        record.update(
            {
                "selection_attempts": selection["attempts"],
                "observed_stay_probability": (
                    selection["stays"] / selection["attempts"]
                    if selection["attempts"]
                    else None
                ),
                "expected_stay_probability": (
                    selection["expected_stay_sum"] / selection["expected_stay_count"]
                    if selection["expected_stay_count"]
                    else None
                ),
            }
        )
        records.append(record)
    return records


def _occupancy_overlap(sequence, nstates, target=None):
    if not sequence:
        return None
    counts = np.bincount(sequence, minlength=nstates).astype(float)
    observed = counts / counts.sum()
    target = (
        np.full(nstates, 1.0 / nstates)
        if target is None
        else np.asarray(target, dtype=float)
    )
    return float(np.minimum(observed, target).sum())


def _edge_records(
    graph, transition_matrices, selection_statistics, overlap=None
):
    aggregate = np.zeros((len(graph), len(graph)), dtype=np.int64)
    for matrix in transition_matrices.values():
        aggregate += matrix
    records = []
    for left in range(len(graph) - 1):
        right = left + 1
        left_total = int(aggregate[left].sum())
        right_total = int(aggregate[right].sum())
        forward = int(aggregate[left, right])
        reverse = int(aggregate[right, left])
        records.append(
            {
                "left_state": left,
                "left_name": graph[left]["name"],
                "right_state": right,
                "right_name": graph[right]["name"],
                "left_to_right_count": forward,
                "right_to_left_count": reverse,
                "left_to_right_probability": forward / left_total if left_total else None,
                "right_to_left_probability": reverse / right_total if right_total else None,
                "expected_left_to_right_probability": (
                    selection_statistics[left]["expected_right_sum"]
                    / selection_statistics[left]["expected_right_count"]
                    if selection_statistics[left]["expected_right_count"]
                    else None
                ),
                "expected_right_to_left_probability": (
                    selection_statistics[right]["expected_left_sum"]
                    / selection_statistics[right]["expected_left_count"]
                    if selection_statistics[right]["expected_left_count"]
                    else None
                ),
                "uwham_overlap_left_to_right": (
                    None if overlap is None else float(overlap[left, right])
                ),
                "uwham_overlap_right_to_left": (
                    None if overlap is None else float(overlap[right, left])
                ),
                "uwham_overlap_minimum": (
                    None
                    if overlap is None
                    else float(min(overlap[left, right], overlap[right, left]))
                ),
            }
        )
    return records


def _branch_excursions(sequence, physical, hottest, branch_states):
    entered = 0
    hottest_visits = 0
    complete_returns = 0
    active = False
    reached_hottest = False
    previous = physical
    branch = set(branch_states)
    for state in sequence:
        if previous == physical and state in branch:
            entered += 1
            active = True
            reached_hottest = False
        if active and state == hottest:
            hottest_visits += 1
            reached_hottest = True
        if active and state == physical:
            if reached_hottest:
                complete_returns += 1
            active = False
            reached_hottest = False
        previous = state
    return {
        "entries": entered,
        "hottest_state_visits_during_excursions": hottest_visits,
        "complete_physical_hottest_physical_returns": complete_returns,
        "unfinished_excursion": active,
    }


def _rest2_diagnostics(graph, sequences):
    a_physical = next(i for i, node in enumerate(graph) if node["name"] == "a_physical")
    b_physical = next(i for i, node in enumerate(graph) if node["name"] == "b_physical")
    branches = {
        "endpoint_a": {
            "physical": a_physical,
            "states": [i for i, node in enumerate(graph) if node["kind"] == "rest2_a"],
        },
        "endpoint_b": {
            "physical": b_physical,
            "states": [i for i, node in enumerate(graph) if node["kind"] == "rest2_b"],
        },
    }
    result = {}
    for label, branch in branches.items():
        states = branch["states"]
        if not states:
            result[label] = {
                "enabled": False,
                "states": [],
                "adaptive": None,
                "production": None,
            }
            continue
        hottest = min(states) if label == "endpoint_a" else max(states)
        state_rows = []
        for index in states:
            scale_key = "a" if label == "endpoint_a" else "b"
            scale = float(graph[index]["rest2"][scale_key])
            state_rows.append(
                {
                    "index": index,
                    "name": graph[index]["name"],
                    "scale": scale,
                    "effective_temperature_ratio": 1.0 / scale,
                    "adaptive_visits": int(sequences["adaptive"].count(index)),
                    "production_visits": int(sequences["production"].count(index)),
                }
            )
        result[label] = {
            "enabled": True,
            "physical_state": branch["physical"],
            "hottest_state": hottest,
            "states": state_rows,
            "adaptive": _branch_excursions(
                sequences["adaptive"], branch["physical"], hottest, states
            ),
            "production": _branch_excursions(
                sequences["production"], branch["physical"], hottest, states
            ),
        }
    return result


def _bias_stability(rows, graph):
    if len(rows) < 2:
        return {
            "history_records": len(rows),
            "maximum_last_update_kbt": None,
            "physical_ddg_last_change_kbt": None,
        }
    names = [node["name"] for node in graph]
    try:
        previous = np.asarray([float(rows[-2][name]) for name in names])
        latest = np.asarray([float(rows[-1][name]) for name in names])
    except (KeyError, TypeError, ValueError):
        return {
            "history_records": len(rows),
            "maximum_last_update_kbt": None,
            "physical_ddg_last_change_kbt": None,
        }
    a = names.index("a_physical")
    b = names.index("b_physical")
    return {
        "history_records": len(rows),
        "maximum_last_update_kbt": float(np.max(np.abs(latest - previous))),
        "physical_ddg_last_change_kbt": float((latest[b] - latest[a]) - (previous[b] - previous[a])),
    }


def analyze_awh_diagnostics(
    graph,
    *,
    trace_rows,
    sampled_states,
    reduced_energies,
    free_energies=None,
    fallback_transitions=None,
    bias_history=None,
    thresholds=None,
    friction=None,
    target=None,
):
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    nstates = len(graph)
    active_target = target
    if active_target is None and friction is not None:
        active_target = friction.get("active_target")
    if active_target is not None and len(active_target) != nstates:
        active_target = None
    sequences = _stage_sequences(trace_rows, nstates)
    selection_statistics = _selection_statistics(trace_rows, nstates)
    transition_matrices = _transition_counts(
        trace_rows, nstates, fallback=fallback_transitions
    )
    state_records = _state_records(
        graph, sequences, selection_statistics, target=active_target
    )
    overlap = None
    effective_samples = None
    if free_energies is not None:
        weighted = mbar_weights(
            reduced_energies, sampled_states, free_energies, nstates
        )
        if weighted is not None:
            weights, counts = weighted
            overlap = weights.T @ weights @ np.diag(counts)
            effective_samples = 1.0 / np.sum(weights**2, axis=0)
            for index, record in enumerate(state_records):
                record["uwham_effective_samples"] = float(effective_samples[index])
    edges = _edge_records(
        graph,
        transition_matrices,
        selection_statistics,
        overlap=overlap,
    )
    adjacent = [
        edge["uwham_overlap_minimum"]
        for edge in edges
        if edge["uwham_overlap_minimum"] is not None
    ]
    minimum_overlap = min(adjacent) if adjacent else None
    a_index = next(i for i, node in enumerate(graph) if node["name"] == "a_physical")
    b_index = next(i for i, node in enumerate(graph) if node["name"] == "b_physical")
    endpoint_ess = {
        "a_physical": (
            None if effective_samples is None else float(effective_samples[a_index])
        ),
        "b_physical": (
            None if effective_samples is None else float(effective_samples[b_index])
        ),
    }
    production_occupancy_overlap = _occupancy_overlap(
        sequences["production"], nstates
    )
    adaptive_occupancy_overlap = _occupancy_overlap(sequences["adaptive"], nstates)
    production_target_overlap = _occupancy_overlap(
        sequences["production"], nstates, active_target
    )
    adaptive_target_overlap = _occupancy_overlap(
        sequences["adaptive"], nstates, active_target
    )
    rest2 = _rest2_diagnostics(graph, sequences)
    warnings = []
    if minimum_overlap is not None and minimum_overlap < thresholds["min_adjacent_overlap"]:
        warnings.append(
            f"minimum adjacent UWHAM overlap {minimum_overlap:.4f} is below "
            f"{thresholds['min_adjacent_overlap']:.4f}"
        )
    if effective_samples is not None:
        for name, value in endpoint_ess.items():
            if value < thresholds["min_endpoint_effective_samples"]:
                warnings.append(
                    f"{name} effective samples {value:.1f} is below "
                    f"{thresholds['min_endpoint_effective_samples']:.1f}"
                )
    if (
        production_target_overlap is not None
        and production_target_overlap < thresholds["min_uniform_occupancy_overlap"]
    ):
        warnings.append(
            f"production occupancy overlap with the configured target "
            f"{production_target_overlap:.3f} is below "
            f"{thresholds['min_uniform_occupancy_overlap']:.3f}"
        )
    for label, branch in rest2.items():
        if not branch["enabled"]:
            continue
        stage = branch["production"] if sequences["production"] else branch["adaptive"]
        returns = stage["complete_physical_hottest_physical_returns"]
        if returns < thresholds["min_rest2_hot_returns"]:
            warnings.append(
                f"{label} has {returns} complete physical-hottest-physical REST2 "
                f"returns; at least {thresholds['min_rest2_hot_returns']} are required"
            )
    return {
        "schema_version": 1,
        "states": state_records,
        "edges": edges,
        "adaptive": {
            "moves": len(sequences["adaptive"]),
            "uniform_occupancy_overlap": adaptive_occupancy_overlap,
            "target_occupancy_overlap": adaptive_target_overlap,
        },
        "production": {
            "moves": len(sequences["production"]),
            "reduced_energy_samples": len(sampled_states),
            "uniform_occupancy_overlap": production_occupancy_overlap,
            "target_occupancy_overlap": production_target_overlap,
        },
        "uwham": {
            "overlap_matrix": None if overlap is None else overlap.tolist(),
            "minimum_adjacent_overlap": minimum_overlap,
            "endpoint_effective_samples": endpoint_ess,
        },
        "rest2": rest2,
        "friction": friction,
        "bias_stability": _bias_stability(bias_history or [], graph),
        "thresholds": thresholds,
        "quality_passed": not warnings,
        "warnings": warnings,
    }


def plot_awh_diagnostics(diagnostics, path, trace_rows=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = diagnostics["states"]
    names = [record["name"] for record in states]
    x = np.arange(len(states))
    adaptive = [record["adaptive_visits"] for record in states]
    production = [record["production_visits"] for record in states]
    edges = diagnostics["edges"]
    edge_x = np.arange(len(edges)) + 0.5
    edge_overlap = [
        np.nan
        if edge["uwham_overlap_minimum"] is None
        else edge["uwham_overlap_minimum"]
        for edge in edges
    ]
    edge_traffic = [
        edge["left_to_right_count"] + edge["right_to_left_count"] for edge in edges
    ]
    figure, axes = plt.subplots(2, 4, figsize=(21, 8), constrained_layout=True)
    trace_rows = trace_rows or []
    trace_moves = []
    trace_states = []
    for row in trace_rows:
        try:
            trace_moves.append(int(row["move"]))
            trace_states.append(int(row["state"]))
        except (KeyError, TypeError, ValueError):
            continue
    axes[0, 0].plot(trace_moves, trace_states, linewidth=0.7)
    axes[0, 0].set_title("State history")
    axes[0, 0].set_xlabel("AWH move")
    axes[0, 0].set_ylabel("State index")
    axes[0, 1].plot(x, adaptive, label="adaptive")
    axes[0, 1].plot(x, production, label="production")
    axes[0, 1].set_title("State visits")
    axes[0, 1].legend()
    axes[0, 2].bar(edge_x, edge_traffic)
    axes[0, 2].set_title("Adjacent transition traffic")
    friction = diagnostics.get("friction") or {}
    friction_states = friction.get("states") or []
    friction_values = [
        record.get("friction_kbt2_ps_per_state2", np.nan)
        for record in friction_states
    ]
    axes[0, 3].plot(x[: len(friction_values)], friction_values, marker="o")
    axes[0, 3].set_title("Generalized-force friction")
    axes[1, 0].plot(edge_x, edge_overlap, marker="o")
    axes[1, 0].axhline(
        diagnostics["thresholds"]["min_adjacent_overlap"],
        color="red",
        linestyle="--",
    )
    axes[1, 0].set_title("Minimum directional UWHAM overlap")
    ess = [
        record.get("uwham_effective_samples", np.nan) for record in states
    ]
    axes[1, 1].plot(x, ess, marker="o")
    axes[1, 1].set_title("UWHAM effective samples")
    stay = [
        record.get("observed_stay_probability", np.nan) for record in states
    ]
    expected_stay = [
        record.get("expected_stay_probability", np.nan) for record in states
    ]
    axes[1, 2].plot(x, stay, label="observed", marker="o")
    axes[1, 2].plot(x, expected_stay, label="expected", marker=".")
    axes[1, 2].set_title("State stay probability")
    axes[1, 2].legend()
    suggested = friction.get("suggested_target")
    active = friction.get("active_target")
    if suggested is not None:
        axes[1, 3].plot(x, suggested, label="suggested", marker="o")
    if active is not None:
        axes[1, 3].plot(x, active, label="active", marker=".")
    axes[1, 3].set_title("AWH target distribution")
    if suggested is not None or active is not None:
        axes[1, 3].legend()
    for axis in list(axes.flat)[1:]:
        axis.set_xlabel("AWH state index")
        axis.grid(alpha=0.2)
    axes[0, 1].set_xticks(x, names, rotation=90, fontsize=6)
    figure.savefig(path, dpi=150)
    plt.close(figure)
