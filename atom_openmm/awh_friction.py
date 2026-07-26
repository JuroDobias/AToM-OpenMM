"""Generalized-force friction diagnostics for the discrete AWH state graph."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


FRICTION_SAMPLE_FIELDS = [
    "move",
    "steps",
    "stage",
    "state",
    "state_name",
    "probability",
    "force_kbt_per_state",
]


def generalized_forces(reduced_energies, states, nstates):
    """Estimate d(reduced potential)/d(graph index) for selected states."""
    energies = {int(index): float(value) for index, value in reduced_energies.items()}
    forces = {}
    for state in states:
        state = int(state)
        if state == 0:
            forces[state] = energies[1] - energies[0]
        elif state == nstates - 1:
            forces[state] = energies[state] - energies[state - 1]
        else:
            forces[state] = 0.5 * (energies[state + 1] - energies[state - 1])
    return forces


def reconcile_friction_samples(path, maximum_move):
    """Remove rows written after the latest durable AWH checkpoint."""
    path = Path(path)
    if not path.exists():
        with path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FRICTION_SAMPLE_FIELDS).writeheader()
        return
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            row
            for row in reader
            if _integer(row.get("move"), default=maximum_move + 1) <= maximum_move
        ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRICTION_SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in FRICTION_SAMPLE_FIELDS}
            for row in rows
        )
    os.replace(temporary, path)


def append_friction_samples(
    path,
    *,
    move,
    steps,
    stage,
    graph,
    probabilities,
    forces,
):
    path = Path(path)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRICTION_SAMPLE_FIELDS)
        if write_header:
            writer.writeheader()
        for state in sorted(forces):
            writer.writerow(
                {
                    "move": int(move),
                    "steps": int(steps),
                    "stage": stage,
                    "state": int(state),
                    "state_name": graph[state]["name"],
                    "probability": float(probabilities[state]),
                    "force_kbt_per_state": float(forces[state]),
                }
            )


class FrictionAccumulator:
    """Online weighted autocorrelation statistics with bounded memory."""

    def __init__(self, nstates, maximum_lag, sample_interval_ps):
        self.nstates = int(nstates)
        self.maximum_lag = int(maximum_lag)
        self.sample_interval_ps = float(sample_interval_ps)
        shape = (self.nstates,)
        lag_shape = (self.nstates, self.maximum_lag)
        self.weight = np.zeros(shape)
        self.weight_squared = np.zeros(shape)
        self.force = np.zeros(shape)
        self.force_squared = np.zeros(shape)
        self.pair_weight = np.zeros(lag_shape)
        self.pair_product = np.zeros(lag_shape)
        self.pair_left = np.zeros(lag_shape)
        self.pair_right = np.zeros(lag_shape)
        self.history = []

    def update(self, samples):
        current = {
            int(state): (float(weight), float(force))
            for state, (weight, force) in samples.items()
            if weight > 0 and np.isfinite(weight) and np.isfinite(force)
        }
        for state, (weight, force) in current.items():
            self.weight[state] += weight
            self.weight_squared[state] += weight * weight
            self.force[state] += weight * force
            self.force_squared[state] += weight * force * force
            for lag, previous in enumerate(reversed(self.history), start=1):
                if lag > self.maximum_lag:
                    break
                if state not in previous:
                    continue
                old_weight, old_force = previous[state]
                pair_weight = weight * old_weight
                index = lag - 1
                self.pair_weight[state, index] += pair_weight
                self.pair_product[state, index] += (
                    pair_weight * old_force * force
                )
                self.pair_left[state, index] += pair_weight * old_force
                self.pair_right[state, index] += pair_weight * force
        self.history.append(current)
        if len(self.history) > self.maximum_lag:
            self.history.pop(0)

    def records(self, graph=None, minimum_effective_samples=0):
        records = []
        for state in range(self.nstates):
            total = self.weight[state]
            effective = (
                total * total / self.weight_squared[state]
                if self.weight_squared[state] > 0
                else 0.0
            )
            mean = self.force[state] / total if total > 0 else None
            variance = (
                max(0.0, self.force_squared[state] / total - mean * mean)
                if total > 0
                else None
            )
            correlations = []
            if variance is not None and variance > 0:
                for lag in range(self.maximum_lag):
                    pair_total = self.pair_weight[state, lag]
                    if pair_total <= 0:
                        correlations.append(None)
                        continue
                    covariance = (
                        self.pair_product[state, lag] / pair_total
                        - (self.pair_left[state, lag] / pair_total)
                        * (self.pair_right[state, lag] / pair_total)
                    )
                    correlations.append(float(covariance / variance))
            integrated = None
            friction = None
            if (
                variance is not None
                and variance > 0
                and effective >= minimum_effective_samples
            ):
                correlation_sum = 0.0
                for index in range(0, len(correlations), 2):
                    first = correlations[index]
                    second = (
                        correlations[index + 1]
                        if index + 1 < len(correlations)
                        else 0.0
                    )
                    if first is None or second is None or first + second <= 0:
                        break
                    correlation_sum += first + second
                integrated = self.sample_interval_ps * (
                    0.5 + correlation_sum
                )
                friction = variance * integrated
            records.append(
                {
                    "index": state,
                    "name": None if graph is None else graph[state]["name"],
                    "total_probability_weight": float(total),
                    "effective_samples": float(effective),
                    "mean_force_kbt_per_state": (
                        None if mean is None else float(mean)
                    ),
                    "force_variance_kbt2_per_state2": (
                        None if variance is None else float(variance)
                    ),
                    "integrated_autocorrelation_ps": (
                        None if integrated is None else float(integrated)
                    ),
                    "friction_kbt2_ps_per_state2": (
                        None if friction is None else float(friction)
                    ),
                    "lag_correlations": correlations,
                }
            )
        return records

    def to_dict(self):
        return {
            "schema_version": 1,
            "nstates": self.nstates,
            "maximum_lag": self.maximum_lag,
            "sample_interval_ps": self.sample_interval_ps,
            "weight": self.weight.tolist(),
            "weight_squared": self.weight_squared.tolist(),
            "force": self.force.tolist(),
            "force_squared": self.force_squared.tolist(),
            "pair_weight": self.pair_weight.tolist(),
            "pair_product": self.pair_product.tolist(),
            "pair_left": self.pair_left.tolist(),
            "pair_right": self.pair_right.tolist(),
            "history": [
                {
                    str(state): [float(weight), float(force)]
                    for state, (weight, force) in row.items()
                }
                for row in self.history
            ],
        }

    @classmethod
    def from_dict(cls, data):
        accumulator = cls(
            data["nstates"],
            data["maximum_lag"],
            data["sample_interval_ps"],
        )
        for name in (
            "weight",
            "weight_squared",
            "force",
            "force_squared",
            "pair_weight",
            "pair_product",
            "pair_left",
            "pair_right",
        ):
            setattr(accumulator, name, np.asarray(data[name], dtype=float))
        accumulator.history = [
            {
                int(state): (float(values[0]), float(values[1]))
                for state, values in row.items()
            }
            for row in data.get("history", [])
        ]
        return accumulator


def friction_target(records, maximum_relative_weight=5.0):
    values = np.asarray(
        [
            np.nan
            if record["friction_kbt2_ps_per_state2"] is None
            else record["friction_kbt2_ps_per_state2"]
            for record in records
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        return None
    scaling = np.sqrt(values)
    scaling /= np.exp(np.mean(np.log(scaling)))
    limit = float(maximum_relative_weight)
    scaling = np.clip(scaling, 1.0 / limit, limit)
    return scaling / scaling.sum()


def friction_summary(
    accumulator,
    graph,
    minimum_effective_samples,
    target=None,
    maximum_relative_weight=5.0,
):
    records = accumulator.records(
        graph, minimum_effective_samples=minimum_effective_samples
    )
    suggested = friction_target(
        records, maximum_relative_weight=maximum_relative_weight
    )
    return {
        "schema_version": 1,
        "coordinate": "ordered_state_graph_index",
        "sample_interval_ps": accumulator.sample_interval_ps,
        "maximum_correlation_lag_moves": accumulator.maximum_lag,
        "minimum_effective_samples": int(minimum_effective_samples),
        "states": records,
        "suggested_target": (
            None if suggested is None else suggested.tolist()
        ),
        "active_target": None if target is None else np.asarray(target).tolist(),
        "ready": suggested is not None,
    }


def _integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
