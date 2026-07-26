"""Single-walker accelerated weight histogram sampling over ATM states."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from openmm import unit
from scipy.special import logsumexp

from atom_openmm.abfe_structprep import set_platform
from atom_openmm.awh_analysis import (
    DEFAULT_THRESHOLDS,
    analyze_awh_diagnostics,
    plot_awh_diagnostics,
    read_bias_history,
    read_reduced_energies,
    read_state_trace,
)
from atom_openmm.awh_friction import (
    FrictionAccumulator,
    append_friction_samples,
    friction_summary,
    friction_target,
    generalized_forces,
    reconcile_friction_samples,
)
from atom_openmm.awh_global_gibbs import OMMWorkerAWHGlobalGibbs
from atom_openmm.awh_trajectory import StateTaggedXTCReporter, write_subset_topology
from atom_openmm.equilibration import AmberMaskResolver
from atom_openmm.neqti import build_atm_state_parameters, split_two_leg_paths
from atom_openmm.ommsystem import OMMSystemRBFE
from atom_openmm.ommworker import OMMWorkerATMSync
from atom_openmm.rest2 import set_multi_rest2_scales
from atom_openmm.rest2_exchange import set_atm_state


R_KJ_MOL_K = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
    unit.kilojoule_per_mole / unit.kelvin
)


class AWHConfigError(ValueError):
    pass


def normalize_awh_options(workflow, atom_options):
    raw = workflow.get("awh") or {}
    if not isinstance(raw, dict):
        raise AWHConfigError("workflow.awh must be a mapping")
    states = build_atm_state_parameters(atom_options)
    split_two_leg_paths(states)
    physical_temperature = float(atom_options["TEMPERATURES"][0])
    rest = raw.get("rest2") or {}
    if not isinstance(rest, dict):
        raise AWHConfigError("workflow.awh.rest2 must be a mapping")
    temperatures = [
        float(value)
        for value in rest.get(
            "effective_temperatures_k",
            [physical_temperature * 2.0 ** (index / 5.0) for index in range(6)],
        )
    ]
    if temperatures[0] != physical_temperature or any(
        b <= a for a, b in zip(temperatures, temperatures[1:])
    ):
        raise AWHConfigError(
            "workflow.awh.rest2.effective_temperatures_k must start at the "
            "physical temperature and increase strictly"
        )
    adaptive = raw.get("adaptive") or {}
    production = raw.get("production") or {}
    analysis = raw.get("analysis") or {}
    if not isinstance(analysis, dict):
        raise AWHConfigError("workflow.awh.analysis must be a mapping")
    trajectory = analysis.get("trajectory") or {}
    friction = analysis.get("friction") or {}
    thresholds = analysis.get("thresholds") or {}
    metric_target = adaptive.get("metric_target") or {}
    state_sampling = raw.get("state_sampling") or {}
    if (
        not isinstance(trajectory, dict)
        or not isinstance(friction, dict)
        or not isinstance(thresholds, dict)
        or not isinstance(metric_target, dict)
        or not isinstance(state_sampling, dict)
    ):
        raise AWHConfigError(
            "AWH trajectory, friction, thresholds, and metric_target settings "
            "must be mappings"
        )
    settings = {
        "state_move_interval_steps": int(raw.get("state_move_interval_steps", 500)),
        "atm_state_count": int(raw.get("atm_state_count", len(states))),
        "random_seed": int(raw.get("random_seed", 2026)),
        "target_distribution": str(raw.get("target_distribution", "uniform")).lower(),
        "initial_error_kj_per_mol": float(raw.get("initial_error_kj_per_mol", 50.0)),
        "diffusion_per_ps": float(raw.get("diffusion_per_ps", 0.005)),
        "resume": bool(raw.get("resume", True)),
        "start_state": str(raw.get("start_state", "a")).lower(),
        "start_annealing_steps_per_state": int(
            raw.get("start_annealing_steps_per_state", 10000)
        ),
        "initial_state_file": raw.get("initial_state_file"),
        "checkpoint_interval_moves": int(raw.get("checkpoint_interval_moves", 100)),
        "state_sampling": {
            "method": str(
                state_sampling.get("method", "legacy_local_gibbs")
            ).lower(),
            "validation_interval_moves": int(
                state_sampling.get("validation_interval_moves", 1000)
            ),
            "validation_states_per_check": int(
                state_sampling.get("validation_states_per_check", 4)
            ),
            "validation_tolerance_kj_per_mol": float(
                state_sampling.get("validation_tolerance_kj_per_mol", 0.05)
            ),
        },
        "adaptive": {
            "min_steps": int(adaptive.get("min_steps", 1_000_000)),
            "max_steps": int(adaptive.get("max_steps", 20_000_000)),
            "min_round_trips": int(adaptive.get("min_round_trips", 10)),
            "min_visits_per_state": int(adaptive.get("min_visits_per_state", 100)),
            "covering_fraction": float(adaptive.get("covering_fraction", 0.8)),
            "learning_rate_kbt": float(
                adaptive.get("learning_rate_kbt", 0.1)
            ),
            "metric_target": {
                "enabled": bool(metric_target.get("enabled", False)),
                "min_round_trips": int(
                    metric_target.get("min_round_trips", 2)
                ),
                "update_interval_moves": int(
                    metric_target.get("update_interval_moves", 1000)
                ),
                "smoothing": float(metric_target.get("smoothing", 0.2)),
                "max_relative_weight": float(
                    metric_target.get("max_relative_weight", 5.0)
                ),
            },
        },
        "production": {
            "steps": int(production.get("steps", 5_000_000)),
            "reduced_energy_interval_moves": int(
                production.get("reduced_energy_interval_moves", 10)
            ),
            "bootstrap_samples": int(production.get("bootstrap_samples", 500)),
        },
        "rest2": {
            "enabled": bool(rest.get("enabled", True)),
            "effective_temperatures_k": temperatures,
            "endpoint_a_solute": str(
                rest.get("endpoint_a_solute", '#unbound:"*"')
            ),
            "endpoint_b_solute": str(
                rest.get("endpoint_b_solute", '#unbound:"*"')
            ),
        },
        "analysis": {
            "trajectory": {
                "enabled": bool(trajectory.get("enabled", True)),
                "interval_moves": int(trajectory.get("interval_moves", 10)),
                "atom_selection": str(
                    trajectory.get(
                        "atom_selection", "!:HOH,WAT,NA,CL,K,CA"
                    )
                ),
            },
            "friction": {
                "enabled": bool(friction.get("enabled", False)),
                "max_correlation_lag_moves": int(
                    friction.get("max_correlation_lag_moves", 50)
                ),
                "min_effective_samples": int(
                    friction.get("min_effective_samples", 200)
                ),
            },
            "thresholds": {
                key: (
                    int(thresholds.get(key, value))
                    if key == "min_rest2_hot_returns"
                    else float(thresholds.get(key, value))
                )
                for key, value in DEFAULT_THRESHOLDS.items()
            },
        },
        "platform": raw.get("platform"),
    }
    if settings["target_distribution"] != "uniform":
        raise AWHConfigError("workflow.awh.target_distribution currently must be 'uniform'")
    if settings["start_state"] not in {"a", "b"}:
        raise AWHConfigError("workflow.awh.start_state must be 'a' or 'b'")
    if settings["state_sampling"]["method"] not in {
        "legacy_local_gibbs",
        "hybrid_global_gibbs",
    }:
        raise AWHConfigError(
            "workflow.awh.state_sampling.method must be "
            "'legacy_local_gibbs' or 'hybrid_global_gibbs'"
        )
    positive = [
        settings["state_move_interval_steps"],
        settings["atm_state_count"],
        settings["checkpoint_interval_moves"],
        settings["start_annealing_steps_per_state"],
        settings["adaptive"]["min_steps"],
        settings["adaptive"]["max_steps"],
        settings["production"]["steps"],
        settings["production"]["reduced_energy_interval_moves"],
        settings["analysis"]["trajectory"]["interval_moves"],
        settings["analysis"]["friction"]["max_correlation_lag_moves"],
        settings["analysis"]["friction"]["min_effective_samples"],
        settings["adaptive"]["metric_target"]["update_interval_moves"],
        settings["state_sampling"]["validation_interval_moves"],
        settings["state_sampling"]["validation_states_per_check"],
    ]
    if any(value < 1 for value in positive):
        raise AWHConfigError("AWH step and interval settings must be positive")
    if settings["adaptive"]["min_steps"] > settings["adaptive"]["max_steps"]:
        raise AWHConfigError("workflow.awh.adaptive.min_steps cannot exceed max_steps")
    if settings["atm_state_count"] < 4:
        raise AWHConfigError("workflow.awh.atm_state_count must be at least 4")
    if not 0 < settings["adaptive"]["covering_fraction"] <= 1:
        raise AWHConfigError("workflow.awh.adaptive.covering_fraction must be in (0, 1]")
    if (
        settings["initial_error_kj_per_mol"] <= 0
        or settings["diffusion_per_ps"] <= 0
        or settings["adaptive"]["learning_rate_kbt"] <= 0
        or settings["state_sampling"]["validation_tolerance_kj_per_mol"] <= 0
    ):
        raise AWHConfigError("AWH initial error and diffusion must be positive")
    metric_target = settings["adaptive"]["metric_target"]
    if (
        metric_target["min_round_trips"] < 0
        or not 0 < metric_target["smoothing"] <= 1
        or metric_target["max_relative_weight"] < 1
    ):
        raise AWHConfigError("workflow.awh.adaptive.metric_target settings are invalid")
    if metric_target["enabled"] and not settings["analysis"]["friction"]["enabled"]:
        raise AWHConfigError(
            "friction analysis must be enabled when metric target scaling is enabled"
        )
    quality = settings["analysis"]["thresholds"]
    if (
        quality["min_adjacent_overlap"] <= 0
        or quality["min_endpoint_effective_samples"] <= 0
        or quality["min_rest2_hot_returns"] < 0
        or not 0 < quality["min_uniform_occupancy_overlap"] <= 1
    ):
        raise AWHConfigError("workflow.awh.analysis thresholds are invalid")
    return settings


def densify_atm_states(states, target_count):
    """Interpolate each ATM half path while preserving the M+/M- boundary."""
    target_count = int(target_count)
    if target_count == len(states):
        return [deepcopy(state) for state in states]
    paths = split_two_leg_paths(states)
    midpoint_plus = paths["leg_a_forward"][-1]
    midpoint_minus = paths["leg_b_reverse"][0]
    first = states[: midpoint_plus + 1]
    second = states[midpoint_minus:]
    original_edges = [len(first) - 1, len(second) - 1]
    available_edges = target_count - 2
    first_edges = max(
        1,
        int(round(available_edges * original_edges[0] / sum(original_edges))),
    )
    second_edges = available_edges - first_edges
    if second_edges < 1:
        second_edges = 1
        first_edges = available_edges - 1

    categorical = {"atmdirection", "atmintermediate"}

    def interpolate(path, count):
        locations = np.linspace(0.0, len(path) - 1, count)
        result = []
        for location in locations:
            lower = min(int(math.floor(location)), len(path) - 1)
            upper = min(lower + 1, len(path) - 1)
            fraction = location - lower
            state = {}
            for key, value in path[lower].items():
                if key in categorical:
                    state[key] = deepcopy(
                        path[lower if fraction < 0.5 else upper][key]
                    )
                else:
                    state[key] = (
                        value * (1.0 - fraction)
                        + path[upper][key] * fraction
                    )
            result.append(state)
        result[0] = deepcopy(path[0])
        result[-1] = deepcopy(path[-1])
        return result

    dense_first = interpolate(first, first_edges + 1)
    dense_second = interpolate(second, second_edges + 1)
    result = dense_first + dense_second
    if len(result) != target_count:
        raise AWHConfigError("internal AWH schedule interpolation count mismatch")
    split_two_leg_paths(result)
    return result


def build_awh_state_graph(atm_states, settings):
    temperatures = settings["rest2"]["effective_temperatures_k"]
    physical_temperature = temperatures[0]
    graph = []
    if settings["rest2"]["enabled"]:
        for temperature in reversed(temperatures[1:]):
            graph.append(
                {
                    "name": f"a_hot_{temperature:g}",
                    "atm_state": 0,
                    "rest2": {"a": physical_temperature / temperature, "b": 1.0},
                    "kind": "rest2_a",
                    "rest2_region": "a",
                    "rest2_scale": physical_temperature / temperature,
                    "effective_temperature_k": temperature,
                }
            )
    for index, state in enumerate(atm_states):
        graph.append(
            {
                "name": "a_physical"
                if index == 0
                else "b_physical"
                if index == len(atm_states) - 1
                else f"atm_{index}",
                "atm_state": index,
                "rest2": {"a": 1.0, "b": 1.0},
                "kind": "physical" if index in (0, len(atm_states) - 1) else "atm",
                "rest2_region": None,
                "rest2_scale": 1.0,
                "effective_temperature_k": physical_temperature,
            }
        )
    if settings["rest2"]["enabled"]:
        for temperature in temperatures[1:]:
            graph.append(
                {
                    "name": f"b_hot_{temperature:g}",
                    "atm_state": len(atm_states) - 1,
                    "rest2": {"a": 1.0, "b": physical_temperature / temperature},
                    "kind": "rest2_b",
                    "rest2_region": "b",
                    "rest2_scale": physical_temperature / temperature,
                    "effective_temperature_k": temperature,
                }
            )
    return graph


class AWHBias:
    """Free-energy bias estimate for the discrete AWH state graph."""

    def __init__(self, nstates, initial_histogram_size, target=None):
        self.target = np.full(nstates, 1.0 / nstates) if target is None else np.asarray(target)
        self.free_energy = np.zeros(nstates)
        self.reference = float(initial_histogram_size) * self.target
        self.covering = np.zeros(nstates)
        self.visits = np.zeros(nstates, dtype=np.int64)
        self.updates = 0

    @property
    def bias(self):
        return self.free_energy + np.log(self.target)

    def probabilities(self, reduced_energies, indices):
        log_weights = np.asarray(
            [self.bias[index] - energy for index, energy in zip(indices, reduced_energies)]
        )
        return np.exp(log_weights - logsumexp(log_weights))

    def update(self, indices, probabilities, learning_rate_kbt=None):
        sampled = np.zeros_like(self.free_energy)
        sampled[np.asarray(indices, dtype=int)] = probabilities
        if learning_rate_kbt is not None:
            self.free_energy -= float(learning_rate_kbt) * (
                sampled - self.target
            )
            self.covering += sampled
            self.updates += 1
            self.free_energy -= self.free_energy[0]
            return
        denominator = self.reference + self.target
        self.free_energy -= np.log((self.reference + sampled) / denominator)
        self.reference += self.target
        self.covering += sampled
        self.updates += 1
        self.free_energy -= self.free_energy[0]

    def to_dict(self):
        return {
            "target": self.target.tolist(),
            "free_energy": self.free_energy.tolist(),
            "reference": self.reference.tolist(),
            "covering": self.covering.tolist(),
            "visits": self.visits.tolist(),
            "updates": self.updates,
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            len(data["free_energy"]),
            1.0,
            target=np.asarray(
                data.get(
                    "target",
                    np.full(len(data["free_energy"]), 1.0 / len(data["free_energy"])),
                ),
                dtype=float,
            ),
        )
        obj.free_energy = np.asarray(data["free_energy"], dtype=float)
        obj.reference = np.asarray(data["reference"], dtype=float)
        obj.covering = np.asarray(data["covering"], dtype=float)
        obj.visits = np.asarray(data["visits"], dtype=np.int64)
        obj.updates = int(data["updates"])
        return obj


class GlobalGibbsDiagnostics:
    """Online diagnostics for deciding whether truncated Gibbs would suffice."""

    RADII = (1, 2, 4, 8, 16)

    def __init__(self):
        self.moves = 0
        self.jump_counts = {}
        self.jump_distance_sum = 0.0
        self.jump_distance_squared_sum = 0.0
        self.expected_jump_distance_sum = 0.0
        self.radius_mass_sum = {radius: 0.0 for radius in self.RADII}
        self.radius_mass_99 = {radius: 0 for radius in self.RADII}
        self.radius_mass_999 = {radius: 0 for radius in self.RADII}
        self.md_seconds = 0.0
        self.energy_seconds = 0.0
        self.selection_seconds = 0.0
        self.validation_checks = 0
        self.maximum_validation_error_kj_per_mol = 0.0

    def update(self, previous, selected, probabilities):
        probabilities = np.asarray(probabilities, dtype=float)
        indices = np.arange(len(probabilities))
        distances = np.abs(indices - int(previous))
        jump = abs(int(selected) - int(previous))
        self.moves += 1
        self.jump_counts[jump] = self.jump_counts.get(jump, 0) + 1
        self.jump_distance_sum += jump
        self.jump_distance_squared_sum += jump * jump
        expected = float(np.dot(probabilities, distances))
        self.expected_jump_distance_sum += expected
        for radius in self.RADII:
            mass = float(probabilities[distances <= radius].sum())
            self.radius_mass_sum[radius] += mass
            self.radius_mass_99[radius] += int(mass >= 0.99)
            self.radius_mass_999[radius] += int(mass >= 0.999)
        return expected

    def record_validation(self, maximum_error):
        self.validation_checks += 1
        self.maximum_validation_error_kj_per_mol = max(
            self.maximum_validation_error_kj_per_mol,
            float(maximum_error),
        )

    def to_dict(self):
        moves = max(1, self.moves)
        mean = self.jump_distance_sum / moves
        variance = max(
            0.0,
            self.jump_distance_squared_sum / moves - mean * mean,
        )
        return {
            "schema_version": 1,
            "moves": self.moves,
            "jump_distance_histogram": {
                str(key): value for key, value in sorted(self.jump_counts.items())
            },
            "mean_jump_distance": mean if self.moves else None,
            "jump_distance_std": math.sqrt(variance) if self.moves else None,
            "mean_conditional_expected_jump_distance": (
                self.expected_jump_distance_sum / moves if self.moves else None
            ),
            "truncated_probability_mass": {
                str(radius): {
                    "mean": self.radius_mass_sum[radius] / moves,
                    "fraction_at_least_0.99": self.radius_mass_99[radius] / moves,
                    "fraction_at_least_0.999": self.radius_mass_999[radius] / moves,
                }
                for radius in self.RADII
            },
            "timing_seconds": {
                "md": self.md_seconds,
                "energy_scan": self.energy_seconds,
                "selection": self.selection_seconds,
                "total_per_move": (
                    (
                        self.md_seconds
                        + self.energy_seconds
                        + self.selection_seconds
                    )
                    / moves
                    if self.moves
                    else None
                ),
            },
            "validation": {
                "checks": self.validation_checks,
                "maximum_error_kj_per_mol": self.maximum_validation_error_kj_per_mol,
            },
        }

    @classmethod
    def from_dict(cls, data):
        value = cls()
        value.moves = int(data.get("moves", 0))
        value.jump_counts = {
            int(key): int(count)
            for key, count in data.get("jump_distance_histogram", {}).items()
        }
        mean = data.get("mean_jump_distance")
        std = data.get("jump_distance_std")
        if mean is not None:
            value.jump_distance_sum = float(mean) * value.moves
            value.jump_distance_squared_sum = (
                float(std or 0.0) ** 2 + float(mean) ** 2
            ) * value.moves
        expected = data.get("mean_conditional_expected_jump_distance")
        if expected is not None:
            value.expected_jump_distance_sum = float(expected) * value.moves
        for radius in value.RADII:
            row = data.get("truncated_probability_mass", {}).get(str(radius), {})
            value.radius_mass_sum[radius] = float(row.get("mean", 0.0)) * value.moves
            value.radius_mass_99[radius] = round(
                float(row.get("fraction_at_least_0.99", 0.0)) * value.moves
            )
            value.radius_mass_999[radius] = round(
                float(row.get("fraction_at_least_0.999", 0.0)) * value.moves
            )
        timing = data.get("timing_seconds", {})
        value.md_seconds = float(timing.get("md", 0.0))
        value.energy_seconds = float(timing.get("energy_scan", 0.0))
        value.selection_seconds = float(timing.get("selection", 0.0))
        validation = data.get("validation", {})
        value.validation_checks = int(validation.get("checks", 0))
        value.maximum_validation_error_kj_per_mol = float(
            validation.get("maximum_error_kj_per_mol", 0.0)
        )
        return value


def estimate_mbar(reduced_energies, sampled_states, nstates, max_iterations=10000):
    u_nk = np.asarray(reduced_energies, dtype=float)
    sampled_states = np.asarray(sampled_states, dtype=int)
    counts = np.bincount(sampled_states, minlength=nstates).astype(float)
    if not len(u_nk) or np.any(counts == 0):
        return None
    f = np.zeros(nstates)
    log_counts = np.log(counts)
    for _ in range(max_iterations):
        log_denominator = logsumexp(log_counts + f - u_nk, axis=1)
        updated = -logsumexp(-u_nk - log_denominator[:, None], axis=0)
        updated -= updated[0]
        if np.max(np.abs(updated - f)) < 1e-10:
            f = updated
            break
        f = updated
    return f


def bootstrap_mbar_ddg(
    reduced_energies,
    sampled_states,
    nstates,
    state_a,
    state_b,
    beta,
    samples,
    rng,
):
    if samples < 1:
        return None
    matrix = np.asarray(reduced_energies, dtype=float)
    states = np.asarray(sampled_states, dtype=int)
    groups = [np.flatnonzero(states == index) for index in range(nstates)]
    if any(not len(group) for group in groups):
        return None
    estimates = []
    for _ in range(samples):
        selected = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        estimate = estimate_mbar(matrix[selected], states[selected], nstates)
        if estimate is not None:
            estimates.append((estimate[state_b] - estimate[state_a]) / beta / 4.184)
    return None if len(estimates) < 2 else float(np.std(estimates, ddof=1))


def _protocol_signature(options, settings):
    def scalar(value):
        raw = getattr(value, "_value", value)
        return float(raw)

    dynamics_settings = deepcopy(
        {key: value for key, value in settings.items() if key != "analysis"}
    )
    metric_target = dynamics_settings.get("adaptive", {}).get("metric_target")
    if metric_target is not None and not metric_target.get("enabled", False):
        dynamics_settings["adaptive"].pop("metric_target")
    state_sampling = dynamics_settings.get("state_sampling")
    if state_sampling is not None and state_sampling.get("method") == "legacy_local_gibbs":
        dynamics_settings.pop("state_sampling")
    payload = {
        "schema_version": 1,
        "schedule": [
            {key: scalar(value) for key, value in state.items()}
            for state in build_atm_state_parameters(options)
        ],
        "settings": dynamics_settings,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_yaml(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _neighbors(index, nstates):
    return list(range(max(0, index - 1), min(nstates, index + 2)))


def _ensure_trace_schema(path, fields, initial_state):
    if not path.exists():
        with path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
        return
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        existing_fields = reader.fieldnames or []
        rows = list(reader)
    if existing_fields == fields:
        return
    previous = int(initial_state)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            migrated = {field: row.get(field, "") for field in fields}
            migrated["previous_state"] = row.get("previous_state", previous)
            try:
                previous = int(row["state"])
            except (KeyError, TypeError, ValueError):
                pass
            writer.writerow(migrated)
    os.replace(temporary, path)


def run_awh(options, awh_options=None, progress_callback=None):
    settings = awh_options or normalize_awh_options({"awh": {}}, options)
    logger = logging.getLogger("atom_openmm.awh")
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO)
    basename = options["BASENAME"]
    atm_states = densify_atm_states(
        build_atm_state_parameters(options), settings["atm_state_count"]
    )
    graph = build_awh_state_graph(atm_states, settings)
    signature = _protocol_signature(options, settings)
    manifest_path = Path("awh_protocol.yaml")
    state_path = Path("awh_checkpoint.yaml")
    xml_path = Path("awh_checkpoint.xml")
    trace_path = Path("awh_state_trace.csv")
    matrix_path = Path("awh_reduced_energies.csv")
    bias_path = Path("awh_bias_history.csv")
    diagnostics_path = Path("awh_diagnostics.yaml")
    diagnostics_plot_path = Path("awh_diagnostics.png")
    trajectory_path = Path("awh_trajectory.xtc")
    trajectory_topology_path = Path("awh_trajectory_topology.pdb")
    trajectory_frames_path = Path("awh_trajectory_frames.csv")
    friction_samples_path = Path("awh_friction_samples.csv")
    friction_path = Path("awh_friction.yaml")
    existing = state_path.exists() or xml_path.exists() or trace_path.exists()
    if settings["resume"] and existing:
        if not manifest_path.exists():
            raise AWHConfigError("AWH resume artifacts exist without awh_protocol.yaml")
        with manifest_path.open() as handle:
            if (yaml.safe_load(handle) or {}).get("signature") != signature:
                raise AWHConfigError("existing AWH artifacts use a different protocol")
    elif not settings["resume"] and existing:
        for path in (
            state_path,
            xml_path,
            trace_path,
            matrix_path,
            bias_path,
            diagnostics_path,
            diagnostics_plot_path,
            trajectory_path,
            trajectory_topology_path,
            trajectory_frames_path,
            friction_samples_path,
            friction_path,
        ):
            path.unlink(missing_ok=True)
    _atomic_yaml(
        manifest_path,
        {
            "schema_version": 1,
            "signature": signature,
            "graph": graph,
            "settings": settings,
        },
    )

    system_options = deepcopy(options)
    initial_file = (
        settings["initial_state_file"]
        or options.get("NEQTI_INITIAL_STATE_FILE")
        or options.get("INITIAL_STATE_FILE")
        or f"{basename}_equil.xml"
    )
    if not Path(initial_file).exists():
        raise AWHConfigError(f"AWH initial state does not exist: {initial_file}")
    system_options["INITIAL_STATE_FILE"] = str(initial_file)
    system_options["REST2_ENABLED"] = settings["rest2"]["enabled"]
    if settings["rest2"]["enabled"]:
        system_options["AWH_REST2_REGIONS"] = {
            "a": {
                "endpoint": "a",
                "selection": settings["rest2"]["endpoint_a_solute"],
            },
            "b": {
                "endpoint": "b",
                "selection": settings["rest2"]["endpoint_b_solute"],
            },
        }
    ommsystem = OMMSystemRBFE(
        basename, system_options, basename + ".pdb", basename + "_sys.xml", logger
    )
    platform_options = deepcopy(options)
    if settings.get("platform"):
        platform_options["OPENMM_PLATFORM"] = settings["platform"]
    platform, platform_properties = set_platform(platform_options)
    node_info = {
        "node_name": "localhost",
        "arch": platform.getName(),
        "slot_number": "0:0",
        "threads_number": "1",
    }
    global_sampling = (
        settings["state_sampling"]["method"] == "hybrid_global_gibbs"
    )
    worker_class = OMMWorkerAWHGlobalGibbs if global_sampling else OMMWorkerATMSync
    worker_kwargs = {}
    if global_sampling:
        worker_kwargs.update(atm_states=atm_states, graph=graph)
    worker = worker_class(
        basename,
        ommsystem,
        system_options,
        node_info=node_info,
        compute=False,
        logger=logger,
        **worker_kwargs,
    )
    worker.platform = platform
    worker.platform_properties = platform_properties

    def apply_node(index):
        node = graph[index]
        set_atm_state(worker.context, ommsystem, atm_states[node["atm_state"]])
        if settings["rest2"]["enabled"]:
            set_multi_rest2_scales(worker.context, node["rest2"], ommsystem.rest2_system)

    def energy(index):
        apply_node(index)
        value = worker.context.getState(getEnergy=True).getPotentialEnergy()
        return value.value_in_unit(unit.kilojoule_per_mole)

    def validate_global_energies(analytical, indices):
        direct = []
        for index in indices:
            direct.append(energy(index))
        apply_node(current)
        errors = np.abs(np.asarray(direct) - np.asarray(analytical)[indices])
        maximum = float(np.max(errors)) if len(errors) else 0.0
        global_diagnostics.record_validation(maximum)
        tolerance = settings["state_sampling"][
            "validation_tolerance_kj_per_mol"
        ]
        if maximum > tolerance:
            worst = int(indices[int(np.argmax(errors))])
            raise AWHConfigError(
                "hybrid global-Gibbs energy validation failed for "
                f"{graph[worst]['name']}: error {maximum:.6g} kJ/mol exceeds "
                f"{tolerance:.6g} kJ/mol"
            )

    beta = 1.0 / (R_KJ_MOL_K * float(options["TEMPERATURES"][0]))
    dynamics_integrator = (
        worker.equilibrium_integrator if global_sampling else worker.integrator
    )
    timestep_ps = dynamics_integrator.getStepSize().value_in_unit(unit.picosecond)
    initial_histogram = max(
        1.0,
        1.0
        / (
            settings["diffusion_per_ps"]
            * settings["state_move_interval_steps"]
            * timestep_ps
            * (settings["initial_error_kj_per_mol"] * beta) ** 2
        ),
    )
    a_index = next(i for i, node in enumerate(graph) if node["name"] == "a_physical")
    b_index = next(i for i, node in enumerate(graph) if node["name"] == "b_physical")
    current = a_index if settings["start_state"] == "a" else b_index
    rng = np.random.default_rng(settings["random_seed"])
    awh = AWHBias(len(graph), initial_histogram)
    friction_settings = settings["analysis"]["friction"]
    friction_accumulator = FrictionAccumulator(
        len(graph),
        friction_settings["max_correlation_lag_moves"],
        settings["state_move_interval_steps"] * timestep_ps,
    )
    global_diagnostics = GlobalGibbsDiagnostics()
    stage = "adaptive"
    total_steps = 0
    production_steps_completed = 0
    round_trips = 0
    last_endpoint = current if current in (a_index, b_index) else None
    crossed = False
    transitions = np.zeros((len(graph), len(graph)), dtype=np.int64)
    restored_move = None
    if settings["resume"] and state_path.exists() and xml_path.exists():
        with state_path.open() as handle:
            saved = yaml.safe_load(handle) or {}
        worker.simulation.loadState(str(xml_path))
        current = int(saved["current_state"])
        total_steps = int(saved["total_steps"])
        production_steps_completed = int(saved.get("production_steps_completed", 0))
        round_trips = int(saved["round_trips"])
        last_endpoint = saved.get("last_endpoint")
        crossed = bool(saved.get("crossed", False))
        transitions = np.asarray(
            saved.get("transitions", transitions.tolist()), dtype=np.int64
        )
        stage = saved["stage"]
        restored_move = int(saved.get("move", saved["awh"]["updates"]))
        awh = AWHBias.from_dict(saved["awh"])
        saved_friction = saved.get("friction")
        if global_sampling and saved.get("global_gibbs"):
            global_diagnostics = GlobalGibbsDiagnostics.from_dict(
                saved["global_gibbs"]
            )
        if friction_settings["enabled"] and saved_friction:
            restored_friction = FrictionAccumulator.from_dict(saved_friction)
            if (
                restored_friction.nstates == len(graph)
                and restored_friction.maximum_lag
                == friction_settings["max_correlation_lag_moves"]
            ):
                friction_accumulator = restored_friction
            else:
                logger.warning(
                    "AWH friction settings changed; restarting friction statistics"
                )
        rng.bit_generator.state = saved["rng_state"]
        logger.info("Resumed AWH %s at %d steps in state %s", stage, total_steps, graph[current]["name"])
    else:
        worker.simulation.loadState(str(initial_file))
        if settings["start_state"] == "b" and settings["initial_state_file"] is None:
            logger.info(
                "Preparing independent AWH B start with %d MD steps per ATM state",
                settings["start_annealing_steps_per_state"],
            )
            for state_index in range(1, len(atm_states)):
                set_atm_state(worker.context, ommsystem, atm_states[state_index])
                if settings["rest2"]["enabled"]:
                    set_multi_rest2_scales(
                        worker.context, {"a": 1.0, "b": 1.0}, ommsystem.rest2_system
                    )
                worker.run(settings["start_annealing_steps_per_state"])
            worker.simulation.saveState("awh_start_B.xml")
            logger.info("Completed independent AWH B-start preparation")
    apply_node(current)
    if global_sampling and global_diagnostics.validation_checks == 0:
        initial_energies = worker.all_graph_energies()
        validate_global_energies(initial_energies, list(range(len(graph))))
        logger.info(
            "Hybrid global-Gibbs initial energy validation passed for %d states",
            len(graph),
        )

    trace_fields = [
        "move",
        "stage",
        "steps",
        "previous_state",
        "previous_state_name",
        "state",
        "state_name",
        "potential_kj_per_mol",
        "left_state",
        "left_probability",
        "stay_probability",
        "right_state",
        "right_probability",
        "selected_probability",
        "jump_distance",
        "conditional_expected_jump_distance",
        "round_trips",
        "minimum_visits",
    ]
    _ensure_trace_schema(
        trace_path,
        trace_fields,
        a_index if settings["start_state"] == "a" else b_index,
    )
    move = awh.updates if restored_move is None else restored_move
    if friction_settings["enabled"]:
        reconcile_friction_samples(friction_samples_path, move)

    trajectory_settings = settings["analysis"]["trajectory"]
    if trajectory_settings["enabled"]:
        position_state = worker.context.getState(
            getPositions=True, enforcePeriodicBox=True
        )
        resolver = AmberMaskResolver(
            worker.simulation.topology,
            position_state.getPositions(),
            keywords=system_options,
            base_dir=system_options.get("WORKDIR", "."),
        )
        trajectory_atoms = resolver.resolve(
            trajectory_settings["atom_selection"],
            "workflow.awh.analysis.trajectory.atom_selection",
        )
        if not trajectory_atoms:
            raise AWHConfigError("AWH trajectory atom selection is empty")
        if not trajectory_topology_path.exists():
            write_subset_topology(
                worker.simulation.topology,
                position_state.getPositions(),
                trajectory_atoms,
                trajectory_topology_path,
            )
        origin_simulation_step = int(worker.simulation.currentStep)
        origin_awh_steps = total_steps

        def trajectory_metadata():
            node = graph[current]
            return {
                "move": move + 1,
                "stage": stage,
                "state": current,
                "state_name": node["name"],
                "kind": node["kind"],
                "atm_state": node["atm_state"],
                "rest2_region": node["rest2_region"],
                "rest2_scale": node["rest2_scale"],
                "effective_temperature_k": node["effective_temperature_k"],
            }

        worker.simulation.reporters.append(
            StateTaggedXTCReporter(
                trajectory_path,
                trajectory_frames_path,
                trajectory_settings["interval_moves"]
                * settings["state_move_interval_steps"],
                trajectory_atoms,
                trajectory_metadata,
                append=settings["resume"],
                origin_simulation_step=origin_simulation_step,
                origin_awh_steps=origin_awh_steps,
            )
        )

    def diagnostics(free_energies=None):
        sampled_states, matrices = read_reduced_energies(matrix_path)
        if free_energies is None:
            free_energies = estimate_mbar(matrices, sampled_states, len(graph))
        friction = None
        if friction_settings["enabled"]:
            friction = friction_summary(
                friction_accumulator,
                graph,
                friction_settings["min_effective_samples"],
                target=awh.target,
                maximum_relative_weight=settings["adaptive"]["metric_target"][
                    "max_relative_weight"
                ],
            )
            _atomic_yaml(friction_path, friction)
        value = analyze_awh_diagnostics(
            graph,
            trace_rows=read_state_trace(trace_path),
            sampled_states=sampled_states,
            reduced_energies=matrices,
            free_energies=free_energies,
            fallback_transitions=transitions,
            bias_history=read_bias_history(bias_path),
            thresholds=settings["analysis"]["thresholds"],
            friction=friction,
        )
        _atomic_yaml(diagnostics_path, value)
        return value

    def checkpoint():
        worker.simulation.saveState(str(xml_path))
        _atomic_yaml(
            state_path,
            {
                "schema_version": 1,
                "signature": signature,
                "stage": stage,
                "move": move,
                "current_state": current,
                "total_steps": total_steps,
                "production_steps_completed": production_steps_completed,
                "round_trips": round_trips,
                "last_endpoint": last_endpoint,
                "crossed": crossed,
                "transitions": transitions.tolist(),
                "awh": awh.to_dict(),
                "friction": (
                    friction_accumulator.to_dict()
                    if friction_settings["enabled"]
                    else None
                ),
                "global_gibbs": (
                    global_diagnostics.to_dict() if global_sampling else None
                ),
                "rng_state": rng.bit_generator.state,
            },
        )

    def summary(status="partial", free_energies=None):
        f = awh.free_energy / beta
        ddg = float((f[b_index] - f[a_index]) / 4.184)
        row_totals = transitions.sum(axis=1, keepdims=True)
        transition_probabilities = np.divide(
            transitions,
            row_totals,
            out=np.zeros_like(transitions, dtype=float),
            where=row_totals > 0,
        )
        quality = diagnostics(free_energies=free_energies)
        return {
            "schema_version": 1,
            "status": status,
            "stage": stage,
            "total_steps": total_steps,
            "production_steps_completed": production_steps_completed,
            "round_trips": round_trips,
            "state_visits": awh.visits.tolist(),
            "minimum_visits": int(np.min(awh.visits)),
            "current_state": current,
            "current_state_name": graph[current]["name"],
            "transition_counts": transitions.tolist(),
            "transition_probabilities": transition_probabilities.tolist(),
            "overlap_score": quality["uwham"]["minimum_adjacent_overlap"],
            "rest2": quality["rest2"],
            "diagnostics": quality,
            "global_gibbs": (
                global_diagnostics.to_dict() if global_sampling else None
            ),
            "analysis": {
                "awh_bias_ddg_kcal_per_mol": ddg,
                "uwham_ddg_kcal_per_mol": None,
                "uwham_bootstrap_std_kcal_per_mol": None,
            },
        }

    def append_trace(
        previous,
        candidates,
        probabilities,
        reduced,
        conditional_expected_jump_distance=None,
    ):
        probability_by_state = dict(zip(candidates, probabilities))
        energy_by_state = dict(zip(candidates, reduced))
        with trace_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=trace_fields).writerow(
                {
                    "move": move,
                    "stage": stage,
                    "steps": total_steps,
                    "previous_state": previous,
                    "previous_state_name": graph[previous]["name"],
                    "state": current,
                    "state_name": graph[current]["name"],
                    "potential_kj_per_mol": energy_by_state[current],
                    "left_state": previous - 1 if previous > 0 else "",
                    "left_probability": probability_by_state.get(previous - 1, ""),
                    "stay_probability": probability_by_state.get(previous, ""),
                    "right_state": (
                        previous + 1 if previous + 1 < len(graph) else ""
                    ),
                    "right_probability": probability_by_state.get(previous + 1, ""),
                    "selected_probability": probability_by_state.get(current, ""),
                    "jump_distance": abs(current - previous),
                    "conditional_expected_jump_distance": (
                        ""
                        if conditional_expected_jump_distance is None
                        else conditional_expected_jump_distance
                    ),
                    "round_trips": round_trips,
                    "minimum_visits": int(np.min(awh.visits)),
                }
            )

    def evaluate_state_move(update_bias):
        energy_started = time.perf_counter()
        if global_sampling:
            candidates = list(range(len(graph)))
            all_energies = worker.all_graph_energies()
            reduced_by_state = {
                index: beta * value for index, value in enumerate(all_energies)
            }
            global_diagnostics.energy_seconds += time.perf_counter() - energy_started
            validation = settings["state_sampling"]
            if (
                (move + 1) % validation["validation_interval_moves"] == 0
            ):
                physical = [
                    index
                    for index, node in enumerate(graph)
                    if not node["kind"].startswith("rest2_")
                ]
                count = min(validation["validation_states_per_check"], len(physical))
                selected = sorted(
                    int(value)
                    for value in rng.choice(physical, size=count, replace=False)
                )
                validate_global_energies(all_energies, selected)
        else:
            candidates = _neighbors(current, len(graph))
            evaluation_radius = 2 if friction_settings["enabled"] else 1
            evaluation_states = list(
                range(
                    max(0, current - evaluation_radius),
                    min(len(graph), current + evaluation_radius + 1),
                )
            )
            reduced_by_state = {
                index: beta * energy(index) for index in evaluation_states
            }
        reduced = [reduced_by_state[index] for index in candidates]
        selection_started = time.perf_counter()
        probabilities = awh.probabilities(reduced, candidates)
        if friction_settings["enabled"]:
            probability_by_state = dict(zip(candidates, probabilities))
            forces = generalized_forces(
                reduced_by_state, candidates, len(graph)
            )
            friction_accumulator.update(
                {
                    state: (probability_by_state[state], forces[state])
                    for state in candidates
                }
            )
            append_friction_samples(
                friction_samples_path,
                move=move + 1,
                steps=total_steps,
                stage=stage,
                graph=graph,
                probabilities=probability_by_state,
                forces=forces,
            )
        if update_bias:
            awh.update(
                candidates,
                probabilities,
                learning_rate_kbt=settings["adaptive"]["learning_rate_kbt"],
            )
            metric_target = settings["adaptive"]["metric_target"]
            if (
                metric_target["enabled"]
                and round_trips >= metric_target["min_round_trips"]
                and (move + 1) % metric_target["update_interval_moves"] == 0
            ):
                records = friction_accumulator.records(
                    graph,
                    minimum_effective_samples=friction_settings[
                        "min_effective_samples"
                    ],
                )
                target = friction_target(
                    records,
                    maximum_relative_weight=metric_target[
                        "max_relative_weight"
                    ],
                )
                if target is not None:
                    smoothing = metric_target["smoothing"]
                    awh.target = (
                        (1.0 - smoothing) * awh.target + smoothing * target
                    )
                    awh.target /= awh.target.sum()
        if global_sampling:
            global_diagnostics.selection_seconds += (
                time.perf_counter() - selection_started
            )
        return candidates, probabilities, reduced

    while stage == "adaptive":
        md_started = time.perf_counter()
        worker.run(settings["state_move_interval_steps"])
        if global_sampling:
            global_diagnostics.md_seconds += time.perf_counter() - md_started
        total_steps += settings["state_move_interval_steps"]
        previous = current
        candidates, probabilities, reduced = evaluate_state_move(True)
        current = int(rng.choice(candidates, p=probabilities))
        expected_jump = None
        if global_sampling:
            expected_jump = global_diagnostics.update(
                previous, current, probabilities
            )
        transitions[previous, current] += 1
        awh.visits[current] += 1
        apply_node(current)
        if current in (a_index, b_index) and current != last_endpoint:
            if last_endpoint is not None:
                crossed = not crossed
                if not crossed:
                    round_trips += 1
            last_endpoint = current
        move += 1
        append_trace(
            previous,
            candidates,
            probabilities,
            reduced,
            expected_jump,
        )
        adaptive = settings["adaptive"]
        covered = np.count_nonzero(awh.visits) / len(graph) >= adaptive["covering_fraction"]
        converged = (
            total_steps >= adaptive["min_steps"]
            and round_trips >= adaptive["min_round_trips"]
            and int(np.min(awh.visits)) >= adaptive["min_visits_per_state"]
            and covered
        )
        if converged:
            stage = "production"
            logger.info("AWH adaptive stage converged after %d steps and %d round trips", total_steps, round_trips)
        if move % settings["checkpoint_interval_moves"] == 0 or converged:
            with bias_path.open("a", newline="") as handle:
                writer = csv.writer(handle)
                if handle.tell() == 0:
                    writer.writerow(
                        ["move", "steps", "stage", *[node["name"] for node in graph]]
                    )
                writer.writerow([move, total_steps, stage, *awh.free_energy.tolist()])
            checkpoint()
            live_summary = summary()
            if progress_callback:
                progress_callback(live_summary)
        if total_steps >= adaptive["max_steps"] and not converged:
            checkpoint()
            result = summary("partial")
            result["warnings"] = [
                "AWH adaptive stage reached max_steps before convergence",
                *result["diagnostics"]["warnings"],
            ]
            plot_awh_diagnostics(
                result["diagnostics"],
                diagnostics_plot_path,
                read_state_trace(trace_path),
            )
            _atomic_yaml(Path("awh_summary.yaml"), result)
            return result

    sampled_states = []
    matrices = []
    while production_steps_completed < settings["production"]["steps"]:
        md_started = time.perf_counter()
        worker.run(settings["state_move_interval_steps"])
        if global_sampling:
            global_diagnostics.md_seconds += time.perf_counter() - md_started
        total_steps += settings["state_move_interval_steps"]
        production_steps_completed += settings["state_move_interval_steps"]
        previous = current
        candidates, probabilities, reduced = evaluate_state_move(False)
        current = int(rng.choice(candidates, p=probabilities))
        expected_jump = None
        if global_sampling:
            expected_jump = global_diagnostics.update(
                previous, current, probabilities
            )
        transitions[previous, current] += 1
        awh.visits[current] += 1
        apply_node(current)
        if current in (a_index, b_index) and current != last_endpoint:
            if last_endpoint is not None:
                crossed = not crossed
                if not crossed:
                    round_trips += 1
            last_endpoint = current
        move += 1
        append_trace(
            previous,
            candidates,
            probabilities,
            reduced,
            expected_jump,
        )
        if move % settings["production"]["reduced_energy_interval_moves"] == 0:
            if global_sampling:
                row = list(reduced)
                sampled_state = previous
            else:
                row = [beta * energy(index) for index in range(len(graph))]
                apply_node(current)
                sampled_state = current
            matrices.append(row)
            sampled_states.append(sampled_state)
            with matrix_path.open("a", newline="") as handle:
                writer = csv.writer(handle)
                if handle.tell() == 0:
                    writer.writerow(["sampled_state", *[node["name"] for node in graph]])
                writer.writerow([sampled_state, *row])
        if move % settings["checkpoint_interval_moves"] == 0:
            checkpoint()
            live_summary = summary()
            if progress_callback:
                progress_callback(live_summary)
    checkpoint()
    if matrix_path.exists():
        with matrix_path.open() as handle:
            rows = list(csv.reader(handle))
        sampled_states = [int(row[0]) for row in rows[1:]]
        matrices = [[float(value) for value in row[1:]] for row in rows[1:]]
    mbar = estimate_mbar(matrices, sampled_states, len(graph))
    result = summary("completed", free_energies=mbar)
    if mbar is not None:
        result["analysis"]["uwham_ddg_kcal_per_mol"] = float(
            (mbar[b_index] - mbar[a_index]) / beta / 4.184
        )
        result["analysis"]["uwham_bootstrap_std_kcal_per_mol"] = bootstrap_mbar_ddg(
            matrices,
            sampled_states,
            len(graph),
            a_index,
            b_index,
            beta,
            settings["production"]["bootstrap_samples"],
            rng,
        )
        difference = abs(
            result["analysis"]["uwham_ddg_kcal_per_mol"]
            - result["analysis"]["awh_bias_ddg_kcal_per_mol"]
        )
        uncertainty = result["analysis"]["uwham_bootstrap_std_kcal_per_mol"]
        tolerance = max(0.5, 2.0 * uncertainty) if uncertainty is not None else 0.5
        result["analysis"]["awh_uwham_difference_kcal_per_mol"] = difference
        result["analysis"]["agreement_tolerance_kcal_per_mol"] = tolerance
        if difference > tolerance:
            result["status"] = "partial"
            result.setdefault("warnings", []).append(
                "AWH bias and fixed-bias UWHAM estimates disagree beyond tolerance"
            )
    else:
        result["status"] = "partial"
        result.setdefault("warnings", []).append(
            "fixed-bias samples did not visit every state; UWHAM is unavailable"
        )
    quality = result["diagnostics"]
    if not quality["quality_passed"]:
        result["status"] = "partial"
        result.setdefault("warnings", []).extend(quality["warnings"])
    plot_awh_diagnostics(
        quality, diagnostics_plot_path, read_state_trace(trace_path)
    )
    _atomic_yaml(Path("awh_summary.yaml"), result)
    return result


def analyze_existing_awh(options, awh_options):
    manifest_path = Path("awh_protocol.yaml")
    checkpoint_path = Path("awh_checkpoint.yaml")
    if not manifest_path.exists() or not checkpoint_path.exists():
        raise AWHConfigError(
            "AWH analysis requires awh_protocol.yaml and awh_checkpoint.yaml"
        )
    with manifest_path.open() as handle:
        manifest = yaml.safe_load(handle) or {}
    with checkpoint_path.open() as handle:
        checkpoint = yaml.safe_load(handle) or {}
    graph = manifest.get("graph")
    if not isinstance(graph, list) or not graph:
        raise AWHConfigError("awh_protocol.yaml has no state graph")
    settings = deepcopy(manifest.get("settings") or awh_options)
    settings["analysis"] = deepcopy(awh_options["analysis"])
    sampled_states, matrices = read_reduced_energies(
        Path("awh_reduced_energies.csv")
    )
    mbar = estimate_mbar(matrices, sampled_states, len(graph))
    beta = 1.0 / (R_KJ_MOL_K * float(options["TEMPERATURES"][0]))
    a_index = next(i for i, node in enumerate(graph) if node["name"] == "a_physical")
    b_index = next(i for i, node in enumerate(graph) if node["name"] == "b_physical")
    free_energy = np.asarray(
        (checkpoint.get("awh") or {}).get("free_energy", np.zeros(len(graph))),
        dtype=float,
    )
    friction = None
    friction_data = checkpoint.get("friction")
    friction_settings = settings["analysis"]["friction"]
    if friction_settings["enabled"] and friction_data:
        friction = friction_summary(
            FrictionAccumulator.from_dict(friction_data),
            graph,
            friction_settings["min_effective_samples"],
            target=(checkpoint.get("awh") or {}).get("target"),
            maximum_relative_weight=settings["adaptive"]["metric_target"][
                "max_relative_weight"
            ],
        )
        _atomic_yaml(Path("awh_friction.yaml"), friction)
    bias_ddg = float(
        (free_energy[b_index] - free_energy[a_index]) / beta / 4.184
    )
    quality = analyze_awh_diagnostics(
        graph,
        trace_rows=read_state_trace(Path("awh_state_trace.csv")),
        sampled_states=sampled_states,
        reduced_energies=matrices,
        free_energies=mbar,
        fallback_transitions=checkpoint.get("transitions"),
        bias_history=read_bias_history(Path("awh_bias_history.csv")),
        thresholds=settings["analysis"]["thresholds"],
        friction=friction,
    )
    _atomic_yaml(Path("awh_diagnostics.yaml"), quality)
    plot_awh_diagnostics(
        quality,
        Path("awh_diagnostics.png"),
        read_state_trace(Path("awh_state_trace.csv")),
    )
    production_complete = int(checkpoint.get("production_steps_completed", 0)) >= int(
        settings["production"]["steps"]
    )
    result = {
        "schema_version": 1,
        "status": "completed" if production_complete else "partial",
        "stage": checkpoint.get("stage", "adaptive"),
        "total_steps": int(checkpoint.get("total_steps", 0)),
        "production_steps_completed": int(
            checkpoint.get("production_steps_completed", 0)
        ),
        "round_trips": int(checkpoint.get("round_trips", 0)),
        "state_visits": list(
            (checkpoint.get("awh") or {}).get("visits", [0] * len(graph))
        ),
        "minimum_visits": int(
            min((checkpoint.get("awh") or {}).get("visits", [0]))
        ),
        "current_state": int(checkpoint.get("current_state", a_index)),
        "current_state_name": graph[
            int(checkpoint.get("current_state", a_index))
        ]["name"],
        "transition_counts": checkpoint.get("transitions"),
        "overlap_score": quality["uwham"]["minimum_adjacent_overlap"],
        "rest2": quality["rest2"],
        "diagnostics": quality,
        "analysis": {
            "awh_bias_ddg_kcal_per_mol": bias_ddg,
            "uwham_ddg_kcal_per_mol": None,
            "uwham_bootstrap_std_kcal_per_mol": None,
        },
        "warnings": list(quality["warnings"]),
    }
    if mbar is None:
        result["status"] = "partial"
        result["warnings"].append(
            "fixed-bias samples did not visit every state; UWHAM is unavailable"
        )
    else:
        uwham_ddg = float((mbar[b_index] - mbar[a_index]) / beta / 4.184)
        uncertainty = bootstrap_mbar_ddg(
            matrices,
            sampled_states,
            len(graph),
            a_index,
            b_index,
            beta,
            settings["production"]["bootstrap_samples"],
            np.random.default_rng(settings["random_seed"]),
        )
        difference = abs(uwham_ddg - bias_ddg)
        tolerance = max(0.5, 2.0 * uncertainty) if uncertainty is not None else 0.5
        result["analysis"].update(
            {
                "uwham_ddg_kcal_per_mol": uwham_ddg,
                "uwham_bootstrap_std_kcal_per_mol": uncertainty,
                "awh_uwham_difference_kcal_per_mol": difference,
                "agreement_tolerance_kcal_per_mol": tolerance,
            }
        )
        if difference > tolerance:
            result["status"] = "partial"
            result["warnings"].append(
                "AWH bias and fixed-bias UWHAM estimates disagree beyond tolerance"
            )
    if not quality["quality_passed"]:
        result["status"] = "partial"
    result["warnings"] = list(dict.fromkeys(result["warnings"]))
    _atomic_yaml(Path("awh_summary.yaml"), result)
    return result
