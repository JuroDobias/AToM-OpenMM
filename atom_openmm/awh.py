"""Single-walker accelerated weight histogram sampling over ATM states."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
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
    thresholds = analysis.get("thresholds") or {}
    if not isinstance(trajectory, dict) or not isinstance(thresholds, dict):
        raise AWHConfigError(
            "workflow.awh.analysis.trajectory and thresholds must be mappings"
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
        "adaptive": {
            "min_steps": int(adaptive.get("min_steps", 1_000_000)),
            "max_steps": int(adaptive.get("max_steps", 20_000_000)),
            "min_round_trips": int(adaptive.get("min_round_trips", 10)),
            "min_visits_per_state": int(adaptive.get("min_visits_per_state", 100)),
            "covering_fraction": float(adaptive.get("covering_fraction", 0.8)),
            "learning_rate_kbt": float(
                adaptive.get("learning_rate_kbt", 0.1)
            ),
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
    ):
        raise AWHConfigError("AWH initial error and diffusion must be positive")
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
    """AWH free-energy/reference-histogram update for a discrete state graph."""

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
            "free_energy": self.free_energy.tolist(),
            "reference": self.reference.tolist(),
            "covering": self.covering.tolist(),
            "visits": self.visits.tolist(),
            "updates": self.updates,
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls(len(data["free_energy"]), 1.0)
        obj.free_energy = np.asarray(data["free_energy"], dtype=float)
        obj.reference = np.asarray(data["reference"], dtype=float)
        obj.covering = np.asarray(data["covering"], dtype=float)
        obj.visits = np.asarray(data["visits"], dtype=np.int64)
        obj.updates = int(data["updates"])
        return obj


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

    dynamics_settings = {
        key: value for key, value in settings.items() if key != "analysis"
    }
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
    worker = OMMWorkerATMSync(
        basename, ommsystem, system_options, node_info=node_info, compute=False, logger=logger
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

    beta = 1.0 / (R_KJ_MOL_K * float(options["TEMPERATURES"][0]))
    timestep_ps = worker.integrator.getStepSize().value_in_unit(unit.picosecond)
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
    stage = "adaptive"
    total_steps = 0
    production_steps_completed = 0
    round_trips = 0
    last_endpoint = current if current in (a_index, b_index) else None
    crossed = False
    transitions = np.zeros((len(graph), len(graph)), dtype=np.int64)
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
        awh = AWHBias.from_dict(saved["awh"])
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
        "round_trips",
        "minimum_visits",
    ]
    _ensure_trace_schema(
        trace_path,
        trace_fields,
        a_index if settings["start_state"] == "a" else b_index,
    )
    move = awh.updates

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
        value = analyze_awh_diagnostics(
            graph,
            trace_rows=read_state_trace(trace_path),
            sampled_states=sampled_states,
            reduced_energies=matrices,
            free_energies=free_energies,
            fallback_transitions=transitions,
            bias_history=read_bias_history(bias_path),
            thresholds=settings["analysis"]["thresholds"],
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
                "current_state": current,
                "total_steps": total_steps,
                "production_steps_completed": production_steps_completed,
                "round_trips": round_trips,
                "last_endpoint": last_endpoint,
                "crossed": crossed,
                "transitions": transitions.tolist(),
                "awh": awh.to_dict(),
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
            "analysis": {
                "awh_bias_ddg_kcal_per_mol": ddg,
                "uwham_ddg_kcal_per_mol": None,
                "uwham_bootstrap_std_kcal_per_mol": None,
            },
        }

    def append_trace(previous, candidates, probabilities, reduced):
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
                    "round_trips": round_trips,
                    "minimum_visits": int(np.min(awh.visits)),
                }
            )

    while stage == "adaptive":
        worker.run(settings["state_move_interval_steps"])
        total_steps += settings["state_move_interval_steps"]
        candidates = _neighbors(current, len(graph))
        previous = current
        reduced = [beta * energy(index) for index in candidates]
        probabilities = awh.probabilities(reduced, candidates)
        awh.update(
            candidates,
            probabilities,
            learning_rate_kbt=settings["adaptive"]["learning_rate_kbt"],
        )
        current = int(rng.choice(candidates, p=probabilities))
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
        append_trace(previous, candidates, probabilities, reduced)
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
        worker.run(settings["state_move_interval_steps"])
        total_steps += settings["state_move_interval_steps"]
        production_steps_completed += settings["state_move_interval_steps"]
        candidates = _neighbors(current, len(graph))
        previous = current
        reduced = [beta * energy(index) for index in candidates]
        probabilities = awh.probabilities(reduced, candidates)
        current = int(rng.choice(candidates, p=probabilities))
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
        append_trace(previous, candidates, probabilities, reduced)
        if move % settings["production"]["reduced_energy_interval_moves"] == 0:
            row = [beta * energy(index) for index in range(len(graph))]
            apply_node(current)
            matrices.append(row)
            sampled_states.append(current)
            with matrix_path.open("a", newline="") as handle:
                writer = csv.writer(handle)
                if handle.tell() == 0:
                    writer.writerow(["sampled_state", *[node["name"] for node in graph]])
                writer.writerow([current, *row])
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
