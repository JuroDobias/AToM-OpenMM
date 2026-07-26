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
    settings = {
        "state_move_interval_steps": int(raw.get("state_move_interval_steps", 500)),
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
        "platform": raw.get("platform"),
    }
    if settings["target_distribution"] != "uniform":
        raise AWHConfigError("workflow.awh.target_distribution currently must be 'uniform'")
    if settings["start_state"] not in {"a", "b"}:
        raise AWHConfigError("workflow.awh.start_state must be 'a' or 'b'")
    positive = [
        settings["state_move_interval_steps"],
        settings["checkpoint_interval_moves"],
        settings["start_annealing_steps_per_state"],
        settings["adaptive"]["min_steps"],
        settings["adaptive"]["max_steps"],
        settings["production"]["steps"],
        settings["production"]["reduced_energy_interval_moves"],
    ]
    if any(value < 1 for value in positive):
        raise AWHConfigError("AWH step and interval settings must be positive")
    if settings["adaptive"]["min_steps"] > settings["adaptive"]["max_steps"]:
        raise AWHConfigError("workflow.awh.adaptive.min_steps cannot exceed max_steps")
    if not 0 < settings["adaptive"]["covering_fraction"] <= 1:
        raise AWHConfigError("workflow.awh.adaptive.covering_fraction must be in (0, 1]")
    if settings["initial_error_kj_per_mol"] <= 0 or settings["diffusion_per_ps"] <= 0:
        raise AWHConfigError("AWH initial error and diffusion must be positive")
    return settings


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

    def update(self, indices, probabilities):
        sampled = np.zeros_like(self.free_energy)
        sampled[np.asarray(indices, dtype=int)] = probabilities
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

    payload = {
        "schema_version": 1,
        "schedule": [
            {key: scalar(value) for key, value in state.items()}
            for state in build_atm_state_parameters(options)
        ],
        "settings": settings,
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


def run_awh(options, awh_options=None, progress_callback=None):
    settings = awh_options or normalize_awh_options({"awh": {}}, options)
    logger = logging.getLogger("atom_openmm.awh")
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO)
    basename = options["BASENAME"]
    atm_states = build_atm_state_parameters(options)
    graph = build_awh_state_graph(atm_states, settings)
    signature = _protocol_signature(options, settings)
    manifest_path = Path("awh_protocol.yaml")
    state_path = Path("awh_checkpoint.yaml")
    xml_path = Path("awh_checkpoint.xml")
    trace_path = Path("awh_state_trace.csv")
    matrix_path = Path("awh_reduced_energies.csv")
    bias_path = Path("awh_bias_history.csv")
    existing = state_path.exists() or xml_path.exists() or trace_path.exists()
    if settings["resume"] and existing:
        if not manifest_path.exists():
            raise AWHConfigError("AWH resume artifacts exist without awh_protocol.yaml")
        with manifest_path.open() as handle:
            if (yaml.safe_load(handle) or {}).get("signature") != signature:
                raise AWHConfigError("existing AWH artifacts use a different protocol")
    elif not settings["resume"] and existing:
        for path in (state_path, xml_path, trace_path, matrix_path, bias_path):
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
        "move", "stage", "steps", "state", "state_name", "potential_kj_per_mol",
        "round_trips", "minimum_visits",
    ]
    if not trace_path.exists():
        with trace_path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=trace_fields).writeheader()

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

    def summary(status="partial"):
        f = awh.free_energy / beta
        ddg = float((f[b_index] - f[a_index]) / 4.184)
        row_totals = transitions.sum(axis=1, keepdims=True)
        transition_probabilities = np.divide(
            transitions,
            row_totals,
            out=np.zeros_like(transitions, dtype=float),
            where=row_totals > 0,
        )
        return {
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
            "analysis": {
                "awh_bias_ddg_kcal_per_mol": ddg,
                "uwham_ddg_kcal_per_mol": None,
                "uwham_bootstrap_std_kcal_per_mol": None,
            },
        }

    move = awh.updates
    while stage == "adaptive":
        worker.run(settings["state_move_interval_steps"])
        total_steps += settings["state_move_interval_steps"]
        candidates = _neighbors(current, len(graph))
        previous = current
        reduced = [beta * energy(index) for index in candidates]
        probabilities = awh.probabilities(reduced, candidates)
        awh.update(candidates, probabilities)
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
        with trace_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=trace_fields).writerow(
                {
                    "move": move,
                    "stage": stage,
                    "steps": total_steps,
                    "state": current,
                    "state_name": graph[current]["name"],
                    "potential_kj_per_mol": reduced[candidates.index(current)],
                    "round_trips": round_trips,
                    "minimum_visits": int(np.min(awh.visits)),
                }
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
            if progress_callback:
                progress_callback(summary())
        if total_steps >= adaptive["max_steps"] and not converged:
            checkpoint()
            result = summary("partial")
            result["warnings"] = ["AWH adaptive stage reached max_steps before convergence"]
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
        move += 1
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
            if progress_callback:
                progress_callback(summary())
    checkpoint()
    if matrix_path.exists():
        with matrix_path.open() as handle:
            rows = list(csv.reader(handle))
        sampled_states = [int(row[0]) for row in rows[1:]]
        matrices = [[float(value) for value in row[1:]] for row in rows[1:]]
    mbar = estimate_mbar(matrices, sampled_states, len(graph))
    result = summary("completed")
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
    _atomic_yaml(Path("awh_summary.yaml"), result)
    return result


def analyze_existing_awh(options, awh_options):
    path = Path("awh_summary.yaml")
    if not path.exists():
        raise AWHConfigError("awh_summary.yaml does not exist")
    with path.open() as handle:
        return yaml.safe_load(handle) or {}
