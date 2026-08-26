from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import yaml
from openmm.app import PDBFile
from openmm.unit import kelvin, kilocalories_per_mole, kilojoules_per_mole, picosecond
from scipy.optimize import brentq

from atom_openmm.async_re import JobManager
from atom_openmm.abfe_structprep import set_platform
from atom_openmm.atm_coordinates import write_atm_swapped_pdb
from atom_openmm.equilibration import neqti_endpoint_steps, neqti_midpoint_steps, run_custom_equilibration
from atom_openmm.ommsystem import OMMSystemRBFE
from atom_openmm.ommworker import OMMWorkerATMSync
from atom_openmm.neqti_integrator import OMMWorkerATMNEQTI
from atom_openmm.neqti_endpoints import (
    create_native_endpoint_system,
    map_native_to_atm_positions,
    transfer_state_to_context,
    write_converted_state,
)
from atom_openmm.rest2 import set_rest2_scale
from atom_openmm.rest2_exchange import (
    create_rest2_exchange_sampler,
    normalize_rest2_sampler_backend,
    observed_transition_diagnostics,
)


KCAL_TO_KJ = 4.184
KB_KCAL_PER_MOL_K = 0.0019872041
BASE_CSV_FIELDS = [
    "trajectory",
    "direction",
    "start_state",
    "end_state",
    "work_kcal_per_mol",
    "work_kj_per_mol",
    "switch_steps",
    "status",
    "error_type",
    "error_message",
]


def _work_column(interval):
    return f"work_interval_{int(interval)}_kcal_per_mol"


def _work_csv_fields(intervals=()):
    fields = list(BASE_CSV_FIELDS)
    insertion = fields.index("switch_steps")
    diagnostic = []
    for interval in intervals:
        diagnostic.extend([_work_column(interval), f"work_interval_{int(interval)}_kj_per_mol"])
    return fields[:insertion] + diagnostic + fields[insertion:]


def _protocol_signature(settings):
    payload = {
        # Keep the base signature version stable so runs without the new
        # optional features remain resumable from schema-5 manifests.
        "schema_version": 5,
        "hamiltonian": "atm_softplus_single_midpoint",
        "paths": settings["paths"],
        "switch_steps_per_segment": settings["switch_steps_per_segment"],
        "preparation_annealing_steps_per_segment": settings.get("preparation_annealing_steps_per_segment", 0),
        "sampling_order": settings.get("sampling_order", "interleaved"),
        "endpoint_system": settings.get("endpoint_system", "atm"),
    }
    if settings.get("work_sample_intervals"):
        payload["work_sample_intervals"] = settings["work_sample_intervals"]
    if settings.get("receptor_exclusion_signature"):
        payload["receptor_exclusion_signature"] = settings["receptor_exclusion_signature"]
    if settings.get("rest2", {}).get("enabled", False):
        rest2 = deepcopy(settings["rest2"])
        if (
            rest2.get("solute") == '#ligand:"*"'
            and not rest2.get("selection_metadata")
        ):
            rest2["solute"] = "both_ligands"
        payload["rest2"] = rest2
    if settings.get("failed_switch_policy") == "count_as_infinite":
        payload["failed_switch_policy"] = "count_as_infinite"
    if settings.get("schedule_optimization", {}).get("enabled", False):
        payload["schedule_optimization"] = settings["schedule_optimization"]
    if settings.get("convergence", {}).get("enabled", False):
        payload["convergence"] = settings["convergence"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _initialize_protocol_manifest(settings, resume):
    path = Path("neqti_protocol.yaml")
    signature = _protocol_signature(settings)
    existing_artifacts = [
        Path("neqti_endpoint_A.xml"),
        Path("neqti_endpoint_B.xml"),
        Path("neqti_leg_a_forward.csv"),
        Path("neqti_leg_a_reverse.csv"),
        Path("neqti_leg_b_forward.csv"),
        Path("neqti_leg_b_reverse.csv"),
        Path("neqti_midpoint_bridge.csv"),
        Path("neqti_midpoint_plus.xml"),
        Path("neqti_midpoint_minus.xml"),
        Path("neqti_midpoint.xml"),
        Path("neqti_a_sampling.chk"),
        Path("neqti_b_sampling.chk"),
        Path("neqti_mplus_sampling.chk"),
        Path("neqti_mminus_sampling.chk"),
        Path("neqti_m_sampling.chk"),
    ]
    if resume and any(item.exists() for item in existing_artifacts):
        if not path.exists():
            raise NEQTIConfigError(
                "Existing NEQTI artifacts predate the single-midpoint protocol; "
                "remove them or set resume: false"
            )
        with open(path) as handle:
            manifest = yaml.safe_load(handle) or {}
        if manifest.get("signature") != signature:
            raise NEQTIConfigError(
                "Existing NEQTI artifacts use a different Hamiltonian, lambda schedule, sampling order, "
                "REST2 selection, receptor exclusion region, or work-estimator schema; "
                "remove them or set resume: false"
            )
    manifest = {
        "schema_version": 6,
        "hamiltonian": "atm_softplus_single_midpoint",
        "paths": settings["paths"],
        "switch_steps_per_segment": settings["switch_steps_per_segment"],
        "preparation_annealing_steps_per_segment": settings.get("preparation_annealing_steps_per_segment", 0),
        "sampling_order": settings.get("sampling_order", "interleaved"),
        "endpoint_system": settings.get("endpoint_system", "atm"),
        "work_sample_intervals": settings.get("work_sample_intervals", []),
        "receptor_exclusion_signature": settings.get("receptor_exclusion_signature"),
        "rest2": settings.get("rest2", {"enabled": False}),
        "failed_switch_policy": settings.get("failed_switch_policy", "abort"),
        "schedule_optimization": settings.get("schedule_optimization", {"enabled": False}),
        "convergence": settings.get("convergence", {"enabled": False}),
        "signature": signature,
    }
    with open(path, "w") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    return signature


class NEQTIConfigError(ValueError):
    pass


class NEQTINonfiniteWorkError(RuntimeError):
    pass


def _is_numerical_switch_failure(exc):
    if isinstance(exc, NEQTINonfiniteWorkError):
        return True
    if not isinstance(exc, mm.OpenMMException):
        return False
    message = str(exc).lower()
    non_numerical_markers = (
        "cuda_error_",
        "error compiling program",
        "error loading cuda module",
        "invalid parameter name",
        "random number seed",
        "requested two different values",
    )
    if any(marker in message for marker in non_numerical_markers):
        return False
    numerical_markers = (
        "nan",
        "not finite",
        "nonfinite",
        "non-finite",
        "constraint tolerance",
        "constraints could not be satisfied",
        "constraint failure",
    )
    return any(marker in message for marker in numerical_markers)


def build_atm_state_parameters(options):
    keys = ["LAMBDAS", "DIRECTION", "INTERMEDIATE", "LAMBDA1", "LAMBDA2", "ALPHA", "U0", "W0COEFF"]
    arrays = {key: options.get(key) for key in keys}
    if any(values is None for values in arrays.values()):
        raise NEQTIConfigError("NEQTI requires the complete async-RE ATM schedule")
    nstates = len(arrays["LAMBDAS"])
    if any(len(values) != nstates for values in arrays.values()):
        raise NEQTIConfigError("ATM schedule arrays must have the same length")
    temperatures = options.get("TEMPERATURES")
    if not isinstance(temperatures, list) or len(temperatures) != 1:
        raise NEQTIConfigError("NEQTI requires exactly one temperature")
    lambda3s = options.get("LAMBDA3") or arrays["LAMBDA2"]
    uh1s = options.get("U1") or arrays["U0"]
    states = []
    for i in range(nstates):
        states.append({
            "lambda": float(arrays["LAMBDAS"][i]),
            "atmdirection": float(arrays["DIRECTION"][i]),
            "atmintermediate": float(arrays["INTERMEDIATE"][i]),
            "lambda1": float(arrays["LAMBDA1"][i]),
            "lambda2": float(arrays["LAMBDA2"][i]),
            "lambda3": float(lambda3s[i]),
            "alpha": float(arrays["ALPHA"][i]) / kilocalories_per_mole,
            "uh": float(arrays["U0"][i]) * kilocalories_per_mole,
            "uh1": float(uh1s[i]) * kilocalories_per_mole,
            "w0": float(arrays["W0COEFF"][i]) * kilocalories_per_mole,
            "temperature": float(temperatures[0]) * kelvin,
            "Umax": float(options["UMAX"]) * kilocalories_per_mole,
            "Ubcore": float(options["UBCORE"]) * kilocalories_per_mole,
            "Acore": float(options["ACORE"]),
            "uoffset": float(options.get("PERTE_OFFSET", 0.0)) * kilocalories_per_mole,
        })
    return states


def split_two_leg_paths(states):
    boundaries = [
        i for i in range(len(states) - 1)
        if states[i]["atmintermediate"] > 0
        and states[i + 1]["atmintermediate"] > 0
        and states[i]["atmdirection"] > 0
        and states[i + 1]["atmdirection"] < 0
    ]
    if len(boundaries) != 1:
        raise NEQTIConfigError("ATM schedule must contain exactly one adjacent M+/M- midpoint pair")
    midpoint_plus = boundaries[0]
    midpoint_minus = midpoint_plus + 1
    return {
        "leg_a_forward": list(range(0, midpoint_plus + 1)),
        "leg_a_reverse": list(range(midpoint_plus, -1, -1)),
        "leg_b_forward": list(range(len(states) - 1, midpoint_minus - 1, -1)),
        "leg_b_reverse": list(range(midpoint_minus, len(states))),
    }


def normalize_neqti_options(workflow, atom_options):
    raw = workflow.get("neqti", {}) or {}
    if not isinstance(raw, dict):
        raise NEQTIConfigError("workflow.neqti must be a mapping")

    if "lambda_schedule" in raw or "state_path" in raw:
        raise NEQTIConfigError("NEQTI now derives its two half paths from the async-RE ATM schedule")
    paths = split_two_leg_paths(build_atm_state_parameters(atom_options))
    work_sample_intervals = raw.get("work_sample_intervals", []) or []
    if not isinstance(work_sample_intervals, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in work_sample_intervals
    ):
        raise NEQTIConfigError("workflow.neqti.work_sample_intervals must be a list of integers")
    if any(value <= 0 for value in work_sample_intervals) or len(set(work_sample_intervals)) != len(work_sample_intervals):
        raise NEQTIConfigError(
            "workflow.neqti.work_sample_intervals must contain unique positive integers"
        )
    work_sample_intervals = sorted(work_sample_intervals)

    switch_integrator = str(raw.get("switch_integrator", "custom")).lower()
    if switch_integrator not in ("custom", "python"):
        raise NEQTIConfigError("workflow.neqti.switch_integrator must be 'custom' or 'python'")
    sampling_order = str(raw.get("sampling_order", "interleaved")).lower()
    if sampling_order not in ("interleaved", "batched"):
        raise NEQTIConfigError("workflow.neqti.sampling_order must be 'interleaved' or 'batched'")
    endpoint_system = str(raw.get("endpoint_system", "atm")).lower()
    if endpoint_system not in ("atm", "native"):
        raise NEQTIConfigError("workflow.neqti.endpoint_system must be 'atm' or 'native'")
    legacy_tolerate_failures = bool(raw.get("tolerate_failed_switches", False))
    failed_switch_policy = str(
        raw.get("failed_switch_policy", "retry" if legacy_tolerate_failures else "abort")
    ).lower()
    if failed_switch_policy not in ("abort", "retry", "count_as_infinite"):
        raise NEQTIConfigError(
            "workflow.neqti.failed_switch_policy must be 'abort', 'retry', or 'count_as_infinite'"
        )

    rest2_raw = raw.get("rest2", {}) or {}
    if not isinstance(rest2_raw, dict):
        raise NEQTIConfigError("workflow.neqti.rest2 must be a mapping")
    rest2_enabled = bool(rest2_raw.get("enabled", False))
    rest2_ensembles = [str(value).lower() for value in rest2_raw.get("ensembles", ["a", "m", "b"])]
    if len(set(rest2_ensembles)) != len(rest2_ensembles) or any(
        value not in ("a", "m", "b") for value in rest2_ensembles
    ):
        raise NEQTIConfigError(
            "workflow.neqti.rest2.ensembles must contain unique values from [a, m, b]"
        )
    physical_temperature = float(atom_options["TEMPERATURES"][0])
    temperatures = [
        float(value) for value in rest2_raw.get(
            "effective_temperatures_k", [physical_temperature, 351, 411, 481, 563, 658, 770, 900]
        )
    ]
    exchange_interval = int(rest2_raw.get("exchange_interval_steps", 500))
    checkpoint_interval = int(rest2_raw.get("checkpoint_interval_cycles", 10))
    rest2_execution = str(rest2_raw.get("execution", "serial"))
    try:
        rest2_sampler_backend = normalize_rest2_sampler_backend(rest2_raw)
    except ValueError as exc:
        raise NEQTIConfigError(f"workflow.neqti.rest2.{exc}") from exc
    rest2_device_indices = rest2_raw.get("device_indices")
    if rest2_device_indices is not None and not isinstance(rest2_device_indices, list):
        raise NEQTIConfigError("workflow.neqti.rest2.device_indices must be a list")
    solute = str(rest2_raw.get("solute", "both_ligands")).strip()
    if solute == "both_ligands":
        solute = '#ligand:"*"'
    initial_steps = int(raw.get("initial_equilibration_steps", 0))
    decorrelation_steps = int(raw.get("decorrelation_steps", atom_options.get("PRODUCTION_STEPS", 1)))
    if rest2_enabled:
        if endpoint_system == "atm" and sampling_order != "interleaved":
            raise NEQTIConfigError("NEQTI REST2 currently requires sampling_order: interleaved")
        if not solute:
            raise NEQTIConfigError("workflow.neqti.rest2.solute must be a non-empty selection")
        if endpoint_system == "atm" and any(
            token in solute for token in ('#bound:"', '#unbound:"')
        ):
            raise NEQTIConfigError(
                "workflow.neqti.rest2.solute cannot use bound/unbound with the shared ATM system; "
                "use ligand_a, ligand_b, or ligand roles"
            )
        if len(temperatures) < 2 or temperatures[0] != physical_temperature:
            raise NEQTIConfigError("REST2 temperatures must start at the physical ATM temperature")
        if any(b <= a for a, b in zip(temperatures, temperatures[1:])):
            raise NEQTIConfigError("REST2 effective temperatures must be strictly increasing")
        if exchange_interval < 1 or checkpoint_interval < 1:
            raise NEQTIConfigError("REST2 exchange and checkpoint intervals must be positive")
        if rest2_execution not in {"serial", "process"}:
            raise NEQTIConfigError("REST2 execution must be 'serial' or 'process'")
        if rest2_sampler_backend == "openmm_native":
            if not hasattr(app, "ReplicaExchangeSampler"):
                raise NEQTIConfigError(
                    "workflow.neqti.rest2.sampler_backend=openmm_native requires "
                    "OpenMM 8.6 or newer"
                )
            if rest2_execution != "serial":
                raise NEQTIConfigError(
                    "workflow.neqti.rest2.sampler_backend=openmm_native requires "
                    "execution: serial"
                )
            if rest2_device_indices is not None and len(rest2_device_indices) > 1:
                raise NEQTIConfigError(
                    "workflow.neqti.rest2.sampler_backend=openmm_native accepts "
                    "at most one device index"
                )
        for name, value in (
            ("initial_equilibration_steps", initial_steps),
            ("decorrelation_steps", decorrelation_steps),
        ):
            if value % exchange_interval:
                raise NEQTIConfigError(
                    f"workflow.neqti.{name} must be divisible by rest2.exchange_interval_steps"
                )
    if endpoint_system == "native":
        if not rest2_enabled:
            raise NEQTIConfigError(
                "workflow.neqti.endpoint_system: native currently requires REST2"
            )
        if set(rest2_ensembles) != {"a", "b"}:
            raise NEQTIConfigError(
                "native endpoint REST2 currently requires rest2.ensembles: [a, b]"
            )
        if int(raw.get("preparation_annealing_steps_per_segment", 0)) < 1:
            raise NEQTIConfigError(
                "native endpoint REST2 requires positive preparation_annealing_steps_per_segment"
            )
    rest2 = {
        "enabled": rest2_enabled,
        "sampler_backend": rest2_sampler_backend,
        "ensembles": rest2_ensembles,
        "solute": solute,
        "effective_temperatures_k": temperatures,
        "exchange_interval_steps": exchange_interval,
        "checkpoint_interval_cycles": checkpoint_interval,
        "execution": rest2_execution,
        "device_indices": rest2_device_indices,
    }
    if rest2_enabled and atom_options.get("SELECTION_METADATA"):
        rest2["selection_metadata"] = deepcopy(atom_options["SELECTION_METADATA"])
        pdb_path = Path(str(atom_options.get("BASENAME", "")) + ".pdb")
        if pdb_path.exists():
            from atom_openmm.equilibration import AmberMaskResolver

            pdb = PDBFile(str(pdb_path))
            contexts = ("a", "b") if endpoint_system == "native" else (None,)
            resolved = {}
            for context in contexts:
                resolver = AmberMaskResolver(
                    pdb.topology,
                    pdb.positions,
                    keywords=atom_options,
                    endpoint=context,
                    base_dir=atom_options.get("WORKDIR", "."),
                )
                key = "shared_atm" if context is None else f"endpoint_{context}"
                resolved[key] = resolver.resolve(solute, "workflow.neqti.rest2.solute")
            rest2["resolved_atom_indices"] = resolved

    switch_steps_per_segment = int(raw.get("switch_steps_per_segment", atom_options.get("PRODUCTION_STEPS", 1)))
    total_switch_steps = switch_steps_per_segment * (len(paths["leg_a_forward"]) - 1)
    invalid_intervals = [value for value in work_sample_intervals if total_switch_steps % value]
    if invalid_intervals:
        raise NEQTIConfigError(
            f"total switch steps ({total_switch_steps}) must be divisible by every work sample interval; "
            f"invalid intervals: {invalid_intervals}"
        )

    optimization_raw = raw.get("schedule_optimization", {}) or {}
    if not isinstance(optimization_raw, dict):
        raise NEQTIConfigError("workflow.neqti.schedule_optimization must be a mapping")
    optimization_enabled = bool(optimization_raw.get("enabled", False))
    optimize_legs = [str(value).lower() for value in optimization_raw.get("optimize_legs", ["a", "b"])]
    if len(set(optimize_legs)) != len(optimize_legs) or any(value not in ("a", "b") for value in optimize_legs):
        raise NEQTIConfigError("schedule_optimization.optimize_legs must contain unique values from [a, b]")
    schedule_optimization = {
        "enabled": optimization_enabled,
        "pilot_samples": int(optimization_raw.get("pilot_samples", 10)),
        "optimize_legs": optimize_legs,
        "score_hysteresis_weight": float(optimization_raw.get("score_hysteresis_weight", 0.7)),
        "score_absolute_weight": float(optimization_raw.get("score_absolute_weight", 0.3)),
        "score_power": float(optimization_raw.get("score_power", 1.5)),
        "score_ewma_alpha": float(optimization_raw.get("score_ewma_alpha", 0.3)),
        "min_segment_steps": int(optimization_raw.get("min_segment_steps", 1000)),
        "max_segment_steps": int(optimization_raw.get("max_segment_steps", 15000)),
        "min_update_factor": float(optimization_raw.get("min_update_factor", 0.5)),
        "max_update_factor": float(optimization_raw.get("max_update_factor", 2.0)),
    }
    if optimization_enabled:
        if endpoint_system != "native" or sampling_order != "interleaved":
            raise NEQTIConfigError(
                "schedule optimization requires endpoint_system: native and sampling_order: interleaved"
            )
        if switch_integrator != "custom":
            raise NEQTIConfigError("schedule optimization requires switch_integrator: custom")
        if schedule_optimization["pilot_samples"] < 1:
            raise NEQTIConfigError("schedule_optimization.pilot_samples must be positive")
        nsegments = len(paths["leg_a_forward"]) - 1
        if schedule_optimization["min_segment_steps"] * nsegments > total_switch_steps:
            raise NEQTIConfigError("schedule_optimization.min_segment_steps is too large")
        if schedule_optimization["max_segment_steps"] * nsegments < total_switch_steps:
            raise NEQTIConfigError("schedule_optimization.max_segment_steps is too small")
        for name in ("score_hysteresis_weight", "score_absolute_weight", "score_power"):
            if schedule_optimization[name] <= 0:
                raise NEQTIConfigError(f"schedule_optimization.{name} must be positive")
        if not 0 < schedule_optimization["score_ewma_alpha"] <= 1:
            raise NEQTIConfigError("schedule_optimization.score_ewma_alpha must be in (0, 1]")
        if not 0 < schedule_optimization["min_update_factor"] <= 1:
            raise NEQTIConfigError("schedule_optimization.min_update_factor must be in (0, 1]")
        if schedule_optimization["max_update_factor"] < 1:
            raise NEQTIConfigError("schedule_optimization.max_update_factor must be >= 1")

    convergence_raw = raw.get("convergence", {}) or {}
    if not isinstance(convergence_raw, dict):
        raise NEQTIConfigError("workflow.neqti.convergence must be a mapping")
    convergence = {
        "enabled": bool(convergence_raw.get("enabled", False)),
        "min_samples_per_direction": int(convergence_raw.get("min_samples_per_direction", 30)),
        "min_overlap_score_per_leg": float(convergence_raw.get("min_overlap_score_per_leg", 0.05)),
        "max_ddg_error_kcal_per_mol": float(convergence_raw.get("max_ddg_error_kcal_per_mol", 0.5)),
        "consecutive_checks": int(convergence_raw.get("consecutive_checks", 3)),
        "max_ddg_range_kcal_per_mol": float(convergence_raw.get("max_ddg_range_kcal_per_mol", 0.25)),
    }
    if convergence["enabled"]:
        if convergence["min_samples_per_direction"] < 2:
            raise NEQTIConfigError("convergence.min_samples_per_direction must be at least 2")
        if convergence["min_samples_per_direction"] > int(raw.get("n_snapshots", atom_options.get("MAX_SAMPLES", 1))):
            raise NEQTIConfigError("convergence.min_samples_per_direction cannot exceed n_snapshots")
        if convergence["min_overlap_score_per_leg"] <= 0 or convergence["max_ddg_error_kcal_per_mol"] <= 0:
            raise NEQTIConfigError("convergence overlap and uncertainty thresholds must be positive")
        if convergence["consecutive_checks"] < 1 or convergence["max_ddg_range_kcal_per_mol"] < 0:
            raise NEQTIConfigError("convergence check count must be positive and DDG range non-negative")

    receptor_exclusion_atoms = [
        int(value) for value in atom_options.get("EXCLUSION_POT_MOL1_INDEXES", [])
    ]
    receptor_exclusion_signature = None
    if receptor_exclusion_atoms:
        encoded_atoms = json.dumps(receptor_exclusion_atoms, separators=(",", ":")).encode()
        receptor_exclusion_signature = {
            "atom_count": len(receptor_exclusion_atoms),
            "sha256": hashlib.sha256(encoded_atoms).hexdigest(),
        }

    return {
        "initial_equilibration_steps": initial_steps,
        "n_snapshots": int(raw.get("n_snapshots", atom_options.get("MAX_SAMPLES", 1))),
        "decorrelation_steps": decorrelation_steps,
        "switch_steps_per_segment": switch_steps_per_segment,
        "work_sample_intervals": work_sample_intervals,
        "receptor_exclusion_signature": receptor_exclusion_signature,
        "preparation_annealing_steps_per_segment": int(raw.get("preparation_annealing_steps_per_segment", 0)),
        "hamiltonian": "atm_softplus_single_midpoint",
        "paths": paths,
        "resume": bool(raw.get("resume", True)),
        "bootstrap_samples": int(raw.get("bootstrap_samples", 200)),
        "random_seed": int(raw.get("random_seed", 2026)),
        "platform": raw.get("platform"),
        "switch_integrator": switch_integrator,
        "sampling_order": sampling_order,
        "endpoint_system": endpoint_system,
        "rest2": rest2,
        "schedule_optimization": schedule_optimization,
        "convergence": convergence,
        "validate_switch_integrator": bool(raw.get("validate_switch_integrator", False)),
        "tolerate_failed_switches": failed_switch_policy == "retry",
        "failed_switch_policy": failed_switch_policy,
        "max_switch_attempts_per_direction": int(
            raw.get("max_switch_attempts_per_direction", raw.get("n_snapshots", atom_options.get("MAX_SAMPLES", 1)))
        ),
    }


def interpolate_state(start, end, fraction):
    par = deepcopy(start)
    for key, start_value in start.items():
        end_value = end[key]
        if key == "temperature":
            par[key] = start_value
        elif key in ("atmdirection", "atmintermediate"):
            par[key] = end_value if fraction >= 1.0 else start_value
        elif hasattr(start_value, "unit"):
            par[key] = start_value + (end_value - start_value) * fraction
        elif isinstance(start_value, (int, float)):
            par[key] = float(start_value) + (float(end_value) - float(start_value)) * fraction
        else:
            par[key] = end_value if fraction >= 1.0 else start_value
    return par


def make_switch_schedule(stateparams, state_path, steps_per_segment):
    if steps_per_segment < 1:
        raise NEQTIConfigError("workflow.neqti.switch_steps_per_segment must be positive")
    schedule = []
    for start_idx, end_idx in zip(state_path[:-1], state_path[1:]):
        start = stateparams[start_idx]
        end = stateparams[end_idx]
        for step in range(1, steps_per_segment + 1):
            schedule.append(interpolate_state(start, end, step / float(steps_per_segment)))
    return schedule


def _select_node_info(options, neqti_options):
    if neqti_options.get("platform"):
        platform = str(neqti_options["platform"])
        return {
            "node_name": "localhost",
            "slot_number": "0:0",
            "threads_number": int(os.getenv("OMP_NUM_THREADS", "1")),
            "arch": platform,
            "user_name": "",
            "tmp_folder": "/tmp",
        }

    helper = object.__new__(JobManager)
    helper._exit = lambda message: (_ for _ in ()).throw(NEQTIConfigError(message))
    nodes = JobManager.set_node_info(helper, options.get("NODEFILE"))
    if not nodes:
        raise NEQTIConfigError("Could not find an OpenMM compute device for NEQTI")
    return nodes[0]


def _read_completed_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "complete"]


def _read_analyzed_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [
            row for row in csv.DictReader(f)
            if row.get("status") in ("complete", "counted_infinite")
        ]


def _read_counted_infinite_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "counted_infinite"]


def _read_failed_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "failed"]


def _ensure_work_csv(path, intervals=()):
    fields = _work_csv_fields(intervals)
    if path.exists():
        with open(path, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != fields:
            raise NEQTIConfigError(
                f"Existing work CSV {path} has a different estimator schema; "
                "restore work_sample_intervals or start a clean run"
            )
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()


def _append_row(path, row, intervals=()):
    fields = _work_csv_fields(intervals)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(checkpoint)
    os.replace(temporary, path)


def _portable_state_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".xml")


def _write_sampling_checkpoint(path, worker):
    path = Path(path)
    _write_checkpoint(path, worker.get_chkpt())
    context = getattr(worker, "context", None)
    if context is None:
        return
    state = context.getState(
        getPositions=True,
        getVelocities=True,
        getParameters=True,
    )
    portable_path = _portable_state_path(path)
    temporary = portable_path.with_suffix(portable_path.suffix + ".tmp")
    temporary.write_text(mm.XmlSerializer.serialize(state))
    os.replace(temporary, portable_path)


def _initialize_sampling_stream(
    worker,
    *,
    direction,
    initial_state_file,
    start_state,
    checkpoint_file,
    completed_count,
    initial_equilibration_steps,
    resume,
    logger,
):
    portable_state = _portable_state_path(checkpoint_file)
    if resume and portable_state.exists():
        worker.simulation.loadState(str(portable_state))
        worker.set_state(start_state)
        logger.info(
            "Resuming NEQTI %s sampling after %d completed trajectories from portable state %s; "
            "skipping initial equilibration",
            direction,
            completed_count,
            portable_state,
        )
        return

    if resume and checkpoint_file.exists():
        try:
            worker.set_chkpt(checkpoint_file.read_bytes())
        except Exception as exc:
            raise NEQTIConfigError(
                f"Could not load {checkpoint_file}. NEQTI sampling checkpoints created before the custom "
                "switching integrator are incompatible; start a clean job or remove old work and checkpoint files."
            ) from exc
        worker.set_state(start_state)
        logger.info(
            "Resuming NEQTI %s sampling after %d completed trajectories from %s; skipping initial equilibration",
            direction,
            completed_count,
            checkpoint_file,
        )
        return

    worker.simulation.loadState(initial_state_file)
    worker.set_state(start_state)
    _write_worker_pdb_pair(worker, f"neqti_{direction}_start.pdb")
    if resume and completed_count > 0:
        logger.info(
            "Resuming NEQTI %s after %d completed trajectories without a sampling checkpoint; "
            "reloading the endpoint and skipping initial equilibration",
            direction,
            completed_count,
        )
    elif initial_equilibration_steps > 0:
        logger.info("NEQTI %s initial equilibration: %d steps", direction, initial_equilibration_steps)
        _run_worker_steps(
            worker,
            initial_equilibration_steps,
            logger,
            f"NEQTI {direction} initial equilibration",
        )
        _write_worker_pdb_pair(worker, f"neqti_{direction}_equilibrated.pdb")
    _write_sampling_checkpoint(checkpoint_file, worker)


def _write_integrated_work(path, rows, prefix):
    with open(path, "w") as f:
        for row in rows:
            f.write(f"{prefix}_{int(row['trajectory'])} {float(row['work_kj_per_mol']):.12g}\n")


def _write_worker_swapped_pdb(worker, path):
    if not getattr(worker, "context", None) or not getattr(worker, "topology", None):
        return
    state = worker.context.getState(getPositions=True)
    box_vectors = state.getPeriodicBoxVectors()
    if box_vectors is not None:
        worker.topology.setPeriodicBoxVectors(box_vectors)
    write_atm_swapped_pdb(worker.topology, state.getPositions(), worker.keywords, path)


def _write_worker_pdb(worker, path):
    if not getattr(worker, "context", None) or not getattr(worker, "topology", None):
        return
    state = worker.context.getState(getPositions=True)
    box_vectors = state.getPeriodicBoxVectors()
    if box_vectors is not None:
        worker.topology.setPeriodicBoxVectors(box_vectors)
    with open(path, "w") as handle:
        PDBFile.writeFile(worker.topology, state.getPositions(), handle, keepIds=True)


def _write_worker_pdb_pair(worker, path):
    path = Path(path)
    _write_worker_pdb(worker, path)
    _write_worker_swapped_pdb(worker, path.with_name(path.stem + "_swapped.pdb"))


def _worker_timestep_ps(worker):
    integrator = getattr(worker, "integrator", None)
    if isinstance(integrator, mm.CompoundIntegrator):
        integrator = integrator.getIntegrator(integrator.getCurrentIntegrator())
    if integrator is None and getattr(worker, "simulation", None) is not None:
        integrator = getattr(worker.simulation, "integrator", None)
        if isinstance(integrator, mm.CompoundIntegrator):
            integrator = integrator.getIntegrator(integrator.getCurrentIntegrator())
    if integrator is None:
        return None
    return float(integrator.getStepSize().value_in_unit(picosecond))


def _effective_ns_per_day(worker, steps, elapsed_seconds):
    timestep_ps = _worker_timestep_ps(worker)
    if timestep_ps is None or elapsed_seconds <= 0.0:
        return None
    return int(steps) * timestep_ps * 86.4 / elapsed_seconds


def _run_worker_steps(worker, steps, logger, label):
    previous_log_run_timing = getattr(worker, "log_run_timing", True)
    worker.log_run_timing = False
    started = time.perf_counter()
    try:
        worker.run(steps)
    finally:
        worker.log_run_timing = previous_log_run_timing
    elapsed = time.perf_counter() - started
    ns_per_day = _effective_ns_per_day(worker, steps, elapsed)
    if ns_per_day is None:
        logger.info("%s complete: %d steps in %.3f s", label, steps, elapsed)
    else:
        logger.info("%s complete: %d steps in %.3f s, %.3f ns/day", label, steps, elapsed, ns_per_day)
    return elapsed


def _potential_kcal(worker):
    if getattr(worker, "context", None) is not None:
        energy = worker.context.getState(getEnergy=True).getPotentialEnergy()
        return energy / kilocalories_per_mole
    pot = worker.get_energy()
    return pot["potential_energy"] / kilocalories_per_mole


def _run_switch(worker, start_par, schedule, steps_per_segment, state_path, logger=None, label=None, work_sample_intervals=()):
    worker.set_state(start_par)
    work = 0.0
    sampled_work = {int(interval): 0.0 for interval in work_sample_intervals}
    total = len(schedule)
    segment_started = time.perf_counter()
    previous_log_run_timing = getattr(worker, "log_run_timing", True)
    worker.log_run_timing = False
    try:
        for step, par in enumerate(schedule, start=1):
            old_energy = _potential_kcal(worker)
            worker.set_state(par)
            new_energy = _potential_kcal(worker)
            increment = new_energy - old_energy
            work += increment
            for interval in sampled_work:
                if step % interval == 0:
                    sampled_work[interval] += interval * increment
            worker.run(1)
            if logger is not None and label and step % steps_per_segment == 0:
                segment = step // steps_per_segment
                elapsed = time.perf_counter() - segment_started
                ns_per_day = _effective_ns_per_day(worker, steps_per_segment, elapsed)
                if ns_per_day is None:
                    logger.info(
                        "NEQTI %s segment %d/%d complete: state %d -> %d, %d/%d steps, work %.6g kcal/mol",
                        label,
                        segment,
                        len(state_path) - 1,
                        state_path[segment - 1],
                        state_path[segment],
                        step,
                        total,
                        work,
                    )
                else:
                    logger.info(
                        "NEQTI %s segment %d/%d complete: state %d -> %d, %d/%d steps, work %.6g kcal/mol, %.3f ns/day",
                        label,
                        segment,
                        len(state_path) - 1,
                        state_path[segment - 1],
                        state_path[segment],
                        step,
                        total,
                        work,
                        ns_per_day,
                    )
                segment_started = time.perf_counter()
    finally:
        worker.log_run_timing = previous_log_run_timing
    return {"exact": work, "sampled": sampled_work}


def _run_switch_custom(worker, start_par, steps_per_segment, state_path, direction, logger=None, label=None):
    integrator = worker.begin_switch(direction, start_par)
    segment_steps = (
        [int(steps_per_segment)] * (len(state_path) - 1)
        if isinstance(steps_per_segment, int)
        else [int(value) for value in steps_per_segment]
    )
    if len(segment_steps) != len(state_path) - 1:
        raise ValueError(f"{direction} segment step count does not match its path")
    total = sum(segment_steps)
    completed_steps = 0
    segment_work = []
    previous_work = 0.0
    try:
        for segment, (start_state, end_state, steps) in enumerate(
            zip(state_path[:-1], state_path[1:], segment_steps), start=1
        ):
            segment_started = time.perf_counter()
            integrator.step(steps)
            elapsed = time.perf_counter() - segment_started
            work = integrator.get_protocol_work() / kilocalories_per_mole
            segment_work.append(float(work - previous_work))
            previous_work = float(work)
            completed_steps += steps
            if logger is not None and label:
                ns_per_day = _effective_ns_per_day(worker, steps, elapsed)
                logger.info(
                    "NEQTI %s segment %d/%d complete: state %d -> %d, %d/%d steps, "
                    "work %.6g kcal/mol, %.3f ns/day",
                    label,
                    segment,
                    len(state_path) - 1,
                    start_state,
                    end_state,
                    completed_steps,
                    total,
                    work,
                    ns_per_day,
                )
        return {
            "exact": integrator.get_protocol_work() / kilocalories_per_mole,
            "segment_work": segment_work,
            "sampled": {
                interval: value / kilocalories_per_mole
                for interval, value in integrator.get_sampled_protocol_work().items()
            },
        }
    finally:
        worker.end_switch()


def _validate_switch_implementations(
    worker,
    *,
    direction,
    snapshot,
    start_state,
    schedule,
    steps_per_segment,
    state_path,
    logger,
    output_file=Path("neqti_switch_validation.yaml"),
):
    if output_file.exists():
        with open(output_file) as handle:
            report = yaml.safe_load(handle) or {}
    else:
        report = {"status": "informational", "directions": {}}
    directions = report.setdefault("directions", {})
    if direction in directions:
        return

    worker.set_chkpt(snapshot)
    worker.set_state(start_state)
    custom_started = time.perf_counter()
    custom_result = _run_switch_custom(
        worker,
        start_state,
        steps_per_segment,
        state_path,
        direction,
        logger,
        f"validation custom {direction}",
    )
    custom_seconds = time.perf_counter() - custom_started

    worker.set_chkpt(snapshot)
    worker.set_state(start_state)
    python_started = time.perf_counter()
    python_result = _run_switch(
        worker,
        start_state,
        schedule,
        steps_per_segment,
        state_path,
        logger,
        f"validation python {direction}",
    )
    python_seconds = time.perf_counter() - python_started
    custom_work = custom_result["exact"] if isinstance(custom_result, dict) else custom_result
    python_work = python_result["exact"] if isinstance(python_result, dict) else python_result
    worker.set_chkpt(snapshot)
    worker.set_state(start_state)

    total_steps = len(schedule)
    directions[direction] = {
        "custom_work_kcal_per_mol": float(custom_work),
        "python_work_kcal_per_mol": float(python_work),
        "work_difference_kcal_per_mol": float(custom_work - python_work),
        "custom_seconds": custom_seconds,
        "python_seconds": python_seconds,
        "custom_ns_per_day": _effective_ns_per_day(worker, total_steps, custom_seconds),
        "python_ns_per_day": _effective_ns_per_day(worker, total_steps, python_seconds),
        "steps": total_steps,
    }
    with open(output_file, "w") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    logger.info("Wrote informational %s switch-integrator comparison to %s", direction, output_file)


def _bootstrap_bar(forward, reverse, temperature, nboots, seed):
    if nboots <= 0 or len(forward) < 2 or len(reverse) < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates = []
    forward = np.asarray(forward, dtype=float)
    reverse = np.asarray(reverse, dtype=float)
    for _ in range(nboots):
        sample_f = rng.choice(forward, size=len(forward), replace=True)
        sample_r = rng.choice(reverse, size=len(reverse), replace=True)
        try:
            estimate = estimate_bar(sample_f, sample_r, temperature)
            if estimate is not None and math.isfinite(estimate):
                estimates.append(estimate)
        except Exception:
            pass
    if not estimates:
        return None
    return float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0


def estimate_bar(forward_work_kcal, reverse_work_kcal, temperature_kelvin):
    forward = np.asarray(forward_work_kcal, dtype=float)
    reverse = np.asarray(reverse_work_kcal, dtype=float)
    if len(forward) == 0 or len(reverse) == 0:
        return None
    finite_forward = forward[np.isfinite(forward)]
    finite_reverse = reverse[np.isfinite(reverse)]
    if len(finite_forward) == 0 or len(finite_reverse) == 0:
        return None
    if np.any(np.isneginf(forward)) or np.any(np.isneginf(reverse)):
        raise ValueError("BAR work values may be finite or +inf, but not -inf")
    beta = 1.0 / (KB_KCAL_PER_MOL_K * float(temperature_kelvin))

    def fermi(x):
        return 1.0 / (1.0 + np.exp(np.clip(x, -700, 700)))

    def equation(df):
        return np.mean(fermi(beta * (forward - df))) - np.mean(fermi(beta * (reverse + df)))

    low = min(float(np.min(finite_forward)), -float(np.max(finite_reverse))) - 100.0 / beta
    high = max(float(np.max(finite_forward)), -float(np.min(finite_reverse))) + 100.0 / beta
    try:
        return float(brentq(equation, low, high, maxiter=200))
    except ValueError:
        return None


def analyze_neqti_work(forward_work_kcal, reverse_work_kcal, temperature_kelvin, bootstrap_samples=200, random_seed=2026):
    dg = estimate_bar(forward_work_kcal, reverse_work_kcal, temperature_kelvin)
    if dg is None:
        return None
    err = _bootstrap_bar(forward_work_kcal, reverse_work_kcal, temperature_kelvin, bootstrap_samples, random_seed)
    return {
        "bar_dg_kcal_per_mol": dg,
        "bar_dg_kj_per_mol": dg * KCAL_TO_KJ,
        "bar_bootstrap_std_kcal_per_mol": err,
        "bar_bootstrap_std_kj_per_mol": None if err is None else err * KCAL_TO_KJ,
    }


def _bar_overlap_score(forward_work, reverse_work, dg, temperature_kelvin):
    if dg is None or len(forward_work) == 0 or len(reverse_work) == 0:
        return None
    beta = 1.0 / (KB_KCAL_PER_MOL_K * float(temperature_kelvin))
    forward_acceptance = np.mean(1.0 / (1.0 + np.exp(np.clip(beta * (np.asarray(forward_work) - dg), -700, 700))))
    reverse_acceptance = np.mean(1.0 / (1.0 + np.exp(np.clip(beta * (np.asarray(reverse_work) + dg), -700, 700))))
    return float(min(1.0, 2.0 * min(forward_acceptance, reverse_acceptance)))


def analyze_two_leg_work(work, temperature_kelvin, bootstrap_samples=200, random_seed=2026):
    components = {}
    for name, forward_key, reverse_key in (
        ("leg_a", "leg_a_forward", "leg_a_reverse"),
        ("leg_b", "leg_b_forward", "leg_b_reverse"),
    ):
        dg = estimate_bar(work[forward_key], work[reverse_key], temperature_kelvin)
        components[name] = {
            "dg_kcal_per_mol": dg,
            "overlap_score": _bar_overlap_score(
                work[forward_key], work[reverse_key], dg, temperature_kelvin
            ),
        }
    if any(component["dg_kcal_per_mol"] is None for component in components.values()):
        return None
    ddg = components["leg_a"]["dg_kcal_per_mol"] - components["leg_b"]["dg_kcal_per_mol"]
    error = None
    if bootstrap_samples > 0 and all(len(values) >= 2 for values in work.values()):
        rng = np.random.default_rng(random_seed)
        estimates = []
        for _ in range(bootstrap_samples):
            sampled = {
                key: rng.choice(values, size=len(values), replace=True)
                for key, values in work.items()
            }
            try:
                leg_a = estimate_bar(
                    sampled["leg_a_forward"], sampled["leg_a_reverse"], temperature_kelvin
                )
                leg_b = estimate_bar(
                    sampled["leg_b_forward"], sampled["leg_b_reverse"], temperature_kelvin
                )
                if leg_a is not None and leg_b is not None:
                    estimates.append(leg_a - leg_b)
            except Exception:
                pass
        if estimates:
            error = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
    return {
        "bar_dg_kcal_per_mol": float(ddg),
        "bar_dg_kj_per_mol": float(ddg) * KCAL_TO_KJ,
        "bar_bootstrap_std_kcal_per_mol": error,
        "bar_bootstrap_std_kj_per_mol": None if error is None else error * KCAL_TO_KJ,
        "components": components,
        "overlap_score": min(component["overlap_score"] for component in components.values()),
    }


def _paired_bootstrap_difference(exact_work, variant_work, temperature_kelvin, bootstrap_samples, random_seed):
    if bootstrap_samples <= 0 or any(len(values) < 2 for values in exact_work.values()):
        return None
    rng = np.random.default_rng(random_seed)
    differences = []
    for _ in range(bootstrap_samples):
        exact_sample = {}
        variant_sample = {}
        for key, exact_values in exact_work.items():
            indices = rng.integers(0, len(exact_values), size=len(exact_values))
            exact_sample[key] = np.asarray(exact_values)[indices]
            variant_sample[key] = np.asarray(variant_work[key])[indices]
        exact = analyze_two_leg_work(exact_sample, temperature_kelvin, bootstrap_samples=0)
        variant = analyze_two_leg_work(variant_sample, temperature_kelvin, bootstrap_samples=0)
        if exact is not None and variant is not None:
            differences.append(
                variant["bar_dg_kcal_per_mol"] - exact["bar_dg_kcal_per_mol"]
            )
    if not differences:
        return None
    return float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0


def analyze_work_estimators(rows, intervals, temperature_kelvin, bootstrap_samples=200, random_seed=2026):
    columns = {"exact": "work_kcal_per_mol"}
    columns.update({f"interval_{value}": _work_column(value) for value in intervals})
    estimator_work = {
        estimator: {
            direction: [float(row[column]) for row in direction_rows]
            for direction, direction_rows in rows.items()
        }
        for estimator, column in columns.items()
    }
    analyses = {}
    for estimator, work in estimator_work.items():
        analysis = analyze_two_leg_work(
            work,
            temperature_kelvin,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )
        analyses[estimator] = analysis
    exact = analyses["exact"]
    for estimator, analysis in analyses.items():
        if estimator == "exact" or analysis is None or exact is None:
            continue
        difference = analysis["bar_dg_kcal_per_mol"] - exact["bar_dg_kcal_per_mol"]
        analysis["difference_from_exact_kcal_per_mol"] = float(difference)
        analysis["difference_from_exact_kj_per_mol"] = float(difference) * KCAL_TO_KJ
        paired_std = _paired_bootstrap_difference(
            estimator_work["exact"],
            estimator_work[estimator],
            temperature_kelvin,
            bootstrap_samples,
            random_seed,
        )
        analysis["paired_bootstrap_difference_std_kcal_per_mol"] = paired_std
        analysis["paired_bootstrap_difference_std_kj_per_mol"] = (
            None if paired_std is None else paired_std * KCAL_TO_KJ
        )
    return analyses


def _allocate_segment_steps(scores, total_steps, current_steps, settings):
    scores = np.asarray(scores, dtype=float)
    current = np.asarray(current_steps, dtype=int)
    powered = np.power(np.maximum(scores, 1.0e-12), settings["score_power"])
    raw = powered / powered.sum() * int(total_steps)
    lower = np.maximum(
        settings["min_segment_steps"],
        np.floor(current * settings["min_update_factor"]).astype(int),
    )
    upper = np.minimum(
        settings["max_segment_steps"],
        np.ceil(current * settings["max_update_factor"]).astype(int),
    )
    values = np.clip(raw, lower, upper)
    for _ in range(100):
        difference = float(total_steps) - float(values.sum())
        if abs(difference) < 1.0e-8:
            break
        adjustable = values < upper if difference > 0 else values > lower
        if not np.any(adjustable):
            raise NEQTIConfigError("schedule optimizer bounds cannot preserve total switch steps")
        weights = powered[adjustable]
        values[adjustable] += difference * weights / weights.sum()
        values = np.clip(values, lower, upper)
    steps = np.floor(values).astype(int)
    remainder = int(total_steps) - int(steps.sum())
    order = np.argsort(-(values - steps))
    while remainder > 0:
        changed = False
        for index in order:
            if steps[index] < upper[index]:
                steps[index] += 1
                remainder -= 1
                changed = True
                if remainder == 0:
                    break
        if not changed:
            raise NEQTIConfigError("schedule optimizer could not allocate remaining steps")
    while remainder < 0:
        changed = False
        for index in reversed(order):
            if steps[index] > lower[index]:
                steps[index] -= 1
                remainder += 1
                changed = True
                if remainder == 0:
                    break
        if not changed:
            raise NEQTIConfigError("schedule optimizer could not remove excess steps")
    return [int(value) for value in steps]


def _optimizer_cycle_scores(forward_segment_work, reverse_segment_work, settings):
    reverse_physical = list(reversed(reverse_segment_work))
    return np.asarray([
        settings["score_hysteresis_weight"] * abs(float(forward) + float(reverse))
        + settings["score_absolute_weight"] * max(abs(float(forward)), abs(float(reverse)))
        + 1.0e-6
        for forward, reverse in zip(forward_segment_work, reverse_physical)
    ])


def _schedule_change_fraction(previous, current):
    return float(np.sum(np.abs(np.asarray(current) - np.asarray(previous))) / (2.0 * sum(previous)))


def _convergence_record(analysis, counts, settings):
    components = {} if analysis is None else analysis.get("components", {})
    overlaps = [components.get(name, {}).get("overlap_score") for name in ("leg_a", "leg_b")]
    error = None if analysis is None else analysis.get("bar_bootstrap_std_kcal_per_mol")
    enough = all(value >= settings["min_samples_per_direction"] for value in counts.values())
    thresholds_pass = (
        enough
        and all(value is not None and value >= settings["min_overlap_score_per_leg"] for value in overlaps)
        and error is not None
        and error <= settings["max_ddg_error_kcal_per_mol"]
    )
    return {
        "sample_count_per_direction": min(counts.values()),
        "ddg_kcal_per_mol": None if analysis is None else analysis.get("bar_dg_kcal_per_mol"),
        "ddg_error_kcal_per_mol": error,
        "leg_a_overlap_score": overlaps[0],
        "leg_b_overlap_score": overlaps[1],
        "thresholds_pass": bool(thresholds_pass),
    }


def _convergence_reached(history, settings):
    required = settings["consecutive_checks"]
    if len(history) < required:
        return False
    recent = history[-required:]
    if not all(record["thresholds_pass"] for record in recent):
        return False
    estimates = [record["ddg_kcal_per_mol"] for record in recent]
    return max(estimates) - min(estimates) <= settings["max_ddg_range_kcal_per_mol"]


def summarize_existing_neqti_work(options, neqti_options, paths, *, bootstrap_samples=None):
    work_files = {
        "leg_a_forward": Path("neqti_leg_a_forward.csv"),
        "leg_a_reverse": Path("neqti_leg_a_reverse.csv"),
        "leg_b_forward": Path("neqti_leg_b_forward.csv"),
        "leg_b_reverse": Path("neqti_leg_b_reverse.csv"),
    }
    missing = [str(path) for path in work_files.values() if not path.exists()]
    if missing:
        raise NEQTIConfigError("Missing NEQTI work files: " + ", ".join(missing))

    rows = {name: _read_analyzed_rows(path) for name, path in work_files.items()}
    for name, values in rows.items():
        _write_integrated_work(Path(f"integ_{name}.dat"), values, name)
    states = build_atm_state_parameters(options)
    temperature_kelvin = states[paths["leg_a_forward"][0]]["temperature"] / kelvin
    work_estimator_analyses = analyze_work_estimators(
        rows,
        neqti_options.get("work_sample_intervals", []),
        temperature_kelvin,
        bootstrap_samples=neqti_options["bootstrap_samples"] if bootstrap_samples is None else bootstrap_samples,
        random_seed=neqti_options["random_seed"],
    )
    analysis = work_estimator_analyses["exact"]
    complete = all(len(values) >= neqti_options["n_snapshots"] for values in rows.values())
    usable_overlap = analysis is not None and analysis["overlap_score"] >= 0.01
    failed_counts = {
        name: len(_read_failed_rows(path)) for name, path in work_files.items()
    }
    counted_infinite_counts = {
        name: len(_read_counted_infinite_rows(path)) for name, path in work_files.items()
    }
    finite_counts = {
        name: sum(math.isfinite(float(row["work_kcal_per_mol"])) for row in values)
        for name, values in rows.items()
    }
    warnings = []
    counted_total = sum(counted_infinite_counts.values())
    if counted_total:
        warnings.append(
            f"{counted_total} numerical switch failures were included as +infinite protocol work."
        )
    if analysis is not None and analysis.get("bar_bootstrap_std_kcal_per_mol") is None:
        warnings.append("BAR uncertainty could not be identified from the available samples.")
    if analysis is None and all(rows.values()):
        warnings.append("BAR has no finite connecting sample in at least one required direction.")
    rest2_summary = None
    if neqti_options.get("rest2", {}).get("enabled", False):
        rest2_summary = {
            "sampler_backend": neqti_options["rest2"].get(
                "sampler_backend", "custom"
            ),
            "states": {},
            "sampled_ensembles": list(neqti_options["rest2"].get("ensembles", ["a", "m", "b"])),
        }
        for ensemble in ("a", "m", "b"):
            metadata_path = Path("neqti_rest2") / ensemble / "state.json"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text())
            backend = str(metadata.get("sampler_backend", "custom"))
            attempts = np.asarray(metadata.get("attempts", []), dtype=int)
            accepts = np.asarray(metadata.get("accepts", []), dtype=int)
            rates = np.divide(
                accepts, attempts,
                out=np.zeros_like(accepts, dtype=float), where=attempts > 0,
            )
            rest2_summary["states"][ensemble] = {
                "completed_cycles": int(metadata.get("cycle", 0)),
                "sampler_backend": backend,
                "openmm_version": metadata.get("openmm_version"),
                "attempts": attempts.tolist(),
                "accepts": accepts.tolist(),
                "acceptance_rates": rates.tolist(),
                "round_trips": [int(value) for value in metadata.get("round_trips", [])],
            }
            if backend == "openmm_native":
                rest2_summary["states"][ensemble]["exchange_diagnostics"] = (
                    observed_transition_diagnostics(
                        metadata_path.parent / "state_trace.csv",
                        len(metadata.get("assignments", [])),
                    )
                )
        rest2_summary["effective_temperatures_k"] = neqti_options["rest2"]["effective_temperatures_k"]
        rest2_summary["exchange_interval_steps"] = neqti_options["rest2"]["exchange_interval_steps"]
        low_acceptance = []
        for ensemble, state_summary in rest2_summary["states"].items():
            for neighbor, (attempt_count, rate) in enumerate(zip(state_summary["attempts"], state_summary["acceptance_rates"])):
                if attempt_count > 0 and rate < 0.05:
                    low_acceptance.append(f"{ensemble}:{neighbor}-{neighbor + 1}")
        rest2_summary["warnings"] = (
            ["REST2 neighbor acceptance below 0.05 for " + ", ".join(low_acceptance)]
            if low_acceptance else []
        )
    convergence_state = {}
    convergence_path = Path("neqti_convergence.yaml")
    if convergence_path.exists():
        with convergence_path.open() as handle:
            convergence_state = yaml.safe_load(handle) or {}
    termination_reason = convergence_state.get("termination_reason")
    convergence_enabled = neqti_options.get("convergence", {}).get("enabled", False)
    optimizer_state = None
    optimizer_path = Path("neqti_schedule_optimization.yaml")
    if optimizer_path.exists():
        with optimizer_path.open() as handle:
            optimizer_state = yaml.safe_load(handle) or None
    if termination_reason == "converged":
        summary_status = "completed"
    elif convergence_enabled and termination_reason == "max_samples":
        summary_status = "partial"
    else:
        summary_status = "completed" if complete and usable_overlap else "partial"
    summary = {
        "jobname": options["BASENAME"],
        "method": "neqti",
        "status": summary_status,
        "termination_reason": termination_reason,
        "forward_samples": len(rows["leg_a_forward"]) + len(rows["leg_b_forward"]),
        "reverse_samples": len(rows["leg_a_reverse"]) + len(rows["leg_b_reverse"]),
        "sample_counts": {name: len(values) for name, values in rows.items()},
        "finite_sample_counts": finite_counts,
        "counted_infinite_work_counts": counted_infinite_counts,
        "failed_switch_counts": failed_counts,
        "temperature_kelvin": float(temperature_kelvin),
        "hamiltonian": "atm_softplus_single_midpoint",
        "endpoint_system": neqti_options.get("endpoint_system", "atm"),
        "paths": paths,
        "settings": neqti_options,
        "analysis": analysis,
        "work_estimator_analyses": work_estimator_analyses,
        "rest2": rest2_summary,
        "convergence": convergence_state or None,
        "schedule_optimization": optimizer_state,
        "warnings": warnings,
    }
    with open("neqti_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    return summary


def analyze_existing_neqti(options, neqti_options=None):
    if neqti_options is None:
        neqti_options = normalize_neqti_options({"neqti": {}}, options)
    return summarize_existing_neqti_work(options, neqti_options, neqti_options["paths"])


def _save_worker_state(worker, state_path, pdb_path=None):
    worker.select_equilibrium() if hasattr(worker, "select_equilibrium") else None
    worker.simulation.saveState(str(state_path))
    _strip_state_if_possible(state_path)
    if pdb_path is not None:
        _write_worker_pdb_pair(worker, pdb_path)


def _strip_state_if_possible(path):
    try:
        from atom_openmm.equilibration import _strip_integrator_parameters

        _strip_integrator_parameters(path)
    except Exception:
        pass


def _run_preparation_anneal(worker, switch_name, start_state, state_path, steps_per_segment, output_state, output_pdb, logger):
    labels = {
        "leg_a_forward": "physical_to_M",
        "leg_a_reverse": "M_to_A",
        "leg_b_reverse": "M_to_B",
    }
    label = labels.get(switch_name, switch_name)
    logger.info(
        "NEQTI preparation anneal %s (%s): %s",
        label,
        switch_name,
        " -> ".join(str(i) for i in state_path),
    )
    _run_switch_custom(
        worker,
        start_state,
        steps_per_segment,
        state_path,
        switch_name,
        logger,
        f"preparation anneal {label} ({switch_name})",
    )
    _save_worker_state(worker, output_state, output_pdb)


def run_neqti(options, neqti_options=None, progress_callback=None):
    if neqti_options is None:
        neqti_options = normalize_neqti_options({"neqti": {}}, options)
    neqti_options.setdefault("sampling_order", "interleaved")
    neqti_options.setdefault("work_sample_intervals", [])

    basename = options["BASENAME"]
    logger = logging.getLogger("atom_openmm.neqti")
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO)

    protocol_signature = _initialize_protocol_manifest(neqti_options, neqti_options["resume"])

    stateparams = build_atm_state_parameters(options)
    paths = neqti_options["paths"]
    schedules = {
        name: make_switch_schedule(stateparams, path, neqti_options["switch_steps_per_segment"])
        for name, path in paths.items()
    }
    states = {
        "a": stateparams[paths["leg_a_forward"][0]],
        "m": stateparams[paths["leg_a_forward"][-1]],
        "m_for_b": stateparams[paths["leg_b_forward"][-1]],
        "b": stateparams[paths["leg_b_forward"][0]],
    }
    initial_state_file = options.get("NEQTI_INITIAL_STATE_FILE") or options.get("INITIAL_STATE_FILE") or basename + "_0.xml"
    state_files = {name: initial_state_file for name in states}
    endpoint_steps = neqti_endpoint_steps(options)
    midpoint_steps = neqti_midpoint_steps(options)
    equilibrated_files = {
        "m": Path("neqti_midpoint.xml"),
        "a": Path("neqti_endpoint_A.xml"),
        "b": Path("neqti_endpoint_B.xml"),
    }
    marker = Path("neqti_equilibrium_states.ok")
    marker_matches = marker.exists() and marker.read_text().strip() == protocol_signature
    if (
        endpoint_steps is not None
        or midpoint_steps is not None
        or neqti_options["preparation_annealing_steps_per_segment"] > 0
        or neqti_options.get("endpoint_system") == "native"
    ):
        if neqti_options["resume"] and marker_matches and all(path.exists() for path in equilibrated_files.values()):
            logger.info("Reusing completed NEQTI endpoint and midpoint states")
        else:
            logger.info("Building NEQTI M, A, and B equilibrium states")
            endpoint_system = OMMSystemRBFE(basename, options, basename + ".pdb", basename + "_sys.xml", logger)
            endpoint_system.create_system()
            platform_options = deepcopy(options)
            if neqti_options.get("platform"):
                platform_options["OPENMM_PLATFORM"] = neqti_options["platform"]
            platform, platform_properties = set_platform(platform_options)
            anneal_steps = neqti_options["preparation_annealing_steps_per_segment"]
            if anneal_steps > 0:
                prep_options = deepcopy(options)
                prep_options["INITIAL_STATE_FILE"] = initial_state_file
                prep_worker = OMMWorkerATMNEQTI(
                    basename,
                    endpoint_system,
                    prep_options,
                    node_info=_select_node_info(options, neqti_options),
                    compute=True,
                    logger=logger,
                    switch_schedules={name: [stateparams[index] for index in path] for name, path in paths.items()},
                    steps_per_segment=anneal_steps,
                    random_seed=neqti_options["random_seed"],
                )
                try:
                    prep_worker.simulation.loadState(initial_state_file)
                    _run_preparation_anneal(
                        prep_worker,
                        "leg_a_forward",
                        states["a"],
                        paths["leg_a_forward"],
                        anneal_steps,
                        "neqti_midpoint_annealed.xml",
                        "neqti_midpoint_annealed.pdb",
                        logger,
                    )
                finally:
                    prep_worker.finish()
                midpoint_source = "neqti_midpoint_annealed.xml"
            else:
                midpoint_source = initial_state_file

            if midpoint_steps is not None:
                run_custom_equilibration(
                    ommsystem=endpoint_system,
                    steps=midpoint_steps,
                    platform=platform,
                    platform_properties=platform_properties,
                    output_dir=Path("equilibration") / "neqti_m",
                    final_state_path=equilibrated_files["m"],
                    final_pdb_path="neqti_midpoint.pdb",
                    initial_state_path=midpoint_source,
                    atm_state=states["m"],
                    logger=logger,
                )
                logger.info("Completed NEQTI midpoint equilibration")
            else:
                equilibrated_files["m"] = Path(midpoint_source)

            endpoint_sources = {"a": str(equilibrated_files["m"]), "b": str(equilibrated_files["m"])}
            if anneal_steps > 0:
                prep_options = deepcopy(options)
                prep_options["INITIAL_STATE_FILE"] = str(equilibrated_files["m"])
                prep_worker = OMMWorkerATMNEQTI(
                    basename,
                    endpoint_system,
                    prep_options,
                    node_info=_select_node_info(options, neqti_options),
                    compute=True,
                    logger=logger,
                    switch_schedules={name: [stateparams[index] for index in path] for name, path in paths.items()},
                    steps_per_segment=anneal_steps,
                    random_seed=neqti_options["random_seed"],
                )
                try:
                    prep_worker.simulation.loadState(str(equilibrated_files["m"]))
                    _run_preparation_anneal(
                        prep_worker,
                        "leg_a_reverse",
                        states["m"],
                        paths["leg_a_reverse"],
                        anneal_steps,
                        "neqti_endpoint_A_annealed.xml",
                        "neqti_endpoint_A_annealed.pdb",
                        logger,
                    )
                    endpoint_sources["a"] = "neqti_endpoint_A_annealed.xml"
                    prep_worker.simulation.loadState(str(equilibrated_files["m"]))
                    _run_preparation_anneal(
                        prep_worker,
                        "leg_b_reverse",
                        states["m_for_b"],
                        paths["leg_b_reverse"],
                        anneal_steps,
                        "neqti_endpoint_B_annealed.xml",
                        "neqti_endpoint_B_annealed.pdb",
                        logger,
                    )
                    endpoint_sources["b"] = "neqti_endpoint_B_annealed.xml"
                finally:
                    prep_worker.finish()

            if neqti_options.get("endpoint_system", "atm") == "native":
                for name in ("a", "b"):
                    native_system = create_native_endpoint_system(
                        endpoint_system, name, rest2=True, logger=logger
                    )
                    native_input = Path(f"neqti_endpoint_{name.upper()}_native_input.xml")
                    write_converted_state(
                        endpoint_sources[name],
                        native_system,
                        native_input,
                        endpoint=name,
                        keywords=options,
                        to_native=True,
                        platform=platform,
                        platform_properties=platform_properties,
                    )
                    if endpoint_steps is None:
                        equilibrated_files[name].write_text(native_input.read_text())
                        continue
                    run_custom_equilibration(
                        ommsystem=native_system,
                        steps=endpoint_steps,
                        platform=platform,
                        platform_properties=platform_properties,
                        output_dir=Path("equilibration") / f"neqti_{name}",
                        final_state_path=equilibrated_files[name],
                        final_pdb_path={"a": "neqti_endpoint_A.pdb", "b": "neqti_endpoint_B.pdb"}[name],
                        initial_state_path=native_input,
                        atm_state=None,
                        swapped_diagnostics_keywords=options,
                        selection_endpoint=name,
                        logger=logger,
                    )
                    logger.info("Completed native NEQTI %s endpoint equilibration", name)
            elif endpoint_steps is not None:
                for name in ("a", "b"):
                    run_custom_equilibration(
                        ommsystem=endpoint_system,
                        steps=endpoint_steps,
                        platform=platform,
                        platform_properties=platform_properties,
                        output_dir=Path("equilibration") / f"neqti_{name}",
                        final_state_path=equilibrated_files[name],
                        final_pdb_path={"a": "neqti_endpoint_A.pdb", "b": "neqti_endpoint_B.pdb"}[name],
                        initial_state_path=endpoint_sources[name],
                        atm_state=states[name],
                        selection_endpoint=name,
                        logger=logger,
                    )
                    logger.info("Completed NEQTI %s equilibration", name)
            marker.write_text(protocol_signature + "\n")
        state_files.update({name: str(path) for name, path in equilibrated_files.items() if path.exists()})

    node_info = _select_node_info(options, neqti_options)
    rest2_enabled = neqti_options.get("rest2", {}).get("enabled", False)
    system_options = deepcopy(options)
    atm_rest2_enabled = rest2_enabled and neqti_options.get("endpoint_system", "atm") == "atm"
    system_options["REST2_ENABLED"] = atm_rest2_enabled
    system_options["REST2_SOLUTE"] = neqti_options.get("rest2", {}).get(
        "solute", '#ligand:"*"'
    )
    ommsystem = OMMSystemRBFE(
        basename, system_options, basename + ".pdb", basename + "_sys.xml", logger
    )
    worker_options = deepcopy(options)
    worker_options["REST2_ENABLED"] = atm_rest2_enabled
    worker_options["INITIAL_STATE_FILE"] = state_files["m"]
    if not neqti_options["resume"] and rest2_enabled:
        # Sampler constructors create their checkpoint banks. Remove stale banks
        # before constructing any resident contexts, not after they are live.
        shutil.rmtree("neqti_rest2", ignore_errors=True)
    use_custom_worker = (
        neqti_options["switch_integrator"] == "custom"
        or neqti_options["validate_switch_integrator"]
    )
    nsegments = len(paths["leg_a_forward"]) - 1
    segment_steps = {
        name: [neqti_options["switch_steps_per_segment"]] * nsegments for name in paths
    }
    if use_custom_worker:
        worker = OMMWorkerATMNEQTI(
            basename,
            ommsystem,
            worker_options,
            node_info=node_info,
            compute=True,
            logger=logger,
            switch_schedules={name: [stateparams[index] for index in path] for name, path in paths.items()},
            steps_per_segment=segment_steps,
            random_seed=neqti_options["random_seed"],
            work_sample_intervals=neqti_options["work_sample_intervals"],
        )
    else:
        worker = OMMWorkerATMSync(
            basename,
            ommsystem,
            worker_options,
            node_info=node_info,
            compute=True,
            logger=logger,
        )

    rest2_sampler = None
    native_endpoint_resources = {}
    if atm_rest2_enabled:
        set_rest2_scale(worker.context, 1.0, ommsystem.rest2_system)
        rest2_sampler = create_rest2_exchange_sampler(
            system=worker.system,
            topology=worker.topology,
            base_integrator=worker.equilibrium_integrator if use_custom_worker else worker.integrator,
            ommsystem=ommsystem,
            state_files={"a": state_files["a"], "m": state_files["m"], "b": state_files["b"]},
            atm_states={"a": states["a"], "m": states["m"], "b": states["b"]},
            config=neqti_options["rest2"],
            platform=worker.platform,
            platform_properties=worker.platform_properties,
            resume=neqti_options["resume"],
            random_seed=neqti_options["random_seed"],
            logger=logger,
        )
        logger.info(
            "NEQTI REST2 enabled with %d replicas spanning %.1f-%.1f K",
            len(neqti_options["rest2"]["effective_temperatures_k"]),
            neqti_options["rest2"]["effective_temperatures_k"][0],
            neqti_options["rest2"]["effective_temperatures_k"][-1],
        )

    if neqti_options.get("endpoint_system") == "native" and neqti_options["sampling_order"] == "interleaved":
        try:
            for ensemble in ("a", "b"):
                native_system = create_native_endpoint_system(
                    ommsystem, ensemble, rest2=True, logger=logger
                )
                sampler = create_rest2_exchange_sampler(
                    system=native_system.system,
                    topology=native_system.topology,
                    base_integrator=native_system.integrator,
                    rest2_system=native_system.rest2_system,
                    state_files={ensemble: state_files[ensemble]},
                    config=neqti_options["rest2"],
                    platform=worker.platform,
                    platform_properties=worker.platform_properties,
                    output_dir="neqti_rest2",
                    resume=neqti_options["resume"],
                    random_seed=neqti_options["random_seed"] + (10000 if ensemble == "b" else 0),
                    logger=logger,
                )
                native_endpoint_resources[ensemble] = (native_system, sampler)
            backend = neqti_options["rest2"].get("sampler_backend", "custom")
            contexts_per_ladder = (
                1 if backend == "openmm_native"
                else len(neqti_options["rest2"]["effective_temperatures_k"])
            )
            logger.info(
                "Native interleaved NEQTI initialized two resident REST2 ladders "
                "(%d context%s each, sampler backend %s) and one ATM worker context",
                contexts_per_ladder,
                "" if contexts_per_ladder == 1 else "s",
                backend,
            )
        except Exception as exc:
            for _system, sampler in native_endpoint_resources.values():
                sampler.close()
            worker.finish()
            raise NEQTIConfigError(
                "Could not allocate resident native endpoint REST2 ladders; use sampling_order: batched on a memory-limited GPU"
            ) from exc

    work_files = {name: Path(f"neqti_{name}.csv") for name in paths}
    checkpoints = {"m": Path("neqti_m_sampling.chk"), "a": Path("neqti_a_sampling.chk"), "b": Path("neqti_b_sampling.chk")}
    if not neqti_options["resume"]:
        for path in [
            *work_files.values(),
            *checkpoints.values(),
            Path("neqti_summary.yaml"),
            Path("neqti_schedule_optimization.yaml"),
            Path("neqti_convergence.yaml"),
        ]:
            if path.exists():
                path.unlink()
        shutil.rmtree("neqti_schedule_optimization", ignore_errors=True)
    for path in work_files.values():
        _ensure_work_csv(path, neqti_options["work_sample_intervals"])

    def execute_switch(switch_name, start_state, schedule, path, label):
        if atm_rest2_enabled:
            set_rest2_scale(worker.context, 1.0, ommsystem.rest2_system)
        if neqti_options["switch_integrator"] == "custom":
            work_result = _run_switch_custom(
                worker,
                start_state,
                segment_steps[switch_name],
                path,
                switch_name,
                logger,
                label,
            )
        else:
            switch_args = (
                worker,
                start_state,
                schedule,
                neqti_options["switch_steps_per_segment"],
                path,
                logger,
                label,
            )
            work_result = (
                _run_switch(
                    *switch_args,
                    work_sample_intervals=neqti_options["work_sample_intervals"],
                )
                if neqti_options["work_sample_intervals"]
                else _run_switch(*switch_args)
            )
        if not isinstance(work_result, dict):
            work_result = {"exact": work_result, "sampled": {}}
        values = [work_result["exact"], *work_result["sampled"].values()]
        if not all(math.isfinite(float(value)) for value in values):
            raise NEQTINonfiniteWorkError(f"Switch returned non-finite protocol work: {work_result}")
        return work_result

    def completed_work_row(switch_name, attempt, work_result):
        exact = float(work_result["exact"])
        row = {
            "trajectory": attempt,
            "direction": switch_name,
            "start_state": paths[switch_name][0],
            "end_state": paths[switch_name][-1],
            "work_kcal_per_mol": f"{exact:.12g}",
            "work_kj_per_mol": f"{exact * KCAL_TO_KJ:.12g}",
            "switch_steps": sum(segment_steps[switch_name]),
            "status": "complete",
        }
        for interval, value in work_result["sampled"].items():
            row[_work_column(interval)] = f"{float(value):.12g}"
            row[f"work_interval_{interval}_kj_per_mol"] = f"{float(value) * KCAL_TO_KJ:.12g}"
        return row

    def current_counts():
        return {name: len(_read_analyzed_rows(path)) for name, path in work_files.items()}

    def emit_progress(with_analysis=False):
        if progress_callback is None:
            return
        counts = current_counts()
        analysis_summary = None
        if with_analysis and all(counts.values()):
            analysis_summary = summarize_existing_neqti_work(
                options,
                neqti_options,
                paths,
                bootstrap_samples=0,
            )
        progress_callback({
            "jobname": basename,
            "method": "neqti",
            "status": "partial",
            "forward_samples": counts["leg_a_forward"] + counts["leg_b_forward"],
            "reverse_samples": counts["leg_a_reverse"] + counts["leg_b_reverse"],
            "sample_counts": counts,
            "finite_sample_counts": {
                name: len(_read_completed_rows(path)) for name, path in work_files.items()
            },
            "counted_infinite_work_counts": {
                name: len(_read_counted_infinite_rows(path)) for name, path in work_files.items()
            },
            "analysis": None if analysis_summary is None else analysis_summary.get("analysis"),
            "work_estimator_analyses": (
                None if analysis_summary is None
                else analysis_summary.get("work_estimator_analyses")
            ),
            "warnings": [] if analysis_summary is None else analysis_summary.get("warnings", []),
        })

    def record_switch_failure(switch_name, attempt, exc):
        policy = neqti_options.get(
            "failed_switch_policy",
            "retry" if neqti_options.get("tolerate_failed_switches", False) else "abort",
        )
        counted = policy == "count_as_infinite" and _is_numerical_switch_failure(exc)
        row = {
            "trajectory": attempt, "direction": switch_name,
            "start_state": paths[switch_name][0], "end_state": paths[switch_name][-1],
            "work_kcal_per_mol": "inf" if counted else "",
            "work_kj_per_mol": "inf" if counted else "",
            "switch_steps": sum(segment_steps[switch_name]),
            "status": "counted_infinite" if counted else "failed",
            "error_type": type(exc).__name__, "error_message": str(exc),
        }
        for interval in neqti_options["work_sample_intervals"]:
            row[_work_column(interval)] = "inf" if counted else ""
            row[f"work_interval_{interval}_kj_per_mol"] = "inf" if counted else ""
        _append_row(work_files[switch_name], row, neqti_options["work_sample_intervals"])
        if counted:
            logger.warning(
                "NEQTI %s trajectory %d failed numerically and was counted as +infinite work: %s",
                switch_name, attempt, exc,
            )
            return
        if policy == "retry":
            logger.warning(
                "NEQTI %s trajectory %d failed and will be retried with a later sample: %s",
                switch_name, attempt, exc,
            )
        else:
            logger.error("NEQTI %s trajectory %d failed: %s", switch_name, attempt, exc)
            raise exc

    endpoint_stream_specs = (
        ("a", "leg_a_forward"),
        ("b", "leg_b_forward"),
    )

    def completed_count(name):
        return len(_read_analyzed_rows(work_files[name]))

    def attempted_count(name):
        return completed_count(name) + len(_read_failed_rows(work_files[name]))

    def needs_attempt(name, attempt):
        return completed_count(name) < neqti_options["n_snapshots"] and attempted_count(name) <= attempt

    def all_targets_reached():
        return all(completed_count(name) >= neqti_options["n_snapshots"] for name in work_files)

    def run_native_endpoint_stream(ensemble, switch_name):
        completed = _read_analyzed_rows(work_files[switch_name])
        if len(completed) >= neqti_options["n_snapshots"]:
            return
        native_system = create_native_endpoint_system(
            ommsystem, ensemble, rest2=True, logger=logger
        )
        sampler = create_rest2_exchange_sampler(
            system=native_system.system,
            topology=native_system.topology,
            base_integrator=native_system.integrator,
            rest2_system=native_system.rest2_system,
            state_files={ensemble: state_files[ensemble]},
            config=neqti_options["rest2"],
            platform=worker.platform,
            platform_properties=worker.platform_properties,
            output_dir="neqti_rest2",
            resume=True,
            random_seed=neqti_options["random_seed"] + (10000 if ensemble == "b" else 0),
            logger=logger,
        )
        try:
            existing_bank = sampler.has_bank(ensemble)
            sampler.activate(ensemble)
            if not existing_bank and not completed and neqti_options["initial_equilibration_steps"] > 0:
                sampler.run_steps(
                    ensemble,
                    neqti_options["initial_equilibration_steps"],
                    f"NEQTI native {ensemble} initial REST2 equilibration",
                )
            elif existing_bank:
                logger.info(
                    "Resuming native NEQTI %s REST2 sampling after %d analyzed trajectories",
                    ensemble,
                    len(completed),
                )
            elif completed:
                logger.info(
                    "Initializing missing native NEQTI %s REST2 bank after %d analyzed "
                    "trajectories; skipping initial equilibration",
                    ensemble,
                    len(completed),
                )

            all_existing = len(completed) + len(_read_failed_rows(work_files[switch_name]))
            for attempt in range(
                all_existing, neqti_options["max_switch_attempts_per_direction"]
            ):
                if len(_read_analyzed_rows(work_files[switch_name])) >= neqti_options["n_snapshots"]:
                    break
                if neqti_options["decorrelation_steps"] > 0:
                    sampler.run_steps(
                        ensemble,
                        neqti_options["decorrelation_steps"],
                        f"NEQTI native {ensemble} trajectory {attempt} decorrelation",
                    )
                native_state = sampler.physical_state(ensemble)
                native_pdb = Path(f"neqti_{ensemble}_native_snapshot_{attempt}.pdb")
                with native_pdb.open("w") as handle:
                    PDBFile.writeFile(
                        native_system.topology,
                        native_state.getPositions(),
                        handle,
                        keepIds=True,
                    )
                transfer_state_to_context(
                    native_state,
                    worker.context,
                    endpoint=ensemble,
                    keywords=options,
                    to_native=False,
                )
                box_vectors = native_state.getPeriodicBoxVectors()
                if box_vectors is not None:
                    worker.topology.setPeriodicBoxVectors(box_vectors)
                worker.set_state(stateparams[paths[switch_name][0]])
                snapshot = worker.get_chkpt()
                _write_worker_pdb_pair(
                    worker, f"neqti_{ensemble}_snapshot_{attempt}.pdb"
                )
                if neqti_options["validate_switch_integrator"]:
                    _validate_switch_implementations(
                        worker,
                        direction=switch_name,
                        snapshot=snapshot,
                        start_state=stateparams[paths[switch_name][0]],
                        schedule=schedules[switch_name],
                        steps_per_segment=neqti_options["switch_steps_per_segment"],
                        state_path=paths[switch_name],
                        logger=logger,
                    )
                try:
                    work_result = execute_switch(
                        switch_name,
                        stateparams[paths[switch_name][0]],
                        schedules[switch_name],
                        paths[switch_name],
                        f"{switch_name} trajectory {attempt}",
                    )
                    _write_worker_pdb_pair(
                        worker,
                        f"neqti_{ensemble}_{switch_name}_snapshot_{attempt}_post_switch.pdb",
                    )
                    _append_row(
                        work_files[switch_name],
                        completed_work_row(switch_name, attempt, work_result),
                        neqti_options["work_sample_intervals"],
                    )
                except Exception as exc:
                    record_switch_failure(switch_name, attempt, exc)
                finally:
                    worker.set_chkpt(snapshot)
                    worker.set_state(states[ensemble])
                emit_progress(with_analysis=True)
        finally:
            sampler.close()

    def run_interleaved_sampling():
        stream_initialized = {"m": False, "a": False, "b": False}

        def native_rest2_snapshot(ensemble, attempt, label_prefix="trajectory"):
            native_system, sampler = native_endpoint_resources[ensemble]
            if not stream_initialized[ensemble]:
                existing_bank = sampler.has_bank(ensemble)
                sampler.activate(ensemble)
                if not existing_bank and neqti_options["initial_equilibration_steps"] > 0:
                    sampler.run_steps(
                        ensemble,
                        neqti_options["initial_equilibration_steps"],
                        f"NEQTI native {ensemble} initial REST2 equilibration",
                    )
                elif existing_bank:
                    logger.info("Resuming resident native NEQTI %s REST2 bank", ensemble)
                stream_initialized[ensemble] = True
            if neqti_options["decorrelation_steps"] > 0:
                sampler.run_steps(
                    ensemble,
                    neqti_options["decorrelation_steps"],
                    f"NEQTI native {ensemble} {label_prefix} {attempt} decorrelation",
                )
            native_state = sampler.physical_state(ensemble)
            transfer_state_to_context(
                native_state,
                worker.context,
                endpoint=ensemble,
                keywords=options,
                to_native=False,
            )
            box_vectors = native_state.getPeriodicBoxVectors()
            if box_vectors is not None:
                worker.topology.setPeriodicBoxVectors(box_vectors)
            worker.set_state(states[ensemble])
            return native_state, native_system

        def transfer_rest2_physical(ensemble):
            state = rest2_sampler.physical_state(ensemble)
            worker.context.setPositions(state.getPositions())
            worker.context.setVelocities(state.getVelocities())
            box_vectors = state.getPeriodicBoxVectors()
            if box_vectors is not None:
                worker.context.setPeriodicBoxVectors(*box_vectors)
                worker.topology.setPeriodicBoxVectors(box_vectors)
            worker.set_state(states[ensemble])
            set_rest2_scale(worker.context, 1.0, ommsystem.rest2_system)

        def rest2_snapshot(ensemble, attempt):
            if neqti_options["decorrelation_steps"] > 0:
                rest2_sampler.run_steps(
                    ensemble,
                    neqti_options["decorrelation_steps"],
                    f"NEQTI {ensemble} trajectory {attempt} decorrelation",
                )
            transfer_rest2_physical(ensemble)
            return worker.get_chkpt()

        def initialize_stream_once(direction, initial_state_file, start_state, checkpoint_file, completed):
            if rest2_sampler is not None:
                if not stream_initialized[direction]:
                    existing_bank = rest2_sampler.has_bank(direction)
                    rest2_sampler.activate(direction)
                    if existing_bank:
                        logger.info(
                            "Resuming NEQTI %s REST2 sampling after %d completed trajectories; "
                            "skipping initial equilibration",
                            direction, completed,
                        )
                    elif completed > 0:
                        logger.info(
                            "Initializing missing NEQTI %s REST2 bank after %d completed trajectories; "
                            "skipping initial equilibration",
                            direction, completed,
                        )
                    elif neqti_options["initial_equilibration_steps"] > 0:
                        rest2_sampler.run_steps(
                            direction,
                            neqti_options["initial_equilibration_steps"],
                            f"NEQTI {direction} initial REST2 equilibration",
                        )
                    stream_initialized[direction] = True
                else:
                    rest2_sampler.activate(direction)
                transfer_rest2_physical(direction)
                return
            if stream_initialized[direction]:
                if checkpoint_file.exists():
                    worker.set_chkpt(checkpoint_file.read_bytes())
                    worker.set_state(start_state)
                return
            _initialize_sampling_stream(
                worker,
                direction=direction,
                initial_state_file=initial_state_file,
                start_state=start_state,
                checkpoint_file=checkpoint_file,
                completed_count=completed,
                initial_equilibration_steps=neqti_options["initial_equilibration_steps"],
                resume=neqti_options["resume"],
                logger=logger,
            )
            stream_initialized[direction] = True

        def apply_segment_steps(updated):
            for direction, values in updated.items():
                segment_steps[direction] = [int(value) for value in values]
                worker.set_switch_segment_steps(direction, segment_steps[direction])

        def run_schedule_optimization():
            settings = neqti_options.get("schedule_optimization", {})
            if not settings.get("enabled", False):
                return
            state_path = Path("neqti_schedule_optimization.yaml")
            output_dir = Path("neqti_schedule_optimization")
            output_dir.mkdir(exist_ok=True)
            pilot_files = {
                name: output_dir / f"pilot_{name}.csv" for name in work_files
            }
            for path in pilot_files.values():
                _ensure_work_csv(path, neqti_options["work_sample_intervals"])
            if state_path.exists() and neqti_options["resume"]:
                with state_path.open() as handle:
                    optimizer_state = yaml.safe_load(handle) or {}
            else:
                optimizer_state = {
                    "schema_version": 1,
                    "status": "running",
                    "completed_pilot_cycles": 0,
                    "pilot_samples": settings["pilot_samples"],
                    "scores": {},
                    "history": [],
                    "segment_steps": deepcopy(segment_steps),
                }
            apply_segment_steps(optimizer_state.get("segment_steps", segment_steps))
            completed_cycles = int(optimizer_state.get("completed_pilot_cycles", 0))
            for path in pilot_files.values():
                with path.open() as handle:
                    existing_rows = list(csv.DictReader(handle))
                committed_rows = [
                    row for row in existing_rows
                    if int(row.get("trajectory", -1)) < completed_cycles
                ]
                if len(committed_rows) != len(existing_rows):
                    with path.open("w", newline="") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=_work_csv_fields(neqti_options["work_sample_intervals"]),
                        )
                        writer.writeheader()
                        writer.writerows(committed_rows)
            if optimizer_state.get("status") == "frozen":
                logger.info(
                    "Restored frozen NEQTI schedules after %d pilot cycles", completed_cycles
                )
                return

            for cycle in range(completed_cycles, settings["pilot_samples"]):
                logger.info(
                    "NEQTI schedule optimization cycle %d/%d starting with A steps %s and B steps %s",
                    cycle + 1,
                    settings["pilot_samples"],
                    segment_steps["leg_a_forward"],
                    segment_steps["leg_b_forward"],
                )
                initialize_stream_once(
                    "m", state_files["m"], states["m"], checkpoints["m"], cycle
                )
                if neqti_options["decorrelation_steps"] > 0:
                    _run_worker_steps(
                        worker,
                        neqti_options["decorrelation_steps"],
                        logger,
                        f"NEQTI optimizer m cycle {cycle} decorrelation",
                    )
                midpoint_snapshot = worker.get_chkpt()
                pilot_results = {}
                for switch_name in ("leg_a_reverse", "leg_b_reverse"):
                    worker.set_chkpt(midpoint_snapshot)
                    result = execute_switch(
                        switch_name,
                        stateparams[paths[switch_name][0]],
                        schedules[switch_name],
                        paths[switch_name],
                        f"optimizer {switch_name} cycle {cycle}",
                    )
                    pilot_results[switch_name] = result
                    _append_row(
                        pilot_files[switch_name],
                        completed_work_row(switch_name, cycle, result),
                        neqti_options["work_sample_intervals"],
                    )
                worker.set_chkpt(midpoint_snapshot)
                worker.set_state(states["m"])
                _write_sampling_checkpoint(checkpoints["m"], worker)

                for ensemble, switch_name in endpoint_stream_specs:
                    native_state, _native_system = native_rest2_snapshot(
                        ensemble, cycle, label_prefix="optimizer cycle"
                    )
                    snapshot = worker.get_chkpt()
                    result = execute_switch(
                        switch_name,
                        stateparams[paths[switch_name][0]],
                        schedules[switch_name],
                        paths[switch_name],
                        f"optimizer {switch_name} cycle {cycle}",
                    )
                    pilot_results[switch_name] = result
                    _append_row(
                        pilot_files[switch_name],
                        completed_work_row(switch_name, cycle, result),
                        neqti_options["work_sample_intervals"],
                    )
                    worker.set_chkpt(snapshot)
                    worker.set_state(states[ensemble])

                history_record = {"cycle": cycle + 1, "legs": {}}
                updated_steps = deepcopy(segment_steps)
                for leg in settings["optimize_legs"]:
                    forward_name = f"leg_{leg}_forward"
                    reverse_name = f"leg_{leg}_reverse"
                    cycle_scores = _optimizer_cycle_scores(
                        pilot_results[forward_name]["segment_work"],
                        pilot_results[reverse_name]["segment_work"],
                        settings,
                    )
                    previous_scores = optimizer_state["scores"].get(leg)
                    aggregate_scores = (
                        cycle_scores
                        if previous_scores is None
                        else settings["score_ewma_alpha"] * cycle_scores
                        + (1.0 - settings["score_ewma_alpha"]) * np.asarray(previous_scores)
                    )
                    optimizer_state["scores"][leg] = aggregate_scores.tolist()
                    previous_steps = segment_steps[forward_name]
                    next_steps = _allocate_segment_steps(
                        aggregate_scores,
                        sum(previous_steps),
                        previous_steps,
                        settings,
                    )
                    updated_steps[forward_name] = next_steps
                    updated_steps[reverse_name] = list(reversed(next_steps))
                    history_record["legs"][leg] = {
                        "cycle_scores": cycle_scores.tolist(),
                        "aggregate_scores": aggregate_scores.tolist(),
                        "previous_steps": list(previous_steps),
                        "next_steps": list(next_steps),
                        "allocation_change_fraction": _schedule_change_fraction(
                            previous_steps, next_steps
                        ),
                    }
                apply_segment_steps(updated_steps)
                optimizer_state["completed_pilot_cycles"] = cycle + 1
                optimizer_state["segment_steps"] = deepcopy(segment_steps)
                optimizer_state["history"].append(history_record)
                optimizer_state["status"] = (
                    "frozen" if cycle + 1 >= settings["pilot_samples"] else "running"
                )
                temporary = state_path.with_suffix(".yaml.tmp")
                with temporary.open("w") as handle:
                    yaml.safe_dump(optimizer_state, handle, sort_keys=False)
                os.replace(temporary, state_path)
                logger.info(
                    "NEQTI schedule optimization cycle %d complete; A steps %s, B steps %s",
                    cycle + 1,
                    segment_steps["leg_a_forward"],
                    segment_steps["leg_b_forward"],
                )

            logger.info("NEQTI schedule optimization frozen after %d pilot cycles", settings["pilot_samples"])

        run_schedule_optimization()

        convergence_settings = neqti_options.get("convergence", {})
        convergence_path = Path("neqti_convergence.yaml")
        if convergence_path.exists() and neqti_options["resume"]:
            with convergence_path.open() as handle:
                convergence_state = yaml.safe_load(handle) or {}
        else:
            convergence_state = {"schema_version": 1, "history": [], "termination_reason": None}
        if convergence_state.get("termination_reason") in ("converged", "max_samples"):
            logger.info(
                "NEQTI production already terminated with reason %s",
                convergence_state["termination_reason"],
            )
            return

        def update_convergence():
            if not convergence_settings.get("enabled", False):
                return False
            counts = current_counts()
            if not all(
                value >= convergence_settings["min_samples_per_direction"]
                for value in counts.values()
            ):
                return False
            analysis_summary = summarize_existing_neqti_work(
                options, neqti_options, paths
            )
            record = _convergence_record(
                analysis_summary.get("analysis"), counts, convergence_settings
            )
            convergence_state["history"].append(record)
            reached = _convergence_reached(
                convergence_state["history"], convergence_settings
            )
            if reached:
                convergence_state["termination_reason"] = "converged"
            temporary = convergence_path.with_suffix(".yaml.tmp")
            with temporary.open("w") as handle:
                yaml.safe_dump(convergence_state, handle, sort_keys=False)
            os.replace(temporary, convergence_path)
            logger.info(
                "NEQTI convergence check at %d samples/direction: DDG=%s, error=%s, overlaps=[%s, %s], reached=%s",
                record["sample_count_per_direction"],
                record["ddg_kcal_per_mol"],
                record["ddg_error_kcal_per_mol"],
                record["leg_a_overlap_score"],
                record["leg_b_overlap_score"],
                reached,
            )
            return reached

        for attempt in range(
            min(attempted_count(name) for name in work_files),
            neqti_options["max_switch_attempts_per_direction"],
        ):
            if all_targets_reached():
                break
            cycle_changed = False

            midpoint_switches = ("leg_a_reverse", "leg_b_reverse")
            if any(needs_attempt(name, attempt) for name in midpoint_switches):
                initialize_stream_once(
                    "m",
                    state_files["m"],
                    states["m"],
                    checkpoints["m"],
                    max(completed_count(name) for name in midpoint_switches),
                )
                if rest2_sampler is None and neqti_options["decorrelation_steps"] > 0:
                    _run_worker_steps(
                        worker,
                        neqti_options["decorrelation_steps"],
                        logger,
                        f"NEQTI m trajectory {attempt} decorrelation",
                    )
                snapshot = (
                    rest2_snapshot("m", attempt)
                    if rest2_sampler is not None else worker.get_chkpt()
                )
                _write_worker_pdb_pair(worker, f"neqti_m_snapshot_{attempt}.pdb")
                for switch_name in midpoint_switches:
                    if not needs_attempt(switch_name, attempt):
                        continue
                    worker.set_chkpt(snapshot)
                    worker.set_state(stateparams[paths[switch_name][0]])
                    if neqti_options["validate_switch_integrator"]:
                        _validate_switch_implementations(
                            worker,
                            direction=switch_name,
                            snapshot=snapshot,
                            start_state=stateparams[paths[switch_name][0]],
                            schedule=schedules[switch_name],
                            steps_per_segment=neqti_options["switch_steps_per_segment"],
                            state_path=paths[switch_name],
                            logger=logger,
                        )
                    try:
                        work_result = execute_switch(
                            switch_name,
                            stateparams[paths[switch_name][0]],
                            schedules[switch_name],
                            paths[switch_name],
                            f"{switch_name} trajectory {attempt}",
                        )
                        _write_worker_pdb_pair(worker, f"neqti_m_{switch_name}_snapshot_{attempt}_post_switch.pdb")
                        _append_row(
                            work_files[switch_name],
                            completed_work_row(switch_name, attempt, work_result),
                            neqti_options["work_sample_intervals"],
                        )
                    except Exception as exc:
                        record_switch_failure(switch_name, attempt, exc)
                    finally:
                        worker.set_chkpt(snapshot)
                        worker.set_state(states["m"])
                    cycle_changed = True
                if rest2_sampler is None:
                    _write_sampling_checkpoint(checkpoints["m"], worker)

            for ensemble, switch_name in endpoint_stream_specs:
                if not needs_attempt(switch_name, attempt):
                    continue
                if native_endpoint_resources:
                    native_state, native_system = native_rest2_snapshot(ensemble, attempt)
                    native_pdb = Path(f"neqti_{ensemble}_native_snapshot_{attempt}.pdb")
                    with native_pdb.open("w") as handle:
                        PDBFile.writeFile(
                            native_system.topology,
                            native_state.getPositions(),
                            handle,
                            keepIds=True,
                        )
                    snapshot = worker.get_chkpt()
                    _write_worker_pdb_pair(worker, f"neqti_{ensemble}_snapshot_{attempt}.pdb")
                    try:
                        work_result = execute_switch(
                            switch_name,
                            stateparams[paths[switch_name][0]],
                            schedules[switch_name],
                            paths[switch_name],
                            f"{switch_name} trajectory {attempt}",
                        )
                        _write_worker_pdb_pair(
                            worker,
                            f"neqti_{ensemble}_{switch_name}_snapshot_{attempt}_post_switch.pdb",
                        )
                        _append_row(
                            work_files[switch_name],
                            completed_work_row(switch_name, attempt, work_result),
                            neqti_options["work_sample_intervals"],
                        )
                    except Exception as exc:
                        record_switch_failure(switch_name, attempt, exc)
                    finally:
                        worker.set_chkpt(snapshot)
                        worker.set_state(states[ensemble])
                    cycle_changed = True
                    continue
                initialize_stream_once(
                    ensemble,
                    state_files[ensemble],
                    states[ensemble],
                    checkpoints[ensemble],
                    completed_count(switch_name),
                )
                if rest2_sampler is None and neqti_options["decorrelation_steps"] > 0:
                    _run_worker_steps(
                        worker,
                        neqti_options["decorrelation_steps"],
                        logger,
                        f"NEQTI {ensemble} trajectory {attempt} decorrelation",
                    )
                snapshot = (
                    rest2_snapshot(ensemble, attempt)
                    if rest2_sampler is not None else worker.get_chkpt()
                )
                _write_worker_pdb_pair(worker, f"neqti_{ensemble}_snapshot_{attempt}.pdb")
                if neqti_options["validate_switch_integrator"]:
                    _validate_switch_implementations(
                        worker,
                        direction=switch_name,
                        snapshot=snapshot,
                        start_state=stateparams[paths[switch_name][0]],
                        schedule=schedules[switch_name],
                        steps_per_segment=neqti_options["switch_steps_per_segment"],
                        state_path=paths[switch_name],
                        logger=logger,
                    )
                try:
                    work_result = execute_switch(
                        switch_name,
                        stateparams[paths[switch_name][0]],
                        schedules[switch_name],
                        paths[switch_name],
                        f"{switch_name} trajectory {attempt}",
                    )
                    _write_worker_pdb_pair(worker, f"neqti_{ensemble}_{switch_name}_snapshot_{attempt}_post_switch.pdb")
                    _append_row(
                        work_files[switch_name],
                        completed_work_row(switch_name, attempt, work_result),
                        neqti_options["work_sample_intervals"],
                    )
                except Exception as exc:
                    record_switch_failure(switch_name, attempt, exc)
                finally:
                    worker.set_chkpt(snapshot)
                    worker.set_state(states[ensemble])
                    if rest2_sampler is None:
                        _write_sampling_checkpoint(checkpoints[ensemble], worker)
                cycle_changed = True

            if cycle_changed:
                emit_progress(with_analysis=True)
                if update_convergence():
                    break

        if convergence_settings.get("enabled", False) and not convergence_state.get("termination_reason"):
            if all_targets_reached():
                convergence_state["termination_reason"] = "max_samples"
                temporary = convergence_path.with_suffix(".yaml.tmp")
                with temporary.open("w") as handle:
                    yaml.safe_dump(convergence_state, handle, sort_keys=False)
                os.replace(temporary, convergence_path)

    if neqti_options["sampling_order"] == "interleaved":
        try:
            run_interleaved_sampling()
        finally:
            if rest2_sampler is not None:
                rest2_sampler.close()
            for _native_system, sampler in native_endpoint_resources.values():
                sampler.close()
            worker.finish()
        return summarize_existing_neqti_work(options, neqti_options, paths)

    try:
        midpoint_switches = ("leg_a_reverse", "leg_b_reverse")
        if any(len(_read_analyzed_rows(work_files[name])) < neqti_options["n_snapshots"] for name in midpoint_switches):
            completed_count = max(len(_read_analyzed_rows(work_files[name])) for name in midpoint_switches)
            _initialize_sampling_stream(
                worker,
                direction="m",
                initial_state_file=state_files["m"],
                start_state=states["m"],
                checkpoint_file=checkpoints["m"],
                completed_count=completed_count,
                initial_equilibration_steps=neqti_options["initial_equilibration_steps"],
                resume=neqti_options["resume"],
                logger=logger,
            )
            all_existing = max(
                len(_read_analyzed_rows(work_files[name])) + len(_read_failed_rows(work_files[name]))
                for name in midpoint_switches
            )
            for attempt in range(all_existing, neqti_options["max_switch_attempts_per_direction"]):
                if all(len(_read_analyzed_rows(work_files[name])) >= neqti_options["n_snapshots"] for name in midpoint_switches):
                    break
                if neqti_options["decorrelation_steps"] > 0:
                    _run_worker_steps(worker, neqti_options["decorrelation_steps"], logger, f"NEQTI m trajectory {attempt} decorrelation")
                snapshot = worker.get_chkpt()
                _write_worker_pdb_pair(worker, f"neqti_m_snapshot_{attempt}.pdb")
                for switch_name in midpoint_switches:
                    if len(_read_analyzed_rows(work_files[switch_name])) >= neqti_options["n_snapshots"]:
                        continue
                    worker.set_chkpt(snapshot)
                    worker.set_state(stateparams[paths[switch_name][0]])
                    if neqti_options["validate_switch_integrator"]:
                        _validate_switch_implementations(worker, direction=switch_name, snapshot=snapshot,
                            start_state=stateparams[paths[switch_name][0]], schedule=schedules[switch_name],
                            steps_per_segment=neqti_options["switch_steps_per_segment"], state_path=paths[switch_name], logger=logger)
                    try:
                        work_result = execute_switch(switch_name, stateparams[paths[switch_name][0]], schedules[switch_name], paths[switch_name], f"{switch_name} trajectory {attempt}")
                        _write_worker_pdb_pair(worker, f"neqti_m_{switch_name}_snapshot_{attempt}_post_switch.pdb")
                        _append_row(
                            work_files[switch_name],
                            completed_work_row(switch_name, attempt, work_result),
                            neqti_options["work_sample_intervals"],
                        )
                    except Exception as exc:
                        record_switch_failure(switch_name, attempt, exc)
                    finally:
                        worker.set_chkpt(snapshot)
                        worker.set_state(states["m"])
                _write_sampling_checkpoint(checkpoints["m"], worker)
                emit_progress()

        for ensemble, switch_name in endpoint_stream_specs:
            if neqti_options.get("endpoint_system") == "native":
                run_native_endpoint_stream(ensemble, switch_name)
                continue
            completed = _read_analyzed_rows(work_files[switch_name])
            if len(completed) >= neqti_options["n_snapshots"]:
                continue
            _initialize_sampling_stream(
                worker,
                direction=ensemble,
                initial_state_file=state_files[ensemble],
                start_state=states[ensemble],
                checkpoint_file=checkpoints[ensemble],
                completed_count=len(completed),
                initial_equilibration_steps=neqti_options["initial_equilibration_steps"],
                resume=neqti_options["resume"],
                logger=logger,
            )
            all_existing = len(completed) + len(_read_failed_rows(work_files[switch_name]))
            for attempt in range(all_existing, neqti_options["max_switch_attempts_per_direction"]):
                if len(_read_analyzed_rows(work_files[switch_name])) >= neqti_options["n_snapshots"]:
                    break
                if neqti_options["decorrelation_steps"] > 0:
                    _run_worker_steps(worker, neqti_options["decorrelation_steps"], logger, f"NEQTI {ensemble} trajectory {attempt} decorrelation")
                snapshot = worker.get_chkpt()
                _write_worker_pdb_pair(worker, f"neqti_{ensemble}_snapshot_{attempt}.pdb")
                if neqti_options["validate_switch_integrator"]:
                    _validate_switch_implementations(worker, direction=switch_name, snapshot=snapshot,
                        start_state=stateparams[paths[switch_name][0]], schedule=schedules[switch_name],
                        steps_per_segment=neqti_options["switch_steps_per_segment"], state_path=paths[switch_name], logger=logger)
                try:
                    work_result = execute_switch(switch_name, stateparams[paths[switch_name][0]], schedules[switch_name], paths[switch_name], f"{switch_name} trajectory {attempt}")
                    _write_worker_pdb_pair(worker, f"neqti_{ensemble}_{switch_name}_snapshot_{attempt}_post_switch.pdb")
                    _append_row(
                        work_files[switch_name],
                        completed_work_row(switch_name, attempt, work_result),
                        neqti_options["work_sample_intervals"],
                    )
                except Exception as exc:
                    record_switch_failure(switch_name, attempt, exc)
                finally:
                    worker.set_chkpt(snapshot)
                    worker.set_state(states[ensemble])
                    _write_sampling_checkpoint(checkpoints[ensemble], worker)
                emit_progress()
    finally:
        worker.finish()

    return summarize_existing_neqti_work(options, neqti_options, paths)
