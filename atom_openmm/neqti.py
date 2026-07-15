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
import openmm as mm
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
from atom_openmm.rest2 import set_rest2_scale
from atom_openmm.rest2_exchange import REST2ExchangeSampler


KCAL_TO_KJ = 4.184
KB_KCAL_PER_MOL_K = 0.0019872041
CSV_FIELDS = [
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


def _protocol_signature(settings):
    payload = {
        "schema_version": 5,
        "hamiltonian": "atm_softplus_single_midpoint",
        "paths": settings["paths"],
        "switch_steps_per_segment": settings["switch_steps_per_segment"],
        "preparation_annealing_steps_per_segment": settings.get("preparation_annealing_steps_per_segment", 0),
        "sampling_order": settings.get("sampling_order", "interleaved"),
    }
    if settings.get("rest2", {}).get("enabled", False):
        payload["rest2"] = settings["rest2"]
    if settings.get("failed_switch_policy") == "count_as_infinite":
        payload["failed_switch_policy"] = "count_as_infinite"
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
                "Existing NEQTI artifacts use a different Hamiltonian, lambda schedule, or sampling order; "
                "remove them or set resume: false"
            )
    manifest = {
        "schema_version": 5,
        "hamiltonian": "atm_softplus_single_midpoint",
        "paths": settings["paths"],
        "switch_steps_per_segment": settings["switch_steps_per_segment"],
        "preparation_annealing_steps_per_segment": settings.get("preparation_annealing_steps_per_segment", 0),
        "sampling_order": settings.get("sampling_order", "interleaved"),
        "rest2": settings.get("rest2", {"enabled": False}),
        "failed_switch_policy": settings.get("failed_switch_policy", "abort"),
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

    switch_integrator = str(raw.get("switch_integrator", "custom")).lower()
    if switch_integrator not in ("custom", "python"):
        raise NEQTIConfigError("workflow.neqti.switch_integrator must be 'custom' or 'python'")
    sampling_order = str(raw.get("sampling_order", "interleaved")).lower()
    if sampling_order not in ("interleaved", "batched"):
        raise NEQTIConfigError("workflow.neqti.sampling_order must be 'interleaved' or 'batched'")
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
    physical_temperature = float(atom_options["TEMPERATURES"][0])
    temperatures = [
        float(value) for value in rest2_raw.get(
            "effective_temperatures_k", [physical_temperature, 351, 411, 481, 563, 658, 770, 900]
        )
    ]
    exchange_interval = int(rest2_raw.get("exchange_interval_steps", 500))
    checkpoint_interval = int(rest2_raw.get("checkpoint_interval_cycles", 10))
    solute = str(rest2_raw.get("solute", "both_ligands"))
    initial_steps = int(raw.get("initial_equilibration_steps", 0))
    decorrelation_steps = int(raw.get("decorrelation_steps", atom_options.get("PRODUCTION_STEPS", 1)))
    if rest2_enabled:
        if sampling_order != "interleaved":
            raise NEQTIConfigError("NEQTI REST2 currently requires sampling_order: interleaved")
        if solute != "both_ligands":
            raise NEQTIConfigError("workflow.neqti.rest2.solute currently supports only 'both_ligands'")
        if len(temperatures) < 2 or temperatures[0] != physical_temperature:
            raise NEQTIConfigError("REST2 temperatures must start at the physical ATM temperature")
        if any(b <= a for a, b in zip(temperatures, temperatures[1:])):
            raise NEQTIConfigError("REST2 effective temperatures must be strictly increasing")
        if exchange_interval < 1 or checkpoint_interval < 1:
            raise NEQTIConfigError("REST2 exchange and checkpoint intervals must be positive")
        for name, value in (
            ("initial_equilibration_steps", initial_steps),
            ("decorrelation_steps", decorrelation_steps),
        ):
            if value % exchange_interval:
                raise NEQTIConfigError(
                    f"workflow.neqti.{name} must be divisible by rest2.exchange_interval_steps"
                )
    rest2 = {
        "enabled": rest2_enabled,
        "solute": solute,
        "effective_temperatures_k": temperatures,
        "exchange_interval_steps": exchange_interval,
        "checkpoint_interval_cycles": checkpoint_interval,
    }

    return {
        "initial_equilibration_steps": initial_steps,
        "n_snapshots": int(raw.get("n_snapshots", atom_options.get("MAX_SAMPLES", 1))),
        "decorrelation_steps": decorrelation_steps,
        "switch_steps_per_segment": int(raw.get("switch_steps_per_segment", atom_options.get("PRODUCTION_STEPS", 1))),
        "preparation_annealing_steps_per_segment": int(raw.get("preparation_annealing_steps_per_segment", 0)),
        "hamiltonian": "atm_softplus_single_midpoint",
        "paths": paths,
        "resume": bool(raw.get("resume", True)),
        "bootstrap_samples": int(raw.get("bootstrap_samples", 200)),
        "random_seed": int(raw.get("random_seed", 2026)),
        "platform": raw.get("platform"),
        "switch_integrator": switch_integrator,
        "sampling_order": sampling_order,
        "rest2": rest2,
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


def _ensure_work_csv(path):
    if path.exists():
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def _append_row(path, row):
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(checkpoint)
    os.replace(temporary, path)


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
    _write_checkpoint(checkpoint_file, worker.get_chkpt())


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


def _run_switch(worker, start_par, schedule, steps_per_segment, state_path, logger=None, label=None):
    worker.set_state(start_par)
    work = 0.0
    total = len(schedule)
    segment_started = time.perf_counter()
    previous_log_run_timing = getattr(worker, "log_run_timing", True)
    worker.log_run_timing = False
    try:
        for step, par in enumerate(schedule, start=1):
            old_energy = _potential_kcal(worker)
            worker.set_state(par)
            new_energy = _potential_kcal(worker)
            work += new_energy - old_energy
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
    return work


def _run_switch_custom(worker, start_par, steps_per_segment, state_path, direction, logger=None, label=None):
    integrator = worker.begin_switch(direction, start_par)
    total = (len(state_path) - 1) * steps_per_segment
    try:
        for segment, (start_state, end_state) in enumerate(zip(state_path[:-1], state_path[1:]), start=1):
            segment_started = time.perf_counter()
            integrator.step(steps_per_segment)
            elapsed = time.perf_counter() - segment_started
            work = integrator.get_protocol_work() / kilocalories_per_mole
            if logger is not None and label:
                ns_per_day = _effective_ns_per_day(worker, steps_per_segment, elapsed)
                logger.info(
                    "NEQTI %s segment %d/%d complete: state %d -> %d, %d/%d steps, "
                    "work %.6g kcal/mol, %.3f ns/day",
                    label,
                    segment,
                    len(state_path) - 1,
                    start_state,
                    end_state,
                    segment * steps_per_segment,
                    total,
                    work,
                    ns_per_day,
                )
        return integrator.get_protocol_work() / kilocalories_per_mole
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
    custom_work = _run_switch_custom(
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
    python_work = _run_switch(
        worker,
        start_state,
        schedule,
        steps_per_segment,
        state_path,
        logger,
        f"validation python {direction}",
    )
    python_seconds = time.perf_counter() - python_started
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
    if dg is None or not forward_work or not reverse_work:
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
    work = {name: [float(row["work_kcal_per_mol"]) for row in values] for name, values in rows.items()}
    states = build_atm_state_parameters(options)
    temperature_kelvin = states[paths["leg_a_forward"][0]]["temperature"] / kelvin
    analysis = analyze_two_leg_work(
        work,
        temperature_kelvin,
        bootstrap_samples=neqti_options["bootstrap_samples"] if bootstrap_samples is None else bootstrap_samples,
        random_seed=neqti_options["random_seed"],
    )
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
        rest2_summary = {"states": {}}
        for ensemble in ("a", "m", "b"):
            metadata_path = Path("neqti_rest2") / ensemble / "state.json"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text())
            attempts = np.asarray(metadata.get("attempts", []), dtype=int)
            accepts = np.asarray(metadata.get("accepts", []), dtype=int)
            rates = np.divide(
                accepts, attempts,
                out=np.zeros_like(accepts, dtype=float), where=attempts > 0,
            )
            rest2_summary["states"][ensemble] = {
                "completed_cycles": int(metadata.get("cycle", 0)),
                "attempts": attempts.tolist(),
                "accepts": accepts.tolist(),
                "acceptance_rates": rates.tolist(),
                "round_trips": [int(value) for value in metadata.get("round_trips", [])],
            }
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
    summary = {
        "jobname": options["BASENAME"],
        "method": "neqti",
        "status": "completed" if complete and usable_overlap else "partial",
        "forward_samples": len(rows["leg_a_forward"]) + len(rows["leg_b_forward"]),
        "reverse_samples": len(rows["leg_a_reverse"]) + len(rows["leg_b_reverse"]),
        "sample_counts": {name: len(values) for name, values in rows.items()},
        "finite_sample_counts": finite_counts,
        "counted_infinite_work_counts": counted_infinite_counts,
        "failed_switch_counts": failed_counts,
        "temperature_kelvin": float(temperature_kelvin),
        "hamiltonian": "atm_softplus_single_midpoint",
        "paths": paths,
        "settings": neqti_options,
        "analysis": analysis,
        "rest2": rest2_summary,
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
    if endpoint_steps is not None or midpoint_steps is not None or neqti_options["preparation_annealing_steps_per_segment"] > 0:
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

            if endpoint_steps is not None:
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
                        logger=logger,
                    )
                    logger.info("Completed NEQTI %s equilibration", name)
            marker.write_text(protocol_signature + "\n")
        state_files.update({name: str(path) for name, path in equilibrated_files.items() if path.exists()})

    node_info = _select_node_info(options, neqti_options)
    rest2_enabled = neqti_options.get("rest2", {}).get("enabled", False)
    system_options = deepcopy(options)
    system_options["REST2_ENABLED"] = rest2_enabled
    ommsystem = OMMSystemRBFE(
        basename, system_options, basename + ".pdb", basename + "_sys.xml", logger
    )
    worker_options = deepcopy(options)
    worker_options["REST2_ENABLED"] = rest2_enabled
    worker_options["INITIAL_STATE_FILE"] = state_files["m"]
    use_custom_worker = (
        neqti_options["switch_integrator"] == "custom"
        or neqti_options["validate_switch_integrator"]
    )
    if use_custom_worker:
        worker = OMMWorkerATMNEQTI(
            basename,
            ommsystem,
            worker_options,
            node_info=node_info,
            compute=True,
            logger=logger,
            switch_schedules={name: [stateparams[index] for index in path] for name, path in paths.items()},
            steps_per_segment=neqti_options["switch_steps_per_segment"],
            random_seed=neqti_options["random_seed"],
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
    if rest2_enabled:
        set_rest2_scale(worker.context, 1.0, ommsystem.rest2_system)
        rest2_sampler = REST2ExchangeSampler(
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

    work_files = {name: Path(f"neqti_{name}.csv") for name in paths}
    checkpoints = {"m": Path("neqti_m_sampling.chk"), "a": Path("neqti_a_sampling.chk"), "b": Path("neqti_b_sampling.chk")}
    if not neqti_options["resume"]:
        for path in [*work_files.values(), *checkpoints.values(), Path("neqti_summary.yaml")]:
            if path.exists():
                path.unlink()
    for path in work_files.values():
        _ensure_work_csv(path)

    def execute_switch(switch_name, start_state, schedule, path, label):
        if rest2_enabled:
            set_rest2_scale(worker.context, 1.0, ommsystem.rest2_system)
        if neqti_options["switch_integrator"] == "custom":
            work_kcal = _run_switch_custom(
                worker,
                start_state,
                neqti_options["switch_steps_per_segment"],
                path,
                switch_name,
                logger,
                label,
            )
        else:
            work_kcal = _run_switch(
                worker,
                start_state,
                schedule,
                neqti_options["switch_steps_per_segment"],
                path,
                logger,
                label,
            )
        if not math.isfinite(float(work_kcal)):
            raise NEQTINonfiniteWorkError(f"Switch returned non-finite protocol work: {work_kcal}")
        return work_kcal

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
            "warnings": [] if analysis_summary is None else analysis_summary.get("warnings", []),
        })

    def record_switch_failure(switch_name, attempt, exc):
        policy = neqti_options.get(
            "failed_switch_policy",
            "retry" if neqti_options.get("tolerate_failed_switches", False) else "abort",
        )
        counted = policy == "count_as_infinite" and _is_numerical_switch_failure(exc)
        _append_row(work_files[switch_name], {
            "trajectory": attempt, "direction": switch_name,
            "start_state": paths[switch_name][0], "end_state": paths[switch_name][-1],
            "work_kcal_per_mol": "inf" if counted else "",
            "work_kj_per_mol": "inf" if counted else "",
            "switch_steps": len(schedules[switch_name]),
            "status": "counted_infinite" if counted else "failed",
            "error_type": type(exc).__name__, "error_message": str(exc),
        })
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

    def run_interleaved_sampling():
        stream_initialized = {"m": False, "a": False, "b": False}

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
                        work_kcal = execute_switch(
                            switch_name,
                            stateparams[paths[switch_name][0]],
                            schedules[switch_name],
                            paths[switch_name],
                            f"{switch_name} trajectory {attempt}",
                        )
                        _write_worker_pdb_pair(worker, f"neqti_m_{switch_name}_snapshot_{attempt}_post_switch.pdb")
                        _append_row(work_files[switch_name], {
                            "trajectory": attempt, "direction": switch_name,
                            "start_state": paths[switch_name][0], "end_state": paths[switch_name][-1],
                            "work_kcal_per_mol": f"{work_kcal:.12g}",
                            "work_kj_per_mol": f"{work_kcal * KCAL_TO_KJ:.12g}",
                            "switch_steps": len(schedules[switch_name]), "status": "complete",
                        })
                    except Exception as exc:
                        record_switch_failure(switch_name, attempt, exc)
                    finally:
                        worker.set_chkpt(snapshot)
                        worker.set_state(states["m"])
                    cycle_changed = True
                if rest2_sampler is None:
                    _write_checkpoint(checkpoints["m"], worker.get_chkpt())

            for ensemble, switch_name in endpoint_stream_specs:
                if not needs_attempt(switch_name, attempt):
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
                    work_kcal = execute_switch(
                        switch_name,
                        stateparams[paths[switch_name][0]],
                        schedules[switch_name],
                        paths[switch_name],
                        f"{switch_name} trajectory {attempt}",
                    )
                    _write_worker_pdb_pair(worker, f"neqti_{ensemble}_{switch_name}_snapshot_{attempt}_post_switch.pdb")
                    _append_row(work_files[switch_name], {
                        "trajectory": attempt, "direction": switch_name,
                        "start_state": paths[switch_name][0], "end_state": paths[switch_name][-1],
                        "work_kcal_per_mol": f"{work_kcal:.12g}",
                        "work_kj_per_mol": f"{work_kcal * KCAL_TO_KJ:.12g}",
                        "switch_steps": len(schedules[switch_name]), "status": "complete",
                    })
                except Exception as exc:
                    record_switch_failure(switch_name, attempt, exc)
                finally:
                    worker.set_chkpt(snapshot)
                    worker.set_state(states[ensemble])
                    if rest2_sampler is None:
                        _write_checkpoint(checkpoints[ensemble], worker.get_chkpt())
                cycle_changed = True

            if cycle_changed:
                emit_progress(with_analysis=True)

    if neqti_options["sampling_order"] == "interleaved":
        try:
            run_interleaved_sampling()
        finally:
            if rest2_sampler is not None:
                rest2_sampler.close()
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
                        work_kcal = execute_switch(switch_name, stateparams[paths[switch_name][0]], schedules[switch_name], paths[switch_name], f"{switch_name} trajectory {attempt}")
                        _write_worker_pdb_pair(worker, f"neqti_m_{switch_name}_snapshot_{attempt}_post_switch.pdb")
                        _append_row(work_files[switch_name], {
                            "trajectory": attempt, "direction": switch_name,
                            "start_state": paths[switch_name][0], "end_state": paths[switch_name][-1],
                            "work_kcal_per_mol": f"{work_kcal:.12g}",
                            "work_kj_per_mol": f"{work_kcal * KCAL_TO_KJ:.12g}",
                            "switch_steps": len(schedules[switch_name]), "status": "complete",
                        })
                    except Exception as exc:
                        record_switch_failure(switch_name, attempt, exc)
                    finally:
                        worker.set_chkpt(snapshot)
                        worker.set_state(states["m"])
                _write_checkpoint(checkpoints["m"], worker.get_chkpt())
                emit_progress()

        for ensemble, switch_name in endpoint_stream_specs:
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
                    work_kcal = execute_switch(switch_name, stateparams[paths[switch_name][0]], schedules[switch_name], paths[switch_name], f"{switch_name} trajectory {attempt}")
                    _write_worker_pdb_pair(worker, f"neqti_{ensemble}_{switch_name}_snapshot_{attempt}_post_switch.pdb")
                    _append_row(work_files[switch_name], {
                        "trajectory": attempt, "direction": switch_name,
                        "start_state": paths[switch_name][0], "end_state": paths[switch_name][-1],
                        "work_kcal_per_mol": f"{work_kcal:.12g}",
                        "work_kj_per_mol": f"{work_kcal * KCAL_TO_KJ:.12g}",
                        "switch_steps": len(schedules[switch_name]), "status": "complete",
                    })
                except Exception as exc:
                    record_switch_failure(switch_name, attempt, exc)
                finally:
                    worker.set_chkpt(snapshot)
                    worker.set_state(states[ensemble])
                    _write_checkpoint(checkpoints[ensemble], worker.get_chkpt())
                emit_progress()
    finally:
        worker.finish()

    return summarize_existing_neqti_work(options, neqti_options, paths)
