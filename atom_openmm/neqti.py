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
from scipy.optimize import brentq, minimize_scalar

from atom_openmm.async_re import JobManager
from atom_openmm.abfe_structprep import set_platform
from atom_openmm.atm_coordinates import write_atm_swapped_pdb
from atom_openmm.equilibration import neqti_endpoint_steps, neqti_midpoint_steps, run_custom_equilibration
from atom_openmm.ommsystem import OMMSystemRBFE
from atom_openmm.ommworker import OMMWorkerATMSync
from atom_openmm.neqti_integrator import OMMWorkerATMNEQTI


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
]


def _protocol_signature(settings):
    payload = {
        "schema_version": 3,
        "hamiltonian": "atm_softplus_two_leg",
        "paths": settings["paths"],
        "switch_steps_per_segment": settings["switch_steps_per_segment"],
    }
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
        Path("neqti_a_sampling.chk"),
        Path("neqti_b_sampling.chk"),
        Path("neqti_mplus_sampling.chk"),
        Path("neqti_mminus_sampling.chk"),
    ]
    if resume and any(item.exists() for item in existing_artifacts):
        if not path.exists():
            raise NEQTIConfigError(
                "Existing NEQTI artifacts predate the two-leg protocol; "
                "remove them or set resume: false"
            )
        with open(path) as handle:
            manifest = yaml.safe_load(handle) or {}
        if manifest.get("signature") != signature:
            raise NEQTIConfigError(
                "Existing NEQTI artifacts use a different Hamiltonian or lambda schedule; "
                "remove them or set resume: false"
            )
    manifest = {
        "schema_version": 3,
        "hamiltonian": "atm_softplus_two_leg",
        "paths": settings["paths"],
        "switch_steps_per_segment": settings["switch_steps_per_segment"],
        "signature": signature,
    }
    with open(path, "w") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    return signature


class NEQTIConfigError(ValueError):
    pass


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

    return {
        "initial_equilibration_steps": int(raw.get("initial_equilibration_steps", 0)),
        "n_snapshots": int(raw.get("n_snapshots", atom_options.get("MAX_SAMPLES", 1))),
        "decorrelation_steps": int(raw.get("decorrelation_steps", atom_options.get("PRODUCTION_STEPS", 1))),
        "switch_steps_per_segment": int(raw.get("switch_steps_per_segment", atom_options.get("PRODUCTION_STEPS", 1))),
        "hamiltonian": "atm_softplus_two_leg",
        "paths": paths,
        "resume": bool(raw.get("resume", True)),
        "bootstrap_samples": int(raw.get("bootstrap_samples", 200)),
        "random_seed": int(raw.get("random_seed", 2026)),
        "platform": raw.get("platform"),
        "switch_integrator": switch_integrator,
        "validate_switch_integrator": bool(raw.get("validate_switch_integrator", False)),
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
            estimates.append(estimate_bar(sample_f, sample_r, temperature))
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
    beta = 1.0 / (KB_KCAL_PER_MOL_K * float(temperature_kelvin))

    def fermi(x):
        return 1.0 / (1.0 + np.exp(np.clip(x, -700, 700)))

    def equation(df):
        return np.mean(fermi(beta * (forward - df))) - np.mean(fermi(beta * (reverse + df)))

    low = min(float(np.min(forward)), -float(np.max(reverse))) - 100.0 / beta
    high = max(float(np.max(forward)), -float(np.min(reverse))) + 100.0 / beta
    try:
        return float(brentq(equation, low, high, maxiter=200))
    except ValueError:
        objective = lambda df: equation(df) ** 2
        result = minimize_scalar(objective, bounds=(low, high), method="bounded")
        if not result.success:
            raise RuntimeError("BAR optimization failed")
        return float(result.x)


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
        ("midpoint_bridge", "bridge_forward", "bridge_reverse"),
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
    ddg = (
        components["leg_a"]["dg_kcal_per_mol"]
        + components["midpoint_bridge"]["dg_kcal_per_mol"]
        - components["leg_b"]["dg_kcal_per_mol"]
    )
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
                estimates.append(
                    estimate_bar(sampled["leg_a_forward"], sampled["leg_a_reverse"], temperature_kelvin)
                    + estimate_bar(sampled["bridge_forward"], sampled["bridge_reverse"], temperature_kelvin)
                    - estimate_bar(sampled["leg_b_forward"], sampled["leg_b_reverse"], temperature_kelvin)
                )
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


def summarize_existing_neqti_work(options, neqti_options, paths):
    work_files = {
        "leg_a_forward": Path("neqti_leg_a_forward.csv"),
        "leg_a_reverse": Path("neqti_leg_a_reverse.csv"),
        "leg_b_forward": Path("neqti_leg_b_forward.csv"),
        "leg_b_reverse": Path("neqti_leg_b_reverse.csv"),
    }
    missing = [str(path) for path in work_files.values() if not path.exists()]
    bridge_file = Path("neqti_midpoint_bridge.csv")
    if not bridge_file.exists():
        missing.append(str(bridge_file))
    if missing:
        raise NEQTIConfigError("Missing NEQTI work files: " + ", ".join(missing))

    rows = {name: _read_completed_rows(path) for name, path in work_files.items()}
    for name, values in rows.items():
        _write_integrated_work(Path(f"integ_{name}.dat"), values, name)
    bridge_rows = _read_completed_rows(bridge_file)
    bridge_forward = [row for row in bridge_rows if row["direction"] == "mplus_to_mminus"]
    bridge_reverse = [row for row in bridge_rows if row["direction"] == "mminus_to_mplus"]
    work = {name: [float(row["work_kcal_per_mol"]) for row in values] for name, values in rows.items()}
    work["bridge_forward"] = [float(row["work_kcal_per_mol"]) for row in bridge_forward]
    work["bridge_reverse"] = [float(row["work_kcal_per_mol"]) for row in bridge_reverse]
    states = build_atm_state_parameters(options)
    temperature_kelvin = states[paths["leg_a_forward"][0]]["temperature"] / kelvin
    analysis = analyze_two_leg_work(
        work,
        temperature_kelvin,
        bootstrap_samples=neqti_options["bootstrap_samples"],
        random_seed=neqti_options["random_seed"],
    )
    complete = (
        all(len(values) >= neqti_options["n_snapshots"] for values in rows.values())
        and len(bridge_forward) >= neqti_options["n_snapshots"]
        and len(bridge_reverse) >= neqti_options["n_snapshots"]
    )
    usable_overlap = analysis is not None and analysis["overlap_score"] >= 0.01
    summary = {
        "jobname": options["BASENAME"],
        "method": "neqti",
        "status": "completed" if complete and usable_overlap else "partial",
        "forward_samples": len(rows["leg_a_forward"]) + len(rows["leg_b_forward"]),
        "reverse_samples": len(rows["leg_a_reverse"]) + len(rows["leg_b_reverse"]),
        "sample_counts": {name: len(values) for name, values in rows.items()},
        "bridge_samples": {"forward": len(bridge_forward), "reverse": len(bridge_reverse)},
        "temperature_kelvin": float(temperature_kelvin),
        "hamiltonian": "atm_softplus_two_leg",
        "paths": paths,
        "settings": neqti_options,
        "analysis": analysis,
    }
    with open("neqti_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    return summary


def analyze_existing_neqti(options, neqti_options=None):
    if neqti_options is None:
        neqti_options = normalize_neqti_options({"neqti": {}}, options)
    return summarize_existing_neqti_work(options, neqti_options, neqti_options["paths"])


def _bridge_work(worker, start_state, target_state):
    worker.set_state(start_state)
    old_energy = _potential_kcal(worker)
    worker.set_state(target_state)
    new_energy = _potential_kcal(worker)
    worker.set_state(start_state)
    return new_energy - old_energy


def run_neqti(options, neqti_options=None, progress_callback=None):
    if neqti_options is None:
        neqti_options = normalize_neqti_options({"neqti": {}}, options)

    basename = options["BASENAME"]
    logger = logging.getLogger("atom_openmm.neqti")
    if not logger.handlers:
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
        "mplus": stateparams[paths["leg_a_forward"][-1]],
        "b": stateparams[paths["leg_b_forward"][0]],
        "mminus": stateparams[paths["leg_b_forward"][-1]],
    }
    initial_state_file = options.get("NEQTI_INITIAL_STATE_FILE") or options.get("INITIAL_STATE_FILE") or basename + "_0.xml"
    state_files = {name: initial_state_file for name in states}
    endpoint_steps = neqti_endpoint_steps(options)
    midpoint_steps = neqti_midpoint_steps(options)
    equilibrated_files = {
        "mplus": Path("neqti_midpoint_plus.xml"),
        "mminus": Path("neqti_midpoint_minus.xml"),
        "a": Path("neqti_endpoint_A.xml"),
        "b": Path("neqti_endpoint_B.xml"),
    }
    marker = Path("neqti_equilibrium_states.ok")
    marker_matches = marker.exists() and marker.read_text().strip() == protocol_signature
    if endpoint_steps is not None or midpoint_steps is not None:
        if neqti_options["resume"] and marker_matches and all(path.exists() for path in equilibrated_files.values()):
            logger.info("Reusing completed NEQTI endpoint and midpoint states")
        else:
            logger.info("Building NEQTI M+, M-, A, and B equilibrium states")
            endpoint_system = OMMSystemRBFE(basename, options, basename + ".pdb", basename + "_sys.xml", logger)
            endpoint_system.create_system()
            platform_options = deepcopy(options)
            if neqti_options.get("platform"):
                platform_options["OPENMM_PLATFORM"] = neqti_options["platform"]
            platform, platform_properties = set_platform(platform_options)
            for name in ("mplus", "mminus", "a", "b"):
                steps = midpoint_steps if name.startswith("m") else endpoint_steps
                if steps is None:
                    continue
                source = initial_state_file
                if name == "a" and equilibrated_files["mplus"].exists():
                    source = str(equilibrated_files["mplus"])
                elif name == "b" and equilibrated_files["mminus"].exists():
                    source = str(equilibrated_files["mminus"])
                run_custom_equilibration(
                    ommsystem=endpoint_system,
                    steps=steps,
                    platform=platform,
                    platform_properties=platform_properties,
                    output_dir=Path("equilibration") / f"neqti_{name}",
                    final_state_path=equilibrated_files[name],
                    final_pdb_path={
                        "mplus": "neqti_midpoint_plus.pdb",
                        "mminus": "neqti_midpoint_minus.pdb",
                        "a": "neqti_endpoint_A.pdb",
                        "b": "neqti_endpoint_B.pdb",
                    }[name],
                    initial_state_path=source,
                    atm_state=states[name],
                )
                logger.info("Completed NEQTI %s equilibration", name)
            marker.write_text(protocol_signature + "\n")
        state_files.update({name: str(path) for name, path in equilibrated_files.items() if path.exists()})

    node_info = _select_node_info(options, neqti_options)
    ommsystem = OMMSystemRBFE(basename, options, basename + ".pdb", basename + "_sys.xml", logger)
    worker_options = deepcopy(options)
    worker_options["INITIAL_STATE_FILE"] = state_files["mplus"]
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

    work_files = {name: Path(f"neqti_{name}.csv") for name in paths}
    bridge_file = Path("neqti_midpoint_bridge.csv")
    checkpoints = {name: Path(f"neqti_{name}_sampling.chk") for name in states}
    if not neqti_options["resume"]:
        for path in [*work_files.values(), bridge_file, *checkpoints.values(), Path("neqti_summary.yaml")]:
            if path.exists():
                path.unlink()
    for path in [*work_files.values(), bridge_file]:
        _ensure_work_csv(path)

    def execute_switch(switch_name, start_state, schedule, path, label):
        if neqti_options["switch_integrator"] == "custom":
            return _run_switch_custom(
                worker,
                start_state,
                neqti_options["switch_steps_per_segment"],
                path,
                switch_name,
                logger,
                label,
            )
        return _run_switch(
            worker,
            start_state,
            schedule,
            neqti_options["switch_steps_per_segment"],
            path,
            logger,
            label,
        )

    stream_specs = (
        ("mplus", "leg_a_reverse", "mminus"),
        ("mminus", "leg_b_reverse", "mplus"),
        ("a", "leg_a_forward", None),
        ("b", "leg_b_forward", None),
    )
    existing_bridge = {
        (row["direction"], int(row["trajectory"])) for row in _read_completed_rows(bridge_file)
    }
    try:
        for ensemble, switch_name, bridge_target in stream_specs:
            completed = _read_completed_rows(work_files[switch_name])
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
            for traj in range(len(completed), neqti_options["n_snapshots"]):
                if neqti_options["decorrelation_steps"] > 0:
                    _run_worker_steps(worker, neqti_options["decorrelation_steps"], logger, f"NEQTI {ensemble} trajectory {traj} decorrelation")
                snapshot = worker.get_chkpt()
                _write_worker_pdb_pair(worker, f"neqti_{ensemble}_snapshot_{traj}.pdb")
                if bridge_target is not None:
                    bridge_direction = f"{ensemble}_to_{bridge_target}"
                    if (bridge_direction, traj) not in existing_bridge:
                        bridge_work = _bridge_work(worker, states[ensemble], states[bridge_target])
                        _append_row(bridge_file, {
                            "trajectory": traj, "direction": bridge_direction,
                            "start_state": paths[switch_name][0], "end_state": paths[switch_name][0],
                            "work_kcal_per_mol": f"{bridge_work:.12g}",
                            "work_kj_per_mol": f"{bridge_work * KCAL_TO_KJ:.12g}", "switch_steps": 0, "status": "complete",
                        })
                        existing_bridge.add((bridge_direction, traj))
                if neqti_options["validate_switch_integrator"]:
                    _validate_switch_implementations(worker, direction=switch_name, snapshot=snapshot,
                        start_state=states[ensemble], schedule=schedules[switch_name],
                        steps_per_segment=neqti_options["switch_steps_per_segment"], state_path=paths[switch_name], logger=logger)
                work_kcal = execute_switch(switch_name, states[ensemble], schedules[switch_name], paths[switch_name], f"{switch_name} trajectory {traj}")
                _append_row(work_files[switch_name], {
                    "trajectory": traj, "direction": switch_name,
                    "start_state": paths[switch_name][0], "end_state": paths[switch_name][-1],
                    "work_kcal_per_mol": f"{work_kcal:.12g}",
                    "work_kj_per_mol": f"{work_kcal * KCAL_TO_KJ:.12g}",
                    "switch_steps": len(schedules[switch_name]), "status": "complete",
                })
                if progress_callback is not None:
                    current_counts = {
                        name: len(_read_completed_rows(path)) for name, path in work_files.items()
                    }
                    progress_callback({
                        "jobname": basename,
                        "method": "neqti",
                        "status": "partial",
                        "forward_samples": current_counts["leg_a_forward"] + current_counts["leg_b_forward"],
                        "reverse_samples": current_counts["leg_a_reverse"] + current_counts["leg_b_reverse"],
                        "sample_counts": current_counts,
                        "analysis": None,
                    })
                worker.set_chkpt(snapshot)
                worker.set_state(states[ensemble])
                _write_checkpoint(checkpoints[ensemble], worker.get_chkpt())
    finally:
        worker.finish()

    return summarize_existing_neqti_work(options, neqti_options, paths)
