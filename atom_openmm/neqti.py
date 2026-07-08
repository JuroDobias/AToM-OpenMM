from __future__ import annotations

import csv
import logging
import math
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from openmm.unit import kelvin, kilocalorie_per_mole, kilocalories_per_mole, kilojoules_per_mole
from scipy.optimize import brentq, minimize_scalar

from atom_openmm.async_re import JobManager
from atom_openmm.ommsystem import OMMSystemRBFE
from atom_openmm.ommworker import OMMWorkerATMSync


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


class NEQTIConfigError(ValueError):
    pass


def build_atm_state_parameters(options):
    lambdas = options.get("LAMBDAS")
    temperatures = options.get("TEMPERATURES")
    intermediates = options.get("INTERMEDIATE")
    directions = options.get("DIRECTION")
    lambda1s = options.get("LAMBDA1")
    lambda2s = options.get("LAMBDA2")
    lambda3s = options.get("LAMBDA3") or options.get("LAMBDA2")
    alphas = options.get("ALPHA")
    uhs = options.get("U0")
    uhs1 = options.get("U1") or options.get("U0")
    w0coeffs = options.get("W0COEFF")

    required = [lambdas, temperatures, intermediates, directions, lambda1s, lambda2s, lambda3s, alphas, uhs, uhs1, w0coeffs]
    if any(value is None for value in required):
        raise NEQTIConfigError("LAMBDAS, TEMPERATURES, DIRECTION, INTERMEDIATE, LAMBDA1, LAMBDA2, ALPHA, U0, and W0COEFF are required")

    nstates = len(lambdas)
    for values in [intermediates, directions, lambda1s, lambda2s, lambda3s, alphas, uhs, uhs1, w0coeffs]:
        if len(values) != nstates:
            raise NEQTIConfigError("ATM schedule arrays must have the same length")
    if len(temperatures) != 1:
        raise NEQTIConfigError("NEQTI v1 supports exactly one temperature")

    stateparams = []
    for lambd, direction, intermediate, lambda1, lambda2, lambda3, alpha, uh, uh1, w0 in zip(
        lambdas, directions, intermediates, lambda1s, lambda2s, lambda3s, alphas, uhs, uhs1, w0coeffs
    ):
        par = {
            "lambda": float(lambd),
            "atmdirection": float(direction),
            "atmintermediate": float(intermediate),
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
            "lambda3": float(lambda3),
            "alpha": float(alpha) / kilocalories_per_mole,
            "uh": float(uh) * kilocalories_per_mole,
            "uh1": float(uh1) * kilocalories_per_mole,
            "w0": float(w0) * kilocalories_per_mole,
            "temperature": float(temperatures[0]) * kelvin,
            "Umax": float(options.get("UMAX")) * kilocalorie_per_mole,
            "Ubcore": float(options.get("UBCORE")) * kilocalorie_per_mole,
            "Acore": float(options.get("ACORE")),
            "uoffset": float(options.get("PERTE_OFFSET", 0.0)) * kilocalorie_per_mole,
        }
        stateparams.append(par)
    return stateparams


def normalize_neqti_options(workflow, atom_options):
    raw = workflow.get("neqti", {}) or {}
    if not isinstance(raw, dict):
        raise NEQTIConfigError("workflow.neqti must be a mapping")

    stateparams = build_atm_state_parameters(atom_options)
    default_path = list(range(len(stateparams)))
    state_path = raw.get("state_path", default_path)
    if not isinstance(state_path, list) or len(state_path) < 2:
        raise NEQTIConfigError("workflow.neqti.state_path must contain at least two state indices")
    state_path = [int(i) for i in state_path]
    if any(i < 0 or i >= len(stateparams) for i in state_path):
        raise NEQTIConfigError("workflow.neqti.state_path contains an out-of-range state index")

    return {
        "initial_equilibration_steps": int(raw.get("initial_equilibration_steps", 0)),
        "n_snapshots": int(raw.get("n_snapshots", atom_options.get("MAX_SAMPLES", 1))),
        "decorrelation_steps": int(raw.get("decorrelation_steps", atom_options.get("PRODUCTION_STEPS", 1))),
        "switch_steps_per_segment": int(raw.get("switch_steps_per_segment", atom_options.get("PRODUCTION_STEPS", 1))),
        "state_path": state_path,
        "resume": bool(raw.get("resume", True)),
        "bootstrap_samples": int(raw.get("bootstrap_samples", 200)),
        "random_seed": int(raw.get("random_seed", 2026)),
        "platform": raw.get("platform"),
    }


def interpolate_state(start, end, fraction):
    par = deepcopy(start)
    for key, start_value in start.items():
        end_value = end[key]
        if key == "temperature":
            par[key] = start_value
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


def _append_row(path, row):
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_integrated_work(path, rows, prefix):
    with open(path, "w") as f:
        for row in rows:
            f.write(f"{prefix}_{int(row['trajectory'])} {float(row['work_kj_per_mol']):.12g}\n")


def _potential_kcal(worker):
    pot = worker.get_energy()
    return pot["potential_energy"] / kilocalories_per_mole


def _run_switch(worker, start_par, schedule):
    worker.set_state(start_par)
    work = 0.0
    for par in schedule:
        old_energy = _potential_kcal(worker)
        worker.set_state(par)
        new_energy = _potential_kcal(worker)
        work += new_energy - old_energy
        worker.run(1)
    return work


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


def run_neqti(options, neqti_options=None):
    if neqti_options is None:
        neqti_options = normalize_neqti_options({"neqti": {}}, options)

    basename = options["BASENAME"]
    logger = logging.getLogger("atom_openmm.neqti")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

    stateparams = build_atm_state_parameters(options)
    state_path = neqti_options["state_path"]
    reverse_path = list(reversed(state_path))
    forward_schedule = make_switch_schedule(stateparams, state_path, neqti_options["switch_steps_per_segment"])
    reverse_schedule = make_switch_schedule(stateparams, reverse_path, neqti_options["switch_steps_per_segment"])
    forward_start = stateparams[state_path[0]]
    reverse_start = stateparams[state_path[-1]]

    node_info = _select_node_info(options, neqti_options)
    ommsystem = OMMSystemRBFE(basename, options, basename + ".pdb", basename + "_sys.xml", logger)
    worker = OMMWorkerATMSync(basename, ommsystem, options, node_info=node_info, compute=True, logger=logger)

    forward_file = Path("neqti_forward.csv")
    reverse_file = Path("neqti_reverse.csv")
    if not neqti_options["resume"]:
        for path in [forward_file, reverse_file, Path("integA.dat"), Path("integB.dat"), Path("neqti_summary.yaml")]:
            if path.exists():
                path.unlink()

    try:
        completed_forward = _read_completed_rows(forward_file)
        completed_reverse = _read_completed_rows(reverse_file)

        worker.set_state(forward_start)
        if neqti_options["initial_equilibration_steps"] > 0:
            worker.run(neqti_options["initial_equilibration_steps"])
        for traj in range(len(completed_forward), neqti_options["n_snapshots"]):
            if neqti_options["decorrelation_steps"] > 0:
                worker.run(neqti_options["decorrelation_steps"])
            snapshot = worker.get_chkpt()
            work_kcal = _run_switch(worker, forward_start, forward_schedule)
            _append_row(
                forward_file,
                {
                    "trajectory": traj,
                    "direction": "forward",
                    "start_state": state_path[0],
                    "end_state": state_path[-1],
                    "work_kcal_per_mol": f"{work_kcal:.12g}",
                    "work_kj_per_mol": f"{work_kcal * KCAL_TO_KJ:.12g}",
                    "switch_steps": len(forward_schedule),
                    "status": "complete",
                },
            )
            worker.set_chkpt(snapshot)
            worker.set_state(forward_start)

        worker.set_state(reverse_start)
        if neqti_options["initial_equilibration_steps"] > 0:
            worker.run(neqti_options["initial_equilibration_steps"])
        for traj in range(len(completed_reverse), neqti_options["n_snapshots"]):
            if neqti_options["decorrelation_steps"] > 0:
                worker.run(neqti_options["decorrelation_steps"])
            snapshot = worker.get_chkpt()
            work_kcal = _run_switch(worker, reverse_start, reverse_schedule)
            _append_row(
                reverse_file,
                {
                    "trajectory": traj,
                    "direction": "reverse",
                    "start_state": state_path[-1],
                    "end_state": state_path[0],
                    "work_kcal_per_mol": f"{work_kcal:.12g}",
                    "work_kj_per_mol": f"{work_kcal * KCAL_TO_KJ:.12g}",
                    "switch_steps": len(reverse_schedule),
                    "status": "complete",
                },
            )
            worker.set_chkpt(snapshot)
            worker.set_state(reverse_start)
    finally:
        worker.finish()

    forward_rows = _read_completed_rows(forward_file)
    reverse_rows = _read_completed_rows(reverse_file)
    _write_integrated_work(Path("integA.dat"), forward_rows, "forward")
    _write_integrated_work(Path("integB.dat"), reverse_rows, "reverse")

    forward_work = [float(row["work_kcal_per_mol"]) for row in forward_rows]
    reverse_work = [float(row["work_kcal_per_mol"]) for row in reverse_rows]
    temperature_kelvin = stateparams[state_path[0]]["temperature"] / kelvin
    analysis = analyze_neqti_work(
        forward_work,
        reverse_work,
        temperature_kelvin,
        bootstrap_samples=neqti_options["bootstrap_samples"],
        random_seed=neqti_options["random_seed"],
    )
    summary = {
        "jobname": basename,
        "method": "neqti",
        "status": "completed" if len(forward_rows) >= neqti_options["n_snapshots"] and len(reverse_rows) >= neqti_options["n_snapshots"] else "partial",
        "forward_samples": len(forward_rows),
        "reverse_samples": len(reverse_rows),
        "temperature_kelvin": float(temperature_kelvin),
        "state_path": state_path,
        "switch_steps": len(forward_schedule),
        "settings": neqti_options,
        "analysis": analysis,
    }
    with open("neqti_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    return summary
