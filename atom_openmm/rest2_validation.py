"""Standalone conventional MD, REST2/HREX, and umbrella validation workflow."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from time import monotonic

import numpy as np
import openmm as mm
from openmm import app, unit
import yaml

from atom_openmm.rest2 import create_rest2_system, set_rest2_scale


LOGGER = logging.getLogger(__name__)
KCAL_TO_KJ = 4.184
R_KJ_MOL_K = 0.00831446261815324


class REST2ValidationError(ValueError):
    pass


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_config(config_file):
    config_file = Path(config_file).resolve()
    with config_file.open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise REST2ValidationError("validation configuration must be a YAML mapping")
    config["_config_file"] = config_file
    config["_base_dir"] = config_file.parent
    workdir = Path(config.get("workdir", "run"))
    if not workdir.is_absolute():
        workdir = config_file.parent / workdir
    config["_workdir"] = workdir.resolve()
    return config


def _resolve(config, value):
    path = Path(value)
    return path if path.is_absolute() else (config["_base_dir"] / path).resolve()


def _platform(config):
    simulation = config.get("simulation", {})
    requested = simulation.get("platform", "auto")
    properties = {str(k): str(v) for k, v in simulation.get("platform_properties", {}).items()}
    if requested == "auto":
        available = {mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())}
        requested = next((name for name in ("CUDA", "OpenCL", "CPU", "Reference") if name in available), None)
    if requested is None:
        raise REST2ValidationError("OpenMM has no available platform")
    platform = mm.Platform.getPlatformByName(requested)
    LOGGER.info("Using OpenMM platform %s", requested)
    return platform, properties


def _simulation_settings(config):
    simulation = config.get("simulation", {})
    return {
        "temperature": float(simulation.get("temperature_k", 300.0)) * unit.kelvin,
        "pressure": float(simulation.get("pressure_atm", 1.0)) * unit.atmosphere,
        "timestep": float(simulation.get("timestep_ps", 0.002)) * unit.picoseconds,
        "friction": float(simulation.get("friction_per_ps", 1.0)) / unit.picoseconds,
        "cutoff": float(simulation.get("nonbonded_cutoff_nm", 0.9)) * unit.nanometer,
        "seed": int(simulation.get("random_seed", 2026)),
    }


def _prepared_paths(config):
    workdir = config["_workdir"]
    return {
        "prmtop": workdir / "prepared" / "1oiy.prmtop",
        "inpcrd": workdir / "prepared" / "1oiy.inpcrd",
        "equilibrated_state": workdir / "prepared" / "equilibrated_state.xml",
        "equilibrated_pdb": workdir / "prepared" / "equilibrated.pdb",
    }


def _prepare_ligand_parameters(config):
    system = config.get("system", {})
    if "ligand_mol2" in system:
        ligand_mol2 = _resolve(config, system["ligand_mol2"])
        ligand_frcmod = _resolve(config, system["ligand_frcmod"])
        return ligand_mol2, ligand_frcmod
    if "ligand_file" not in system:
        raise REST2ValidationError(
            "system must define ligand_mol2/ligand_frcmod or ligand_file"
        )
    ligand_file = _resolve(config, system["ligand_file"])
    if ligand_file.suffix.lower() != ".sdf" or not ligand_file.exists():
        raise REST2ValidationError("system.ligand_file must be an existing SDF file")
    output = config["_workdir"] / "prepared" / "parameters"
    output.mkdir(parents=True, exist_ok=True)
    ligand_mol2 = output / "ligand.mol2"
    ligand_frcmod = output / "ligand.frcmod"
    parameterization = system.get("parameterization", {})
    charge_model = str(parameterization.get("charge_model", "bcc"))
    net_charge = int(system.get("net_charge", parameterization.get("net_charge", 0)))
    residue_name = str(system.get("residue_name", "UNL"))
    LOGGER.info(
        "Parameterizing %s with GAFF2/%s and net charge %d",
        ligand_file.name, charge_model, net_charge,
    )
    subprocess.run([
        "antechamber", "-i", str(ligand_file), "-fi", "sdf",
        "-o", str(ligand_mol2), "-fo", "mol2", "-at", "gaff2",
        "-c", charge_model, "-nc", str(net_charge), "-rn", residue_name,
        "-s", "2",
    ], cwd=output, check=True)
    subprocess.run([
        "parmchk2", "-i", str(ligand_mol2), "-f", "mol2",
        "-o", str(ligand_frcmod), "-s", "gaff2",
    ], cwd=output, check=True)
    return ligand_mol2, ligand_frcmod


def _write_tleap(config, output, ligand_mol2, ligand_frcmod):
    system = config.get("system", {})
    if not ligand_mol2.exists() or not ligand_frcmod.exists():
        raise REST2ValidationError("ligand MOL2 and frcmod inputs must exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    commands = [
        f"source {system.get('ligand_forcefield', 'leaprc.gaff2')}",
        f"source {system.get('water_forcefield', 'leaprc.water.tip3p')}",
        f'LIG = loadmol2 "{ligand_mol2}"',
        f'loadamberparams "{ligand_frcmod}"',
        "check LIG",
        f"solvatebox LIG {system.get('solvent_box', 'TIP3PBOX')} {float(system.get('padding_a', 10.0)):.3f}",
    ]
    if system.get("neutralize", True):
        commands.extend(["addions2 LIG Na+ 0", "addions2 LIG Cl- 0"])
    paths = _prepared_paths(config)
    commands.extend([
        f'saveamberparm LIG "{paths["prmtop"]}" "{paths["inpcrd"]}"',
        "quit",
    ])
    output.write_text("\n".join(commands) + "\n")


def _load_amber(config, barostat=False):
    paths = _prepared_paths(config)
    if not paths["prmtop"].exists() or not paths["inpcrd"].exists():
        raise REST2ValidationError("prepared Amber files are missing; run the prepare stage")
    settings = _simulation_settings(config)
    topology_file = app.AmberPrmtopFile(str(paths["prmtop"]))
    coordinates = app.AmberInpcrdFile(str(paths["inpcrd"]))
    system = topology_file.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=settings["cutoff"],
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True,
    )
    if barostat:
        system.addForce(mm.MonteCarloBarostat(settings["pressure"], settings["temperature"], 25))
    return topology_file.topology, coordinates, system


def _load_equilibrated_state(config):
    path = _prepared_paths(config)["equilibrated_state"]
    if not path.exists():
        raise REST2ValidationError("equilibrated state is missing; run the prepare stage")
    return mm.XmlSerializer.deserialize(path.read_text())


def _set_state(context, state, set_velocities=True):
    context.setPositions(state.getPositions())
    context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    if set_velocities:
        try:
            context.setVelocities(state.getVelocities())
        except Exception:
            pass


def _solute_atoms(topology, config):
    residue_names = set(config.get("rest2", {}).get("solute_residues", ["LIG"]))
    atoms = [atom.index for atom in topology.atoms() if atom.residue.name in residue_names]
    if not atoms:
        found = sorted({atom.residue.name for atom in topology.atoms()})
        raise REST2ValidationError(
            f"REST2 solute residues {sorted(residue_names)} selected no atoms; found {found}"
        )
    return atoms


def _ligand_atom_indices(topology, config):
    names = set(config.get("rest2", {}).get("solute_residues", ["LIG"]))
    return [atom.index for atom in topology.atoms() if atom.residue.name in names]


def _torsion_indices(topology, config):
    ligand_atoms = _ligand_atom_indices(topology, config)
    atom_ids = config.get("torsion", {}).get("atom_ids", [1, 22, 23, 13])
    if len(atom_ids) != 4 or any(int(value) < 1 or int(value) > len(ligand_atoms) for value in atom_ids):
        raise REST2ValidationError("torsion.atom_ids must contain four valid 1-based ligand atom IDs")
    return tuple(ligand_atoms[int(value) - 1] for value in atom_ids)


def torsion_angle_degrees(positions_nm, indices):
    xyz = np.asarray(positions_nm, dtype=float)[list(indices)]
    b0 = -(xyz[1] - xyz[0])
    b1 = xyz[2] - xyz[1]
    b2 = xyz[3] - xyz[2]
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _positions_nm(state):
    return state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)


def _integrator(config, seed):
    settings = _simulation_settings(config)
    integrator = mm.LangevinMiddleIntegrator(
        settings["temperature"], settings["friction"], settings["timestep"]
    )
    integrator.setRandomNumberSeed(int(seed))
    return integrator


def prepare(config):
    paths = _prepared_paths(config)
    paths["prmtop"].parent.mkdir(parents=True, exist_ok=True)
    tleap_file = paths["prmtop"].parent / "tleap.in"
    ligand_mol2, ligand_frcmod = _prepare_ligand_parameters(config)
    _write_tleap(config, tleap_file, ligand_mol2, ligand_frcmod)
    LOGGER.info("Preparing solvated ligand with tleap")
    subprocess.run(["tleap", "-f", str(tleap_file)], cwd=paths["prmtop"].parent, check=True)
    topology, coordinates, system = _load_amber(config, barostat=True)
    platform, properties = _platform(config)
    integrator = _integrator(config, _simulation_settings(config)["seed"])
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(coordinates.positions)
    if coordinates.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*coordinates.boxVectors)
    equilibration = config.get("preparation", {})
    LOGGER.info("Minimizing prepared system")
    simulation.minimizeEnergy(
        tolerance=float(equilibration.get("minimization_tolerance_kj_mol_nm", 10.0))
        * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=int(equilibration.get("minimization_max_iterations", 500)),
    )
    simulation.context.setVelocitiesToTemperature(
        _simulation_settings(config)["temperature"], _simulation_settings(config)["seed"]
    )
    npt_steps = int(equilibration.get("npt_steps", 250000))
    report_interval = int(equilibration.get("report_interval_steps", 5000))
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout, report_interval, step=True, potentialEnergy=True, temperature=True,
        volume=True, speed=True, remainingTime=True, totalSteps=npt_steps,
    ))
    LOGGER.info("Running NPT preparation for %d steps", npt_steps)
    simulation.step(npt_steps)
    state = simulation.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True
    )
    paths["equilibrated_state"].write_text(mm.XmlSerializer.serialize(state))
    with paths["equilibrated_pdb"].open("w") as handle:
        app.PDBFile.writeFile(topology, state.getPositions(), handle, keepIds=True)
    return paths


def _add_torsion_restraint(system, indices, target_degrees, k_kcal_mol_rad2):
    force = mm.CustomTorsionForce(
        "0.5*k*d^2; d=min(abs(theta-theta0), 2*pi-abs(theta-theta0)); pi=3.141592653589793"
    )
    force.addPerTorsionParameter("theta0")
    force.addPerTorsionParameter("k")
    force.addTorsion(
        *indices,
        [math.radians(target_degrees), float(k_kcal_mol_rad2) * KCAL_TO_KJ],
    )
    force.setName("REST2 validation torsion umbrella")
    system.addForce(force)


def _initialize_rotamer(config, topology, system, state, target_degrees, seed, steps):
    restrained = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    indices = _torsion_indices(topology, config)
    k = float(config.get("torsion", {}).get("initialization_k_kcal_mol_rad2", 50.0))
    _add_torsion_restraint(restrained, indices, target_degrees, k)
    platform, properties = _platform(config)
    simulation = app.Simulation(topology, restrained, _integrator(config, seed), platform, properties)
    _set_state(simulation.context, state, set_velocities=False)
    simulation.minimizeEnergy(maxIterations=500)
    simulation.context.setVelocitiesToTemperature(_simulation_settings(config)["temperature"], seed)
    simulation.step(int(steps))
    return simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)


def _append_csv(path, header, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def _rest2_sampler_backend(config):
    backend = str(config.get("rest2", {}).get("sampler_backend", "custom")).strip().lower()
    aliases = {"native": "openmm_native", "openmm": "openmm_native"}
    backend = aliases.get(backend, backend)
    if backend not in {"custom", "openmm_native"}:
        raise REST2ValidationError(
            "rest2.sampler_backend must be 'custom' or 'openmm_native'"
        )
    if backend == "openmm_native" and not hasattr(app, "ReplicaExchangeSampler"):
        raise REST2ValidationError(
            "rest2.sampler_backend=openmm_native requires OpenMM 8.6 or newer"
        )
    return backend


def _native_rest2_states(scales, rest2_system):
    return [
        {
            rest2_system.scale_parameter: float(scale),
            rest2_system.sqrt_scale_parameter: math.sqrt(float(scale)),
        }
        for scale in scales
    ]


def _truncate_csv_after_cycle(path, cycle):
    path = Path(path)
    if not path.exists():
        return
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return
    kept = [rows[0]]
    kept.extend(row for row in rows[1:] if row and int(row[0]) <= int(cycle))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        csv.writer(handle).writerows(kept)
    os.replace(temporary, path)


class _NativeREST2CompatibilityReporter:
    """Write the established validation outputs from OpenMM's native sampler."""

    def __init__(
        self, output, topology, sampler, scales, temperatures, torsion_indices,
        basin_limits, exchange_interval, timestep, resume,
    ):
        self.output = Path(output)
        self.sampler = sampler
        self.scales = list(scales)
        self.temperatures = list(temperatures)
        self.torsion_indices = torsion_indices
        self.basin_limits = basin_limits
        self.exchange_interval = int(exchange_interval)
        self.timestep = timestep
        self.state_csv = self.output / "state_trace.csv"
        self.torsion_csv = self.output / "physical_torsion.csv"
        self.heated_torsion_csv = self.output / "state_torsions.csv"
        start_cycle = int(sampler.currentIteration)
        if resume:
            for path in (self.state_csv, self.torsion_csv, self.heated_torsion_csv):
                _truncate_csv_after_cycle(path, start_cycle)
        self.previous_angles = [
            torsion_angle_degrees(_positions_nm(state), torsion_indices)
            for state in sampler.replicaConformation
        ]
        trajectory = self.output / "physical.dcd"
        append = bool(resume and trajectory.exists())
        self.trajectory_handle = trajectory.open("r+b" if append else "wb")
        self.dcd = app.DCDFile(
            self.trajectory_handle, topology, timestep * self.exchange_interval,
            firstStep=start_cycle * self.exchange_interval, interval=1, append=append,
        )

    def close(self):
        if self.trajectory_handle is not None:
            self.trajectory_handle.close()
            self.trajectory_handle = None

    def __call__(self, sampler):
        cycle = int(sampler.currentIteration)
        step = cycle * self.exchange_interval
        assignments = [int(value) for value in sampler.replicaStateIndex]
        end_angles = [
            torsion_angle_degrees(_positions_nm(state), self.torsion_indices)
            for state in sampler.replicaConformation
        ]
        for walker, state_index in enumerate(assignments):
            before = self.previous_angles[walker]
            after = end_angles[walker]
            _append_csv(
                self.heated_torsion_csv,
                [
                    "cycle", "step", "walker", "state_index", "scale",
                    "effective_temperature_k", "start_torsion_deg", "end_torsion_deg",
                    "basin_a_before", "basin_a_after", "basin_changed",
                ],
                [
                    cycle, step, walker, state_index, self.scales[state_index],
                    self.temperatures[state_index], before, after,
                    int(_in_basin_a(before, self.basin_limits)),
                    int(_in_basin_a(after, self.basin_limits)),
                    int(
                        _in_basin_a(before, self.basin_limits)
                        != _in_basin_a(after, self.basin_limits)
                    ),
                ],
            )
        _append_csv(
            self.state_csv,
            ["cycle", *[f"walker_{i}" for i in range(len(assignments))]],
            [cycle, *assignments],
        )
        physical_walker = assignments.index(0)
        physical_state = sampler.replicaConformation[physical_walker]
        self.dcd.writeModel(
            physical_state.getPositions(),
            periodicBoxVectors=physical_state.getPeriodicBoxVectors(),
        )
        energy = sampler.replicaStateEnergy[physical_walker][0]
        _append_csv(
            self.torsion_csv,
            ["cycle", "step", "walker", "torsion_deg", "potential_kj_mol"],
            [
                cycle, step, physical_walker, end_angles[physical_walker],
                energy.value_in_unit(unit.kilojoule_per_mole),
            ],
        )
        self.previous_angles = end_angles


def run_md(config, resume=False):
    topology, _, system = _load_amber(config, barostat=False)
    initial = _load_equilibrated_state(config)
    settings = config.get("ordinary_md", {})
    runs = int(settings.get("runs", 2))
    steps = int(settings.get("steps", 1000000))
    sample_interval = int(settings.get("sample_interval_steps", 500))
    init_steps = int(settings.get("rotamer_initialization_steps", 25000))
    targets = settings.get("initial_torsions_deg", [-40.0, 140.0])
    output = config["_workdir"] / "ordinary_md"
    output.mkdir(parents=True, exist_ok=True)
    platform, properties = _platform(config)
    torsion_indices = _torsion_indices(topology, config)
    seed0 = _simulation_settings(config)["seed"]
    for run_index in range(runs):
        checkpoint = output / f"run_{run_index + 1}.chk"
        progress_file = output / f"run_{run_index + 1}.json"
        csv_file = output / f"run_{run_index + 1}_torsion.csv"
        trajectory = output / f"run_{run_index + 1}.dcd"
        if not resume:
            for path in (checkpoint, progress_file, csv_file, trajectory):
                path.unlink(missing_ok=True)
        completed = 0
        simulation = app.Simulation(
            topology, system, _integrator(config, seed0 + run_index + 1), platform, properties
        )
        if resume and checkpoint.exists() and progress_file.exists():
            simulation.loadCheckpoint(str(checkpoint))
            completed = int(json.loads(progress_file.read_text()).get("steps", 0))
        else:
            target = float(targets[run_index % len(targets)])
            initialized = _initialize_rotamer(
                config, topology, system, initial, target, seed0 + 1000 + run_index, init_steps
            )
            _set_state(simulation.context, initialized)
        simulation.reporters.append(
            app.DCDReporter(str(trajectory), sample_interval, append=bool(resume and trajectory.exists()))
        )
        LOGGER.info("Ordinary MD run %d: continuing from %d/%d steps", run_index + 1, completed, steps)
        while completed < steps:
            block = min(sample_interval, steps - completed)
            simulation.step(block)
            completed += block
            state = simulation.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
            angle = torsion_angle_degrees(_positions_nm(state), torsion_indices)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            _append_csv(csv_file, ["step", "torsion_deg", "potential_kj_mol"], [completed, angle, energy])
            simulation.saveCheckpoint(str(checkpoint))
            progress_file.write_text(json.dumps({"steps": completed}) + "\n")


def exchange_log_acceptance(beta, u_ii, u_ij, u_jj, u_ji):
    """Log Metropolis ratio for swapping two REST state assignments."""
    return -float(beta) * ((u_ij + u_ji) - (u_ii + u_jj))


def _context_energy_kj(context):
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _attempt_exchange(context_i, context_j, state_i, state_j, scales, beta, rng, rest2):
    u_ii = _context_energy_kj(context_i)
    u_jj = _context_energy_kj(context_j)
    set_rest2_scale(context_i, scales[state_j], rest2)
    set_rest2_scale(context_j, scales[state_i], rest2)
    u_ij = _context_energy_kj(context_i)
    u_ji = _context_energy_kj(context_j)
    log_acceptance = exchange_log_acceptance(beta, u_ii, u_ij, u_jj, u_ji)
    accepted = math.log(rng.random()) < min(0.0, log_acceptance)
    if not accepted:
        set_rest2_scale(context_i, scales[state_i], rest2)
        set_rest2_scale(context_j, scales[state_j], rest2)
    return accepted, log_acceptance


def _rest2_checkpoint(output, contexts, assignments, cycle, attempts, accepts, rng):
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for index, context in enumerate(contexts):
        temporary = checkpoint_dir / f"walker_{index}.chk.tmp"
        temporary.write_bytes(context.createCheckpoint())
        os.replace(temporary, checkpoint_dir / f"walker_{index}.chk")
    payload = {
        "cycle": cycle,
        "assignments": assignments,
        "attempts": attempts.tolist(),
        "accepts": accepts.tolist(),
        "rng_state": rng.bit_generator.state,
    }
    temporary = checkpoint_dir / "state.json.tmp"
    temporary.write_text(json.dumps(payload) + "\n")
    os.replace(temporary, checkpoint_dir / "state.json")


def _native_transition_diagnostics(state_csv, replicas):
    with Path(state_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = [f"walker_{index}" for index in range(replicas)]
    trace = np.asarray(
        [[int(row[name]) for name in columns] for row in rows], dtype=int
    )
    counts = np.zeros(max(0, replicas - 1), dtype=int)
    total_changes = 0
    if len(trace) > 1:
        for previous, current in zip(trace[:-1], trace[1:]):
            for old_state, new_state in zip(previous, current):
                if old_state == new_state:
                    continue
                total_changes += 1
                lower = min(old_state, new_state)
                upper = max(old_state, new_state)
                if upper - lower == 1:
                    counts[lower] += 1
    denominator = max(1, (len(trace) - 1) * replicas)
    return {
        "metric": "observed_state_transitions",
        "total_state_changes": int(total_changes),
        "neighbor_transition_counts": counts.tolist(),
        "neighbor_transitions_per_replica_iteration": (counts / denominator).tolist(),
        "note": "OpenMM's global exchange sampler does not expose proposal acceptance counters.",
    }


def _run_rest2_native(
    config, topology, rest2, initial, output, temperatures, scales,
    exchange_interval, steps_per_replica, cycles, seed0, platform, properties,
    resume, ensemble_metadata,
):
    states = _native_rest2_states(scales, rest2)
    integrator = _integrator(config, seed0 + 2000)
    simulation = app.Simulation(topology, rest2.system, integrator, platform, properties)
    _set_state(simulation.context, initial, set_velocities=False)
    simulation.context.setVelocitiesToTemperature(
        _simulation_settings(config)["temperature"], seed0 + 3000
    )
    sampler = app.ReplicaExchangeSampler(states, simulation, exchange_interval)
    for index in range(len(scales)):
        _set_state(simulation.context, initial, set_velocities=False)
        simulation.context.setVelocitiesToTemperature(
            _simulation_settings(config)["temperature"], seed0 + 3000 + index
        )
        sampler.replicaConformation[index] = simulation.context.getState(
            getPositions=True, getVelocities=True, parameters=True,
            integratorParameters=True, enforcePeriodicBox=True,
        )

    checkpoint_interval = int(
        config.get("rest2", {}).get("checkpoint_interval_cycles", 10)
    )
    native_dir = output / "native_sampler"
    checkpoint_files_exist = (
        native_dir.joinpath("log.csv").exists()
        and all(
            native_dir.joinpath(f"checkpoint_{index}.xml").exists()
            for index in range(len(scales))
        )
    )
    native_resume = bool(resume and checkpoint_files_exist)
    if not native_resume and native_dir.exists():
        shutil.rmtree(native_dir)
    checkpoint_reporter = app.ReplicaExchangeReporter(
        str(native_dir), checkpoint_interval, sampler,
        checkpoints=True, resume=native_resume,
    )
    if native_resume:
        LOGGER.info(
            "Native OpenMM REST2 continuing from exchange cycle %d/%d",
            sampler.currentIteration, cycles,
        )
    elif resume:
        for path in (
            output / "state_trace.csv", output / "physical_torsion.csv",
            output / "state_torsions.csv", output / "physical.dcd",
        ):
            path.unlink(missing_ok=True)

    torsion_indices = _torsion_indices(topology, config)
    basin_limits = config.get("torsion", {}).get("basin_a_deg", [-90.0, 90.0])
    compatibility_reporter = _NativeREST2CompatibilityReporter(
        output, topology, sampler, scales, temperatures, torsion_indices,
        basin_limits, exchange_interval, _simulation_settings(config)["timestep"],
        native_resume,
    )
    sampler.reporters.extend([compatibility_reporter, checkpoint_reporter])
    random.seed(seed0 + 4000 + int(sampler.currentIteration))
    remaining = cycles - int(sampler.currentIteration)
    if remaining < 0:
        raise REST2ValidationError(
            f"native checkpoint cycle {sampler.currentIteration} exceeds requested {cycles}"
        )
    existing_summary = output / "summary.yaml"
    if remaining == 0 and existing_summary.exists():
        compatibility_reporter.close()
        LOGGER.info("Native OpenMM REST2 already complete at cycle %d/%d", cycles, cycles)
        return yaml.safe_load(existing_summary.read_text())
    LOGGER.info(
        "Running native OpenMM REST2 from cycle %d/%d with %d replicas",
        sampler.currentIteration, cycles, len(scales),
    )
    started = monotonic()
    try:
        sampler.simulate(remaining)
    finally:
        compatibility_reporter.close()
    elapsed = monotonic() - started
    integrated_ns = (
        remaining * exchange_interval * len(scales)
        * float(config.get("simulation", {}).get("timestep_ps", 0.002)) / 1000.0
    )
    performance = {
        "wall_seconds_this_run": float(elapsed),
        "integrated_ns_this_run": float(integrated_ns),
        "aggregate_ns_per_day_this_run": (
            float(integrated_ns * 86400.0 / elapsed) if elapsed > 0 else None
        ),
    }
    summary = {
        "sampler_backend": "openmm_native",
        "openmm_version": mm.__version__,
        "effective_temperatures_k": temperatures,
        "scales": scales,
        "acceptance_rates": [],
        "exchange_diagnostics": _native_transition_diagnostics(
            output / "state_trace.csv", len(scales)
        ),
        "completed_cycles": cycles,
        "performance": performance,
    }
    if ensemble_metadata:
        summary["ensemble"] = dict(ensemble_metadata)
    _atomic_yaml(output / "summary.yaml", summary)
    LOGGER.info(
        "Native OpenMM REST2 complete: %d cycles, %.3f aggregate ns/day",
        cycles, performance["aggregate_ns_per_day_this_run"] or 0.0,
    )
    return summary


def run_rest2(
    config, resume=False, _output=None, _initial_state=None,
    _seed_offset=0, _ensemble_metadata=None,
):
    rest_config = config.get("rest2", {})
    backend = _rest2_sampler_backend(config)
    ensembles = rest_config.get("ensembles", [])
    if _output is None and ensembles:
        if not isinstance(ensembles, list) or not ensembles:
            raise REST2ValidationError("rest2.ensembles must be a non-empty list")
        root = config["_workdir"] / "rest2"
        if not resume and root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        topology, _, physical_system = _load_amber(config, barostat=False)
        equilibrated = _load_equilibrated_state(config)
        initialization_steps = int(rest_config.get("initialization_steps", 50000))
        seen = set()
        manifest = []
        for index, ensemble in enumerate(ensembles):
            if not isinstance(ensemble, dict) or "id" not in ensemble or "initial_torsion_deg" not in ensemble:
                raise REST2ValidationError(
                    "each REST2 ensemble requires id and initial_torsion_deg"
                )
            ensemble_id = str(ensemble["id"])
            if not ensemble_id.replace("-", "").replace("_", "").isalnum() or ensemble_id in seen:
                raise REST2ValidationError(f"invalid or duplicate REST2 ensemble id: {ensemble_id}")
            seen.add(ensemble_id)
            target = float(ensemble["initial_torsion_deg"])
            output = root / ensemble_id
            custom_checkpoint = output / "checkpoints" / "state.json"
            native_checkpoint = output / "native_sampler" / "log.csv"
            checkpoint_exists = (
                custom_checkpoint.exists() if backend == "custom"
                else native_checkpoint.exists()
            )
            if resume and checkpoint_exists:
                initial = equilibrated
            else:
                initial = _initialize_rotamer(
                    config, topology, physical_system, equilibrated, target,
                    _simulation_settings(config)["seed"] + 8000 + index,
                    initialization_steps,
                )
                output.mkdir(parents=True, exist_ok=True)
                with (output / "initial.pdb").open("w") as handle:
                    app.PDBFile.writeFile(topology, initial.getPositions(), handle, keepIds=True)
            metadata = {"id": ensemble_id, "initial_torsion_deg": target}
            run_rest2(
                config, resume=resume, _output=output, _initial_state=initial,
                _seed_offset=(index + 1) * 100000, _ensemble_metadata=metadata,
            )
            gc.collect()
            manifest.append({**metadata, "directory": ensemble_id})
        _atomic_yaml(root / "ensembles.yaml", {"ensembles": manifest})
        return

    topology, _, physical_system = _load_amber(config, barostat=False)
    initial = _initial_state or _load_equilibrated_state(config)
    temperatures = [float(value) for value in rest_config.get(
        "effective_temperatures_k", [300, 345, 396, 455, 522, 600]
    )]
    physical_temperature = float(config.get("simulation", {}).get("temperature_k", 300.0))
    if temperatures[0] != physical_temperature or any(b <= a for a, b in zip(temperatures, temperatures[1:])):
        raise REST2ValidationError("REST2 temperatures must start at the physical temperature and increase")
    scales = [physical_temperature / value for value in temperatures]
    rest2 = create_rest2_system(physical_system, _solute_atoms(topology, config))
    exchange_interval = int(rest_config.get("exchange_interval_steps", 500))
    steps_per_replica = int(rest_config.get("steps_per_replica", 1000000))
    cycles = int(math.ceil(steps_per_replica / exchange_interval))
    output = Path(_output) if _output is not None else config["_workdir"] / "rest2"
    if not resume and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    platform, properties = _platform(config)
    seed0 = _simulation_settings(config)["seed"] + int(_seed_offset)
    if backend == "openmm_native":
        return _run_rest2_native(
            config, topology, rest2, initial, output, temperatures, scales,
            exchange_interval, steps_per_replica, cycles, seed0, platform, properties,
            resume, _ensemble_metadata,
        )
    contexts = []
    integrators = []
    for index, scale in enumerate(scales):
        integrator = _integrator(config, seed0 + 2000 + index)
        context = mm.Context(rest2.system, integrator, platform, properties)
        _set_state(context, initial, set_velocities=False)
        context.setVelocitiesToTemperature(_simulation_settings(config)["temperature"], seed0 + 3000 + index)
        set_rest2_scale(context, scale, rest2)
        contexts.append(context)
        integrators.append(integrator)

    assignments = list(range(len(scales)))
    attempts = np.zeros(len(scales) - 1, dtype=int)
    accepts = np.zeros(len(scales) - 1, dtype=int)
    start_cycle = 0
    rng = np.random.default_rng(seed0 + 4000)
    checkpoint_dir = output / "checkpoints"
    state_file = checkpoint_dir / "state.json"
    if resume and state_file.exists():
        state_data = json.loads(state_file.read_text())
        start_cycle = int(state_data["cycle"])
        assignments = [int(value) for value in state_data["assignments"]]
        attempts = np.asarray(state_data["attempts"], dtype=int)
        accepts = np.asarray(state_data["accepts"], dtype=int)
        rng.bit_generator.state = state_data["rng_state"]
        for index, context in enumerate(contexts):
            context.loadCheckpoint((checkpoint_dir / f"walker_{index}.chk").read_bytes())
        LOGGER.info("REST2 continuing from exchange cycle %d/%d", start_cycle, cycles)

    beta = 1.0 / (R_KJ_MOL_K * physical_temperature)
    torsion_indices = _torsion_indices(topology, config)
    exchange_csv = output / "exchanges.csv"
    state_csv = output / "state_trace.csv"
    torsion_csv = output / "physical_torsion.csv"
    heated_torsion_csv = output / "state_torsions.csv"
    checkpoint_interval = int(rest_config.get("checkpoint_interval_cycles", 10))
    basin_limits = config.get("torsion", {}).get("basin_a_deg", [-90.0, 90.0])
    trajectory = output / "physical.dcd"
    append_trajectory = bool(resume and trajectory.exists())
    started = monotonic()
    with trajectory.open("r+b" if append_trajectory else "wb") as trajectory_handle:
        dcd = app.DCDFile(
            trajectory_handle, topology,
            _simulation_settings(config)["timestep"] * exchange_interval,
            firstStep=start_cycle, interval=1, append=append_trajectory,
        )
        for cycle in range(start_cycle, cycles):
            block = min(exchange_interval, steps_per_replica - cycle * exchange_interval)
            start_angles = []
            for context in contexts:
                state = context.getState(getPositions=True, enforcePeriodicBox=True)
                start_angles.append(torsion_angle_degrees(_positions_nm(state), torsion_indices))
            for integrator in integrators:
                integrator.step(block)
            end_states = [
                context.getState(getPositions=True, enforcePeriodicBox=True)
                for context in contexts
            ]
            end_angles = [
                torsion_angle_degrees(_positions_nm(state), torsion_indices)
                for state in end_states
            ]
            for walker, state_index in enumerate(assignments):
                _append_csv(
                    heated_torsion_csv,
                    [
                        "cycle", "step", "walker", "state_index", "scale",
                        "effective_temperature_k", "start_torsion_deg", "end_torsion_deg",
                        "basin_a_before", "basin_a_after", "basin_changed",
                    ],
                    [
                        cycle + 1, min((cycle + 1) * exchange_interval, steps_per_replica),
                        walker, state_index, scales[state_index], temperatures[state_index],
                        start_angles[walker], end_angles[walker],
                        int(_in_basin_a(start_angles[walker], basin_limits)),
                        int(_in_basin_a(end_angles[walker], basin_limits)),
                        int(
                            _in_basin_a(start_angles[walker], basin_limits)
                            != _in_basin_a(end_angles[walker], basin_limits)
                        ),
                    ],
                )
            parity = cycle % 2
            for lower_state in range(parity, len(scales) - 1, 2):
                upper_state = lower_state + 1
                walker_i = assignments.index(lower_state)
                walker_j = assignments.index(upper_state)
                accepted, log_acceptance = _attempt_exchange(
                    contexts[walker_i], contexts[walker_j], lower_state, upper_state,
                    scales, beta, rng, rest2,
                )
                attempts[lower_state] += 1
                if accepted:
                    accepts[lower_state] += 1
                    assignments[walker_i], assignments[walker_j] = upper_state, lower_state
                _append_csv(
                    exchange_csv,
                    ["cycle", "lower_state", "upper_state", "walker_i", "walker_j", "accepted", "log_acceptance"],
                    [cycle + 1, lower_state, upper_state, walker_i, walker_j, int(accepted), log_acceptance],
                )
            _append_csv(state_csv, ["cycle", *[f"walker_{i}" for i in range(len(scales))]], [cycle + 1, *assignments])
            physical_walker = assignments.index(0)
            state = contexts[physical_walker].getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
            dcd.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())
            _append_csv(
                torsion_csv,
                ["cycle", "step", "walker", "torsion_deg", "potential_kj_mol"],
                [cycle + 1, min((cycle + 1) * exchange_interval, steps_per_replica), physical_walker,
                 torsion_angle_degrees(_positions_nm(state), torsion_indices),
                 state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)],
            )
            if (cycle + 1) % checkpoint_interval == 0 or cycle + 1 == cycles:
                _rest2_checkpoint(output, contexts, assignments, cycle + 1, attempts, accepts, rng)
                rates = np.divide(accepts, attempts, out=np.zeros_like(accepts, dtype=float), where=attempts > 0)
                LOGGER.info("REST2 cycle %d/%d, neighbor acceptance %s", cycle + 1, cycles, np.round(rates, 3).tolist())
    elapsed = monotonic() - started
    integrated_ns = (
        (cycles - start_cycle) * exchange_interval * len(scales)
        * float(config.get("simulation", {}).get("timestep_ps", 0.002)) / 1000.0
    )
    summary = {
        "sampler_backend": "custom",
        "openmm_version": mm.__version__,
        "effective_temperatures_k": temperatures,
        "scales": scales,
        "attempts": attempts.tolist(),
        "accepts": accepts.tolist(),
        "acceptance_rates": np.divide(accepts, attempts, out=np.zeros_like(accepts, dtype=float), where=attempts > 0).tolist(),
        "completed_cycles": cycles,
        "performance": {
            "wall_seconds_this_run": float(elapsed),
            "integrated_ns_this_run": float(integrated_ns),
            "aggregate_ns_per_day_this_run": (
                float(integrated_ns * 86400.0 / elapsed) if elapsed > 0 else None
            ),
        },
    }
    if _ensemble_metadata:
        summary["ensemble"] = dict(_ensemble_metadata)
    _atomic_yaml(output / "summary.yaml", summary)
    context = None
    context_i = None
    context_j = None
    integrator = None
    del contexts
    del integrators
    gc.collect()


def _window_centers(config):
    umbrella = config.get("umbrella", {})
    if "centers_deg" in umbrella:
        return [float(value) for value in umbrella["centers_deg"]]
    spacing = float(umbrella.get("spacing_deg", 15.0))
    return list(np.arange(-180.0, 180.0, spacing))


def run_umbrella(config, resume=False):
    topology, _, physical_system = _load_amber(config, barostat=False)
    initial = _load_equilibrated_state(config)
    umbrella = config.get("umbrella", {})
    equilibration_steps = int(umbrella.get("equilibration_steps", 50000))
    production_steps = int(umbrella.get("production_steps", 250000))
    sample_interval = int(umbrella.get("sample_interval_steps", 500))
    k = float(umbrella.get("k_kcal_mol_rad2", 50.0))
    output = config["_workdir"] / "umbrella"
    if not resume and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    platform, properties = _platform(config)
    indices = _torsion_indices(topology, config)
    seed0 = _simulation_settings(config)["seed"]
    for window, center in enumerate(_window_centers(config)):
        csv_file = output / f"window_{window:02d}.csv"
        done_file = output / f"window_{window:02d}.done"
        if resume and done_file.exists():
            continue
        csv_file.unlink(missing_ok=True)
        system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(physical_system))
        _add_torsion_restraint(system, indices, center, k)
        simulation = app.Simulation(topology, system, _integrator(config, seed0 + 5000 + window), platform, properties)
        _set_state(simulation.context, initial, set_velocities=False)
        simulation.minimizeEnergy(maxIterations=500)
        simulation.context.setVelocitiesToTemperature(_simulation_settings(config)["temperature"], seed0 + 6000 + window)
        LOGGER.info("Umbrella window %d/%d at %.1f degrees", window + 1, len(_window_centers(config)), center)
        simulation.step(equilibration_steps)
        completed = 0
        while completed < production_steps:
            block = min(sample_interval, production_steps - completed)
            simulation.step(block)
            completed += block
            state = simulation.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
            _append_csv(
                csv_file, ["step", "center_deg", "torsion_deg", "potential_kj_mol"],
                [completed, center, torsion_angle_degrees(_positions_nm(state), indices),
                 state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)],
            )
        done_file.write_text(_utc_now() + "\n")


def periodic_delta_degrees(values, centers):
    return (np.asarray(values) - np.asarray(centers) + 180.0) % 360.0 - 180.0


def periodic_wham(angle_series, centers_deg, k_kcal_mol_rad2, temperature_k, bin_width_deg=5.0, tolerance=1e-10, max_iterations=100000):
    edges = np.arange(-180.0, 180.0 + bin_width_deg, bin_width_deg)
    centers = 0.5 * (edges[:-1] + edges[1:])
    histograms = np.asarray([np.histogram(np.asarray(values), bins=edges)[0] for values in angle_series], dtype=float)
    counts = histograms.sum(axis=1)
    if np.any(counts == 0):
        raise REST2ValidationError("every umbrella window must contain samples")
    beta = 1.0 / (R_KJ_MOL_K * float(temperature_k))
    delta_rad = np.radians(periodic_delta_degrees(centers[None, :], np.asarray(centers_deg)[:, None]))
    reduced_bias = beta * 0.5 * float(k_kcal_mol_rad2) * KCAL_TO_KJ * delta_rad ** 2
    f = np.zeros(len(angle_series))
    numerator = histograms.sum(axis=0)
    for _ in range(max_iterations):
        log_terms = np.log(counts)[:, None] + f[:, None] - reduced_bias
        maximum = np.max(log_terms, axis=0)
        denominator = np.exp(maximum) * np.exp(log_terms - maximum).sum(axis=0)
        probability = numerator / denominator
        probability /= probability.sum()
        new_f = -np.log(np.maximum((probability[None, :] * np.exp(-reduced_bias)).sum(axis=1), 1e-300))
        new_f -= new_f[0]
        if np.max(np.abs(new_f - f)) < tolerance:
            f = new_f
            break
        f = new_f
    else:
        raise REST2ValidationError("periodic WHAM did not converge")
    pmf = -np.log(np.maximum(probability, 1e-300)) / beta
    pmf -= np.min(pmf)
    return centers, probability, pmf, histograms, f


def _read_column(path, column):
    with Path(path).open(newline="") as handle:
        return np.asarray([float(row[column]) for row in csv.DictReader(handle)], dtype=float)


def _basin_a_fraction(angles, limits):
    low, high = (float(value) for value in limits)
    angles = np.asarray(angles)
    return float(np.mean((angles >= low) & (angles <= high)))


def _in_basin_a(angle, limits):
    return float(limits[0]) <= float(angle) <= float(limits[1])


def _block_bootstrap_fraction(angles, limits, samples, rng):
    angles = np.asarray(angles)
    if len(angles) < 2 or samples <= 0:
        return None
    block = max(1, int(round(math.sqrt(len(angles)))))
    values = []
    starts = np.arange(max(1, len(angles) - block + 1))
    for _ in range(samples):
        pieces = [angles[start:start + block] for start in rng.choice(starts, size=int(math.ceil(len(angles) / block)), replace=True)]
        values.append(_basin_a_fraction(np.concatenate(pieces)[:len(angles)], limits))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _resample_blocks(values, rng):
    values = np.asarray(values)
    block = max(1, int(round(math.sqrt(len(values)))))
    starts = np.arange(max(1, len(values) - block + 1))
    pieces = [values[start:start + block] for start in rng.choice(
        starts, size=int(math.ceil(len(values) / block)), replace=True
    )]
    return np.concatenate(pieces)[:len(values)]


def _bootstrap_wham(
    series, centers, k, temperature, bin_width, basin_limits, samples, rng
):
    if samples <= 0:
        return None
    basin_values = []
    pmfs = []
    for _ in range(samples):
        resampled = [_resample_blocks(window, rng) for window in series]
        bins, probability, pmf, _, _ = periodic_wham(
            resampled, centers, k, temperature, bin_width
        )
        mask = (bins >= basin_limits[0]) & (bins <= basin_limits[1])
        basin_values.append(float(probability[mask].sum()))
        pmfs.append(pmf)
    return {
        "basin_ci": [float(np.percentile(basin_values, 2.5)), float(np.percentile(basin_values, 97.5))],
        "pmf_low": np.percentile(pmfs, 2.5, axis=0),
        "pmf_high": np.percentile(pmfs, 97.5, axis=0),
    }


def _basin_transition_count(angles, limits):
    states = (np.asarray(angles) >= limits[0]) & (np.asarray(angles) <= limits[1])
    return int(np.count_nonzero(states[1:] != states[:-1])) if len(states) > 1 else 0


def _round_trips(state_trace, replicas):
    trips = 0
    for walker in range(replicas):
        states = state_trace[:, walker]
        seen_low = False
        seen_high_after_low = False
        for state in states:
            if state == 0:
                if seen_high_after_low:
                    trips += 1
                    seen_high_after_low = False
                seen_low = True
            elif state == replicas - 1 and seen_low:
                seen_high_after_low = True
    return trips


def distribution_metrics(reference_probability, observed_probability):
    reference = np.asarray(reference_probability, dtype=float)
    observed = np.asarray(observed_probability, dtype=float)
    if reference.shape != observed.shape or reference.ndim != 1:
        raise REST2ValidationError("distribution comparison requires equal one-dimensional arrays")
    if np.any(reference < 0.0) or np.any(observed < 0.0):
        raise REST2ValidationError("distribution probabilities must be nonnegative")
    if reference.sum() <= 0.0 or observed.sum() <= 0.0:
        raise REST2ValidationError("distribution probabilities must have positive sums")
    reference = reference / reference.sum()
    observed = observed / observed.sum()
    mixture = 0.5 * (reference + observed)

    def relative_entropy(probability, target):
        mask = probability > 0.0
        return float(np.sum(probability[mask] * np.log(probability[mask] / target[mask])))

    return {
        "total_variation_distance": float(0.5 * np.abs(reference - observed).sum()),
        "probability_overlap": float(np.minimum(reference, observed).sum()),
        "jensen_shannon_divergence_nats": float(
            0.5 * relative_entropy(reference, mixture)
            + 0.5 * relative_entropy(observed, mixture)
        ),
    }


def _probability_to_pmf(probability, temperature_k):
    probability = np.asarray(probability, dtype=float)
    pmf = np.full_like(probability, np.nan)
    mask = probability > 0.0
    pmf[mask] = -R_KJ_MOL_K * float(temperature_k) * np.log(probability[mask])
    pmf[mask] -= np.min(pmf[mask])
    return pmf


def _write_rest2_umbrella_comparison(
    workdir, bin_centers, umbrella_probability, umbrella_pmf,
    rest_angle_series, temperature_k, bin_width_deg,
):
    edges = np.arange(-180.0, 180.0 + float(bin_width_deg), float(bin_width_deg))
    probabilities = {}
    counts = {}
    for name, angles in rest_angle_series.items():
        histogram, _ = np.histogram(np.asarray(angles), bins=edges)
        counts[name] = histogram
        probabilities[name] = histogram / histogram.sum()
    pooled_angles = np.concatenate(list(rest_angle_series.values()))
    pooled_counts, _ = np.histogram(pooled_angles, bins=edges)
    counts["pooled"] = pooled_counts
    probabilities["pooled"] = pooled_counts / pooled_counts.sum()
    metrics = {
        name: distribution_metrics(umbrella_probability, probability)
        for name, probability in probabilities.items()
    }
    rest_pmfs = {
        name: _probability_to_pmf(probability, temperature_k)
        for name, probability in probabilities.items()
    }

    analysis_dir = Path(workdir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "rest2_umbrella_distribution.csv"
    names = (
        [name for name in probabilities if name != "pooled"]
        if len(rest_angle_series) > 1 else []
    )
    header = [
        "torsion_deg", "umbrella_probability", "umbrella_pmf_kj_mol",
        "rest2_pooled_count", "rest2_pooled_probability", "rest2_pooled_pmf_kj_mol",
    ]
    for name in names:
        header.extend([
            f"rest2_{name}_count", f"rest2_{name}_probability",
            f"rest2_{name}_pmf_kj_mol",
        ])
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, center in enumerate(bin_centers):
            row = [
                center, umbrella_probability[index], umbrella_pmf[index],
                counts["pooled"][index], probabilities["pooled"][index],
                rest_pmfs["pooled"][index],
            ]
            for name in names:
                row.extend([
                    counts[name][index], probabilities[name][index], rest_pmfs[name][index]
                ])
            writer.writerow(row)

    os.environ.setdefault("MPLCONFIGDIR", str(analysis_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(bin_centers, umbrella_probability, color="black", linewidth=2, label="Umbrella/WHAM")
    axes[0].step(bin_centers, probabilities["pooled"], where="mid", linewidth=1.8, label="REST2 pooled")
    for name in names:
        axes[0].step(bin_centers, probabilities[name], where="mid", alpha=0.65, label=f"REST2 {name}")
    axes[0].set_ylabel("Probability per bin")
    axes[0].legend(frameon=False)
    axes[1].plot(bin_centers, umbrella_pmf, color="black", linewidth=2, label="Umbrella/WHAM")
    axes[1].step(bin_centers, rest_pmfs["pooled"], where="mid", linewidth=1.8, label="REST2 pooled")
    for name in names:
        axes[1].step(bin_centers, rest_pmfs[name], where="mid", alpha=0.65, label=f"REST2 {name}")
    axes[1].set_xlabel("Torsion (degrees)")
    axes[1].set_ylabel("Relative PMF (kJ/mol)")
    axes[1].set_xlim(-180.0, 180.0)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    plot_path = analysis_dir / "rest2_umbrella_distribution.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return {
        "bin_width_deg": float(bin_width_deg),
        "metrics": metrics,
        "csv": str(csv_path.relative_to(workdir)),
        "plot": str(plot_path.relative_to(workdir)),
    }


def _analyze_rest2_output(config, directory, limits, bootstrap_samples, rng):
    torsion_file = directory / "physical_torsion.csv"
    if not torsion_file.exists():
        return None
    burn_in_steps = int(config.get("analysis", {}).get("rest2_burn_in_steps", 0))
    with torsion_file.open(newline="") as handle:
        torsion_rows = list(csv.DictReader(handle))
    angles = np.asarray([
        float(row["torsion_deg"]) for row in torsion_rows
        if int(row["step"]) > burn_in_steps
    ])
    if not len(angles):
        raise REST2ValidationError(
            f"REST2 burn-in removes every physical sample in {directory}"
        )
    with (directory / "state_trace.csv").open(newline="") as handle:
        trace_rows = list(csv.DictReader(handle))
    walker_columns = [name for name in trace_rows[0] if name.startswith("walker_")] if trace_rows else []
    exchange_interval = int(config.get("rest2", {}).get("exchange_interval_steps", 500))
    burn_cycle = int(math.ceil(burn_in_steps / exchange_interval))
    trace_rows = [row for row in trace_rows if int(row["cycle"]) > burn_cycle]
    trace = np.asarray([
        [int(row[name]) for name in walker_columns] for row in trace_rows
    ], dtype=int)
    summary = yaml.safe_load((directory / "summary.yaml").read_text())
    acceptance_rates = summary.get("acceptance_rates", [])
    analysis = {
        "samples": len(angles),
        "burn_in_steps": burn_in_steps,
        "basin_a_fraction": _basin_a_fraction(angles, limits),
        "basin_a_fraction_95ci": _block_bootstrap_fraction(
            angles, limits, bootstrap_samples, rng
        ),
        "sampler_backend": summary.get("sampler_backend", "custom"),
        "openmm_version": summary.get("openmm_version"),
        "acceptance_rates": acceptance_rates,
        "round_trips": _round_trips(trace, len(walker_columns)) if len(trace) else 0,
        "basin_transitions": _basin_transition_count(angles, limits),
    }
    if summary.get("exchange_diagnostics"):
        analysis["exchange_diagnostics"] = summary["exchange_diagnostics"]
    if summary.get("performance"):
        analysis["performance"] = summary["performance"]
    if summary.get("ensemble"):
        analysis["initial_torsion_deg"] = summary["ensemble"]["initial_torsion_deg"]
    state_torsions = directory / "state_torsions.csv"
    if state_torsions.exists():
        with state_torsions.open(newline="") as handle:
            propagation_rows = [
                row for row in csv.DictReader(handle)
                if int(row["step"]) > burn_in_steps
            ]
        timestep_ps = float(config.get("simulation", {}).get("timestep_ps", 0.002))
        block_ns = timestep_ps * exchange_interval / 1000.0
        by_state = {}
        for row in propagation_rows:
            state_index = int(row["state_index"])
            state = by_state.setdefault(state_index, {
                "state_index": state_index,
                "scale": float(row["scale"]),
                "effective_temperature_k": float(row["effective_temperature_k"]),
                "propagation_blocks": 0,
                "basin_changes": 0,
            })
            state["propagation_blocks"] += 1
            state["basin_changes"] += int(row["basin_changed"])
        for state in by_state.values():
            sampled_ns = state["propagation_blocks"] * block_ns
            state["sampled_ns"] = sampled_ns
            state["basin_changes_per_ns"] = (
                state["basin_changes"] / sampled_ns if sampled_ns else None
            )
        analysis["propagation_by_state"] = [
            by_state[index] for index in sorted(by_state)
        ]
    analysis["_angles"] = angles
    return analysis


def analyze(config):
    workdir = config["_workdir"]
    limits = config.get("torsion", {}).get("basin_a_deg", [-90.0, 90.0])
    bootstrap_samples = int(config.get("analysis", {}).get("bootstrap_samples", 200))
    rng = np.random.default_rng(_simulation_settings(config)["seed"] + 7000)
    rest_angle_series = {}
    result = {
        "schema_version": 1,
        "tool": "atom_openmm_rest2_validation",
        "status": "completed",
        "last_update": _utc_now(),
        "torsion_atom_ids": config.get("torsion", {}).get("atom_ids", [1, 22, 23, 13]),
        "basin_a_deg": limits,
        "ordinary_md": {}, "rest2": {}, "umbrella": {}, "quality": {"status": "usable", "warnings": []},
    }
    md_files = sorted((workdir / "ordinary_md").glob("run_*_torsion.csv"))
    md_series = [_read_column(path, "torsion_deg") for path in md_files]
    if md_series:
        combined = np.concatenate(md_series)
        result["ordinary_md"] = {
            "runs": len(md_series), "samples": len(combined),
            "basin_a_fraction": _basin_a_fraction(combined, limits),
            "basin_a_fraction_95ci": _block_bootstrap_fraction(combined, limits, bootstrap_samples, rng),
            "per_run_basin_a_fraction": [_basin_a_fraction(values, limits) for values in md_series],
            "basin_transitions": int(sum(_basin_transition_count(values, limits) for values in md_series)),
        }
    rest_root = workdir / "rest2"
    if (rest_root / "physical_torsion.csv").exists():
        rest_analysis = _analyze_rest2_output(
            config, rest_root, limits, bootstrap_samples, rng
        )
        rest_angle_series["physical"] = rest_analysis.pop("_angles")
        result["rest2"] = rest_analysis
    elif (rest_root / "ensembles.yaml").exists():
        manifest = yaml.safe_load((rest_root / "ensembles.yaml").read_text())
        ensemble_results = {}
        all_angles = []
        for entry in manifest["ensembles"]:
            ensemble = _analyze_rest2_output(
                config, rest_root / entry["directory"], limits, bootstrap_samples, rng
            )
            if ensemble is not None:
                angles = ensemble.pop("_angles")
                all_angles.append(angles)
                rest_angle_series[entry["id"]] = angles
                ensemble_results[entry["id"]] = ensemble
        if ensemble_results:
            combined = np.concatenate(all_angles)
            values = list(ensemble_results.values())
            rate_sets = [value["acceptance_rates"] for value in values]
            performances = [
                value["performance"] for value in values if value.get("performance")
            ]
            total_wall_seconds = sum(
                float(value.get("wall_seconds_this_run", 0.0))
                for value in performances
            )
            total_integrated_ns = sum(
                float(value.get("integrated_ns_this_run", 0.0))
                for value in performances
            )
            result["rest2"] = {
                "samples": len(combined),
                "burn_in_steps": int(config.get("analysis", {}).get("rest2_burn_in_steps", 0)),
                "basin_a_fraction": _basin_a_fraction(combined, limits),
                "basin_a_fraction_95ci": _block_bootstrap_fraction(
                    combined, limits, bootstrap_samples, rng
                ),
                "sampler_backend": values[0].get("sampler_backend", "custom"),
                "openmm_version": values[0].get("openmm_version"),
                "acceptance_rates": (
                    np.mean(rate_sets, axis=0).tolist()
                    if rate_sets and all(len(rates) for rates in rate_sets) else []
                ),
                "round_trips": min(value["round_trips"] for value in values),
                "basin_transitions": sum(value["basin_transitions"] for value in values),
                "ensembles": ensemble_results,
                "ensemble_basin_a_range": [
                    min(value["basin_a_fraction"] for value in values),
                    max(value["basin_a_fraction"] for value in values),
                ],
            }
            if performances:
                result["rest2"]["performance"] = {
                    "wall_seconds_this_run": total_wall_seconds,
                    "integrated_ns_this_run": total_integrated_ns,
                    "aggregate_ns_per_day_this_run": (
                        total_integrated_ns * 86400.0 / total_wall_seconds
                        if total_wall_seconds > 0.0 else None
                    ),
                }
    window_files = sorted((workdir / "umbrella").glob("window_*.csv"))
    if window_files:
        series = [_read_column(path, "torsion_deg") for path in window_files]
        centers = [float(next(csv.DictReader(path.open()))["center_deg"]) for path in window_files]
        umbrella = config.get("umbrella", {})
        bin_centers, probability, pmf, histograms, _ = periodic_wham(
            series, centers, float(umbrella.get("k_kcal_mol_rad2", 50.0)),
            float(config.get("simulation", {}).get("temperature_k", 300.0)),
            float(config.get("analysis", {}).get("pmf_bin_width_deg", 5.0)),
        )
        basin_mask = (bin_centers >= limits[0]) & (bin_centers <= limits[1])
        umbrella_fraction = float(probability[basin_mask].sum())
        overlap = []
        normalized = histograms / histograms.sum(axis=1, keepdims=True)
        for first, second in zip(normalized, np.roll(normalized, -1, axis=0)):
            overlap.append(float(np.minimum(first, second).sum()))
        umbrella_bootstrap = _bootstrap_wham(
            series, centers, float(umbrella.get("k_kcal_mol_rad2", 50.0)),
            float(config.get("simulation", {}).get("temperature_k", 300.0)),
            float(config.get("analysis", {}).get("pmf_bin_width_deg", 5.0)),
            limits, bootstrap_samples, rng,
        )
        pmf_csv = workdir / "analysis" / "umbrella_pmf.csv"
        pmf_csv.parent.mkdir(parents=True, exist_ok=True)
        with pmf_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["torsion_deg", "probability", "pmf_kj_mol", "pmf_95ci_low_kj_mol", "pmf_95ci_high_kj_mol"])
            pmf_low = umbrella_bootstrap["pmf_low"] if umbrella_bootstrap else [None] * len(pmf)
            pmf_high = umbrella_bootstrap["pmf_high"] if umbrella_bootstrap else [None] * len(pmf)
            writer.writerows(zip(bin_centers, probability, pmf, pmf_low, pmf_high))
        result["umbrella"] = {
            "windows": len(series), "samples": int(sum(len(values) for values in series)),
            "basin_a_fraction": umbrella_fraction,
            "basin_a_fraction_95ci": umbrella_bootstrap["basin_ci"] if umbrella_bootstrap else None,
            "minimum_adjacent_histogram_overlap": min(overlap) if overlap else None,
            "pmf_csv": str(pmf_csv.relative_to(workdir)),
        }
        if rest_angle_series:
            result["distribution_comparison"] = _write_rest2_umbrella_comparison(
                workdir, bin_centers, probability, pmf, rest_angle_series,
                float(config.get("simulation", {}).get("temperature_k", 300.0)),
                float(config.get("analysis", {}).get("pmf_bin_width_deg", 5.0)),
            )
    warnings = result["quality"]["warnings"]
    if result["rest2"]:
        rates = result["rest2"]["acceptance_rates"]
        if rates and any(rate < 0.15 or rate > 0.50 for rate in rates):
            warnings.append("One or more neighboring REST2 acceptance rates are outside 0.15-0.50.")
        if result["rest2"]["round_trips"] < 3:
            warnings.append("Fewer than three complete REST2 ladder round trips were observed.")
        if result["rest2"]["basin_transitions"] < 10:
            warnings.append("Fewer than ten physical-replica torsional basin transitions were observed.")
        ensemble_range = result["rest2"].get("ensemble_basin_a_range")
        if ensemble_range and ensemble_range[1] - ensemble_range[0] > 0.10:
            warnings.append(
                "Independent REST2 ensembles differ in basin-A population by more than 0.10."
            )
    if result["umbrella"] and result["umbrella"]["minimum_adjacent_histogram_overlap"] < 0.03:
        warnings.append("One or more adjacent umbrella windows have histogram overlap below 0.03.")
    if result["rest2"] and result["umbrella"]:
        difference = abs(result["rest2"]["basin_a_fraction"] - result["umbrella"]["basin_a_fraction"])
        result["comparison"] = {"rest2_umbrella_absolute_difference": difference}
        rest_ci = result["rest2"].get("basin_a_fraction_95ci")
        umbrella_ci = result["umbrella"].get("basin_a_fraction_95ci")
        intervals_overlap = bool(
            rest_ci and umbrella_ci and max(rest_ci[0], umbrella_ci[0]) <= min(rest_ci[1], umbrella_ci[1])
        )
        result["comparison"]["confidence_intervals_overlap"] = intervals_overlap
        if difference > 0.10 and not intervals_overlap:
            warnings.append("REST2 and umbrella basin populations differ by more than 0.10.")
    if warnings:
        result["quality"]["status"] = "partial"
    _atomic_yaml(workdir / "result.yaml", result)
    return result


def run(config_file, stage="all", resume=False):
    config = _load_config(config_file)
    config["_workdir"].mkdir(parents=True, exist_ok=True)
    stages = ["prepare", "md", "rest2", "umbrella", "analyze"] if stage == "all" else [stage]
    started = monotonic()
    for current in stages:
        LOGGER.info("Starting REST2 validation stage: %s", current)
        if current == "prepare":
            prepared = _prepared_paths(config)
            required = ("prmtop", "inpcrd", "equilibrated_state")
            if resume and all(prepared[name].exists() for name in required):
                LOGGER.info("Preparation artifacts exist; preserving them for resumed production")
            else:
                prepare(config)
        elif current == "md":
            run_md(config, resume=resume)
        elif current == "rest2":
            run_rest2(config, resume=resume)
        elif current == "umbrella":
            run_umbrella(config, resume=resume)
        elif current == "analyze":
            analyze(config)
        LOGGER.info("Finished stage %s (elapsed %.1f s)", current, monotonic() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="REST2 validation YAML")
    parser.add_argument("--stage", choices=("prepare", "md", "rest2", "umbrella", "analyze", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)-8s - %(name)s - %(message)s")
    run(args.config, stage=args.stage, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
