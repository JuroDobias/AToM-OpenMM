from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from sys import stdout
from typing import Any

import openmm as mm
from openmm import XmlSerializer
from openmm.app import PDBFile, Simulation, StateDataReporter, XTCReporter
from openmm.unit import (
    bar,
    kelvin,
    kilojoule_per_mole,
    nanometer,
    picosecond,
)

from atom_openmm.atm_coordinates import write_atm_swapped_pdb
from atom_openmm.atm_coordinates import atm_swapped_positions
from atom_openmm.selections import SelectionError, compile_selection_expression


KCAL_MOL_A2_TO_KJ_MOL_NM2 = 418.4
A_TO_NM = 0.1


class EquilibrationConfigError(ValueError):
    pass


class ATMSwappedPDBReporter:
    def __init__(self, file, report_interval, keywords, enforce_periodic_box=False):
        self._report_interval = int(report_interval)
        self._keywords = keywords
        self._enforce_periodic_box = enforce_periodic_box
        self._out = open(file, "w")
        self._topology = None
        self._next_model = 1

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return {"steps": steps, "periodic": self._enforce_periodic_box, "include": ["positions"]}

    def report(self, simulation, state):
        positions = atm_swapped_positions(state.getPositions(asNumpy=True), self._keywords)
        if positions is None:
            return
        topology = simulation.topology
        if self._next_model == 1:
            PDBFile.writeHeader(topology, self._out)
            self._topology = topology
        PDBFile.writeModel(topology, positions, self._out, self._next_model)
        self._next_model += 1
        self._out.flush()

    def __del__(self):
        try:
            if self._topology is not None:
                PDBFile.writeFooter(self._topology, self._out)
            self._out.close()
        except Exception:
            pass


class AmberMaskResolver:
    def __init__(self, topology, positions, *, keywords=None, endpoint=None, base_dir=None):
        try:
            import parmed as pmd
        except ImportError as exc:
            raise ImportError("Amber mask selections require `parmed` to be installed.") from exc

        self._structure = pmd.openmm.load_topology(topology, xyz=positions)
        self._n_atoms = len(self._structure.atoms)
        self._keywords = keywords or {}
        self._endpoint = endpoint
        self._base_dir = base_dir

    def resolve(self, mask: str, label: str) -> list[int]:
        if not isinstance(mask, str) or not mask.strip():
            raise EquilibrationConfigError(f"{label} must be a non-empty Amber mask string")
        try:
            compiled = compile_selection_expression(
                mask.strip(),
                self._keywords,
                endpoint=self._endpoint,
                base_dir=self._base_dir,
            )
            from parmed.amber.mask import AmberMask

            flags = AmberMask(self._structure, compiled).Selection()
        except SelectionError as exc:
            raise EquilibrationConfigError(f"{label}: {exc}") from exc
        except Exception as exc:
            raise EquilibrationConfigError(f"{label}: invalid Amber mask {mask!r}: {exc}") from exc

        if len(flags) != self._n_atoms:
            raise EquilibrationConfigError(
                f"{label}: Amber mask {mask!r} returned {len(flags)} flags for {self._n_atoms} atoms"
            )
        selected = [i for i, flag in enumerate(flags) if bool(flag)]
        if not selected:
            raise EquilibrationConfigError(f"{label}: Amber mask {mask!r} selected 0 atoms")
        return selected


def get_equilibration_config(options: dict[str, Any]) -> dict[str, Any]:
    raw = options.get("EQUILIBRATION_PROTOCOL") or {}
    if not isinstance(raw, dict):
        raise EquilibrationConfigError("EQUILIBRATION_PROTOCOL must be a mapping")
    return raw


def pre_atm_steps(options: dict[str, Any]) -> list[dict[str, Any]] | None:
    cfg = get_equilibration_config(options)
    return _steps_from_section(cfg.get("pre_atm"), "EQUILIBRATION_PROTOCOL.pre_atm")


def async_midpoint_steps(options: dict[str, Any]) -> list[dict[str, Any]] | None:
    cfg = get_equilibration_config(options)
    async_cfg = cfg.get("async_re") or {}
    if not isinstance(async_cfg, dict):
        raise EquilibrationConfigError("EQUILIBRATION_PROTOCOL.async_re must be a mapping")
    return _steps_from_section(async_cfg.get("midpoint"), "EQUILIBRATION_PROTOCOL.async_re.midpoint")


def neqti_endpoint_steps(options: dict[str, Any]) -> list[dict[str, Any]] | None:
    cfg = get_equilibration_config(options)
    neqti_cfg = cfg.get("neqti") or {}
    if not isinstance(neqti_cfg, dict):
        raise EquilibrationConfigError("EQUILIBRATION_PROTOCOL.neqti must be a mapping")
    return _steps_from_section(neqti_cfg.get("endpoint"), "EQUILIBRATION_PROTOCOL.neqti.endpoint")


def neqti_midpoint_steps(options: dict[str, Any]) -> list[dict[str, Any]] | None:
    cfg = get_equilibration_config(options)
    neqti_cfg = cfg.get("neqti") or {}
    if not isinstance(neqti_cfg, dict):
        raise EquilibrationConfigError("EQUILIBRATION_PROTOCOL.neqti must be a mapping")
    midpoint = _steps_from_section(neqti_cfg.get("midpoint"), "EQUILIBRATION_PROTOCOL.neqti.midpoint")
    return midpoint if midpoint is not None else neqti_endpoint_steps(options)


def _steps_from_section(section: Any, label: str) -> list[dict[str, Any]] | None:
    if section is None:
        return None
    if isinstance(section, dict) and section.get("mode", "custom") == "default":
        return None
    steps = section.get("steps") if isinstance(section, dict) else None
    if steps is None:
        return None
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        raise EquilibrationConfigError(f"{label}.steps must be a list of mappings")
    if not steps:
        raise EquilibrationConfigError(f"{label}.steps must not be empty")
    _validate_unique_step_ids(steps, label)
    return steps


def normalize_equilibration_protocol(workflow: dict[str, Any]) -> dict[str, Any] | None:
    raw = workflow.get("equilibration")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise EquilibrationConfigError("workflow.equilibration must be a mapping")
    normalized = deepcopy(raw)
    _validate_equilibration_protocol(normalized, "workflow.equilibration")
    return normalized


def _validate_equilibration_protocol(protocol: dict[str, Any], label: str):
    _steps_from_section(protocol.get("pre_atm"), f"{label}.pre_atm")
    async_cfg = protocol.get("async_re") or {}
    if not isinstance(async_cfg, dict):
        raise EquilibrationConfigError(f"{label}.async_re must be a mapping")
    _steps_from_section(async_cfg.get("midpoint"), f"{label}.async_re.midpoint")
    neqti_cfg = protocol.get("neqti") or {}
    if not isinstance(neqti_cfg, dict):
        raise EquilibrationConfigError(f"{label}.neqti must be a mapping")
    _steps_from_section(neqti_cfg.get("endpoint"), f"{label}.neqti.endpoint")
    _steps_from_section(neqti_cfg.get("midpoint"), f"{label}.neqti.midpoint")


def _clone_system(system):
    return XmlSerializer.deserialize(XmlSerializer.serialize(system))


def _load_state(path: str | Path):
    with open(path) as handle:
        return XmlSerializer.deserialize(handle.read())


def _state_positions(path: str | Path):
    return _load_state(path).getPositions(asNumpy=True)


def _save_final_pdb(simulation: Simulation, path: str | Path):
    state = simulation.context.getState(getPositions=True)
    box_vectors = state.getPeriodicBoxVectors()
    if box_vectors is not None:
        simulation.topology.setPeriodicBoxVectors(box_vectors)
    with open(path, "w") as handle:
        PDBFile.writeFile(simulation.topology, state.getPositions(), handle, keepIds=True)


def _strip_integrator_parameters(path: str | Path):
    tree = ET.parse(path)
    root = tree.getroot()
    for node in root.findall("Parameters"):
        node.attrib.clear()
    for node in list(root.findall("IntegratorParameters")):
        root.remove(node)
    tree.write(path, encoding="unicode", xml_declaration=True)


def _save_compatible_state(
    *,
    ommsystem,
    source_state,
    platform,
    platform_properties,
    final_state_path,
    final_pdb_path,
    atm_state,
):
    integrator = mm.VerletIntegrator(0.001 * picosecond)
    simulation = Simulation(ommsystem.topology, ommsystem.system, integrator, platform, platform_properties)
    simulation.context.setPositions(source_state.getPositions())
    box_vectors = source_state.getPeriodicBoxVectors()
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    velocities = source_state.getVelocities()
    if velocities is not None:
        simulation.context.setVelocities(velocities)
    _apply_atm_state(simulation.context, ommsystem, atm_state)
    simulation.saveState(str(final_state_path))
    _strip_integrator_parameters(final_state_path)
    _save_final_pdb(simulation, final_pdb_path)
    if atm_state is not None:
        positions = simulation.context.getState(getPositions=True).getPositions()
        swapped_path = Path(final_pdb_path).with_name(Path(final_pdb_path).stem + "_swapped.pdb")
        write_atm_swapped_pdb(simulation.topology, positions, ommsystem.keywords, swapped_path)


def _build_integrator(step_cfg: dict[str, Any], default_temperature):
    kind = str(step_cfg.get("integrator", "langevin_middle")).lower()
    timestep = float(step_cfg.get("timestep_ps", 0.002)) * picosecond
    if kind == "verlet":
        return mm.VerletIntegrator(timestep)
    if kind in ("langevin_middle", "langevinmiddle"):
        thermostat = step_cfg.get("thermostat") or {}
        temperature = float(
            thermostat.get(
                "initial_temperature_k",
                thermostat.get(
                    "temperature_k", default_temperature / kelvin
                ),
            )
        ) * kelvin
        friction = float(thermostat.get("friction_per_ps", 1.0)) / picosecond
        return mm.LangevinMiddleIntegrator(temperature, friction, timestep)
    raise EquilibrationConfigError(f"Unsupported integrator `{kind}`")


def _resolve_step_restraints(steps, topology, positions, *, keywords=None, endpoint=None, base_dir=None):
    resolver = AmberMaskResolver(
        topology,
        positions,
        keywords=keywords,
        endpoint=endpoint,
        base_dir=base_dir,
    )
    resolved = []
    for i, step in enumerate(steps):
        step_resolved = {}
        if "positional_restraints" in step:
            pos_cfg = step["positional_restraints"]
            step_resolved["positional_atom_indices"] = resolver.resolve(
                pos_cfg["mask"],
                f"steps[{i}].positional_restraints.mask",
            )
        resolved.append(step_resolved)
    return resolved


def _add_positional_restraints(system, reference_positions, atom_indices, cfg):
    k = float(cfg["k_kcal_mol_a2"]) * KCAL_MOL_A2_TO_KJ_MOL_NM2
    tol = float(cfg.get("tolerance_a", 0.0)) * A_TO_NM
    expr = "0.5*k*step(dr-tol)*(dr-tol)^2; dr=periodicdistance(x,y,z,x0,y0,z0)"
    force = mm.CustomExternalForce(expr)
    force.setName("PositionalRestraints")
    force.addGlobalParameter("k", k)
    force.addGlobalParameter("tol", tol)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    positions_nm = reference_positions.value_in_unit(nanometer)
    for atom in atom_indices:
        x0, y0, z0 = positions_nm[atom]
        force.addParticle(int(atom), [float(x0), float(y0), float(z0)])
    system.addForce(force)


def _apply_restraints(system, step_cfg, reference_positions, resolved):
    if "positional_restraints" in step_cfg:
        _add_positional_restraints(
            system,
            reference_positions,
            resolved["positional_atom_indices"],
            step_cfg["positional_restraints"],
        )


def _set_barostat(system, step_cfg):
    ensemble = str(step_cfg.get("ensemble", "NVT")).upper()
    barostats = [system.getForce(i) for i in range(system.getNumForces()) if isinstance(system.getForce(i), mm.MonteCarloBarostat)]
    if ensemble == "NVT":
        for barostat in barostats:
            barostat.setFrequency(0)
        return
    if ensemble != "NPT":
        raise EquilibrationConfigError(f"Unsupported ensemble `{ensemble}`")
    barostat_cfg = step_cfg.get("barostat") or {}
    pressure = float(barostat_cfg.get("pressure_bar", 1.0)) * bar
    frequency = int(barostat_cfg.get("frequency", 25))
    temperature_k = float((step_cfg.get("thermostat") or {}).get("temperature_k", 300.0))
    if barostats:
        barostats[0].setDefaultPressure(pressure)
        barostats[0].setDefaultTemperature(temperature_k * kelvin)
        barostats[0].setFrequency(frequency)
    else:
        system.addForce(mm.MonteCarloBarostat(pressure, temperature_k * kelvin, frequency))


def _step_restraint_label(step_cfg):
    if "positional_restraints" not in step_cfg:
        return "none"
    cfg = step_cfg["positional_restraints"]
    return (
        f"mask={cfg['mask']!r}, k={float(cfg['k_kcal_mol_a2']):g} kcal/mol/A^2, "
        f"tolerance={float(cfg.get('tolerance_a', 0.0)):g} A"
    )


def _step_barostat_label(step_cfg):
    if str(step_cfg.get("ensemble", "NVT")).upper() != "NPT":
        return "disabled"
    cfg = step_cfg.get("barostat") or {}
    pressure = float(cfg.get("pressure_bar", 1.0))
    frequency = int(cfg.get("frequency", 25))
    return f"enabled, pressure={pressure:g} bar, frequency={frequency}"


def _step_id(step_cfg, index):
    step_id = step_cfg.get("id", f"step_{index}")
    if not isinstance(step_id, str) or not step_id:
        raise EquilibrationConfigError(f"steps[{index}].id must be a non-empty string")
    return step_id


def _validate_unique_step_ids(steps, label):
    seen = {}
    for index, step_cfg in enumerate(steps):
        step_id = _step_id(step_cfg, index)
        previous = seen.get(step_id)
        if previous is not None:
            raise EquilibrationConfigError(
                f"{label}.steps has duplicate step id {step_id!r} at steps {previous + 1} and {index + 1}; "
                "step ids must be unique within each custom equilibration section"
            )
        seen[step_id] = index


def _validate_step(step_cfg, index):
    step_type = step_cfg.get("type")
    if step_type not in ("minimization", "md"):
        raise EquilibrationConfigError(f"steps[{index}].type must be 'minimization' or 'md'")
    if step_type == "md":
        int(step_cfg["n_steps"])
        thermostat = step_cfg.get("thermostat") or {}
        if "initial_temperature_k" in thermostat:
            kind = str(
                step_cfg.get("integrator", "langevin_middle")
            ).lower()
            if kind not in ("langevin_middle", "langevinmiddle"):
                raise EquilibrationConfigError(
                    f"steps[{index}].thermostat.initial_temperature_k "
                    "requires a Langevin integrator"
                )
            if float(thermostat["initial_temperature_k"]) <= 0:
                raise EquilibrationConfigError(
                    f"steps[{index}].thermostat.initial_temperature_k "
                    "must be positive"
                )
            if float(
                thermostat.get("temperature_k", 300.0)
            ) <= 0:
                raise EquilibrationConfigError(
                    f"steps[{index}].thermostat.temperature_k must be "
                    "positive"
                )
            if int(
                thermostat.get("temperature_update_interval_steps", 5000)
            ) < 1:
                raise EquilibrationConfigError(
                    f"steps[{index}].thermostat."
                    "temperature_update_interval_steps must be positive"
                )
    if step_type == "minimization":
        float(step_cfg.get("tolerance_kj_mol_nm", 10.0))
        int(step_cfg.get("max_iterations", 0))
    if "positional_restraints" in step_cfg:
        cfg = step_cfg["positional_restraints"]
        if not isinstance(cfg, dict):
            raise EquilibrationConfigError(f"steps[{index}].positional_restraints must be a mapping")
        for key in ("mask", "k_kcal_mol_a2"):
            if key not in cfg:
                raise EquilibrationConfigError(f"steps[{index}].positional_restraints.{key} is required")


def _build_reporters(step_cfg, step_dir, *, keywords=None):
    reporters = []
    reporter_cfg = step_cfg.get("reporters") or {}
    state_cfg = reporter_cfg.get("state")
    if state_cfg:
        interval = int(state_cfg.get("interval", step_cfg.get("reporter_interval", step_cfg.get("n_steps", 1))))
        reporters.append(
            StateDataReporter(
                str(step_dir / "state.csv"),
                interval,
                step=True,
                potentialEnergy=True,
                temperature=True,
                volume=True,
                speed=True,
            )
        )
    traj_cfg = reporter_cfg.get("traj")
    if traj_cfg:
        fmt = str(traj_cfg.get("format", "xtc")).lower()
        interval = int(traj_cfg.get("interval", step_cfg.get("n_steps", 1)))
        if fmt != "xtc":
            raise EquilibrationConfigError("Only XTC trajectory reporters are supported for custom equilibration")
        reporters.append(XTCReporter(str(step_dir / "trajectory.xtc"), interval, enforcePeriodicBox=False))
        if bool(traj_cfg.get("swapped_pdb", False)):
            if keywords is None:
                raise EquilibrationConfigError("reporters.traj.swapped_pdb requires ATM keyword context")
            reporters.append(ATMSwappedPDBReporter(str(step_dir / "trajectory_swapped.pdb"), interval, keywords))
    return reporters


def _apply_atm_state(context, ommsystem, par):
    if not par:
        return
    atmforce = ommsystem.atmforce
    context.setParameter(atmforce.Lambda1(), par["lambda1"])
    context.setParameter(atmforce.Lambda2(), par["lambda2"])
    if ommsystem.multisoftplus:
        context.setParameter("Lambda3", par["lambda3"])
    context.setParameter(atmforce.Alpha(), par["alpha"] * kilojoule_per_mole)
    context.setParameter(atmforce.Uh(), par["uh"] / kilojoule_per_mole)
    if ommsystem.multisoftplus:
        context.setParameter("Uh1", par["uh1"] / kilojoule_per_mole)
    context.setParameter(atmforce.W0(), par["w0"] / kilojoule_per_mole)
    context.setParameter(atmforce.Direction(), par["atmdirection"])
    context.setParameter(atmforce.Umax(), par[atmforce.Umax()] / kilojoule_per_mole)
    context.setParameter(atmforce.Ubcore(), par[atmforce.Ubcore()] / kilojoule_per_mole)
    context.setParameter(atmforce.Acore(), par[atmforce.Acore()])
    context.setParameter("UOffset", par["uoffset"] / kilojoule_per_mole)


def run_custom_equilibration(
    *,
    ommsystem,
    steps: list[dict[str, Any]],
    platform,
    platform_properties: dict[str, str],
    output_dir: str | Path,
    final_state_path: str | Path,
    final_pdb_path: str | Path,
    initial_state_path: str | Path | None = None,
    atm_state: dict[str, Any] | None = None,
    swapped_diagnostics_keywords: dict[str, Any] | None = None,
    selection_endpoint: str | None = None,
    logger=None,
):
    logger = logger or logging.getLogger("atom_openmm.equilibration")
    if not steps:
        raise EquilibrationConfigError("Custom equilibration requires at least one step")
    for i, step in enumerate(steps):
        _validate_step(step, i)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_restraints = _resolve_step_restraints(
        steps,
        ommsystem.topology,
        ommsystem.positions,
        keywords=ommsystem.keywords,
        endpoint=selection_endpoint,
        base_dir=ommsystem.keywords.get("WORKDIR", "."),
    )
    base_positions = ommsystem.positions
    prev_state_path = Path(initial_state_path) if initial_state_path else None
    default_temperature = getattr(ommsystem, "temperature", 300.0 * kelvin)
    manifest = {"steps": {}}

    logger.info("Running custom equilibration protocol in %s (%d steps)", output_dir, len(steps))
    for i, step_cfg in enumerate(steps):
        step_id = _step_id(step_cfg, i)
        step_dir = output_dir / step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        if prev_state_path is None:
            input_label = "initial positions"
        else:
            input_label = str(prev_state_path)
        logger.info(
            "Custom equilibration step %d/%d %s: starting from %s",
            i + 1,
            len(steps),
            step_id,
            input_label,
        )

        step_system = _clone_system(ommsystem.system)
        reference_positions = _state_positions(prev_state_path) if prev_state_path and prev_state_path.exists() else base_positions
        _apply_restraints(step_system, step_cfg, reference_positions, resolved_restraints[i])
        _set_barostat(step_system, step_cfg)
        integrator = _build_integrator(step_cfg, default_temperature)
        simulation = Simulation(ommsystem.topology, step_system, integrator, platform, platform_properties)
        simulation.context.setPositions(base_positions)
        if ommsystem.boxvectors is not None:
            simulation.context.setPeriodicBoxVectors(*ommsystem.boxvectors)
        if prev_state_path and prev_state_path.exists():
            simulation.loadState(str(prev_state_path))
        _apply_atm_state(simulation.context, ommsystem, atm_state)
        simulation.context.applyConstraints(0.00001)
        diagnostic_keywords = (
            ommsystem.keywords if atm_state is not None else swapped_diagnostics_keywords
        )
        for reporter in _build_reporters(
            step_cfg, step_dir, keywords=diagnostic_keywords
        ):
            simulation.reporters.append(reporter)
        if not simulation.reporters:
            interval = max(1, int(step_cfg.get("reporter_interval", step_cfg.get("n_steps", 1))))
            simulation.reporters.append(
                StateDataReporter(stdout, interval, step=True, potentialEnergy=True, temperature=True, volume=True, speed=True)
            )
        if step_cfg["type"] == "md" and step_cfg.get("reset_velocities", False):
            thermostat = step_cfg.get("thermostat") or {}
            temperature = float(
                thermostat.get(
                    "initial_temperature_k",
                    thermostat.get(
                        "temperature_k", default_temperature / kelvin
                    ),
                )
            ) * kelvin
            simulation.context.setVelocitiesToTemperature(temperature)

        wall_start = time.perf_counter()
        if step_cfg["type"] == "minimization":
            logger.info(
                "Custom equilibration step %d/%d %s: minimization, tolerance=%g kJ/mol/nm, max_iterations=%d, restraints=%s",
                i + 1,
                len(steps),
                step_id,
                float(step_cfg.get("tolerance_kj_mol_nm", 10.0)),
                int(step_cfg.get("max_iterations", 0)),
                _step_restraint_label(step_cfg),
            )
            simulation.minimizeEnergy(
                tolerance=float(step_cfg.get("tolerance_kj_mol_nm", 10.0)) * kilojoule_per_mole / nanometer,
                maxIterations=int(step_cfg.get("max_iterations", 0)),
            )
            completed_steps = 0
        else:
            completed_steps = int(step_cfg["n_steps"])
            timestep_ps = float(step_cfg.get("timestep_ps", 0.002))
            ensemble = str(step_cfg.get("ensemble", "NVT")).upper()
            logger.info(
                "Custom equilibration step %d/%d %s: %s MD, steps=%d, timestep=%g ps, restraints=%s, barostat=%s",
                i + 1,
                len(steps),
                step_id,
                ensemble,
                completed_steps,
                timestep_ps,
                _step_restraint_label(step_cfg),
                _step_barostat_label(step_cfg),
            )
            thermostat = step_cfg.get("thermostat") or {}
            initial_temperature = thermostat.get("initial_temperature_k")
            if initial_temperature is None:
                simulation.step(completed_steps)
            else:
                initial_temperature = float(initial_temperature)
                final_temperature = float(
                    thermostat.get(
                        "temperature_k", default_temperature / kelvin
                    )
                )
                update_interval = int(
                    thermostat.get("temperature_update_interval_steps", 5000)
                )
                if update_interval < 1:
                    raise EquilibrationConfigError(
                        "thermostat.temperature_update_interval_steps must "
                        "be positive"
                    )
                integrated = 0
                while integrated < completed_steps:
                    block = min(update_interval, completed_steps - integrated)
                    integrated += block
                    temperature = initial_temperature + (
                        final_temperature - initial_temperature
                    ) * integrated / completed_steps
                    integrator.setTemperature(temperature * kelvin)
                    simulation.step(block)
                logger.info(
                    "Custom equilibration step %d/%d %s: temperature ramp "
                    "completed from %g K to %g K",
                    i + 1,
                    len(steps),
                    step_id,
                    initial_temperature,
                    final_temperature,
                )
            logger.info("Custom equilibration step %d/%d %s: MD integration finished", i + 1, len(steps), step_id)

        step_state_path = step_dir / "final_state.xml"
        step_pdb_path = step_dir / "final_state.pdb"
        logger.info("Custom equilibration step %d/%d %s: saving XML state to %s", i + 1, len(steps), step_id, step_state_path)
        simulation.saveState(str(step_state_path))
        _strip_integrator_parameters(step_state_path)
        logger.info("Custom equilibration step %d/%d %s: writing PDB to %s", i + 1, len(steps), step_id, step_pdb_path)
        _save_final_pdb(simulation, step_pdb_path)
        if diagnostic_keywords is not None:
            swapped_path = step_dir / "final_state_swapped.pdb"
            logger.info(
                "Custom equilibration step %d/%d %s: writing swapped PDB to %s",
                i + 1,
                len(steps),
                step_id,
                swapped_path,
            )
            positions = simulation.context.getState(getPositions=True).getPositions()
            write_atm_swapped_pdb(
                simulation.topology, positions, diagnostic_keywords, swapped_path
            )
        wall_seconds = time.perf_counter() - wall_start
        manifest["steps"][step_id] = {
            "type": step_cfg["type"],
            "completed_steps": completed_steps,
            "wall_seconds": wall_seconds,
            "final_state": str(step_state_path),
            "final_pdb": str(step_pdb_path),
            "final_swapped_pdb": str(step_dir / "final_state_swapped.pdb") if diagnostic_keywords is not None else None,
        }
        if step_cfg["type"] == "md" and wall_seconds > 0.0:
            ns_per_day = completed_steps * timestep_ps * 86.4 / wall_seconds
            manifest["steps"][step_id]["ns_per_day"] = ns_per_day
            logger.info(
                "Custom equilibration step %d/%d %s: completed in %.2f s, %.3f ns/day -> %s",
                i + 1,
                len(steps),
                step_id,
                wall_seconds,
                ns_per_day,
                step_state_path,
            )
        else:
            logger.info(
                "Custom equilibration step %d/%d %s: completed in %.2f s -> %s",
                i + 1,
                len(steps),
                step_id,
                wall_seconds,
                step_state_path,
            )
        with open(output_dir / "manifest.json", "w") as handle:
            json.dump(manifest, handle, indent=2)
        prev_state_path = step_state_path

    if prev_state_path is None:
        raise EquilibrationConfigError("Custom equilibration did not produce a final state")
    state = _load_state(prev_state_path)
    _save_compatible_state(
        ommsystem=ommsystem,
        source_state=state,
        platform=platform,
        platform_properties=platform_properties,
        final_state_path=final_state_path,
        final_pdb_path=final_pdb_path,
        atm_state=atm_state,
    )
    logger.info("Custom equilibration protocol complete: state=%s, pdb=%s", final_state_path, final_pdb_path)
    return {"final_state": str(final_state_path), "final_pdb": str(final_pdb_path), "steps": manifest["steps"]}
