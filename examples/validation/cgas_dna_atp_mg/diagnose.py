"""GPU startup check using the prepared cGAS system and a saved minimized state."""

import argparse
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit
import yaml

from atom_openmm import md_validation as validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--variant", type=int, default=0, choices=(0, 3, 6, 9))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--restraint-k", type=float, default=5.0)
    parser.add_argument("--reset-velocities", action="store_true")
    args = parser.parse_args()
    config = validation._load_config(args.config)
    paths = validation._prepared_paths(config)
    manifest = yaml.safe_load(paths["manifest"].read_text())
    for key in ("system", "topology"):
        if validation._sha256(paths[key]) != manifest["checksums"][key]:
            raise RuntimeError(f"canonical {key} checksum changed")
    topology_file = app.PDBxFile(str(paths["topology"]))
    topology, positions = topology_file.topology, topology_file.positions
    system = mm.XmlSerializer.deserialize(paths["system"].read_text())
    spec = validation.task_spec(config, args.variant)
    if spec["panteva_m12_6_4"]:
        validation.apply_panteva_m1264(
            system, topology, atom_classes=manifest["atom_classes"],
            polarizability_table=validation._resolve(
                config, config["panteva_m1264"]["polarizability_table"]),
        )
    if spec["hu2024_atp"]:
        hu = config["hu2024_atp"]
        validation.apply_hu2024_atp(
            system, topology, prepi=validation._resolve(config, hu["prepi"]),
            frcmod=validation._resolve(config, hu["frcmod"]),
            mod_py=validation._resolve(config, hu["mod_py"]),
            atom_classes=manifest["atom_classes"],
        )
    state = mm.XmlSerializer.deserialize(args.state.read_text())
    if args.restraint_k:
        validation._add_restraints(system, topology, positions, args.restraint_k)
    for timestep in (1.0, 2.0):
        integrator = mm.LangevinMiddleIntegrator(
            args.temperature_k * unit.kelvin, 1 / unit.picosecond,
            timestep * unit.femtoseconds,
        )
        integrator.setRandomNumberSeed(20260914 + args.variant * 1009)
        platform, properties = validation._platform(config)
        simulation = app.Simulation(topology, system, integrator, platform, properties)
        simulation.context.setState(state)
        energy = simulation.context.getState(getEnergy=True, getForces=True)
        forces = np.asarray(energy.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer))
        magnitude = np.linalg.norm(forces, axis=1)
        worst = int(np.argmax(magnitude))
        atoms = list(topology.atoms())
        atom = atoms[worst]
        print("variant", spec["variant"], "timestep_fs", timestep,
              "energy_kj", energy.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
              "max_force", magnitude[worst], "worst_atom", worst,
              atom.residue.name, atom.residue.id, atom.name, flush=True)
        if args.reset_velocities:
            simulation.context.setVelocitiesToTemperature(
                args.temperature_k * unit.kelvin, 20260914,
            )
        for step in range(args.steps):
            try:
                simulation.step(1)
            except Exception as exc:
                print("failed at step", step + 1, "timestep", timestep,
                      "reason", exc, flush=True)
                break
            if (step + 1) % 100 == 0:
                value = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                print("step", step + 1, "energy_kj", value.value_in_unit(
                    unit.kilojoule_per_mole), flush=True)
        del simulation


if __name__ == "__main__":
    main()
