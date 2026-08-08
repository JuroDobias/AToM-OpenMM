#!/usr/bin/env python
import argparse
import csv
import re
from pathlib import Path

import openmm as mm
from openmm import app, unit
import yaml

from atom_openmm.covalent_softcore import create_softcore_hamiltonian
from atom_openmm.neqti_integrator import _apply_amber_reciprocal_transforms


LAMBDAS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
KCAL_TO_KJ = 4.184


def _nonbonded(system):
    return next(force for force in system.getForces() if isinstance(force, mm.NonbondedForce))


def _parse_amber(path):
    text = path.read_text()
    energy_matches = re.findall(r"EPtot\s*=\s*([-+0-9.Ee]+)", text)
    derivative_matches = re.findall(r"DV/DL\s*=\s*([-+0-9.Ee]+)", text)
    if not energy_matches or not derivative_matches:
        raise RuntimeError(f"could not parse fixed-coordinate energy from {path}")
    alpha = re.search(r"Ewald Coefficient\s*=\s*([-+0-9.Ee]+)", text)
    grid = re.search(r"NFFT1\s*=\s*(\d+)\s+NFFT2\s*=\s*(\d+)\s+NFFT3\s*=\s*(\d+)", text)
    if alpha is None or grid is None:
        raise RuntimeError(f"could not parse PME settings from {path}")
    return {
        "energy_kcal_per_mol": float(energy_matches[0]),
        "dudlambda_kcal_per_mol": float(derivative_matches[0]),
        "ewald_alpha_per_a": float(alpha.group(1)),
        "pme_grid": tuple(int(value) for value in grid.groups()),
    }


def _endpoints(root, pme):
    topology = app.AmberPrmtopFile(str(root / "system.parm7"))
    endpoint_a = topology.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        ewaldErrorTolerance=1.0e-5,
    )
    endpoint_b = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(endpoint_a))
    ligand_atoms = [
        atom.index
        for residue in topology.topology.residues()
        if residue.name == "MTH"
        for atom in residue.atoms()
    ]
    if len(ligand_atoms) != 5:
        raise RuntimeError(f"expected five MTH atoms, found {len(ligand_atoms)}")
    for system in (endpoint_a, endpoint_b):
        force = _nonbonded(system)
        force.setUseDispersionCorrection(False)
        force.setPMEParameters(
            pme["ewald_alpha_per_a"] * 10.0,
            *pme["pme_grid"],
        )
    force_b = _nonbonded(endpoint_b)
    ligand_set = set(ligand_atoms)
    for atom in ligand_atoms:
        _, sigma, _ = force_b.getParticleParameters(atom)
        force_b.setParticleParameters(atom, 0.0, sigma, 0.0)
    for index in range(force_b.getNumExceptions()):
        atom1, atom2, _, sigma, _ = force_b.getExceptionParameters(index)
        if int(atom1) in ligand_set or int(atom2) in ligand_set:
            force_b.setExceptionParameters(index, atom1, atom2, 0.0, sigma, 0.0)
    return topology, endpoint_a, endpoint_b, ligand_atoms


def _context(system, platform_name):
    platform = mm.Platform.getPlatformByName(platform_name)
    properties = {}
    if platform_name == "CUDA":
        properties = {"Precision": "double", "DeterministicForces": "true"}
    return mm.Context(system, mm.VerletIntegrator(0.001), platform, properties)


def _energy(context):
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilocalorie_per_mole
    )


def _set_lambda(context, parameter_values, alchemical_lambda):
    resolved = {}
    for name, values in parameter_values.items():
        resolved[name] = (
            (1.0 - alchemical_lambda) * values[0]
            + alchemical_lambda * values[-1]
        )
    _apply_amber_reciprocal_transforms(resolved)
    for name, value in resolved.items():
        context.setParameter(name, value)


def _request_parameter_derivatives(system):
    endpoint_groups = {}
    for force in system.getForces():
        if force.getName() == "CovalentAmberGTIEndpointElectrostaticsA":
            force.setForceGroup(20)
            endpoint_groups["A"] = 20
        elif force.getName() == "CovalentAmberGTIEndpointElectrostaticsB":
            force.setForceGroup(21)
            endpoint_groups["B"] = 21
        if not hasattr(force, "addEnergyParameterDerivative"):
            continue
        existing = {
            force.getEnergyParameterDerivativeName(index)
            for index in range(force.getNumEnergyParameterDerivatives())
        }
        for index in range(force.getNumGlobalParameters()):
            name = force.getGlobalParameterName(index)
            if name not in existing:
                force.addEnergyParameterDerivative(name)
    return endpoint_groups


def _smoothstep2_derivative(value):
    return 30.0 * value**2 * (value - 1.0) ** 2


def _analytic_derivative(context, alchemical_lambda, endpoint_groups):
    derivatives = dict(
        context.getState(
            getParameterDerivatives=True
        ).getEnergyParameterDerivatives()
    )
    value_a = 1.0 - alchemical_lambda
    value_b = alchemical_lambda
    weight_a = value_a**3 * (10.0 + value_a * (-15.0 + 6.0 * value_a))
    weight_b = value_b**3 * (10.0 + value_b * (-15.0 + 6.0 * value_b))
    dweight_a = -_smoothstep2_derivative(value_a)
    dweight_b = _smoothstep2_derivative(value_b)
    slopes = {
        "COVALENT_CHARGE_A": -1.0,
        "COVALENT_CHARGE_B": 1.0,
        "COVALENT_MAPPED_CHARGE": 1.0,
        "COVALENT_STERICS": 1.0,
        "COVALENT_STERICS_A": -1.0,
        "COVALENT_STERICS_B": 1.0,
    }
    value = sum(
        float(derivatives.get(name, 0.0)) * slope
        for name, slope in slopes.items()
    )
    for label, weight, slope in (
        ("A", weight_a, dweight_a),
        ("B", weight_b, dweight_b),
    ):
        if weight == 0.0 or slope == 0.0:
            continue
        weighted_energy = context.getState(
            getEnergy=True,
            groups={endpoint_groups[label]},
        ).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        value += slope * weighted_energy / weight
    return value / KCAL_TO_KJ


def compare(root, platform_name):
    amber = {
        value: _parse_amber(root / f"amber_{value:.2f}.out") for value in LAMBDAS
    }
    pme = amber[LAMBDAS[0]]
    topology, endpoint_a, endpoint_b, ligand_atoms = _endpoints(root, pme)
    hamiltonian = create_softcore_hamiltonian(
        endpoint_a,
        endpoint_b,
        ligand_atoms,
        [],
        function="amber_ssc2",
        coulomb_function="amber_ssc2",
        ssc2_alpha_lj=0.5,
        ssc2_beta_coul=1.0,
        ssc2_switch_width_nm=0.2,
        total_steps=100,
        path_mode="concerted",
        stage_interpolation="linear",
        use_long_range_correction=False,
    )
    endpoint_groups = _request_parameter_derivatives(hamiltonian.system)
    coordinates = app.AmberInpcrdFile(str(root / "system.rst7"))
    context = _context(hamiltonian.system, platform_name)
    context.setPositions(coordinates.positions)
    context.setPeriodicBoxVectors(*coordinates.boxVectors)

    rows = []
    for value in LAMBDAS:
        _set_lambda(context, hamiltonian.parameter_values, value)
        openmm_energy = _energy(context)
        derivative = _analytic_derivative(context, value, endpoint_groups)
        rows.append(
            {
                "lambda": value,
                "amber_energy_kcal_per_mol": amber[value]["energy_kcal_per_mol"],
                "openmm_energy_kcal_per_mol": openmm_energy,
                "amber_dudlambda_kcal_per_mol": amber[value]["dudlambda_kcal_per_mol"],
                "openmm_dudlambda_kcal_per_mol": derivative,
            }
        )
    amber_origin = rows[0]["amber_energy_kcal_per_mol"]
    openmm_origin = rows[0]["openmm_energy_kcal_per_mol"]
    for row in rows:
        row["amber_relative_kcal_per_mol"] = row["amber_energy_kcal_per_mol"] - amber_origin
        row["openmm_relative_kcal_per_mol"] = row["openmm_energy_kcal_per_mol"] - openmm_origin
        row["relative_error_kcal_per_mol"] = (
            row["openmm_relative_kcal_per_mol"] - row["amber_relative_kcal_per_mol"]
        )
        row["derivative_error_kcal_per_mol"] = (
            row["openmm_dudlambda_kcal_per_mol"] - row["amber_dudlambda_kcal_per_mol"]
        )
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    max_energy_error = max(abs(row["relative_error_kcal_per_mol"]) for row in rows)
    max_derivative_error = max(abs(row["derivative_error_kcal_per_mol"]) for row in rows)
    result = {
        "schema_version": 1,
        "status": "passed" if max_energy_error <= 0.02 and max_derivative_error <= 0.1 else "failed",
        "system": "neutral methane annihilation in TIP3P water",
        "amber_executable": "pmemd.cuda_DPFP 26",
        "openmm_platform": f"{platform_name} double precision",
        "lambda_values": list(LAMBDAS),
        "pme": {
            "cutoff_nm": 1.0,
            "ewald_alpha_per_nm": pme["ewald_alpha_per_a"] * 10.0,
            "grid": list(pme["pme_grid"]),
        },
        "max_relative_energy_error_kcal_per_mol": max_energy_error,
        "max_dudlambda_error_kcal_per_mol": max_derivative_error,
        "tolerances": {
            "relative_energy_kcal_per_mol": 0.02,
            "dudlambda_kcal_per_mol": 0.1,
        },
    }
    (root / "result.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    print(yaml.safe_dump(result, sort_keys=False), end="")
    return 0 if result["status"] == "passed" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--platform", choices=("CUDA", "CPU"), default="CUDA")
    args = parser.parse_args()
    raise SystemExit(compare(args.root.resolve(), args.platform))


if __name__ == "__main__":
    main()
