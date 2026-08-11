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


def _nonbonded(system):
    return next(force for force in system.getForces() if isinstance(force, mm.NonbondedForce))


def _last_xvg_row(path):
    rows = [line.split() for line in path.read_text().splitlines() if line and line[0] not in "#@"]
    if not rows:
        raise RuntimeError(f"no data rows in {path}")
    return [float(value) for value in rows[-1]]


def _parse_pme(log_path):
    text = log_path.read_text()
    alpha_patterns = (
        r"Ewald coefficient[^=]*=\s*([-+0-9.Ee]+)",
        r"ewald.*alpha[^=]*=\s*([-+0-9.Ee]+)",
    )
    grid_patterns = (
        r"fourier grid dimensions[^0-9]*(\d+)\s+(\d+)\s+(\d+)",
        r"Using a fourier grid of\s+(\d+)x(\d+)x(\d+)",
        r"grid\s+(\d+)\s+(\d+)\s+(\d+)",
    )
    alpha = next((re.search(pattern, text, re.I) for pattern in alpha_patterns if re.search(pattern, text, re.I)), None)
    gaussian_width = re.search(r"Gaussian width \(1/beta\) of\s*([-+0-9.Ee]+)\s*nm", text, re.I)
    grid = next((re.search(pattern, text, re.I) for pattern in grid_patterns if re.search(pattern, text, re.I)), None)
    if grid is None:
        dimensions = [
            re.search(rf"fourier-{axis}\s*=\s*(\d+)", text, re.I)
            for axis in ("nx", "ny", "nz")
        ]
        if all(dimensions):
            grid_values = tuple(int(match.group(1)) for match in dimensions)
        else:
            grid_values = None
    else:
        grid_values = tuple(int(value) for value in grid.groups())
    if alpha is not None:
        alpha_value = float(alpha.group(1))
    elif gaussian_width is not None:
        alpha_value = 1.0 / float(gaussian_width.group(1))
    else:
        alpha_value = None
    if alpha_value is None or grid_values is None:
        raise RuntimeError(f"could not parse PME settings from {log_path}")
    return alpha_value, grid_values


def _parse_gromacs(root, value):
    run = root / f"gmx_{value:.2f}"
    potential = _last_xvg_row(run / "potential.xvg")[-1]
    dhdl_path = run / "dhdl.xvg"
    legends = re.findall(r'@\s+s(\d+)\s+legend\s+"([^"]+)"', dhdl_path.read_text())
    row = _last_xvg_row(dhdl_path)
    derivative_columns = [int(index) + 1 for index, label in legends if "dH/d" in label or "derivative" in label.lower()]
    if not derivative_columns and len(row) == 3:
        # With one derivative and dhdl-print-energy=potential, GROMACS 2026.3
        # writes time, potential, and dH/dlambda without per-series legends.
        derivative_columns = [2]
    if not derivative_columns:
        raise RuntimeError(f"could not identify dH/dlambda column in {dhdl_path}")
    derivative = sum(row[index] for index in derivative_columns)
    return potential, derivative


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
        atom.index for residue in topology.topology.residues()
        if residue.name == "MTH" for atom in residue.atoms()
    ]
    if len(ligand_atoms) != 5:
        raise RuntimeError(f"expected five MTH atoms, found {len(ligand_atoms)}")
    for system in (endpoint_a, endpoint_b):
        force = _nonbonded(system)
        force.setUseDispersionCorrection(False)
        force.setPMEParameters(pme[0], *pme[1])
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
    if platform_name == "CUDA":
        properties = {"Precision": "double", "DeterministicForces": "true"}
    elif platform_name == "OpenCL":
        properties = {"Precision": "double"}
    else:
        properties = {}
    return mm.Context(system, mm.VerletIntegrator(0.001), platform, properties)


def _set_lambda(context, parameter_values, value):
    parameters = {
        name: (1.0 - value) * values[0] + value * values[-1]
        for name, values in parameter_values.items()
    }
    _apply_amber_reciprocal_transforms(parameters)
    for name, parameter in parameters.items():
        context.setParameter(name, parameter)


def _energy(context):
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _derivative(context, parameter_values, value):
    # The total periodic energy is much larger than the alchemical term.
    # A broad local stencil avoids amplifying PME/reduction noise.
    delta = 0.05
    left = max(0.0, value - delta)
    right = min(1.0, value + delta)
    _set_lambda(context, parameter_values, left)
    energy_left = _energy(context)
    _set_lambda(context, parameter_values, right)
    energy_right = _energy(context)
    return (energy_right - energy_left) / (right - left)


def compare(root, platform_name):
    gromacs = {value: _parse_gromacs(root, value) for value in LAMBDAS}
    pme = _parse_pme(root / "gmx_0.00" / "md.log")
    topology, endpoint_a, endpoint_b, ligand_atoms = _endpoints(root, pme)
    hamiltonian = create_softcore_hamiltonian(
        endpoint_a, endpoint_b, ligand_atoms, [],
        function="gapsys", coulomb_function="gapsys",
        gapsys_scale_linpoint_lj=0.85, gapsys_scale_linpoint_q=0.30,
        gapsys_sigma_nm=0.30, total_steps=100, path_mode="concerted",
        use_long_range_correction=False,
    )
    coordinates = app.AmberInpcrdFile(str(root / "system.rst7"))
    context = _context(hamiltonian.system, platform_name)
    context.setPositions(coordinates.positions)
    context.setPeriodicBoxVectors(*coordinates.boxVectors)
    rows = []
    for value in LAMBDAS:
        _set_lambda(context, hamiltonian.parameter_values, value)
        openmm_energy = _energy(context)
        openmm_derivative = _derivative(context, hamiltonian.parameter_values, value)
        gmx_energy, gmx_derivative = gromacs[value]
        rows.append({
            "lambda": value,
            "gromacs_energy_kj_per_mol": gmx_energy,
            "openmm_energy_kj_per_mol": openmm_energy,
            "gromacs_dudlambda_kj_per_mol": gmx_derivative,
            "openmm_dudlambda_kj_per_mol": openmm_derivative,
        })
    gmx_origin = rows[0]["gromacs_energy_kj_per_mol"]
    openmm_origin = rows[0]["openmm_energy_kj_per_mol"]
    for row in rows:
        row["relative_error_kcal_per_mol"] = (
            (row["openmm_energy_kj_per_mol"] - openmm_origin)
            - (row["gromacs_energy_kj_per_mol"] - gmx_origin)
        ) / 4.184
        row["derivative_error_kcal_per_mol"] = (
            row["openmm_dudlambda_kj_per_mol"] - row["gromacs_dudlambda_kj_per_mol"]
        ) / 4.184
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    max_energy = max(abs(row["relative_error_kcal_per_mol"]) for row in rows)
    max_derivative = max(abs(row["derivative_error_kcal_per_mol"]) for row in rows)
    result = {
        "schema_version": 1,
        "status": "passed" if max_energy <= 0.02 and max_derivative <= 0.10 else "failed",
        "system": "neutral methane annihilation in periodic TIP3P water",
        "gromacs": "Gapsys CPU free-energy kernel",
        "openmm_platform": f"{platform_name} double precision",
        "lambda_values": list(LAMBDAS),
        "pme": {"cutoff_nm": 1.0, "ewald_alpha_per_nm": pme[0], "grid": list(pme[1])},
        "max_relative_energy_error_kcal_per_mol": max_energy,
        "max_dudlambda_error_kcal_per_mol": max_derivative,
        "tolerances": {"relative_energy_kcal_per_mol": 0.02, "dudlambda_kcal_per_mol": 0.10},
    }
    (root / "result.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    print(yaml.safe_dump(result, sort_keys=False), end="")
    return 0 if result["status"] == "passed" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--platform", choices=("CUDA", "OpenCL", "CPU"), default="CUDA")
    args = parser.parse_args()
    raise SystemExit(compare(args.root.resolve(), args.platform))


if __name__ == "__main__":
    main()
