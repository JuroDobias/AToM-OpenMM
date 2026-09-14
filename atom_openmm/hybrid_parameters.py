from __future__ import annotations

from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit
from openff.toolkit import Molecule
from openff.units import unit as offunit
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator

from atom_openmm.covalent_parameters import (
    CovalentParameterBundle,
    CovalentParameterError,
    _system_charges,
    constrain_charge_sum,
)


HybridParameterBundle = CovalentParameterBundle
HybridParameterError = CovalentParameterError


def parameterize_ligand(
    sdf: Path,
    *,
    ligand_forcefield: str = "espaloma-0.3.2",
    ligand_charge_model: str = "nn",
    allow_undefined_stereo: bool = False,
) -> HybridParameterBundle:
    molecule = Molecule.from_file(
        str(sdf), allow_undefined_stereo=bool(allow_undefined_stereo)
    )
    molecule.generate_unique_atom_names()
    if ligand_forcefield.startswith("espaloma"):
        if ligand_charge_model != "nn":
            raise HybridParameterError(
                "Espaloma ligand parameterization requires ligand_charge_model: nn"
            )
        parameterized_molecule = molecule
        generator = EspalomaTemplateGenerator(
            molecules=[parameterized_molecule],
            forcefield=ligand_forcefield,
            template_generator_kwargs={"charge_method": "nn"},
        )
        charge_model = "espaloma_nn"
    elif ligand_forcefield.startswith("gaff-"):
        if ligand_charge_model not in {"am1-bcc", "bcc"}:
            raise HybridParameterError(
                "GAFF ligand parameterization requires ligand_charge_model: am1-bcc"
            )
        # GAFFTemplateGenerator generates a conformer internally.  Parameterize a
        # copy so the docked conformer on the returned molecule remains unchanged.
        parameterized_molecule = Molecule(molecule)
        generator = GAFFTemplateGenerator(
            molecules=[parameterized_molecule], forcefield=ligand_forcefield
        )
        charge_model = "am1-bcc"
    else:
        raise HybridParameterError(
            "ligand_forcefield must be an Espaloma or GAFF force field"
        )
    forcefield = app.ForceField()
    forcefield.registerTemplateGenerator(generator.generator)
    system = forcefield.createSystem(
        parameterized_molecule.to_topology().to_openmm(),
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    charges = _system_charges(system)
    formal_charge = float(molecule.total_charge.m_as(offunit.elementary_charge))
    charges, correction = constrain_charge_sum(
        charges, np.arange(molecule.n_atoms), formal_charge
    )
    nonbonded = next(
        force for force in system.getForces() if isinstance(force, mm.NonbondedForce)
    )
    for index, charge in enumerate(charges):
        _, sigma, epsilon = nonbonded.getParticleParameters(index)
        nonbonded.setParticleParameters(index, charge, sigma, epsilon)
    for atom, parameterized_atom in zip(molecule.atoms, parameterized_molecule.atoms):
        atom.name = parameterized_atom.name
    molecule.partial_charges = charges * offunit.elementary_charge
    provenance = {
        "charge_model": charge_model,
        "ligand_forcefield": ligand_forcefield,
        "net_charge_e": float(charges.sum()),
        "uniform_charge_correction_e": float(correction),
        "source": str(Path(sdf).resolve()),
        "allow_undefined_stereo": bool(allow_undefined_stereo),
    }
    return HybridParameterBundle(
        molecule=molecule,
        system=system,
        charges_e=charges,
        cache_key=f"{Path(sdf).resolve()}:{ligand_forcefield}:{charge_model}",
        provenance=provenance,
    )
