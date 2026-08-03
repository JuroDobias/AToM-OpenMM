from __future__ import annotations

from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit
from openff.toolkit import Molecule
from openff.units import unit as offunit
from openmmforcefields.generators import EspalomaTemplateGenerator

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
) -> HybridParameterBundle:
    if not ligand_forcefield.startswith("espaloma"):
        raise HybridParameterError(
            "noncovalent hybrid topology currently requires an Espaloma ligand force field"
        )
    if ligand_charge_model != "nn":
        raise HybridParameterError(
            "noncovalent hybrid topology currently requires ligand_charge_model: nn"
        )
    molecule = Molecule.from_file(str(sdf), allow_undefined_stereo=False)
    generator = EspalomaTemplateGenerator(
        molecules=[molecule],
        forcefield=ligand_forcefield,
        template_generator_kwargs={"charge_method": "nn"},
    )
    forcefield = app.ForceField()
    forcefield.registerTemplateGenerator(generator.generator)
    system = forcefield.createSystem(
        molecule.to_topology().to_openmm(),
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
    molecule.partial_charges = charges * offunit.elementary_charge
    provenance = {
        "charge_model": "espaloma_nn",
        "ligand_forcefield": ligand_forcefield,
        "net_charge_e": float(charges.sum()),
        "uniform_charge_correction_e": float(correction),
        "source": str(Path(sdf).resolve()),
    }
    return HybridParameterBundle(
        molecule=molecule,
        system=system,
        charges_e=charges,
        cache_key=f"{Path(sdf).resolve()}:{ligand_forcefield}:nn",
        provenance=provenance,
    )
