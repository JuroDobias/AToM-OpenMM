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
    VirtualSiteParameter,
    _system_charges,
    constrain_charge_sum,
)
from atom_openmm.ligand_parameterization import (
    _find_sigma_holes,
    _sigma_hole_distance_a,
    _sigma_hole_name,
    normalize_fixed_sigma_hole_settings,
    normalize_sigma_hole_settings,
)


HybridParameterBundle = CovalentParameterBundle
HybridParameterError = CovalentParameterError


def _apply_fixed_sigma_holes(molecule, system, charges_e, settings):
    if settings is None:
        return np.asarray(charges_e, dtype=float), ()
    settings = normalize_fixed_sigma_hole_settings(settings)
    sigma_settings = {
        key: settings[key]
        for key in ("halogens", "smarts", "distance_a", "distances_a")
        if key in settings
    }
    protocol = {"sigma_holes": normalize_sigma_hole_settings(sigma_settings)}
    matches = _find_sigma_holes(molecule, protocol)
    if not matches:
        return np.asarray(charges_e, dtype=float), ()
    charges = np.asarray(charges_e, dtype=float).copy()
    sites = []
    for index, parents in enumerate(matches, start=1):
        halogen = int(parents[1])
        symbol = molecule.atoms[halogen].symbol
        charge = float(
            settings["charge_e"]
            if "charge_e" in settings else settings["charges_e"][symbol]
        )
        charges[halogen] -= charge
        sites.append(VirtualSiteParameter(
            name=_sigma_hole_name(molecule, parents, index),
            kind="sigma_hole",
            parent_atom_indices=tuple(int(value) for value in parents),
            distance_a=_sigma_hole_distance_a(molecule, parents, protocol["sigma_holes"]),
            charge_e=charge,
        ))
    nonbonded = next(
        force for force in system.getForces()
        if isinstance(force, mm.NonbondedForce)
    )
    for index, value in enumerate(charges):
        _, sigma, epsilon = nonbonded.getParticleParameters(index)
        nonbonded.setParticleParameters(
            index, float(value) * unit.elementary_charge, sigma, epsilon
        )
    return charges, tuple(sites)


def parameterize_ligand(
    sdf: Path,
    *,
    ligand_forcefield: str = "espaloma-0.3.2",
    ligand_charge_model: str = "nn",
    allow_undefined_stereo: bool = False,
    ligand_parameter_cache: Path | None = None,
    ligand_parameter_protocol: str = "gaff2-resp-cl-ep-v1",
    ligand_sigma_holes: dict | None = None,
) -> HybridParameterBundle:
    if ligand_sigma_holes is not None:
        ligand_sigma_holes = normalize_fixed_sigma_hole_settings(
            ligand_sigma_holes
        )
    if ligand_charge_model == "resp-sigma-hole":
        if ligand_parameter_cache is None:
            raise HybridParameterError(
                "resp-sigma-hole requires workflow.setup.ligand_parameter_cache"
            )
        from atom_openmm.ligand_parameterization import load_cached_parameters

        parameters = load_cached_parameters(
            Path(sdf), cache_dir=Path(ligand_parameter_cache),
            protocol_id=str(ligand_parameter_protocol),
        )
        cached_forcefield = parameters.provenance.get("ligand_forcefield")
        if cached_forcefield != ligand_forcefield:
            raise HybridParameterError(
                f"cached ligand force field {cached_forcefield!r} differs from "
                f"workflow ligand_forcefield {ligand_forcefield!r}"
            )
        return parameters
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
    charges, virtual_sites = _apply_fixed_sigma_holes(
        molecule, system, charges, ligand_sigma_holes
    )
    for atom, parameterized_atom in zip(molecule.atoms, parameterized_molecule.atoms):
        atom.name = parameterized_atom.name
    molecule.partial_charges = charges * offunit.elementary_charge
    provenance = {
        "charge_model": charge_model,
        "ligand_forcefield": ligand_forcefield,
        "atomic_charge_sum_e": float(charges.sum()),
        "net_charge_e": float(
            charges.sum() + sum(site.charge_e for site in virtual_sites)
        ),
        "uniform_charge_correction_e": float(correction),
        "source": str(Path(sdf).resolve()),
        "allow_undefined_stereo": bool(allow_undefined_stereo),
        "fixed_sigma_holes": (
            None if ligand_sigma_holes is None else {
                **ligand_sigma_holes,
                "site_count": len(virtual_sites),
            }
        ),
    }
    return HybridParameterBundle(
        molecule=molecule,
        system=system,
        charges_e=charges,
        cache_key=(
            f"{Path(sdf).resolve()}:{ligand_forcefield}:{charge_model}:"
            f"{ligand_sigma_holes!r}"
        ),
        provenance=provenance,
        virtual_sites=virtual_sites,
    )
