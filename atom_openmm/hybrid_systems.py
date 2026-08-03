from __future__ import annotations

from pathlib import Path

import numpy as np
from openmm import app, unit
from openmmforcefields.generators import EspalomaTemplateGenerator

from atom_openmm.covalent_parameters import CovalentParameterError
from atom_openmm.covalent_systems import PreparedCovalentSystem, _nonbonded_force


PreparedPhysicalEnvironment = PreparedCovalentSystem
HybridSystemError = CovalentParameterError


def _forcefield_files(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise HybridSystemError("force-field settings must be strings or lists of strings")


def create_physical_ligand_environment(
    parameters,
    *,
    receptor: Path | None,
    setup: dict,
    solvation_seed: int,
) -> PreparedPhysicalEnvironment:
    molecule = parameters.molecule
    topology = molecule.to_topology().to_openmm()
    positions = molecule.conformers[0].to_openmm()
    modeller = app.Modeller(topology, positions)
    if receptor is not None:
        receptor_pdb = app.PDBFile(str(receptor))
        modeller.add(receptor_pdb.topology, receptor_pdb.positions)

    protein_files = _forcefield_files(
        setup.get("protein_forcefield"), ["amber14-all.xml"]
    )
    solvent_files = _forcefield_files(
        setup.get("solvent_forcefield"), ["amber14/tip3p.xml"]
    )
    forcefield = app.ForceField(*(protein_files + solvent_files))
    generator = EspalomaTemplateGenerator(
        molecules=[molecule],
        forcefield=setup.get("ligand_forcefield", "espaloma-0.3.2"),
        template_generator_kwargs={"charge_method": "from-molecule"},
    )
    forcefield.registerTemplateGenerator(generator.generator)
    modeller.addExtraParticles(forcefield)
    solvent_model = setup.get("solvent_model")
    if solvent_model is None:
        solvent_model = (
            "tip4pew"
            if any("opc.xml" in item for item in solvent_files)
            else "tip3p"
        )
    padding_a = float(setup.get("solvent_padding_a", 10.0))
    ionic_strength = float(setup.get("ionic_strength_molar", 0.15))
    from atom_openmm.covalent_systems import _seeded_python_random

    with _seeded_python_random(solvation_seed):
        modeller.addSolvent(
            forcefield,
            model=solvent_model,
            padding=padding_a * unit.angstrom,
            ionicStrength=ionic_strength * unit.molar,
            neutralize=True,
        )
    cutoff_a = float(setup.get("nonbonded_cutoff_a", 9.0))
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_a * unit.angstrom,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True,
    )
    force = _nonbonded_force(system)
    observed = np.asarray(
        [
            force.getParticleParameters(index)[0].value_in_unit(unit.elementary_charge)
            for index in range(molecule.n_atoms)
        ]
    )
    if not np.allclose(observed, parameters.charges_e, atol=1.0e-6):
        raise HybridSystemError(
            "physical environment does not preserve the parameterized ligand charges"
        )
    provenance = {
        **parameters.provenance,
        "environment": "complex" if receptor is not None else "solvent",
        "receptor": None if receptor is None else str(Path(receptor).resolve()),
        "protein_forcefield": protein_files,
        "solvent_forcefield": solvent_files,
        "solvent_model": solvent_model,
        "padding_a": padding_a,
        "ionic_strength_molar": ionic_strength,
        "nonbonded_cutoff_a": cutoff_a,
        "solvation_seed": int(solvation_seed),
        "solute_atom_count": molecule.n_atoms,
        "total_particle_count": system.getNumParticles(),
    }
    return PreparedPhysicalEnvironment(
        modeller.topology,
        modeller.positions,
        system,
        molecule.n_atoms,
        provenance,
    )
