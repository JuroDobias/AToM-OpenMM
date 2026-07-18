import numpy as np
import openmm as mm
from openff.toolkit import ForceField, Molecule
from openff.units import unit as offunit

from atom_openmm.covalent_parameters import CovalentParameterBundle
from atom_openmm.covalent_systems import create_solvated_capped_reference
from atom_openmm.covalent_hybrid import build_covalent_hybrid_molecule
from atom_openmm.covalent_systems import solvate_capped_reference_hybrid


def _test_solvated_reference_preserves_supplied_charges(tmp_path):
    molecule = Molecule.from_smiles("CO")
    molecule.generate_conformers(n_conformers=1)
    charges = np.linspace(-0.1, 0.1, molecule.n_atoms)
    charges -= charges.sum() / molecule.n_atoms
    molecule.partial_charges = charges * offunit.elementary_charge
    system = ForceField("openff-2.2.1.offxml").create_openmm_system(
        molecule.to_topology(), charge_from_molecules=[molecule]
    )
    bundle = CovalentParameterBundle(
        molecule=molecule,
        system=system,
        charges_e=charges,
        cache_key="test",
        provenance={"charge_model": "test"},
    )
    prepared = create_solvated_capped_reference(
        bundle,
        padding_a=3.0,
        ionic_strength_molar=0.0,
        template_cache=tmp_path / "templates.json",
    )
    force = next(force for force in prepared.system.getForces() if isinstance(force, mm.NonbondedForce))
    observed = np.asarray(
        [
            force.getParticleParameters(index)[0].value_in_unit(mm.unit.elementary_charge)
            for index in range(molecule.n_atoms)
        ]
    )
    assert np.allclose(observed, charges)
    assert prepared.system.getNumParticles() == prepared.topology.getNumAtoms()


def _bundle(smiles):
    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=1)
    charges = np.linspace(-0.1, 0.1, molecule.n_atoms)
    charges -= charges.sum() / molecule.n_atoms
    molecule.partial_charges = charges * offunit.elementary_charge
    system = ForceField("openff-2.2.1.offxml").create_openmm_system(
        molecule.to_topology(), charge_from_molecules=[molecule]
    )
    return CovalentParameterBundle(molecule, system, charges, smiles, {})


def _test_hybrid_reference_uses_one_solvent_environment(tmp_path):
    bundle_a = _bundle("CCO")
    bundle_b = _bundle("CCCO")
    physical_a = create_solvated_capped_reference(
        bundle_a, padding_a=3.0, ionic_strength_molar=0.0,
        template_cache=tmp_path / "hybrid-templates.json",
    )
    hybrid = build_covalent_hybrid_molecule(bundle_a, bundle_b)
    prepared = solvate_capped_reference_hybrid(hybrid, physical_a)
    assert prepared.endpoint_a.getNumParticles() == prepared.endpoint_b.getNumParticles()
    assert prepared.endpoint_a.getNumParticles() == prepared.topology.getNumAtoms()
    assert prepared.endpoint_a.getDefaultPeriodicBoxVectors() == prepared.endpoint_b.getDefaultPeriodicBoxVectors()
