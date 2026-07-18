import numpy as np
from openff.toolkit import ForceField, Molecule
from openff.units import unit as offunit

from atom_openmm.covalent_hybrid import build_covalent_hybrid_molecule
from atom_openmm.covalent_parameters import CovalentParameterBundle


def _bundle(smiles):
    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=1)
    charges = np.zeros(molecule.n_atoms)
    molecule.partial_charges = charges * offunit.elementary_charge
    system = ForceField("openff-2.2.1.offxml").create_openmm_system(
        molecule.to_topology(), charge_from_molecules=[molecule]
    )
    return CovalentParameterBundle(molecule, system, charges, smiles, {})


def _test_hybrid_molecule_has_identical_endpoint_particles():
    hybrid = build_covalent_hybrid_molecule(_bundle("CCO"), _bundle("CCCO"))
    assert hybrid.endpoint_a.getNumParticles() == hybrid.endpoint_b.getNumParticles()
    assert hybrid.endpoint_a.getNumParticles() == hybrid.topology.getNumAtoms()
    assert hybrid.unique_a or hybrid.unique_b
    assert len(hybrid.map_a_to_b) >= 2
    assert hybrid.endpoint_a.getNumConstraints() == hybrid.endpoint_b.getNumConstraints()
    assert hybrid.endpoint_a.getNumConstraints() > 0


def _test_required_mapping_expands_without_global_mcs():
    left = _bundle("CCO")
    right = _bundle("CCCO")
    hybrid = build_covalent_hybrid_molecule(left, right, required_pairs=[(0, 0), (1, 1)])
    assert hybrid.map_a_to_b[0] == 0
    assert hybrid.map_a_to_b[1] == 1
    assert len(hybrid.map_a_to_b) > 2
