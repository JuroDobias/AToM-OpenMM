import numpy as np
import openmm as mm
from openmm import unit
from openff.toolkit import ForceField, Molecule
from openff.units import unit as offunit
from rdkit import Chem

from atom_openmm.covalent_hybrid import (
    DummyBondedScales,
    _add_unique_vacuum_nonbonded,
    _hybrid_topology,
    _inactive_scales,
    build_covalent_hybrid_molecule,
)
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


def test_hybrid_topology_sanitizes_atom_names_for_mmcif():
    molecule = Chem.MolFromSmiles("CO")
    molecule.GetAtomWithIdx(0).SetProp("_Name", "C 1")
    molecule.GetAtomWithIdx(1).SetProp("_Name", "O'2")

    topology = _hybrid_topology(molecule, molecule, {0: 0, 1: 1}, {0: 0, 1: 1})

    assert [atom.name for atom in topology.atoms()] == ["C1", "O2"]


def test_endpoint_excludes_all_cross_branch_nonbonded_pairs():
    hybrid = build_covalent_hybrid_molecule(_bundle("CCO"), _bundle("CCN"))
    nonbonded = next(
        force
        for force in hybrid.endpoint_a.getForces()
        if isinstance(force, mm.NonbondedForce)
    )
    exceptions = {
        tuple(
            sorted(
                (
                    int(nonbonded.getExceptionParameters(index)[0]),
                    int(nonbonded.getExceptionParameters(index)[1]),
                )
            )
        )
        for index in range(nonbonded.getNumExceptions())
    }
    expected = {
        tuple(
            sorted(
                (
                    hybrid.map_a_to_hybrid[atom_a],
                    hybrid.map_b_to_hybrid[atom_b],
                )
            )
        )
        for atom_a in hybrid.unique_a
        for atom_b in hybrid.unique_b
    }

    assert expected <= exceptions


def _test_hybrid_molecule_has_identical_endpoint_particles():
    hybrid = build_covalent_hybrid_molecule(_bundle("CCO"), _bundle("CCCO"))
    assert hybrid.endpoint_a.getNumParticles() == hybrid.endpoint_b.getNumParticles()
    assert hybrid.endpoint_a.getNumParticles() == hybrid.topology.getNumAtoms()
    assert hybrid.unique_a or hybrid.unique_b
    assert len(hybrid.map_a_to_b) >= 2
    assert hybrid.endpoint_a.getNumConstraints() == hybrid.endpoint_b.getNumConstraints()
    assert hybrid.endpoint_a.getNumConstraints() > 0


def _test_required_mapping_selects_an_anchored_mcs():
    left = _bundle("CCO")
    right = _bundle("CCN")
    hybrid = build_covalent_hybrid_molecule(left, right, required_pairs=[(0, 0), (1, 1)])
    assert hybrid.map_a_to_b[0] == 0
    assert hybrid.map_a_to_b[1] == 1
    assert len(hybrid.map_a_to_b) > 2


def test_explicit_atom_map_does_not_expand_to_unrestricted_mcs():
    left = _bundle("CCCO")
    right = _bundle("CCCCO")

    hybrid = build_covalent_hybrid_molecule(
        left,
        right,
        atom_map={0: 0, 1: 1},
        required_pairs=[(0, 0), (1, 1)],
    )

    assert hybrid.map_a_to_b[0] == 0
    assert hybrid.map_a_to_b[1] == 1
    assert 2 not in hybrid.map_a_to_b
    assert 2 in hybrid.unique_a
    assert 2 in hybrid.unique_b


def _test_junction_proper_torsions_are_preserved_by_default():
    molecule = Chem.MolFromSmiles("CCCC")
    _, torsion_scale = _inactive_scales(
        molecule,
        unique={3},
        scales=DummyBondedScales(),
    )

    assert torsion_scale((0, 1, 2, 3)) == 1.0
    assert torsion_scale((0, 1, 2, 3)) == DummyBondedScales().proper_torsion


def _test_junction_proper_torsions_can_be_disabled_explicitly():
    molecule = Chem.MolFromSmiles("CCCC")
    _, torsion_scale = _inactive_scales(
        molecule,
        unique={3},
        scales=DummyBondedScales(junction_proper_torsion=0.0),
    )

    assert torsion_scale((0, 1, 2, 3)) == 0.0


def _test_unique_vacuum_force_exactly_replaces_internal_nonbonded_energy():
    source = mm.System()
    endpoint = mm.System()
    original = mm.NonbondedForce()
    replacement = mm.NonbondedForce()
    original.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    replacement.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    parameters = [
        (-0.20, 0.30, 0.40),
        (0.15, 0.31, 0.35),
        (0.10, 0.32, 0.30),
        (-0.05, 0.33, 0.25),
    ]
    for charge, sigma, epsilon in parameters:
        source.addParticle(12.0)
        endpoint.addParticle(12.0)
        original.addParticle(charge, sigma, epsilon)
        replacement.addParticle(charge, sigma, epsilon)
    original.addException(0, 1, 0.0, 0.3, 0.0)
    original.addException(1, 2, 0.0, 0.3, 0.0)
    original.addException(2, 3, 0.0, 0.3, 0.0)
    original.addException(0, 3, -0.005, 0.315, 0.08)
    for index in range(original.getNumExceptions()):
        replacement.addException(*original.getExceptionParameters(index))
    source.addForce(original)
    endpoint.addForce(replacement)
    _add_unique_vacuum_nonbonded(
        endpoint,
        ((source, {index: index for index in range(4)}, set(range(4))),),
        replacement,
    )
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.30, 0.1, 0.0], [0.44, 0.1, 0.1]]
    ) * unit.nanometer

    def energy(system):
        context = mm.Context(system, mm.VerletIntegrator(1.0 * unit.femtosecond))
        context.setPositions(positions)
        value = context.getState(getEnergy=True).getPotentialEnergy()
        del context
        return value.value_in_unit(unit.kilojoule_per_mole)

    assert np.isclose(energy(source), energy(endpoint), atol=1.0e-8)
