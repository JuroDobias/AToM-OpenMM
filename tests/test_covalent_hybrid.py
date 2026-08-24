import numpy as np
import openmm as mm
from openmm import unit
from openff.toolkit import ForceField, Molecule
from openff.units import unit as offunit
from rdkit import Chem
from rdkit.Chem import AllChem

from atom_openmm.covalent_hybrid import (
    DummyBondedScales,
    _add_unique_vacuum_nonbonded,
    _hybrid_topology,
    _inactive_scales,
    build_covalent_hybrid_molecule,
    complete_covalent_atom_map,
)
from atom_openmm.covalent_parameters import CovalentParameterBundle
from atom_openmm.covalent_softcore import create_softcore_hamiltonian
from atom_openmm.hybrid_mapping import build_hybrid_atom_map


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


def test_transmutation_uses_physical_endpoint_and_heavier_switching_masses():
    left = _bundle("CC(=O)NC1=CC=CC=C1")
    right = _bundle("CS(=O)(=O)NC1=CC=CC=C1")
    mapping, metadata = build_hybrid_atom_map(
        left,
        right,
        {
            "method": "paired_smarts_transmutation",
            "ligand_a_smarts": "[C:1]-[C:2](=[O:3])-[NH:4]",
            "ligand_b_smarts": "[C:1]-[S:2](=[O:3])(=[O:5])-[NH:4]",
            "inactive_bonded_labels": {"ligand_b": [5]},
        },
    )
    transmuted = {tuple(pair) for pair in metadata["transmuted_pairs_0based"]}
    inactive_b = set(metadata["inactive_bonded_atoms_b_0based"])
    hybrid = build_covalent_hybrid_molecule(
        left,
        right,
        atom_map=mapping,
        transmuted_pairs=transmuted,
        inactive_bonded_atoms_b=inactive_b,
    )
    atom_a, atom_b = next(iter(transmuted))
    center = hybrid.map_a_to_hybrid[atom_a]
    assert np.isclose(
        hybrid.endpoint_a.getParticleMass(center).value_in_unit(unit.dalton), 12.01078
    )
    assert np.isclose(
        hybrid.endpoint_b.getParticleMass(center).value_in_unit(unit.dalton), 32.0655
    )

    unique_a = [hybrid.map_a_to_hybrid[index] for index in hybrid.unique_a]
    unique_b = [hybrid.map_b_to_hybrid[index] for index in hybrid.unique_b]
    switching = create_softcore_hamiltonian(
        hybrid.endpoint_a,
        hybrid.endpoint_b,
        unique_a,
        unique_b,
        total_steps=100,
        path_mode="concerted",
    )
    assert np.isclose(
        switching.system.getParticleMass(center).value_in_unit(unit.dalton), 32.0655
    )

    oxygen = hybrid.map_b_to_hybrid[next(iter(inactive_b))]
    bonds = next(
        force for force in hybrid.endpoint_a.getForces()
        if isinstance(force, mm.HarmonicBondForce)
    )
    assert any(
        oxygen in map(int, bonds.getBondParameters(index)[:2])
        and bonds.getBondParameters(index)[3].value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer**2
        ) > 0
        for index in range(bonds.getNumBonds())
    )
    angles = next(
        force for force in hybrid.endpoint_a.getForces()
        if isinstance(force, mm.HarmonicAngleForce)
    )
    assert all(
        angles.getAngleParameters(index)[4].value_in_unit(
            unit.kilojoule_per_mole / unit.radian**2
        ) == 0
        for index in range(angles.getNumAngles())
        if oxygen in map(int, angles.getAngleParameters(index)[:3])
    )


def test_terminal_z_matrix_retains_one_angle_and_one_proper_torsion_group():
    left = _bundle("NC(=O)c1ccccc1")
    right = _bundle("NS(=O)(=O)c1ccccc1")
    mapping, metadata = build_hybrid_atom_map(
        left,
        right,
        {
            "method": "paired_smarts_transmutation",
            "ligand_a_smarts": "[c:1]-[C:2](=[O:3])-[NH2:4]",
            "ligand_b_smarts": "[c:1]-[S:2](=[O:3])(=[O:5])-[NH2:4]",
            "inactive_bonded_labels": {"ligand_b": [5]},
            "inactive_bonded_geometry": "terminal_z_matrix",
        },
    )
    inactive_b = set(metadata["inactive_bonded_atoms_b_0based"])
    hybrid = build_covalent_hybrid_molecule(
        left,
        right,
        atom_map=mapping,
        transmuted_pairs={
            tuple(pair) for pair in metadata["transmuted_pairs_0based"]
        },
        inactive_bonded_atoms_b=inactive_b,
        inactive_bonded_geometry="terminal_z_matrix",
    )

    assert hybrid.inactive_bonded_geometry == "terminal_z_matrix"
    assert len(hybrid.inactive_z_matrix_terms) == 1
    selected = hybrid.inactive_z_matrix_terms[0]
    assert selected.endpoint == "b"
    assert selected.dummy_atom == next(iter(inactive_b))
    assert selected.torsion_barrier_kj_mol > 0.0
    rdkit = right.molecule.to_rdkit()
    assert all(
        rdkit.GetAtomWithIdx(atom).GetAtomicNum() > 1
        for atom in selected.torsion_atoms[:3]
    )
    assert rdkit.GetBondBetweenAtoms(*selected.torsion_atoms[:2]).IsInRing()

    oxygen = selected.hybrid_dummy_atom
    angles = next(
        force for force in hybrid.endpoint_a.getForces()
        if isinstance(force, mm.HarmonicAngleForce)
    )
    nonzero_angles = [
        tuple(map(int, angles.getAngleParameters(index)[:3]))
        for index in range(angles.getNumAngles())
        if oxygen in map(int, angles.getAngleParameters(index)[:3])
        and angles.getAngleParameters(index)[4].value_in_unit(
            unit.kilojoule_per_mole / unit.radian**2
        ) > 0.0
    ]
    assert len(nonzero_angles) == 1
    assert set(nonzero_angles[0]) == set(selected.hybrid_angle_atoms)

    torsions = next(
        force for force in hybrid.endpoint_a.getForces()
        if isinstance(force, mm.PeriodicTorsionForce)
    )
    nonzero_torsions = [
        tuple(map(int, torsions.getTorsionParameters(index)[:4]))
        for index in range(torsions.getNumTorsions())
        if oxygen in map(int, torsions.getTorsionParameters(index)[:4])
        and torsions.getTorsionParameters(index)[6].value_in_unit(
            unit.kilojoule_per_mole
        ) > 0.0
    ]
    assert len(nonzero_torsions) == len(selected.torsion_terms)
    assert all(
        atoms == selected.hybrid_torsion_atoms
        or atoms[::-1] == selected.hybrid_torsion_atoms
        for atoms in nonzero_torsions
    )


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


def test_required_hydrogens_are_checked_after_mapping_completion():
    left = Chem.AddHs(Chem.MolFromSmiles("CO"))
    right = Chem.Mol(left)
    AllChem.EmbedMolecule(left, randomSeed=1)
    AllChem.EmbedMolecule(right, randomSeed=2)
    carbon = 0
    hydrogen = next(
        atom.GetIdx()
        for atom in left.GetAtomWithIdx(carbon).GetNeighbors()
        if atom.GetAtomicNum() == 1
    )

    mapping = complete_covalent_atom_map(
        left,
        right,
        {0: 0, 1: 1},
        required_pairs=[(0, 0), (hydrogen, hydrogen)],
    )

    assert mapping[hydrogen] == hydrogen


def _test_retained_dummy_core_adds_only_state_specific_vacuum_pairs():
    left = _bundle("CCO")
    right = _bundle("CCN")
    common = {0: 0, 1: 1}
    default = build_covalent_hybrid_molecule(left, right, atom_map=common)
    retained = build_covalent_hybrid_molecule(
        left,
        right,
        atom_map=common,
        dummy_core_nonbonded="retain",
    )

    def vacuum_pairs(system):
        force = next((
            force
            for force in system.getForces()
            if force.getName() == "CovalentUniqueVacuumNonbondedForce"
        ), None)
        if force is None:
            return set()
        return {
            tuple(sorted(map(int, force.getBondParameters(index)[:2])))
            for index in range(force.getNumBonds())
        }

    default_a = vacuum_pairs(default.endpoint_a)
    default_b = vacuum_pairs(default.endpoint_b)
    retained_a = vacuum_pairs(retained.endpoint_a)
    retained_b = vacuum_pairs(retained.endpoint_b)
    common_hybrid = {
        retained.map_a_to_hybrid[index] for index in retained.map_a_to_b
    }
    unique_a = {retained.map_a_to_hybrid[index] for index in retained.unique_a}
    unique_b = {retained.map_b_to_hybrid[index] for index in retained.unique_b}

    assert default.dummy_core_nonbonded == "off"
    assert default_a == default_b
    assert default_a < retained_a
    assert default_b < retained_b
    assert retained_a - default_a
    assert retained_b - default_b
    assert all(
        set(pair) & unique_b and set(pair) & common_hybrid
        for pair in retained_a - default_a
    )
    assert all(
        set(pair) & unique_a and set(pair) & common_hybrid
        for pair in retained_b - default_b
    )
    assert not any(
        set(pair) & unique_a and set(pair) & unique_b
        for pair in retained_a | retained_b
    )


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


def _test_selective_rotatable_torsion_scales_classify_central_bonds():
    scales = DummyBondedScales(
        proper_torsion=0.8,
        junction_proper_torsion=0.7,
        junction_rotatable_torsion=0.1,
        internal_rotatable_torsion=0.2,
    )
    _, junction_scale = _inactive_scales(
        Chem.MolFromSmiles("CCCC"), unique={2, 3}, scales=scales
    )
    _, internal_scale = _inactive_scales(
        Chem.MolFromSmiles("CCCCC"), unique={2, 3, 4}, scales=scales
    )
    _, ring_scale = _inactive_scales(
        Chem.MolFromSmiles("C1CCCCC1"), unique=set(range(6)), scales=scales
    )
    _, double_bond_scale = _inactive_scales(
        Chem.MolFromSmiles("CC=CC"), unique={2, 3}, scales=scales
    )

    assert junction_scale((0, 1, 2, 3)) == 0.1
    assert internal_scale((1, 2, 3, 4)) == 0.2
    assert ring_scale((0, 1, 2, 3)) == 0.8
    assert double_bond_scale((0, 1, 2, 3)) == 0.7


def _test_selective_rotatable_torsion_scales_inherit_legacy_values():
    molecule = Chem.MolFromSmiles("CCCCC")
    scales = DummyBondedScales(
        proper_torsion=0.4,
        junction_proper_torsion=0.3,
    )
    _, junction_scale = _inactive_scales(
        molecule, unique={3, 4}, scales=scales
    )
    _, internal_scale = _inactive_scales(
        molecule, unique={2, 3, 4}, scales=scales
    )

    assert junction_scale((1, 2, 3, 4)) == 0.3
    assert internal_scale((1, 2, 3, 4)) == 0.4


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
