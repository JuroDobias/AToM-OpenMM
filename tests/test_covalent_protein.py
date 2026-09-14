from types import SimpleNamespace

import openmm as mm
from openmm import unit
from rdkit import Chem

from atom_openmm.covalent_hybrid import (
    DummyBondedScales,
    _inactive_bonded_scalers,
)
from atom_openmm.covalent_protein import _graft_endpoint, _ordered_system_indices


def test_ordered_system_indices_uses_only_present_ligand_atoms():
    assert _ordered_system_indices({23: 100, 25: 102, 24: 101}) == [100, 101, 102]


def _endpoint_source(atom_count, bonds, *, constrained=()):
    system = mm.System()
    for _ in range(atom_count):
        system.addParticle(12.0 * unit.dalton)
    bond_force = mm.HarmonicBondForce()
    for atom1, atom2 in bonds:
        bond_force.addBond(
            atom1,
            atom2,
            0.14 * unit.nanometer,
            1000.0 * unit.kilojoule_per_mole / unit.nanometer**2,
        )
    system.addForce(bond_force)
    angle_force = mm.HarmonicAngleForce()
    for atom in range(atom_count - 2):
        angle_force.addAngle(
            atom,
            atom + 1,
            atom + 2,
            2.0 * unit.radian,
            100.0 * unit.kilojoule_per_mole / unit.radian**2,
        )
    system.addForce(angle_force)
    torsion_force = mm.PeriodicTorsionForce()
    for atom in range(atom_count - 3):
        torsion_force.addTorsion(
            atom,
            atom + 1,
            atom + 2,
            atom + 3,
            1,
            0.0 * unit.radian,
            5.0 * unit.kilojoule_per_mole,
        )
    system.addForce(torsion_force)
    nonbonded = mm.NonbondedForce()
    for _ in range(atom_count):
        nonbonded.addParticle(0.0, 0.3, 0.0)
    system.addForce(nonbonded)
    for atom1, atom2 in constrained:
        system.addConstraint(atom1, atom2, 0.14 * unit.nanometer)
    return system


def _has_bond(system, pair):
    expected = set(pair)
    for force in system.getForces():
        if not isinstance(force, mm.HarmonicBondForce):
            continue
        for index in range(force.getNumBonds()):
            atom1, atom2, *_ = force.getBondParameters(index)
            if {int(atom1), int(atom2)} == expected:
                return True
    return False


def _has_constraint(system, pair):
    expected = set(pair)
    return any(
        {
            int(system.getConstraintParameters(index)[0]),
            int(system.getConstraintParameters(index)[1]),
        }
        == expected
        for index in range(system.getNumConstraints())
    )


def _has_bonded_term_containing_pair(system, pair):
    expected = set(pair)
    for force, count, getter, width in (
        (mm.HarmonicAngleForce, "getNumAngles", "getAngleParameters", 3),
        (mm.PeriodicTorsionForce, "getNumTorsions", "getTorsionParameters", 4),
    ):
        for candidate in system.getForces():
            if not isinstance(candidate, force):
                continue
            for index in range(getattr(candidate, count)()):
                atoms = [int(value) for value in getattr(candidate, getter)(index)[:width]]
                if any(set(atoms[offset : offset + 2]) == expected for offset in range(width - 1)):
                    return True
    return False


def _has_angle(system, atoms):
    expected = tuple(atoms)
    for force in system.getForces():
        if not isinstance(force, mm.HarmonicAngleForce):
            continue
        for index in range(force.getNumAngles()):
            observed = tuple(int(value) for value in force.getAngleParameters(index)[:3])
            if observed == expected or observed == expected[::-1]:
                return True
    return False


def _has_torsion(system, atoms):
    expected = tuple(atoms)
    for force in system.getForces():
        if not isinstance(force, mm.PeriodicTorsionForce):
            continue
        for index in range(force.getNumTorsions()):
            observed = tuple(int(value) for value in force.getTorsionParameters(index)[:4])
            if observed == expected or observed == expected[::-1]:
                return True
    return False


def test_inactive_protein_endpoint_excludes_alchemical_bond_and_constraint():
    base = _endpoint_source(2, [(0, 1)])
    molecule_a = Chem.MolFromSmiles("CCC")
    molecule_b = Chem.MolFromSmiles("CCCC")
    parameters_a = SimpleNamespace(
        system=_endpoint_source(3, [(0, 1), (1, 2)]),
        molecule=SimpleNamespace(to_rdkit=lambda: molecule_a),
    )
    parameters_b = SimpleNamespace(
        system=_endpoint_source(
            4,
            [(0, 1), (1, 2), (2, 3)],
            constrained=[(2, 3)],
        ),
        molecule=SimpleNamespace(to_rdkit=lambda: molecule_b),
    )
    common = dict(
        base_system=base,
        parameters_a=parameters_a,
        parameters_b=parameters_b,
        hybrid=SimpleNamespace(
            dummy_core_nonbonded="off",
            inactive_bonded_atoms_a=(),
            inactive_bonded_atoms_b=(),
            inactive_z_matrix_terms=(),
        ),
        source_to_global_a={0: 2, 1: 3, 2: 4},
        source_to_global_b={0: 2, 1: 3, 2: 4, 3: 5},
        receptor_atoms={"SG": 0, "HG": 1},
        unique_a=set(),
        unique_b={3},
        active_atoms_a={0, 1, 2},
        active_atoms_b={0, 1, 2, 3},
        dummy_bonded_scales=DummyBondedScales(),
        alchemical_bonds_a=set(),
        alchemical_bonds_b={(2, 3)},
    )

    endpoint_a = _graft_endpoint(state="a", **common)
    endpoint_b = _graft_endpoint(state="b", **common)

    assert not _has_bond(endpoint_a, (4, 5))
    assert not _has_constraint(endpoint_a, (4, 5))
    assert not _has_bonded_term_containing_pair(endpoint_a, (4, 5))
    assert _has_bond(endpoint_b, (4, 5))
    assert _has_constraint(endpoint_b, (4, 5))
    assert _has_bonded_term_containing_pair(endpoint_b, (4, 5))


def test_protein_endpoint_uses_resolved_terminal_z_matrix_policy():
    base = _endpoint_source(2, [(0, 1)])
    molecule_a = Chem.MolFromSmiles("CCCC")
    molecule_b = Chem.MolFromSmiles("CCCCC")
    system_b = _endpoint_source(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    angle_force = next(
        force for force in system_b.getForces()
        if isinstance(force, mm.HarmonicAngleForce)
    )
    angle_force.addAngle(
        1, 3, 4, 2.0 * unit.radian,
        100.0 * unit.kilojoule_per_mole / unit.radian**2,
    )
    torsion_force = next(
        force for force in system_b.getForces()
        if isinstance(force, mm.PeriodicTorsionForce)
    )
    torsion_force.addTorsion(
        0, 2, 3, 4, 1, 0.0 * unit.radian,
        5.0 * unit.kilojoule_per_mole,
    )
    parameters_a = SimpleNamespace(
        system=_endpoint_source(4, [(0, 1), (1, 2), (2, 3)]),
        molecule=SimpleNamespace(to_rdkit=lambda: molecule_a),
    )
    parameters_b = SimpleNamespace(
        system=system_b,
        molecule=SimpleNamespace(to_rdkit=lambda: molecule_b),
    )
    z_matrix = SimpleNamespace(
        endpoint="b",
        dummy_atom=4,
        angle_atoms=(2, 3, 4),
        torsion_atoms=(1, 2, 3, 4),
    )
    endpoint_a = _graft_endpoint(
        base_system=base,
        parameters_a=parameters_a,
        parameters_b=parameters_b,
        hybrid=SimpleNamespace(
            dummy_core_nonbonded="off",
            inactive_bonded_atoms_a=(),
            inactive_bonded_atoms_b=(4,),
            inactive_z_matrix_terms=(z_matrix,),
        ),
        source_to_global_a={0: 2, 1: 3, 2: 4, 3: 5},
        source_to_global_b={0: 2, 1: 3, 2: 4, 3: 5, 4: 6},
        receptor_atoms={"SG": 0, "HG": 1},
        unique_a=set(),
        unique_b={4},
        active_atoms_a={0, 1, 2, 3},
        active_atoms_b={0, 1, 2, 3, 4},
        state="a",
        dummy_bonded_scales=DummyBondedScales(),
        alchemical_bonds_a=set(),
        alchemical_bonds_b=set(),
    )

    assert _has_bond(endpoint_a, (5, 6))
    assert _has_angle(endpoint_a, (4, 5, 6))
    assert _has_torsion(endpoint_a, (3, 4, 5, 6))
    assert not _has_angle(endpoint_a, (3, 5, 6))
    assert not _has_torsion(endpoint_a, (2, 4, 5, 6))


def test_shared_inactive_policy_preserves_unselected_junction_scaling():
    molecule = Chem.MolFromSmiles("CCCC")
    scales = DummyBondedScales(
        junction_angle=0.25,
        junction_proper_torsion=0.5,
    )
    _, angle_scale, torsion_scale = _inactive_bonded_scalers(
        molecule,
        {3},
        set(),
        {},
        scales,
    )

    assert angle_scale((1, 2, 3)) == 0.25
    assert torsion_scale((0, 1, 2, 3)) == 0.5
