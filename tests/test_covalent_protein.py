from atom_openmm.covalent_protein import _ordered_system_indices


def test_ordered_system_indices_uses_only_present_ligand_atoms():
    assert _ordered_system_indices({23: 100, 25: 102, 24: 101}) == [100, 101, 102]
