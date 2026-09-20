from types import SimpleNamespace

import pytest
from openff.toolkit import Molecule
from rdkit import Chem

from atom_openmm.hybrid_mapping import (
    HybridMappingError,
    _automatic_junction_bonds,
    _strict_explicit_pairs,
    build_hybrid_atom_map,
)


def _test_automatic_junction_uses_z_matrix_for_mapped_heavy_chain():
    molecule = Chem.MolFromSmiles("CCCC")

    selected, roots, resolved, warnings = _automatic_junction_bonds(
        molecule, {0, 1, 2}, "ligand_b"
    )

    assert selected == {3}
    assert roots == {3}
    assert resolved[0]["boundary_atoms_0based"] == [2, 3]
    assert resolved[0]["inactive_geometry"] == "terminal_z_matrix"
    assert resolved[0]["selection_source"] == "automatic"
    assert warnings == []


def _test_automatic_junction_falls_back_without_heavy_reference_chain():
    molecule = Chem.MolFromSmiles("CC")

    selected, roots, resolved, warnings = _automatic_junction_bonds(
        molecule, {0}, "ligand_a"
    )

    assert selected == {1}
    assert roots == set()
    assert resolved[0]["inactive_geometry"] == "bond_only"
    assert resolved[0]["fallback_reason"] == "no_mapped_heavy_reference_chain"
    assert warnings[0]["reason"] == "no_mapped_heavy_reference_chain"


def _test_automatic_junction_opens_declared_annulation_bond():
    molecule = Chem.MolFromSmiles("c1ccc2c(c1)CCC2")

    selected, roots, resolved, warnings = _automatic_junction_bonds(
        molecule,
        set(range(6)),
        "ligand_b",
        alchemical_bonds={(3, 8)},
    )

    assert selected == {6, 7, 8}
    assert roots == {6}
    assert resolved[0]["boundary_atoms_0based"] == [4, 6]
    assert warnings == []


def _parameters(smiles):
    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=1)
    return SimpleNamespace(molecule=molecule)


def _test_mapping_restores_openff_kekule_purine_aromaticity():
    ligand_a = _parameters(
        "NC(=O)c1ccc(Nc2nc(OCC3CCCCC3)c3nc[nH]c3n2)cc1"
    )
    ligand_b = _parameters(
        "COc1cc(Nc2nc(OCC3CCCCC3)c3nc[nH]c3n2)ccc1S(N)(=O)=O"
    )

    _, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {
            "method": "mcs_core_smarts",
            "smarts": "c1cc(Nc2nc3c(ncn3)c(OCC3CCCCC3)n2)ccc1",
        },
    )

    assert metadata["aromaticity_model"] == "rdkit"
    assert metadata["aromatic_atom_count_before"] == {
        "ligand_a": 12,
        "ligand_b": 12,
    }
    assert metadata["aromatic_atom_count_after"] == {
        "ligand_a": 15,
        "ligand_b": 15,
    }
    assert metadata["mapped_heavy_atom_count"] == 24


def test_paired_smarts_maps_amide_to_sulfonamide_transmutation():
    ligand_a = _parameters("CC(=O)NC1=CC=CC=C1")
    ligand_b = _parameters("CS(=O)(=O)NC1=CC=CC=C1")

    mapping, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {
            "method": "paired_smarts_transmutation",
            "ligand_a_smarts": "[C:1]-[C:2](=[O:3])-[NH:4]",
            "ligand_b_smarts": "[C:1]-[S:2](=[O:3])(=[O:5])-[NH:4]",
            "inactive_bonded_labels": {"ligand_b": [5]},
        },
    )

    assert metadata["method"] == "paired_smarts_transmutation"
    assert len(metadata["transmuted_pairs_0based"]) == 1
    atom_a, atom_b = metadata["transmuted_pairs_0based"][0]
    assert ligand_a.molecule.atoms[atom_a].atomic_number == 6
    assert ligand_b.molecule.atoms[atom_b].atomic_number == 16
    assert mapping[atom_a] == atom_b
    assert len(metadata["inactive_bonded_atoms_b_0based"]) == 1


def _test_explicit_pairs_preserve_manual_element_transmutation():
    ligand_a = _parameters("CC(=O)NC1=CC=CC=C1")
    ligand_b = _parameters("CS(=O)(=O)NC1=CC=CC=C1")
    pairs = [[0, 0], [1, 1], [2, 2], [3, 4]]

    mapping, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {"method": "explicit_pairs", "pairs_0based": pairs},
    )

    assert metadata["method"] == "explicit_pairs"
    assert mapping[1] == 1
    assert metadata["transmuted_pairs_0based"] == [[1, 1]]
    assert metadata["requested_pairs_0based"] == pairs
    assert metadata["mapped_heavy_atom_count"] == len(pairs)


def _test_explicit_pairs_preserve_hydrogen_to_fluorine_transmutation():
    ligand_a = _parameters("C")
    ligand_b = _parameters("CF")
    carbon_a = next(
        atom.molecule_atom_index
        for atom in ligand_a.molecule.atoms
        if atom.atomic_number == 6
    )
    carbon_b = next(
        atom.molecule_atom_index
        for atom in ligand_b.molecule.atoms
        if atom.atomic_number == 6
    )
    hydrogen_a = next(
        atom.molecule_atom_index
        for atom in ligand_a.molecule.atoms
        if atom.atomic_number == 1
    )
    fluorine_b = next(
        atom.molecule_atom_index
        for atom in ligand_b.molecule.atoms
        if atom.atomic_number == 9
    )
    requested = [[carbon_a, carbon_b], [hydrogen_a, fluorine_b]]

    mapping, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {"method": "explicit_pairs", "pairs_0based": requested},
    )

    assert mapping[hydrogen_a] == fluorine_b
    assert metadata["requested_pairs_0based"] == requested
    assert [hydrogen_a, fluorine_b] in metadata["transmuted_pairs_0based"]
    assert len(metadata["auto_completed_hydrogen_pairs_0based"]) == 3
    assert all(
        ligand_a.molecule.atoms[atom_a].atomic_number == 1
        and ligand_b.molecule.atoms[atom_b].atomic_number == 1
        for atom_a, atom_b in metadata["auto_completed_hydrogen_pairs_0based"]
    )


def _test_explicit_pairs_can_force_stereocenter_hydrogens_unique():
    ligand_a = _parameters("CC(O)C")
    ligand_b = _parameters("CC(O)C")
    molecule_a = ligand_a.molecule.to_rdkit()
    molecule_b = ligand_b.molecule.to_rdkit()
    heavy_pairs = [
        [index, index]
        for index, atom in enumerate(molecule_a.GetAtoms())
        if atom.GetAtomicNum() != 1
    ]
    center = 1
    hydrogen_a = next(
        atom.GetIdx() for atom in molecule_a.GetAtomWithIdx(center).GetNeighbors()
        if atom.GetAtomicNum() == 1
    )
    hydrogen_b = next(
        atom.GetIdx() for atom in molecule_b.GetAtomWithIdx(center).GetNeighbors()
        if atom.GetAtomicNum() == 1
    )

    mapping, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {
            "method": "explicit_pairs",
            "pairs_0based": heavy_pairs,
            "force_unique_atoms_a_0based": [hydrogen_a],
            "force_unique_atoms_b_0based": [hydrogen_b],
        },
    )

    assert hydrogen_a not in mapping
    assert hydrogen_b not in mapping.values()
    assert metadata["force_unique_atoms_a_0based"] == [hydrogen_a]
    assert metadata["force_unique_atoms_b_0based"] == [hydrogen_b]
    assert hydrogen_a in metadata["inactive_bonded_atoms_a_0based"]
    assert hydrogen_b in metadata["inactive_bonded_atoms_b_0based"]
    assert all(
        hydrogen_a != atom_a and hydrogen_b != atom_b
        for atom_a, atom_b in metadata["auto_completed_hydrogen_pairs_0based"]
    )


def _stereocenter_and_hydrogen(parameters):
    molecule = parameters.molecule.to_rdkit()
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    center = next(
        atom
        for atom in molecule.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )
    hydrogen = next(
        atom.GetIdx() for atom in center.GetNeighbors() if atom.GetAtomicNum() == 1
    )
    return molecule, center.GetIdx(), hydrogen


def test_explicit_pairs_automatically_force_inverted_stereo_hydrogens_unique():
    ligand_a = _parameters("C[C@H](O)F")
    ligand_b = _parameters("C[C@@H](O)F")
    molecule_a, center_a, hydrogen_a = _stereocenter_and_hydrogen(ligand_a)
    molecule_b, center_b, hydrogen_b = _stereocenter_and_hydrogen(ligand_b)
    heavy_pairs = [
        [atom_a.GetIdx(), atom_b.GetIdx()]
        for atom_a, atom_b in zip(molecule_a.GetAtoms(), molecule_b.GetAtoms())
        if atom_a.GetAtomicNum() != 1 and atom_b.GetAtomicNum() != 1
    ]

    mapping, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {"method": "explicit_pairs", "pairs_0based": heavy_pairs},
    )

    assert hydrogen_a not in mapping
    assert hydrogen_b not in mapping.values()
    assert metadata["automatically_forced_unique_stereo_hydrogens"] == {
        "ligand_a": [hydrogen_a],
        "ligand_b": [hydrogen_b],
    }
    assert metadata["force_unique_atoms_a_0based"] == [hydrogen_a]
    assert metadata["force_unique_atoms_b_0based"] == [hydrogen_b]
    assert metadata["inverted_stereocenters_0based"] == [{
        "ligand_a_center_0based": center_a,
        "ligand_b_center_0based": center_b,
        "ligand_a_hydrogen_0based": hydrogen_a,
        "ligand_b_hydrogen_0based": hydrogen_b,
        "comparison": "mapped_local_tetrahedral_parity",
        "hydrogen_action": "automatically_forced_unique",
    }]
    assert hydrogen_a in metadata["inactive_bonded_atoms_a_0based"]
    assert hydrogen_b in metadata["inactive_bonded_atoms_b_0based"]
    assert hydrogen_a in metadata["inactive_z_matrix_root_atoms_a_0based"]
    assert hydrogen_b in metadata["inactive_z_matrix_root_atoms_b_0based"]


def test_inverted_stereo_detection_keeps_unmatched_neighbor_branch_unique():
    ligand_a = _parameters("C[C@H](O)F")
    ligand_b = _parameters("C[C@@H](O)F")
    molecule_a, _center_a, hydrogen_a = _stereocenter_and_hydrogen(ligand_a)
    molecule_b, _center_b, hydrogen_b = _stereocenter_and_hydrogen(ligand_b)
    oxygen_a = next(
        atom.GetIdx() for atom in molecule_a.GetAtoms() if atom.GetAtomicNum() == 8
    )
    oxygen_b = next(
        atom.GetIdx() for atom in molecule_b.GetAtoms() if atom.GetAtomicNum() == 8
    )
    heavy_pairs = [
        [atom_a.GetIdx(), atom_b.GetIdx()]
        for atom_a, atom_b in zip(molecule_a.GetAtoms(), molecule_b.GetAtoms())
        if atom_a.GetAtomicNum() != 1
        and atom_b.GetAtomicNum() != 1
        and atom_a.GetIdx() != oxygen_a
        and atom_b.GetIdx() != oxygen_b
    ]

    mapping, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {"method": "explicit_pairs", "pairs_0based": heavy_pairs},
    )

    assert oxygen_a not in mapping
    assert oxygen_b not in mapping.values()
    assert hydrogen_a not in mapping
    assert hydrogen_b not in mapping.values()
    assert metadata["automatically_forced_unique_stereo_hydrogens"] == {
        "ligand_a": [hydrogen_a],
        "ligand_b": [hydrogen_b],
    }


def test_explicit_pairs_complete_retained_stereo_hydrogen():
    ligand_a = _parameters("C[C@H](O)F")
    ligand_b = _parameters("C[C@H](O)F")
    molecule_a, _center_a, hydrogen_a = _stereocenter_and_hydrogen(ligand_a)
    molecule_b, _center_b, hydrogen_b = _stereocenter_and_hydrogen(ligand_b)
    heavy_pairs = [
        [atom_a.GetIdx(), atom_b.GetIdx()]
        for atom_a, atom_b in zip(molecule_a.GetAtoms(), molecule_b.GetAtoms())
        if atom_a.GetAtomicNum() != 1 and atom_b.GetAtomicNum() != 1
    ]

    mapping, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {"method": "explicit_pairs", "pairs_0based": heavy_pairs},
    )

    assert mapping[hydrogen_a] == hydrogen_b
    assert metadata["automatically_forced_unique_stereo_hydrogens"] == {
        "ligand_a": [],
        "ligand_b": [],
    }
    assert metadata["inverted_stereocenters_0based"] == []


def test_mapped_local_stereo_is_independent_of_endpoint_atom_order():
    ligand_a = _parameters("C[C@H](O)F")
    original_b = _parameters("C[C@H](O)F")
    molecule_b = original_b.molecule.to_rdkit()
    order = list(reversed(range(molecule_b.GetNumAtoms())))
    inverse_order = {old: new for new, old in enumerate(order)}
    reordered_b = Chem.RenumberAtoms(molecule_b, order)
    ligand_b = SimpleNamespace(
        molecule=Molecule.from_rdkit(
            reordered_b,
            allow_undefined_stereo=False,
            hydrogens_are_explicit=True,
        )
    )
    molecule_a, _center_a, hydrogen_a = _stereocenter_and_hydrogen(ligand_a)
    molecule_b, _center_b, hydrogen_b = _stereocenter_and_hydrogen(ligand_b)
    heavy_pairs = [
        [atom.GetIdx(), inverse_order[atom.GetIdx()]]
        for atom in molecule_a.GetAtoms()
        if atom.GetAtomicNum() != 1
    ]

    mapping, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {"method": "explicit_pairs", "pairs_0based": heavy_pairs},
    )

    assert mapping[hydrogen_a] == hydrogen_b
    assert metadata["inverted_stereocenters_0based"] == []


def test_explicit_pairs_ignore_cip_relabeling_without_local_inversion():
    ligand_a = _parameters("C[C@H](F)Cl")
    ligand_b = _parameters("C[C@H](F)N")
    molecule_a, center_a, hydrogen_a = _stereocenter_and_hydrogen(ligand_a)
    molecule_b, center_b, hydrogen_b = _stereocenter_and_hydrogen(ligand_b)
    assert molecule_a.GetAtomWithIdx(center_a).GetProp("_CIPCode") == "R"
    assert molecule_b.GetAtomWithIdx(center_b).GetProp("_CIPCode") == "S"
    heavy_pairs = [
        [atom_a.GetIdx(), atom_b.GetIdx()]
        for atom_a, atom_b in zip(molecule_a.GetAtoms(), molecule_b.GetAtoms())
        if atom_a.GetAtomicNum() != 1 and atom_b.GetAtomicNum() != 1
    ]

    mapping, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {"method": "explicit_pairs", "pairs_0based": heavy_pairs},
    )

    assert mapping[hydrogen_a] == hydrogen_b
    assert metadata["inverted_stereocenters_0based"] == []


def test_explicit_inverted_stereo_hydrogen_pair_remains_authoritative():
    ligand_a = _parameters("C[C@H](O)F")
    ligand_b = _parameters("C[C@@H](O)F")
    molecule_a, center_a, hydrogen_a = _stereocenter_and_hydrogen(ligand_a)
    molecule_b, center_b, hydrogen_b = _stereocenter_and_hydrogen(ligand_b)
    pairs = [
        [atom_a.GetIdx(), atom_b.GetIdx()]
        for atom_a, atom_b in zip(molecule_a.GetAtoms(), molecule_b.GetAtoms())
        if atom_a.GetAtomicNum() != 1 and atom_b.GetAtomicNum() != 1
    ] + [[hydrogen_a, hydrogen_b]]

    mapping, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {"method": "explicit_pairs", "pairs_0based": pairs},
    )

    assert mapping[hydrogen_a] == hydrogen_b
    assert metadata["automatically_forced_unique_stereo_hydrogens"] == {
        "ligand_a": [],
        "ligand_b": [],
    }
    assert metadata["inverted_stereocenters_0based"] == [{
        "ligand_a_center_0based": center_a,
        "ligand_b_center_0based": center_b,
        "ligand_a_hydrogen_0based": hydrogen_a,
        "ligand_b_hydrogen_0based": hydrogen_b,
        "comparison": "mapped_local_tetrahedral_parity",
        "hydrogen_action": "explicit_mapping_preserved",
    }]


@pytest.mark.parametrize(
    ("extra", "message"),
    (
        ({"force_unique_atoms_a_0based": [0]}, "must be hydrogens"),
        ({"force_unique_atoms_b_0based": [0]}, "must be hydrogens"),
        ({"force_unique_atoms_a_0based": [999]}, "outside the molecule"),
    ),
)
def _test_explicit_pairs_reject_invalid_force_unique_atoms(extra, message):
    ligand_a = _parameters("CC")
    ligand_b = _parameters("CC")
    settings = {
        "method": "explicit_pairs",
        "pairs_0based": [[0, 0], [1, 1]],
        **extra,
    }
    with pytest.raises(HybridMappingError, match=message):
        build_hybrid_atom_map(ligand_a, ligand_b, settings)


def _test_explicit_pairs_reject_mapped_force_unique_hydrogen():
    ligand_a = _parameters("C")
    ligand_b = _parameters("C")
    hydrogen = next(
        atom.molecule_atom_index for atom in ligand_a.molecule.atoms
        if atom.atomic_number == 1
    )
    with pytest.raises(HybridMappingError, match="explicitly mapped"):
        build_hybrid_atom_map(
            ligand_a,
            ligand_b,
            {
                "method": "explicit_pairs",
                "pairs_0based": [[0, 0], [hydrogen, hydrogen]],
                "force_unique_atoms_a_0based": [hydrogen],
            },
        )


def _test_force_unique_atoms_require_explicit_pairs():
    ligand_a = _parameters("CC")
    ligand_b = _parameters("CC")
    with pytest.raises(HybridMappingError, match="requires mapping.method explicit_pairs"):
        build_hybrid_atom_map(
            ligand_a,
            ligand_b,
            {"method": "mcs", "force_unique_atoms_a_0based": [2]},
        )


def _test_explicit_pairs_support_explicit_inactive_atoms():
    ligand_a = _parameters("CC(=O)NC1=CC=CC=C1")
    ligand_b = _parameters("CS(=O)(=O)NC1=CC=CC=C1")

    _, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {
            "method": "explicit_pairs",
            "pairs_0based": [[0, 0], [1, 1], [2, 2], [3, 4]],
            "inactive_bonded_atoms_b_0based": [3],
            "inactive_bonded_geometry": "terminal_z_matrix",
        },
    )

    assert 3 in metadata["inactive_bonded_atoms_b_0based"]
    automatic = [
        entry
        for entry in metadata["resolved_junction_bonds"]
        if entry["selection_source"] == "automatic"
    ]
    assert automatic
    assert metadata["inactive_bonded_geometry"] == "terminal_z_matrix"


def _test_explicit_pairs_automatically_activate_z_matrix_geometry():
    ligand_a = _parameters("CCC")
    ligand_b = _parameters("CCCC")

    _, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {
            "method": "explicit_pairs",
            "pairs_0based": [[0, 0], [1, 1], [2, 2]],
        },
    )

    assert 3 in metadata["inactive_bonded_atoms_b_0based"]
    assert 3 in metadata["inactive_z_matrix_root_atoms_b_0based"]
    assert metadata["inactive_bonded_geometry"] == "terminal_z_matrix"
    assert metadata["resolved_junction_bonds"][0]["selection_source"] == "automatic"


@pytest.mark.parametrize(
    "pairs",
    (
        [[True, 0]],
        [[1.2, 0]],
        [[-1, 0]],
        [[0, 0], [0, 1]],
    ),
)
def _test_explicit_pairs_reject_invalid_indices(pairs):
    with pytest.raises(HybridMappingError):
        _strict_explicit_pairs(pairs)


def _test_explicit_pairs_require_connected_heavy_core():
    ligand_a = _parameters("CCC")
    ligand_b = _parameters("CCC")

    with pytest.raises(HybridMappingError, match="connected in both ligands"):
        build_hybrid_atom_map(
            ligand_a,
            ligand_b,
            {"method": "explicit_pairs", "pairs_0based": [[0, 0], [2, 2]]},
        )


def _test_explicit_pairs_reject_mapped_inactive_atom():
    ligand_a = _parameters("CC")
    ligand_b = _parameters("CC")

    with pytest.raises(HybridMappingError, match="endpoint-unique"):
        build_hybrid_atom_map(
            ligand_a,
            ligand_b,
            {
                "method": "explicit_pairs",
                "pairs_0based": [[0, 0], [1, 1]],
                "inactive_bonded_atoms_a_0based": [1],
            },
        )


def _test_explicit_pairs_terminal_z_matrix_requires_selected_atom():
    ligand_a = _parameters("CC")
    ligand_b = _parameters("CC")

    with pytest.raises(HybridMappingError, match="at least one inactive"):
        build_hybrid_atom_map(
            ligand_a,
            ligand_b,
            {
                "method": "explicit_pairs",
                "pairs_0based": [[0, 0], [1, 1]],
                "inactive_bonded_geometry": "terminal_z_matrix",
            },
        )


def test_explicit_junction_bond_derives_branch_and_mode():
    ligand_a = _parameters("CC(=O)NC1=CC=CC=C1")
    ligand_b = _parameters("CS(=O)(=O)NC1=CC=CC=C1")
    _, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {
            "method": "explicit_pairs",
            "pairs_0based": [[0, 0], [1, 1], [2, 2], [3, 4]],
            "junction_bonds": {
                "ligand_b": [
                    {
                        "atoms_0based": [1, 3],
                        "inactive_geometry": "terminal_z_matrix",
                    }
                ]
            },
        },
    )
    assert metadata["inactive_bonded_atoms_b_0based"] == [3]
    assert metadata["inactive_z_matrix_root_atoms_b_0based"] == [3]
    assert metadata["resolved_junction_bonds"][0]["boundary_atoms_0based"] == [1, 3]


def test_explicit_full_junction_records_retained_root():
    ligand_a = _parameters("CC")
    ligand_b = _parameters("CCC")
    _, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {
            "method": "explicit_pairs",
            "pairs_0based": [[0, 0], [1, 1]],
            "junction_bonds": {
                "ligand_b": [
                    {
                        "atoms_0based": [1, 2],
                        "inactive_geometry": "full_junction",
                    }
                ]
            },
        },
    )

    assert metadata["inactive_full_junction_root_atoms_a_0based"] == []
    assert metadata["inactive_full_junction_root_atoms_b_0based"] == [2]
    assert metadata["inactive_z_matrix_root_atoms_b_0based"] == []
    assert metadata["inactive_bonded_geometry"] == "full_junction"


def test_paired_smarts_junction_uses_resolved_mapping_labels():
    ligand_a = _parameters("CC(=O)NC1=CC=CC=C1")
    ligand_b = _parameters("CS(=O)(=O)NC1=CC=CC=C1")
    _, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {
            "method": "paired_smarts_transmutation",
            "ligand_a_smarts": "[C:1]-[C:2](=[O:3])-[NH:4]",
            "ligand_b_smarts": "[C:1]-[S:2](=[O:3])(=[O:5])-[NH:4]",
            "junction_bonds": {
                "ligand_b": [
                    {
                        "mapping_labels": [2, 5],
                        "inactive_geometry": "terminal_z_matrix",
                    }
                ]
            },
        },
    )
    resolved = metadata["resolved_junction_bonds"][0]
    assert resolved["boundary_atoms_0based"] == [1, 3]
    assert resolved["inactive_geometry"] == "terminal_z_matrix"


def test_junction_bond_rejects_ambiguous_smarts():
    ligand_a = _parameters("CC(=O)NC1=CC=CC=C1")
    ligand_b = _parameters("CS(=O)(=O)NC1=CC=CC=C1")
    with pytest.raises(HybridMappingError, match="exactly one ordered bond"):
        build_hybrid_atom_map(
            ligand_a,
            ligand_b,
            {
                "method": "explicit_pairs",
                "pairs_0based": [[0, 0], [1, 1], [2, 2], [3, 4]],
                "junction_bonds": {
                    "ligand_b": [
                        {
                            "smarts": "[S:1](=[O:2])(=[O:3])",
                            "bond_labels": [1, 2],
                        }
                    ]
                },
            },
        )


def test_explicit_soft_bond_allows_ring_closure():
    ligand_a = _parameters("CCCCCC")
    ligand_b = _parameters("C1CCCCC1")
    _, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {
            "method": "explicit_pairs",
            "pairs_0based": [[index, index] for index in range(6)],
            "alchemical_bonds": {
                "ligand_b": [{"atoms_0based": [0, 5], "mode": "soft_bond"}]
            },
        },
    )
    assert metadata["alchemical_bonds"]["ligand_b"][0]["atoms_0based"] == [0, 5]


def test_explicit_soft_bonds_allow_one_opening_and_one_closure():
    ligand_a = _parameters("C1CCCCC1")
    ligand_b = _parameters("CC1CCCC1")

    _, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {
            "method": "explicit_pairs",
            "pairs_0based": [[index, index] for index in range(6)],
            "alchemical_bonds": {
                "ligand_a": [{"atoms_0based": [0, 5], "mode": "soft_bond"}],
                "ligand_b": [{"atoms_0based": [1, 5], "mode": "soft_bond"}],
            },
        },
    )

    assert metadata["alchemical_bonds"] == {
        "ligand_a": [{"atoms_0based": [0, 5], "mode": "soft_bond"}],
        "ligand_b": [{"atoms_0based": [1, 5], "mode": "soft_bond"}],
    }


def test_explicit_soft_bonds_still_reject_two_in_one_endpoint():
    ligand_a = _parameters("C1CCCCC1")
    ligand_b = _parameters("CCCCCC")

    with pytest.raises(
        HybridMappingError,
        match="allows one changing bond in ligand_a",
    ):
        build_hybrid_atom_map(
            ligand_a,
            ligand_b,
            {
                "method": "explicit_pairs",
                "pairs_0based": [[index, index] for index in range(6)],
                "alchemical_bonds": {
                    "ligand_a": [
                        {"atoms_0based": [0, 5], "mode": "soft_bond"},
                        {"atoms_0based": [1, 2], "mode": "soft_bond"},
                    ]
                },
            },
        )


def test_explicit_soft_bond_allows_mapped_to_unique_annulation_closure():
    ligand_a = _parameters("c1ccccc1")
    ligand_b = _parameters("c1ccc2c(c1)CCC2")

    mapping, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {
            "method": "explicit_pairs",
            "pairs_0based": [[index, index] for index in range(6)],
            "alchemical_bonds": {
                "ligand_b": [{"atoms_0based": [3, 8], "mode": "soft_bond"}]
            },
        },
    )

    assert 3 in mapping
    assert 8 not in mapping.values()
    assert metadata["alchemical_bonds"]["ligand_b"][0]["atoms_0based"] == [3, 8]


def _test_explicit_soft_bond_allows_unique_to_unique_annulation_closure():
    ligand_a = _parameters("c1ccccc1")
    ligand_b = _parameters("c1ccc2c(c1)CCC2")

    mapping, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {
            "method": "explicit_pairs",
            "pairs_0based": [[index, index] for index in range(6)],
            "alchemical_bonds": {
                "ligand_b": [{"atoms_0based": [7, 8], "mode": "soft_bond"}]
            },
        },
    )

    assert 7 not in mapping.values()
    assert 8 not in mapping.values()
    assert metadata["alchemical_bonds"]["ligand_b"][0]["atoms_0based"] == [7, 8]


def test_soft_closure_opens_ring_for_z_matrix_anchor():
    ligand_a = _parameters("c1ccccc1")
    ligand_b = _parameters("c1ccc2c(c1)CCC2")

    _, metadata = build_hybrid_atom_map(
        ligand_a,
        ligand_b,
        {
            "method": "explicit_pairs",
            "pairs_0based": [[index, index] for index in range(6)],
            "junction_bonds": {
                "ligand_b": [{
                    "atoms_0based": [4, 6],
                    "inactive_geometry": "terminal_z_matrix",
                }]
            },
            "alchemical_bonds": {
                "ligand_b": [{"atoms_0based": [3, 8], "mode": "soft_bond"}]
            },
        },
    )

    assert metadata["inactive_bonded_atoms_b_0based"] == [6, 7, 8]
    assert metadata["inactive_z_matrix_root_atoms_b_0based"] == [6]


def test_junction_bonds_reject_legacy_fields():
    settings = {
        "junction_bonds": {"ligand_a": [{"atoms_0based": [0, 1]}]},
        "inactive_bonded_geometry": "bond_only",
    }
    with pytest.raises(HybridMappingError, match="cannot be combined"):
        # Structural normalization succeeds; conflict is mapping-context validation.
        ligand = _parameters("CC")
        build_hybrid_atom_map(
            ligand,
            ligand,
            {"method": "explicit_pairs", "pairs_0based": [[0, 0], [1, 1]], **settings},
        )
