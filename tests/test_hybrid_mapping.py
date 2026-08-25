from types import SimpleNamespace

import pytest
from openff.toolkit import Molecule

from atom_openmm.hybrid_mapping import (
    HybridMappingError,
    _strict_explicit_pairs,
    build_hybrid_atom_map,
)


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

    assert metadata["inactive_bonded_atoms_b_0based"] == [3]
    assert metadata["inactive_bonded_geometry"] == "terminal_z_matrix"


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
