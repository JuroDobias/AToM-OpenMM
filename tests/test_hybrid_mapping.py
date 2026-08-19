from types import SimpleNamespace

from openff.toolkit import Molecule

from atom_openmm.hybrid_mapping import build_hybrid_atom_map


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
