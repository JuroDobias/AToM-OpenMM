from pathlib import Path

import numpy as np
from rdkit import Chem

from atom_openmm.covalent_dataset import (
    build_capped_thiohemiacetal,
    retain_protein_waters,
)


def _aldehyde():
    molecule = Chem.AddHs(Chem.MolFromSmiles("CC=O"))
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(index, (float(index), 0.0, 0.0))
    molecule.AddConformer(conformer)
    return molecule


def _cys_coordinates():
    return {
        "N": np.asarray([-1.0, 0.0, 0.0]),
        "CA": np.asarray([0.0, 0.0, 0.0]),
        "C": np.asarray([0.8, 1.2, 0.0]),
        "O": np.asarray([0.4, 2.3, 0.0]),
        "CB": np.asarray([0.6, -1.3, 0.0]),
        "SG": np.asarray([2.3, -1.4, 0.0]),
        "HG": np.asarray([3.0, -0.5, 0.0]),
    }


def _test_build_capped_thiohemiacetal_transfers_thiol_hydrogen():
    ligand = _aldehyde()
    product, metadata = build_capped_thiohemiacetal(ligand, _cys_coordinates())
    sulfur = product.GetAtomWithIdx(metadata["cys_sulfur_atom_index"])
    oxygen = product.GetAtomWithIdx(metadata["product_oxygen_atom_index"])
    carbon = product.GetAtomWithIdx(metadata["electrophile_carbon_atom_index"])
    hydrogen = metadata["transferred_hydrogen_atom_index"]

    assert hydrogen not in [atom.GetIdx() for atom in sulfur.GetNeighbors()]
    assert hydrogen in [atom.GetIdx() for atom in oxygen.GetNeighbors()]
    assert sulfur.GetIdx() in [atom.GetIdx() for atom in carbon.GetNeighbors()]
    assert product.GetBondBetweenAtoms(carbon.GetIdx(), oxygen.GetIdx()).GetBondType() == Chem.BondType.SINGLE


def _test_retain_protein_waters_uses_oxygen_distance(tmp_path):
    pdb = tmp_path / "input.pdb"
    pdb.write_text(
        "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      2  O   WAT A   2       4.900   0.000   0.000  1.00  0.00           O  \n"
        "ATOM      3  H1  WAT A   2       5.800   0.000   0.000  1.00  0.00           H  \n"
        "ATOM      4  H2  WAT A   2       4.900   0.900   0.000  1.00  0.00           H  \n"
        "TER\n"
        "ATOM      5  O   WAT A   3       5.100   0.000   0.000  1.00  0.00           O  \n"
        "ATOM      6  H1  WAT A   3       6.000   0.000   0.000  1.00  0.00           H  \n"
        "ATOM      7  H2  WAT A   3       5.100   0.900   0.000  1.00  0.00           H  \n"
        "TER\nEND\n"
    )
    output = tmp_path / "output.pdb"
    assert retain_protein_waters(pdb, output, 5.0) == 1
    text = output.read_text()
    assert "A   2" in text
    assert "A   3" not in text
