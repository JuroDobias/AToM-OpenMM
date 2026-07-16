import pytest


def _metadata(tmp_path):
    from rdkit import Chem

    result = {}
    offset = 10
    for identity, smiles in (("ligand_a", "CCO"), ("ligand_b", "CCN")):
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        path = tmp_path / f"{identity}.sdf"
        writer = Chem.SDWriter(str(path))
        writer.write(molecule)
        writer.close()
        indices = list(range(offset, offset + molecule.GetNumAtoms()))
        result[identity] = {
            "structure_file": path.name,
            "system_atom_indices": indices,
        }
        offset += 20
    return {"SELECTION_METADATA": result}


def _test_compile_selection_preserves_amber_operators_and_smarts_operators(tmp_path):
    from atom_openmm.selections import compile_selection_expression

    compiled = compile_selection_expression(
        '(!:HOH & #ligand_a:"[#6;!R]") | @CA',
        _metadata(tmp_path),
        endpoint="a",
        base_dir=tmp_path,
    )

    assert compiled.startswith("(!:HOH & (@11,12))")
    assert compiled.endswith("| @CA")


def _test_ligand_union_requires_and_selects_both_ligands(tmp_path):
    from atom_openmm.selections import resolve_smarts_atoms

    selected = resolve_smarts_atoms(
        '#ligand:"[#6]"', _metadata(tmp_path), base_dir=tmp_path
    )

    assert selected == [10, 11, 30, 31]


def _test_wildcard_roles_reverse_between_endpoints(tmp_path):
    from atom_openmm.selections import resolve_smarts_atoms

    keywords = _metadata(tmp_path)
    ligand_a = keywords["SELECTION_METADATA"]["ligand_a"]["system_atom_indices"]
    ligand_b = keywords["SELECTION_METADATA"]["ligand_b"]["system_atom_indices"]

    assert resolve_smarts_atoms('#bound:"*"', keywords, endpoint="a", base_dir=tmp_path) == ligand_a
    assert resolve_smarts_atoms('#unbound:"*"', keywords, endpoint="a", base_dir=tmp_path) == ligand_b
    assert resolve_smarts_atoms('#bound:"*"', keywords, endpoint="b", base_dir=tmp_path) == ligand_b
    assert resolve_smarts_atoms('#unbound:"*"', keywords, endpoint="b", base_dir=tmp_path) == ligand_a


def _test_midpoint_rejects_endpoint_dependent_role(tmp_path):
    from atom_openmm.selections import SelectionError, compile_selection_expression

    with pytest.raises(SelectionError, match="undefined at the ATM midpoint"):
        compile_selection_expression('#unbound:"*"', _metadata(tmp_path), base_dir=tmp_path)


def _test_all_symmetry_matches_are_unioned(tmp_path):
    from atom_openmm.selections import resolve_smarts_atoms

    selected = resolve_smarts_atoms(
        '#ligand_a:"C"', _metadata(tmp_path), endpoint="a", base_dir=tmp_path
    )

    assert selected == [10, 11]


def _test_mapping_mismatch_is_reported(tmp_path):
    from atom_openmm.selections import SelectionError, resolve_smarts_atoms

    keywords = _metadata(tmp_path)
    keywords["SELECTION_METADATA"]["ligand_a"]["system_atom_indices"].pop()
    with pytest.raises(SelectionError, match="mapping mismatch"):
        resolve_smarts_atoms('#ligand_a:"*"', keywords, base_dir=tmp_path)
