from atom_openmm.soft_bond_sulfur_mapping_screen import (
    NEW_TRANSMUTATION,
    SELECTED_PATHS,
    _mapped_workflow,
    selected_variants,
)


def test_sulfur_mapping_is_added_without_changing_soft_bonds():
    source = {
        "workflow": {
            "alchemy": {
                "mapping": {
                    "method": "explicit_pairs",
                    "pairs_0based": [[17, 17], [21, 20]],
                    "ccw_mapping_revision_id": 59,
                    "alchemical_bonds": {
                        "ligand_a": [{"atoms_0based": [19, 20], "mode": "soft_bond"}],
                        "ligand_b": [{"atoms_0based": [18, 19], "mode": "soft_bond"}],
                    },
                }
            }
        }
    }
    mapped = _mapped_workflow(source)
    mapping = mapped["workflow"]["alchemy"]["mapping"]
    assert NEW_TRANSMUTATION in mapping["pairs_0based"]
    assert "ccw_mapping_revision_id" not in mapping
    assert mapping["alchemical_bonds"] == source["workflow"]["alchemy"]["mapping"]["alchemical_bonds"]


def test_selected_sulfur_mapping_paths_are_stable_and_complete():
    variants = selected_variants(total_steps=50000)
    assert tuple(item["name"] for item in variants) == SELECTED_PATHS
    assert all(item["total_steps"] == 50000 for item in variants)
