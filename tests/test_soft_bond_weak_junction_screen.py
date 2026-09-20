from atom_openmm.soft_bond_weak_junction_screen import (
    JUNCTIONS,
    selected_variant,
    weak_junction_workflow,
)


def test_weak_junction_workflow_replaces_z_matrices_and_sets_scales():
    source = {
        "workflow": {
            "alchemy": {
                "mapping": {
                    "pairs_0based": [[0, 0]],
                    "inactive_bonded_atoms_a_0based": [1],
                    "inactive_bonded_geometry": "terminal_z_matrix",
                }
            },
            "setup": {},
        }
    }
    workflow = weak_junction_workflow(source)["workflow"]
    mapping = workflow["alchemy"]["mapping"]
    assert "inactive_bonded_geometry" not in mapping
    for endpoint, expected in JUNCTIONS.items():
        entries = mapping["junction_bonds"][endpoint]
        assert [tuple(entry["atoms_0based"]) for entry in entries] == list(expected)
        assert {entry["inactive_geometry"] for entry in entries} == {"bond_only"}
    scales = workflow["setup"]["dummy_bonded_scales"]
    assert scales["bond"] == 1.0
    assert scales["junction_angle"] == 0.2
    assert scales["junction_proper_torsion"] == 0.2
    assert scales["proper_torsion"] == 1.0


def test_weak_junction_path_is_uniform_500_ps_baseline():
    variant = selected_variant()
    assert variant["name"] == "weak_junction_020_500ps"
    assert variant["total_steps"] == 250000
    for node in variant["nodes"]:
        assert node["controls"]["branch_angles_a"] == 1.0
        assert node["controls"]["branch_angles_b"] == 1.0
        assert node["controls"]["branch_torsions_a"] == 1.0
        assert node["controls"]["branch_torsions_b"] == 1.0
