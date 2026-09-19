from atom_openmm.covalent_softcore import (
    EXPLICIT_ENDPOINT_A_CONTROLS,
    EXPLICIT_ENDPOINT_B_CONTROLS,
    resolve_softcore_path,
)
from atom_openmm.soft_bond_branch_relaxation_screen import variants


def test_branch_relaxation_variants_preserve_endpoints_and_soften_only_interior():
    generated = variants(total_steps=1000)
    assert [item["name"] for item in generated] == [
        "baseline_500ps",
        "branch_torsion_005_500ps",
        "branch_angle_025_torsion_005_500ps",
        "branch_angle_010_torsion_000_500ps",
    ]
    for variant in generated:
        assert variant["nodes"][0]["controls"] == EXPLICIT_ENDPOINT_A_CONTROLS
        assert variant["nodes"][-1]["controls"] == EXPLICIT_ENDPOINT_B_CONTROLS
        resolve_softcore_path(
            total_steps=variant["total_steps"],
            control_nodes=variant["nodes"],
            segments_per_interval=variant["segments_per_interval"],
        )

    moderate = generated[2]
    relaxed = next(
        node for node in moderate["nodes"]
        if node["label"] == "relax_unique_branches"
    )
    restored = next(
        node for node in moderate["nodes"]
        if node["label"] == "restore_unique_branches"
    )
    assert relaxed["controls"]["branch_angles_a"] == 0.25
    assert relaxed["controls"]["branch_torsions_b"] == 0.05
    assert restored["controls"]["branch_angles_a"] == 1.0
    assert restored["controls"]["branch_torsions_b"] == 1.0
