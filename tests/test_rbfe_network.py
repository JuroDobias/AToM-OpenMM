from pathlib import Path

import pytest
import yaml

from atom_openmm.rbfe_network import analyze_network


def _write_result(root, ligand_a, ligand_b, value, error=0.1):
    directory = root / f"{ligand_a}--{ligand_b}" / "run" / "pair"
    directory.mkdir(parents=True)
    (directory / "result.yaml").write_text(
        yaml.safe_dump(
            {
                "status": "completed",
                "result": {
                    "ddg_kcal_per_mol": value,
                    "ddg_error_kcal_per_mol": error,
                },
            }
        )
    )


def _test_network_fit_cycles_and_indirect_target(tmp_path):
    values = {
        "P": 0.0,
        "A": 1.0,
        "S": 2.0,
        "M": 0.5,
        "MA": 1.7,
        "MS": 2.8,
    }
    edges = [
        ("P", "A"),
        ("P", "S"),
        ("P", "M"),
        ("M", "MA"),
        ("M", "MS"),
        ("A", "MA"),
        ("S", "MS"),
    ]
    for ligand_a, ligand_b in edges:
        _write_result(
            tmp_path,
            ligand_a,
            ligand_b,
            values[ligand_b] - values[ligand_a],
        )
    config = {
        "reference_node": "P",
        "target": ["A", "MS"],
        "nodes": list(values),
        "edges": [
            {
                "ligand_a": ligand_a,
                "ligand_b": ligand_b,
                "directory": f"{ligand_a}--{ligand_b}",
            }
            for ligand_a, ligand_b in edges
        ],
        "cycles": [
            {
                "id": "amide",
                "terms": [
                    {"edge": "P--A", "coefficient": 1},
                    {"edge": "A--MA", "coefficient": 1},
                    {"edge": "M--MA", "coefficient": -1},
                    {"edge": "P--M", "coefficient": -1},
                ],
            },
            {
                "id": "sulfonamide",
                "terms": [
                    {"edge": "P--S", "coefficient": 1},
                    {"edge": "S--MS", "coefficient": 1},
                    {"edge": "M--MS", "coefficient": -1},
                    {"edge": "P--M", "coefficient": -1},
                ],
            },
        ],
    }
    config_path = tmp_path / "network.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    result = analyze_network(config_path)

    assert result["status"] == "completed"
    assert result["matrix_rank"] == 5
    assert result["target"]["ddg_kcal_per_mol"] == pytest.approx(1.8)
    assert result["target"]["ddg_error_kcal_per_mol"] > 0.0
    assert [cycle["closure_kcal_per_mol"] for cycle in result["cycles"]] == pytest.approx(
        [0.0, 0.0]
    )
    assert Path(tmp_path / "network_result.yaml").is_file()
