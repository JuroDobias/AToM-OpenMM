from pathlib import Path
import csv

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


def _write_separated_result(root, ligand_a, ligand_b, value):
    directory = root / f"{ligand_a}--{ligand_b}" / "run" / "pair"
    directory.mkdir(parents=True)
    component = {"dg_kcal_per_mol": value, "overlap_score": 0.2}
    (directory / "result.yaml").write_text(yaml.safe_dump({
        "status": "completed",
        "alchemy_model": "separated_topology",
        "result": {
            "ddg_kcal_per_mol": value,
            "ddg_error_kcal_per_mol": 0.2,
            "components": {"complex": component, "solvent": component},
        },
    }))
    for environment in ("complex", "solvent"):
        for direction, node, sign in (
            ("forward", ligand_a, 1.0),
            ("reverse", ligand_b, -1.0),
        ):
            with (directory / f"{environment}_{direction}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "sample", "work_kcal_per_mol", "active_node",
                    "active_snapshot_id", "inactive_node",
                    "inactive_vacuum_snapshot_id",
                ])
                writer.writeheader()
                for sample, offset in enumerate((-0.2, 0.0, 0.2), 1):
                    writer.writerow({
                        "sample": sample,
                        "work_kcal_per_mol": sign * value + offset,
                        "active_node": node,
                        "active_snapshot_id": f"nodes/{node}/{environment}/{sample}.xml",
                        "inactive_node": ligand_b if direction == "forward" else ligand_a,
                        "inactive_vacuum_snapshot_id": (
                            f"nodes/{ligand_b if direction == 'forward' else ligand_a}"
                            f"/vacuum/{sample}.xml"
                        ),
                    })


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


def _test_separated_network_uses_node_correlated_cycle_bootstrap(tmp_path):
    edges = [("A", "B", 0.4), ("B", "C", 0.6), ("A", "C", 1.0)]
    for ligand_a, ligand_b, value in edges:
        _write_separated_result(tmp_path, ligand_a, ligand_b, value)
    config = {
        "reference_node": "A",
        "target": ["A", "C"],
        "temperature_k": 300.0,
        "node_bootstrap_samples": 20,
        "random_seed": 7,
        "nodes": ["A", "B", "C"],
        "edges": [
            {"ligand_a": a, "ligand_b": b, "directory": f"{a}--{b}"}
            for a, b, _ in edges
        ],
        "cycles": [{
            "id": "triangle",
            "terms": [
                {"edge": "A--B", "coefficient": 1},
                {"edge": "B--C", "coefficient": 1},
                {"edge": "A--C", "coefficient": -1},
            ],
        }],
    }
    path = tmp_path / "network.yaml"
    path.write_text(yaml.safe_dump(config))

    result = analyze_network(path)

    bootstrap = result["cycles"][0]["node_correlated_bootstrap"]
    assert bootstrap["binding"]["samples"] == 20
    assert bootstrap["complex"]["std_kcal_per_mol"] >= 0.0
    assert bootstrap["solvent"]["std_kcal_per_mol"] >= 0.0
