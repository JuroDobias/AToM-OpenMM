from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

from atom_openmm.rbfe_graph_proposal import propose_graph
from atom_openmm.rbfe_repeat_network import generate_repeats


def _write_sdf(path, smiles):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=7) == 0
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()


def _test_graph_proposal_connects_single_attachment_changes(tmp_path):
    _write_sdf(tmp_path / "a.sdf", "c1ccccc1")
    _write_sdf(tmp_path / "b.sdf", "COc1ccccc1")
    _write_sdf(tmp_path / "c.sdf", "NC(=O)c1ccccc1")
    config = {
        "nodes": [
            {"id": name, "file": f"{name}.sdf"} for name in ("a", "b", "c")
        ],
        "targets": [{"ligand_a": "b", "ligand_b": "c"}],
    }
    path = tmp_path / "inputs.yaml"
    path.write_text(yaml.safe_dump(config))

    result = propose_graph(path)

    assert result["targets"][0]["status"] == "connected"
    assert result["suggested_path_edges"]
    assert all(edge["attachment_site_count"] == 1 for edge in result["suggested_path_edges"])


def _test_repeat_generator_preserves_existing_repeat_files(tmp_path):
    source = tmp_path / "source"
    edge = source / "A--B"
    edge.mkdir(parents=True)
    (source / "network.yaml").write_text(yaml.safe_dump({
        "reference_node": "A",
        "target": ["A", "B"],
        "nodes": ["A", "B"],
        "edges": [{"ligand_a": "A", "ligand_b": "B", "directory": "A--B"}],
    }))
    (source / "submit_all.sh").write_text("#!/bin/bash\n")
    (edge / "workflow.yaml").write_text(yaml.safe_dump({
        "workflow": {"random_seed": 1},
    }))
    prepared = edge / "run" / "receptor-A-B" / "prepared"
    prepared.mkdir(parents=True)
    (prepared / "manifest.yaml").write_text("schema_version: 1\n")
    output = tmp_path / "repeats"

    generate_repeats(source, output, repeats=3, seed_base=100)
    marker = output / "replicate_1" / "A--B" / "marker.txt"
    marker.write_text("keep")
    edge_c = source / "B--C"
    edge_c.mkdir()
    (edge_c / "workflow.yaml").write_text(yaml.safe_dump({
        "workflow": {"random_seed": 1},
    }))
    expanded = yaml.safe_load((source / "network.yaml").read_text())
    expanded["nodes"].append("C")
    expanded["edges"].append({
        "ligand_a": "B", "ligand_b": "C", "directory": "B--C",
    })
    (source / "network.yaml").write_text(yaml.safe_dump(expanded))
    generate_repeats(source, output, repeats=3, seed_base=100)

    seeds = []
    for index in range(1, 4):
        workflow = yaml.safe_load(
            (output / f"replicate_{index}" / "A--B" / "workflow.yaml").read_text()
        )
        seeds.append(workflow["workflow"]["random_seed"])
    assert len(set(seeds)) == 3
    assert marker.read_text() == "keep"
    linked = output / "replicate_1" / "A--B" / "run" / "receptor-A-B" / "prepared"
    assert linked.is_symlink()
    assert (linked / "manifest.yaml").is_file()
    child_network = yaml.safe_load(
        (output / "replicate_1" / "network.yaml").read_text()
    )
    assert len(child_network["edges"]) == 2
    assert Path(output / "repeat_network.yaml").is_file()


def _test_repeat_generator_reuses_finite_edges_and_submits_only_new_jobs(tmp_path):
    source = tmp_path / "source"
    for edge in ("A--B", "B--C"):
        directory = source / edge
        directory.mkdir(parents=True)
        (directory / "workflow.yaml").write_text(yaml.safe_dump({
            "workflow": {"random_seed": 1},
        }))
        (directory / "run.sh").write_text("#!/bin/bash\n")
    network = {
        "reference_node": "A",
        "target": ["A", "C"],
        "nodes": ["A", "B", "C"],
        "edges": [
            {"id": "A--B", "ligand_a": "A", "ligand_b": "B", "directory": "A--B"},
            {"id": "B--C", "ligand_a": "B", "ligand_b": "C", "directory": "B--C"},
        ],
    }
    (source / "network.yaml").write_text(yaml.safe_dump(network))
    (source / "submit_all.sh").write_text("#!/bin/bash\n")
    previous = tmp_path / "previous"
    old_network = {**network, "nodes": ["A", "B"], "edges": [network["edges"][0]], "target": ["A", "B"]}
    for index in range(1, 4):
        repeat = previous / f"replicate_{index}"
        result = repeat / "A--B" / "run" / "pair" / "result.yaml"
        result.parent.mkdir(parents=True)
        result.write_text(yaml.safe_dump({
            "status": "completed",
            "result": {
                "ddg_kcal_per_mol": 1.0 + index / 10,
                "ddg_error_kcal_per_mol": 0.2,
            },
        }))
        (repeat / "network.yaml").write_text(yaml.safe_dump(old_network))
    output = tmp_path / "output"

    manifest = generate_repeats(
        source, output, repeats=3, reuse_repeats_from=previous
    )

    assert manifest["new_job_count"] == 3
    for index in range(1, 4):
        reused = output / f"replicate_{index}" / "A--B"
        assert reused.is_symlink()
        submit = (output / f"replicate_{index}" / "submit_all.sh").read_text()
        assert "B--C" in submit
        assert "A--B" not in submit
