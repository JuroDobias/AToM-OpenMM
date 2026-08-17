#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import yaml

from atom_openmm.rbfe_graph_proposal import discover_ligand_file

try:
    from .generate_hybrid_cdk2_cohort import _run_script, _workflow
except ImportError:
    from generate_hybrid_cdk2_cohort import _run_script, _workflow


def _reference_datasets(csv_path):
    rows = [
        row for row in csv.DictReader(Path(csv_path).open())
        if row["Protein"] == "CDK2"
    ]
    return {
        "experiment": {
            "reference_node": "1h1q",
            "edges": [
                {
                    "ligand_a": row["Ligand1"],
                    "ligand_b": row["Ligand2"],
                    "ddg_kcal_per_mol": float(row["experimental_ddG"]),
                }
                for row in rows
            ],
        },
        "published_atm_gaff2": {
            "reference_node": "1h1q",
            "edges": [
                {
                    "ligand_a": row["Ligand1"],
                    "ligand_b": row["Ligand2"],
                    "ddg_kcal_per_mol": float(row["ATM_ddG"]),
                    "ddg_error_kcal_per_mol": float(row["ATM_error"]),
                }
                for row in rows
            ],
        },
    }


def generate(graph_path, source_cohort, benchmark_root, output, source_dir_name):
    graph_path = Path(graph_path).resolve()
    graph = yaml.safe_load(graph_path.read_text()) or {}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receptor = Path(benchmark_root) / "ATM_Validation/CDK2/receptor/CDK2_new_2_edit.pdb"
    table = Path(benchmark_root) / "ATM_Validation/DDG_ATM_GAFF2.csv"
    if not receptor.is_file() or not table.is_file():
        raise FileNotFoundError("CDK2 receptor or DDG_ATM_GAFF2.csv is missing")
    nodes = {
        str(node["id"]): {**node, "id": str(node["id"])} for node in graph["nodes"]
    }
    sources = {
        identifier: discover_ligand_file(source_cohort, node, graph_path.parent)
        for identifier, node in nodes.items()
    }
    edges = graph.get("simulated_edges", graph.get("edges", []))
    network_edges = []
    for edge_index, item in enumerate(edges, 1):
        ligand_a = str(item["ligand_a"] if isinstance(item, dict) else item[0])
        ligand_b = str(item["ligand_b"] if isinstance(item, dict) else item[1])
        edge_id = str(item.get("id", f"{ligand_a}--{ligand_b}")) if isinstance(item, dict) else f"{ligand_a}--{ligand_b}"
        target = output / edge_id
        if not target.exists():
            (target / "ligands").mkdir(parents=True)
            shutil.copy2(receptor, target / "receptor.pdb")
            for ligand in (ligand_a, ligand_b):
                shutil.copy2(sources[ligand], target / "ligands" / f"{ligand}-p.sdf")
            workflow = _workflow(
                ligand_a, ligand_b,
                n_snapshots=100,
                switch_time_ps=100.0,
                adaptive_switching=True,
                convergence=True,
                dummy_core_nonbonded="off",
                schedule_optimization=True,
                unrestrained_npt_steps=1000000,
                random_seed=20260817 + edge_index,
            )
            workflow["workflow"]["alchemy"]["mapping"] = {"method": "mcs"}
            scales = workflow["workflow"]["setup"]["dummy_bonded_scales"]
            scales["junction_rotatable_torsion"] = 0.1
            scales["internal_rotatable_torsion"] = 0.1
            (target / "workflow.yaml").write_text(
                yaml.safe_dump(workflow, sort_keys=False)
            )
            run = target / "run.sh"
            run.write_text(_run_script(edge_id, source_dir_name=source_dir_name))
            run.chmod(0o755)
        network_edges.append({
            "id": edge_id,
            "ligand_a": ligand_a,
            "ligand_b": ligand_b,
            "directory": edge_id,
        })
    network = {
        "schema_version": 2,
        "reference_node": str(graph.get("reference_node", "1h1q")),
        "nodes": list(nodes.values()),
        "edges": network_edges,
        "targets": graph["targets"],
        "auto_cycles": bool(graph.get("auto_cycles", True)),
        "reference_datasets": _reference_datasets(table),
    }
    (output / "network.yaml").write_text(yaml.safe_dump(network, sort_keys=False))
    submit = output / "submit_all.sh"
    submit.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(
            f'(cd "$(dirname "$0")/{edge["id"]}" && sbatch run.sh)'
            for edge in network_edges
        )
        + "\n"
    )
    submit.chmod(0o755)
    (output / "README.md").write_text(
        "# Repeat-ready CDK2 graph template\n\n"
        f"This curated graph contains {len(nodes)} nodes, {len(network_edges)} simulated "
        f"single-site edges, {len(graph['targets'])} published prediction targets, and "
        "automatic fundamental-cycle analysis. Generate independent repeats with "
        "`atom-rbfe-generate-repeats TEMPLATE OUTPUT --repeats 3`. Existing edge "
        "directories are never overwritten when this graph is expanded.\n"
    )
    return network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--source-cohort", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir-name", default="AToM-OpenMM-unified")
    args = parser.parse_args()
    generate(
        args.graph, args.source_cohort, args.benchmark_root, args.output,
        args.source_dir_name,
    )


if __name__ == "__main__":
    main()
