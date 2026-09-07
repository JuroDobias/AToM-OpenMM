#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from atom_openmm.rbfe_repeat_network import generate_repeats
from generate_hybrid_mcl1_ring_cycles import (
    _molecules,
    _reference_datasets,
    _write_template,
)


GROUPS = {
    "whole_27_47": (("27", "47", "whole_ring"),),
    "soft_27_42": (("27", "42", "soft_closure"),),
    "whole_27_42": (("27", "42", "whole_ring"),),
    "transmutation_triangle": (
        ("42", "23", "topology_transmutation"),
        ("42", "26", "topology_transmutation"),
        ("23", "26", "topology_transmutation"),
    ),
}


def _repair_group_network(path, reference_node, table):
    network = yaml.safe_load(path.read_text())
    network["reference_node"] = reference_node
    network["reference_datasets"] = _reference_datasets(table)
    path.write_text(yaml.safe_dump(network, sort_keys=False))


def generate(benchmark_root, base_workflow, output, repeats=3):
    benchmark_root = Path(benchmark_root).resolve()
    output = Path(output).resolve()
    ligand_file = benchmark_root / "ATM_Validation/MCL-1/ligands/MCL1_ligands.sdf"
    receptor = benchmark_root / "ATM_Validation/MCL-1/receptor/MCL1_new_2.pdb"
    table = benchmark_root / "ATM_Validation/DDG_ATM_GAFF2.csv"
    molecules = _molecules(ligand_file)
    workflow = yaml.safe_load(Path(base_workflow).read_text())
    workflow.pop("atom_options", None)

    templates = output / "templates"
    runs = output / "runs"
    for group_index, (group, edges) in enumerate(GROUPS.items()):
        template = templates / group
        _write_template(
            template,
            workflow,
            molecules,
            receptor,
            table,
            edges,
            control=group.startswith("whole_"),
        )
        reference_node = edges[0][0]
        _repair_group_network(template / "network.yaml", reference_node, table)
        generate_repeats(
            template,
            runs / group,
            repeats=repeats,
            seed_base=20260907 + group_index * 1000,
            share_prepared=False,
        )

    submit = output / "submit_all.sh"
    submit.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(
            f'"$(dirname "$0")/runs/{group}/submit_all.sh"'
            for group in GROUPS
        )
        + "\n"
    )
    submit.chmod(0o755)
    (output / "README.md").write_text(
        "# MCL-1 transmutation graph expansion\n\n"
        "This cohort first completes the whole-ring version of `27 -> 47`, then "
        "adds ligand 42 radially by both soft ring closure and whole-ring mapping. "
        "The `42 -> 23 -> 26 -> 42` triangle tests direct mapped `[nH]`, sulfur, "
        "and oxygen transmutations. Each calculation has three independent repeats: "
        "six calculation variants and 18 jobs in total.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--base-workflow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    generate(args.benchmark_root, args.base_workflow, args.output, args.repeats)


if __name__ == "__main__":
    main()
