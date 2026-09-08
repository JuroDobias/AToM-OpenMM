#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from atom_openmm.rbfe_repeat_network import generate_repeats
from generate_hybrid_mcl1_ring_cycles import (
    _embedded_base_map,
    _molecules,
    _write_template,
)


GROUPS = {
    "forward_swapped_anchor": {
        "ligand_a": "27",
        "ligand_b": "43",
        "reverse": False,
        "swap_anchor": True,
    },
    "reverse_original_anchor": {
        "ligand_a": "43",
        "ligand_b": "27",
        "reverse": True,
        "swap_anchor": False,
    },
    "reverse_swapped_anchor": {
        "ligand_a": "43",
        "ligand_b": "27",
        "reverse": True,
        "swap_anchor": True,
    },
}


def _soft_closure_mapping(base, annulated, *, reverse, swap_anchor):
    pairs, original_anchor, original_soft = _embedded_base_map(base, annulated)
    anchor = original_soft if swap_anchor else original_anchor
    soft = original_anchor if swap_anchor else original_soft
    if reverse:
        pairs = [[atom_b, atom_a] for atom_a, atom_b in pairs]
        endpoint = "ligand_a"
    else:
        endpoint = "ligand_b"
    return {
        "method": "explicit_pairs",
        "pairs_0based": pairs,
        "junction_bonds": {
            endpoint: [
                {
                    "atoms_0based": anchor,
                    "inactive_geometry": "terminal_z_matrix",
                }
            ]
        },
        "alchemical_bonds": {
            endpoint: [{"atoms_0based": soft, "mode": "soft_bond"}]
        },
    }


def _patch_mapping(template, mapping):
    workflow_path = next(template.glob("*/workflow.yaml"))
    workflow = yaml.safe_load(workflow_path.read_text())
    workflow["workflow"]["alchemy"]["mapping"] = mapping
    workflow_path.write_text(yaml.safe_dump(workflow, sort_keys=False))


def generate(benchmark_root, base_workflow, output, repeats=3):
    benchmark_root = Path(benchmark_root).resolve()
    output = Path(output).resolve()
    ligand_file = benchmark_root / "ATM_Validation/MCL-1/ligands/MCL1_ligands.sdf"
    receptor = benchmark_root / "ATM_Validation/MCL-1/receptor/MCL1_new_2.pdb"
    table = benchmark_root / "ATM_Validation/DDG_ATM_GAFF2.csv"
    molecules = _molecules(ligand_file)
    workflow = yaml.safe_load(Path(base_workflow).read_text())
    workflow.pop("atom_options", None)

    for group_index, (group, settings) in enumerate(GROUPS.items()):
        ligand_a = settings["ligand_a"]
        ligand_b = settings["ligand_b"]
        template = output / "templates" / group
        # MCS is only a placeholder; it is replaced before repeats are generated.
        _write_template(
            template,
            copy.deepcopy(workflow),
            molecules,
            receptor,
            table,
            [(ligand_a, ligand_b, "mcs")],
            control=True,
        )
        mapping = _soft_closure_mapping(
            molecules["27"],
            molecules["43"],
            reverse=settings["reverse"],
            swap_anchor=settings["swap_anchor"],
        )
        _patch_mapping(template, mapping)
        generate_repeats(
            template,
            output / "runs" / group,
            repeats=repeats,
            seed_base=20260908 + group_index * 1000,
            share_prepared=False,
        )

    submit = output / "submit_all.sh"
    submit.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(
            f'"$(dirname "$0")/runs/{group}/submit_all.sh"' for group in GROUPS
        )
        + "\n"
    )
    submit.chmod(0o755)
    (output / "README.md").write_text(
        "# MCL-1 soft-closure symmetry validation\n\n"
        "The completed `27 -> 43` original-anchor triplicate is the reference. "
        "This cohort runs the opposite anchor in the forward construction and "
        "both anchor choices after rebuilding the hybrid topology as `43 -> 27`. "
        "Each new construction has three independent repeats, for nine jobs.\n"
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
