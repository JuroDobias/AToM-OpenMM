"""Generate a cGAS soft-bond screen with dynamic unique-branch relaxation."""

from copy import deepcopy
from pathlib import Path

import yaml

from .covalent_workflow import _write_yaml_atomic
from .soft_bond_dual_closure_screen import _variants as dual_closure_variants


def _with_branch_relaxation(source, name, *, angle_scale, torsion_scale):
    variant = deepcopy(source)
    variant["name"] = name
    nodes = []
    for node in variant["nodes"]:
        if node["label"] == "regular_bonded_half":
            controls = dict(nodes[-1]["controls"])
            controls.update(
                branch_angles_a=angle_scale,
                branch_angles_b=angle_scale,
                branch_torsions_a=torsion_scale,
                branch_torsions_b=torsion_scale,
            )
            nodes.append({"label": "relax_unique_branches", "controls": controls})
        nodes.append(deepcopy(node))
        if node["label"] == "central_topology_b":
            controls = dict(nodes[-1]["controls"])
            controls.update(
                branch_angles_a=1.0,
                branch_angles_b=1.0,
                branch_torsions_a=1.0,
                branch_torsions_b=1.0,
            )
            nodes.append({"label": "restore_unique_branches", "controls": controls})

    for index, node in enumerate(nodes):
        node["at"] = index / (len(nodes) - 1)
    variant["nodes"] = nodes
    variant["segments_per_interval"] = [3] * (len(nodes) - 1)
    return variant


def variants(total_steps=250000):
    baseline = next(
        item for item in dual_closure_variants(total_steps=total_steps)
        if item["name"] == "prepare_then_attach_half"
    )
    baseline = deepcopy(baseline)
    baseline["name"] = "baseline_500ps"
    return [
        baseline,
        _with_branch_relaxation(
            baseline,
            "branch_torsion_005_500ps",
            angle_scale=1.0,
            torsion_scale=0.05,
        ),
        _with_branch_relaxation(
            baseline,
            "branch_angle_025_torsion_005_500ps",
            angle_scale=0.25,
            torsion_scale=0.05,
        ),
        _with_branch_relaxation(
            baseline,
            "branch_angle_010_torsion_000_500ps",
            angle_scale=0.10,
            torsion_scale=0.0,
        ),
    ]


def generate(root, bank, *, repo=None):
    root = Path(root).resolve()
    bank = Path(bank).resolve()
    if root.exists():
        raise ValueError("Use a new output directory")
    if not (bank / "complete.yaml").is_file():
        raise ValueError("A complete snapshot bank is required")
    root.mkdir(parents=True)
    repo = Path(repo).resolve() if repo else Path(__file__).resolve().parents[1]
    config = {
        "seed": 20260919,
        "bank": str(bank),
        "skip_optimizer": True,
        "variants": variants(),
    }
    _write_yaml_atomic(root / "screen.yaml", config)
    for variant in config["variants"]:
        name = variant["name"]
        (root / f"{name}.slurm").write_text(f'''#!/bin/bash
#SBATCH --job-name=br-{name[:28]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-b
#SBATCH --time=12:00:00
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myatom_openmm86
export PYTHONPATH="{repo}${{PYTHONPATH:+:$PYTHONPATH}}"
python -m atom_openmm.soft_bond_screen run . --variant {name}
''')
    (root / "README.md").write_text(
        "# Dynamic unique-branch relaxation screen\n\n"
        "Four uniform 500 ps paths use the same held-out cGAS reference snapshots. "
        "The baseline is unchanged. Other paths dynamically soften proper torsions, "
        "or angles and proper torsions, across both unique branches while closure "
        "topology is exchanged. Bond lengths and impropers remain physical. All "
        "branch controls return to 1.0 at both physical endpoints.\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()
    generate(args.root, args.bank, repo=args.repo)
