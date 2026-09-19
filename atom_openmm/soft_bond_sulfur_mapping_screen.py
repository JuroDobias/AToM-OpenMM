"""Generate a cGAS soft-bond screen mapping pyridine C18 to thiazole S18."""

from copy import deepcopy
from pathlib import Path
import shutil

import yaml

from .covalent_workflow import _write_yaml_atomic
from .soft_bond_dual_closure_screen import _variants as dual_variants
from .soft_bond_screen import path_variants


NEW_TRANSMUTATION = [18, 18]
SELECTED_PATHS = (
    "p21_torsion_first_half_crossfade_together",
    "p17_torsion_first_full_crossfade_together",
    "p13_angular_first_half_crossfade_together",
    "attach_then_prepare_half",
)


def _mapped_workflow(source):
    document = deepcopy(source)
    workflow = document["workflow"]
    mapping = workflow["alchemy"]["mapping"]
    pairs = [list(pair) for pair in mapping["pairs_0based"]]
    if NEW_TRANSMUTATION not in pairs:
        pairs.append(list(NEW_TRANSMUTATION))
    mapping["pairs_0based"] = pairs
    mapping.pop("ccw_mapping_revision_id", None)
    mapping["mapping_note"] = (
        "Experimental C18->S18 transmutation reduces both unmatched ring branches"
    )
    return document


def selected_variants(total_steps=50000):
    available = {
        item["name"]: item
        for item in (
            *path_variants(total_steps=total_steps),
            *dual_variants(total_steps=total_steps),
        )
    }
    return [deepcopy(available[name]) for name in SELECTED_PATHS]


def generate(root, source, *, repo=None):
    root = Path(root).resolve()
    source = Path(source).resolve()
    if root.exists():
        raise ValueError("Use a new output directory")
    source_workflow = source / "source_workflow.yaml"
    if not source_workflow.is_file():
        raise ValueError("source must contain source_workflow.yaml")
    root.mkdir(parents=True)
    workflow = _mapped_workflow(yaml.safe_load(source_workflow.read_text()))
    _write_yaml_atomic(root / "source_workflow.yaml", workflow)
    config = {
        "seed": 20260920,
        "duration_ps": 100,
        # Mapped C->S LJ offsets produce a small dynamic-LRC endpoint constant
        # on CPU. Forces remain identical; retain and report the signed offset.
        "endpoint_audit_energy_tolerance_kj_mol": 1.0,
        "variants": selected_variants(),
    }
    _write_yaml_atomic(root / "screen.yaml", config)
    repo = Path(repo).resolve() if repo else Path(__file__).resolve().parents[1]

    jobs = [("bank", "python -m atom_openmm.soft_bond_screen bank .")]
    jobs.extend(
        (item["name"], f"python -m atom_openmm.soft_bond_screen run . --variant {item['name']}")
        for item in config["variants"]
    )
    for name, command in jobs:
        (root / f"{name}.slurm").write_text(f'''#!/bin/bash
#SBATCH --job-name=cs-{name[:28]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-b
#SBATCH --time=04:00:00
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myatom_openmm86
export PYTHONPATH="{repo}${{PYTHONPATH:+:$PYTHONPATH}}"
{command}
''')
    (root / "README.md").write_text(
        "# cGAS C18-to-S18 mapped-transmutation screen\n\n"
        "This experiment adds the explicit A atom 18 aromatic-carbon to B atom "
        "18 aromatic-sulfur mapping. It reduces the unmatched A ring branch from "
        "three to two heavy atoms and the B branch from two to one. A fresh "
        "30-snapshot-per-endpoint reference bank is generated. Four 100 ps paths "
        "use 10 schedule-optimization snapshots and 20 held-out snapshots per "
        "direction. REST2 is disabled.\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()
    generate(args.root, args.source, repo=args.repo)
