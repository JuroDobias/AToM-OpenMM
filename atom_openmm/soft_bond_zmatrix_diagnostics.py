"""Generate the held-out cGAS terminal-Z-matrix switching diagnostic run."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from atom_openmm.covalent_workflow import _write_yaml_atomic
from atom_openmm.soft_bond_branch_relaxation_screen import variants


def diagnostic_variant():
    variant = next(
        item for item in variants(total_steps=250000)
        if item["name"] == "baseline_500ps"
    )
    result = deepcopy(variant)
    result["name"] = "zmatrix_mechanics_500ps"
    return result


def generate(root, source, bank, *, repo=None):
    root = Path(root).resolve()
    source = Path(source).resolve()
    bank = Path(bank).resolve()
    if root.exists():
        raise ValueError("Use a new output directory")
    if not (bank / "complete.yaml").is_file():
        raise ValueError("bank must contain complete.yaml")
    workflow = yaml.safe_load(source.read_text())
    mapping = workflow["workflow"]["alchemy"]["mapping"]
    if mapping.get("junction_bonds"):
        raise ValueError("diagnostic source must use automatic terminal Z-matrix junctions")
    root.mkdir(parents=True)
    _write_yaml_atomic(root / "source_workflow.yaml", workflow)
    _write_yaml_atomic(root / "screen.yaml", {
        "seed": 20260921,
        "bank": str(bank),
        "skip_optimizer": True,
        "evaluation_count": 5,
        "variants": [diagnostic_variant()],
        "diagnostics": {
            "enabled": True,
            "interval_steps": 100,
            "phases": ["evaluation"],
            "save_ligand_trajectory": True,
            "bonded_scope": "unique_atoms",
            "energy_detail": "total_plus_bonded",
        },
    })
    repo = Path(repo).resolve() if repo else Path(__file__).resolve().parents[1]
    (root / "run.slurm").write_text(f'''#!/bin/bash
#SBATCH --job-name=cgas-zmat-diag
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
python -m atom_openmm.soft_bond_screen run . --variant zmatrix_mechanics_500ps
''')
    (root / "README.md").write_text(
        "# cGAS terminal-Z-matrix switching mechanics\n\n"
        "Reuses the original terminal-Z-matrix reference bank. Five held-out paired "
        "500 ps switches use the fixed prepare-then-attach-half schedule. Every 100 "
        "steps, the run records exact potential/work data, all bonded terms touching "
        "endpoint-unique atoms, and the complete hybrid-ligand coordinates.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()
    generate(args.root, args.source, args.bank, repo=args.repo)
