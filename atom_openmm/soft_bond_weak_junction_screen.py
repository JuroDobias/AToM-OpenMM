"""Generate a cGAS reference screen with weak, fully retained junction geometry."""

from copy import deepcopy
from pathlib import Path

import yaml

from .covalent_workflow import _write_yaml_atomic
from .soft_bond_branch_relaxation_screen import variants as branch_variants


JUNCTIONS = {
    "ligand_a": (
        (17, 18),
        (21, 20),
    ),
    "ligand_b": (
        (17, 18),
        (20, 19),
    ),
}


def weak_junction_workflow(document):
    result = deepcopy(document)
    workflow = result["workflow"]
    mapping = workflow["alchemy"]["mapping"]
    mapping["junction_bonds"] = {
        endpoint: [
            {
                "atoms_0based": list(atoms),
                "inactive_geometry": "full_junction",
            }
            for atoms in entries
        ]
        for endpoint, entries in JUNCTIONS.items()
    }
    mapping.pop("inactive_bonded_atoms_a_0based", None)
    mapping.pop("inactive_bonded_atoms_b_0based", None)
    mapping.pop("inactive_bonded_geometry", None)
    mapping.pop("ccw_mapping_revision_id", None)
    mapping["mapping_note"] = (
        "Original ring-contraction mapping; all four junctions retain full geometry"
    )
    workflow.setdefault("setup", {})["dummy_bonded_scales"] = {
        "bond": 1.0,
        "angle": 1.0,
        "proper_torsion": 1.0,
        "junction_angle": 0.2,
        "junction_proper_torsion": 0.2,
        "junction_rotatable_torsion": 0.2,
    }
    return result


def selected_variant():
    variant = next(
        item for item in branch_variants(total_steps=250000)
        if item["name"] == "baseline_500ps"
    )
    variant = deepcopy(variant)
    variant["name"] = "weak_junction_020_500ps"
    return variant


def generate(root, source, *, repo=None):
    root = Path(root).resolve()
    source = Path(source).resolve()
    if root.exists():
        raise ValueError("Use a new output directory")
    source_workflow = source / "source_workflow.yaml"
    if not source_workflow.is_file():
        raise ValueError("source must contain source_workflow.yaml")
    root.mkdir(parents=True)
    workflow = weak_junction_workflow(yaml.safe_load(source_workflow.read_text()))
    _write_yaml_atomic(root / "source_workflow.yaml", workflow)
    config = {
        "seed": 20260920,
        "duration_ps": 500,
        "bank_equilibration": {
            "minimization_tolerance_kj_mol_nm": 10,
            "minimization_max_iterations": 2000,
            "nvt_steps": 100000,
            "nvt_timestep_fs": 1.0,
            "npt_steps": 2500000,
            "npt_timestep_fs": 2.0,
        },
        "skip_optimizer": True,
        "variants": [selected_variant()],
    }
    _write_yaml_atomic(root / "screen.yaml", config)
    repo = Path(repo).resolve() if repo else Path(__file__).resolve().parents[1]
    jobs = (
        ("bank", "python -m atom_openmm.soft_bond_screen bank .", "12:00:00"),
        (
            "weak_junction_020_500ps",
            "python -m atom_openmm.soft_bond_screen run . --variant weak_junction_020_500ps",
            "12:00:00",
        ),
    )
    for name, command, walltime in jobs:
        (root / f"{name}.slurm").write_text(f'''#!/bin/bash
#SBATCH --job-name=wj-{name[:28]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-b
#SBATCH --time={walltime}
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myatom_openmm86
export PYTHONPATH="{repo}${{PYTHONPATH:+:$PYTHONPATH}}"
{command}
''')
    (root / "README.md").write_text(
        "# cGAS weak retained-junction reference experiment\n\n"
        "Original ms_491 -> ms_539 mapping. The four automatically detected "
        "Z-matrix junctions are explicitly replaced by full-junction geometry. "
        "Junction bonds remain full strength; junction angles and proper/rotatable "
        "proper torsions are retained at 0.2 in inactive branches. Internal branch "
        "bonded terms remain physical. Each endpoint receives 0.1 ns NVT and 5 ns "
        "NPT before 30 snapshots separated by 0.2 ns. The first 10 bank snapshots "
        "remain reserved, and 20 held-out snapshots per direction use the uniform "
        "500 ps prepare-then-attach-half path without schedule optimization.\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()
    generate(args.root, args.source, repo=args.repo)
