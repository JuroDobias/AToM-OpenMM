#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

try:
    from .generate_hybrid_cdk2_cohort import _run_script
except ImportError:
    from generate_hybrid_cdk2_cohort import _run_script


def geometric_temperatures(minimum_k, maximum_k, replicas):
    minimum_k = float(minimum_k)
    maximum_k = float(maximum_k)
    replicas = int(replicas)
    if minimum_k <= 0.0 or maximum_k <= minimum_k or replicas < 2:
        raise ValueError(
            "temperature ladder requires 0 < minimum < maximum and replicas >= 2"
        )
    ratio = (maximum_k / minimum_k) ** (1.0 / (replicas - 1))
    return [
        round(minimum_k * ratio**index, 3)
        for index in range(replicas)
    ]


def generate(
    source_edge,
    output,
    source_dir_name,
    maximum_k,
    replicas,
    junction_proper_torsion_scale,
    junction_rotatable_torsion_scale=None,
    internal_rotatable_torsion_scale=None,
):
    source_edge = Path(source_edge).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(source_edge / "receptor.pdb", output / "receptor.pdb")
    shutil.copytree(source_edge / "ligands", output / "ligands")
    workflow = yaml.safe_load((source_edge / "workflow.yaml").read_text())
    neqti = workflow["workflow"]["neqti"]
    setup = workflow["workflow"]["setup"]
    neqti["n_snapshots"] = 20
    neqti.pop("adaptive_switching", None)
    neqti.pop("convergence", None)
    temperatures = geometric_temperatures(300.0, maximum_k, replicas)
    neqti["rest2"]["effective_temperatures_k"] = temperatures
    neqti["rest2"]["coordinate_reporter"] = {
        "enabled": True,
        "interval_cycles": 25,
        "state_indices": "all",
    }
    neqti["random_seed"] = int(neqti.get("random_seed", 2026)) + 1000
    setup.setdefault("dummy_bonded_scales", {})[
        "junction_proper_torsion"
    ] = float(junction_proper_torsion_scale)
    if junction_rotatable_torsion_scale is not None:
        setup["dummy_bonded_scales"]["junction_rotatable_torsion"] = float(
            junction_rotatable_torsion_scale
        )
    if internal_rotatable_torsion_scale is not None:
        setup["dummy_bonded_scales"]["internal_rotatable_torsion"] = float(
            internal_rotatable_torsion_scale
        )
    (output / "workflow.yaml").write_text(
        yaml.safe_dump(workflow, sort_keys=False)
    )
    run = output / "run.sh"
    run.write_text(_run_script("1oiy--32", source_dir_name=source_dir_name))
    run.chmod(0o755)
    (output / "README.md").write_text(
        "# 1oiy -> 32 high-temperature REST2 diagnostic\n\n"
        f"Geometric REST2 ladder: {temperatures} K ({replicas} replicas). "
        "Inactive junction proper-torsion scale: "
        f"{junction_proper_torsion_scale}. "
        "Selective junction/internal rotatable-torsion scales: "
        f"{junction_rotatable_torsion_scale}/"
        f"{internal_rotatable_torsion_scale}. "
        "Coordinates for every thermodynamic state are written every 25 exchange "
        "cycles, independently of the 20 physical states used for fixed 100 ps "
        "switches. This run starts from a clean workdir.\n"
    )
    return temperatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-edge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir-name", required=True)
    parser.add_argument("--maximum-k", type=float, default=1000.0)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument(
        "--junction-proper-torsion-scale", type=float, default=1.0
    )
    parser.add_argument("--junction-rotatable-torsion-scale", type=float)
    parser.add_argument("--internal-rotatable-torsion-scale", type=float)
    args = parser.parse_args()
    temperatures = generate(
        args.source_edge,
        args.output,
        args.source_dir_name,
        args.maximum_k,
        args.replicas,
        args.junction_proper_torsion_scale,
        args.junction_rotatable_torsion_scale,
        args.internal_rotatable_torsion_scale,
    )
    print("REST2 temperatures (K):", temperatures)


if __name__ == "__main__":
    main()
