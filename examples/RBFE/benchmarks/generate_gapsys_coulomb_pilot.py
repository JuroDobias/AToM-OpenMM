#!/usr/bin/env python
"""Generate matched Gapsys Coulomb switch replays for two CDK2 edges."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


EDGES = ("17--21", "1oiy--32")
PROTOCOLS = {
    "gapsys_concerted_100ps": {
        "function": "gapsys",
        "coulomb_function": "gapsys",
        "gapsys_scale_linpoint_lj": 0.85,
        "gapsys_scale_linpoint_q": 0.30,
        "gapsys_sigma_nm": 0.30,
        "long_range_correction": "endpoint_correction",
        "total_steps": 50000,
        "path": {"mode": "concerted"},
    },
    "gapsys_full_vdw_midpoint_100ps": {
        "function": "gapsys",
        "coulomb_function": "gapsys",
        "gapsys_scale_linpoint_lj": 0.85,
        "gapsys_scale_linpoint_q": 0.30,
        "gapsys_sigma_nm": 0.30,
        "long_range_correction": "endpoint_correction",
        "total_steps": 50000,
        "path": {
            "nodes": [0.5],
            "vdw_a": [1.0, 1.0, 0.0],
            "charge_a": [1.0, 0.5, 0.0],
        },
    },
}


def _run_script(name, source_dir_name):
    return f"""#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name={name[:28]}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-b|gen-d
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH -t 12:00:00

set -euo pipefail
cd "${{SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}}"
[[ -f results/result.yaml ]] && exit 0
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myatom
export LD_LIBRARY_PATH="$HOME/myAToM/openmm-build-env/lib:$HOME/myAToM/openmm-endpoint-gates-install/lib:${{LD_LIBRARY_PATH:-}}"
export OPENMM_PLUGIN_DIR="$HOME/myAToM/openmm-endpoint-gates-install/lib/plugins"
export PYTHONPATH="$HOME/myAToM/{source_dir_name}${{PYTHONPATH:+:$PYTHONPATH}}"
python -m atom_openmm.hybrid_switch_benchmark run benchmark.yaml
"""


def generate(snapshot_bank, output, source_dir_name):
    snapshot_bank = Path(snapshot_bank)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for edge in EDGES:
        for protocol_name, softcore in PROTOCOLS.items():
            name = f"{edge}_{protocol_name}"
            directory = output / name
            directory.mkdir()
            payload = {
                "snapshot_bank": str(snapshot_bank),
                "output_dir": "results",
                "edges": [edge],
                "environments": ["complex", "solvent"],
                "n_snapshots": 20,
                "temperature_k": 300.0,
                "timestep_fs": 2.0,
                "bootstrap_samples": 500,
                "random_seed": 20260810,
                "platform": "CUDA",
                "protocols": [
                    {
                        "name": protocol_name,
                        "stage_interpolation": "linear",
                        "duration_ps": 100.0,
                        "softcore": softcore,
                    }
                ],
            }
            (directory / "benchmark.yaml").write_text(
                yaml.safe_dump(payload, sort_keys=False)
            )
            script = directory / "run.sh"
            script.write_text(_run_script(name, source_dir_name))
            script.chmod(0o755)
            jobs.append(name)
    submit = output / "submit.sh"
    submit.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'cd "$(dirname "$0")"\n'
        + "\n".join(f'(cd "{name}" && sbatch run.sh)' for name in jobs)
        + "\n"
    )
    submit.chmod(0o755)
    (output / "README.md").write_text(
        "# Gapsys Coulomb CDK2 pilot\n\n"
        "Four matched jobs replay the same 20 endpoint snapshots for 17--21 and "
        "1oiy--32. Each edge compares a 100 ps concerted Gapsys charge/LJ path "
        "with a 100 ps path that keeps both vdW branches fully coupled at the "
        "midpoint. Adaptive switching and schedule optimization are absent from "
        "the replay benchmark.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source-dir-name", default="AToM-OpenMM-separated-topology"
    )
    args = parser.parse_args()
    generate(args.snapshot_bank, args.output, args.source_dir_name)


if __name__ == "__main__":
    main()
