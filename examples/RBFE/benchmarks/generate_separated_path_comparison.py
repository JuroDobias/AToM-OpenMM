#!/usr/bin/env python
"""Generate matched separated-topology switching-path pilots."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


PATHS = {
    "ssc2_full_vdw_midpoint": {
        "function": "amber_ssc2",
        "coulomb_function": "linear_pme",
        "stage_interpolation": "linear",
        "ssc2_alpha_lj": 0.5,
        "ssc2_switch_width_nm": 0.2,
        "total_steps": 50000,
        "path": {
            "nodes": [0.5],
            "vdw_a": [1.0, 1.0, 0.0],
            "charge_a": [1.0, 0.5, 0.0],
        },
        "long_range_correction": "dynamic",
    },
    "ssc2_linear_vdw_charge": {
        "function": "amber_ssc2",
        "coulomb_function": "linear_pme",
        "stage_interpolation": "linear",
        "ssc2_alpha_lj": 0.5,
        "ssc2_switch_width_nm": 0.2,
        "total_steps": 50000,
        "path": {
            "nodes": [0.5],
            "vdw_a": [1.0, 0.5, 0.0],
            "charge_a": [1.0, 0.5, 0.0],
        },
        "long_range_correction": "dynamic",
    },
    "beutler_charged_handoff": {
        "function": "beutler",
        "coulomb_function": "linear_pme",
        "stage_interpolation": "linear",
        "alpha": 0.3,
        "sigma_nm": 0.25,
        "power": 1,
        "total_steps": 50000,
        "path": {
            "nodes": [0.3, 0.7],
            "vdw_a": [1.0, 1.0, 1.0, 0.0],
            "charge_a": [1.0, 1.0, 0.0, 0.0],
        },
        "long_range_correction": "dynamic",
    },
}


def _resolve_input(path, base):
    path = Path(path)
    return str(path if path.is_absolute() else (base / path).resolve())


def _run_script(name, source_dir_name):
    return f"""#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name=sep-path-{name[:12]}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-b|gen-d
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH -t 12:00:00

set -euo pipefail
cd "${{SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}}"
[[ -f run/receptor-21-32/result.yaml ]] && exit 0
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myatom
export PYTHONPATH="$HOME/myAToM/{source_dir_name}${{PYTHONPATH:+:$PYTHONPATH}}"
export LD_LIBRARY_PATH="$HOME/myAToM/openmm-build-env/lib:$HOME/myAToM/openmm-endpoint-gates-install/lib:${{LD_LIBRARY_PATH:-}}"
export OPENMM_PLUGIN_DIR="$HOME/myAToM/openmm-endpoint-gates-install/lib/plugins"
PYTHON_BIN="${{ATOM_PYTHON:-$HOME/myAToM/atm-gates-venv/bin/python}}"
"$PYTHON_BIN" -m atom_openmm.rbfe_workflow workflow.yaml
"""


def generate(base_workflow, output, source_dir_name):
    base_workflow = Path(base_workflow).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = yaml.safe_load(base_workflow.read_text())
    workflow = payload["workflow"]
    base = base_workflow.parent
    workflow["receptor"] = _resolve_input(workflow["receptor"], base)
    workflow["ligands"] = {
        name: _resolve_input(path, base)
        for name, path in workflow["ligands"].items()
    }
    bank = workflow["alchemy"]["node_bank"]
    bank["path"] = _resolve_input(bank["path"], base)

    for name, softcore in PATHS.items():
        directory = output / name
        directory.mkdir()
        variant = copy.deepcopy(payload)
        variant_workflow = variant["workflow"]
        variant_workflow["workdir"] = "run"
        neqti = variant_workflow["neqti"]
        neqti["n_snapshots"] = 20
        neqti["softcore"] = copy.deepcopy(softcore)
        neqti["adaptive_switching"] = {"enabled": False}
        neqti["random_seed"] = 2026
        (directory / "workflow.yaml").write_text(
            yaml.safe_dump(variant, sort_keys=False)
        )
        script = directory / "run.sh"
        script.write_text(_run_script(name, source_dir_name))
        script.chmod(0o755)

    submit = output / "submit.sh"
    submit.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'cd "$(dirname "$0")"\n'
        + "\n".join(
            f"(cd {name} && sbatch run.sh)" for name in PATHS
        )
        + "\n"
    )
    submit.chmod(0o755)
    (output / "README.md").write_text(
        "# Separated-topology path comparison\n\n"
        "All three 21--32 pilots reuse the same physical node bank, snapshot "
        "permutation, random seed, 20 samples per direction, and 100 ps switch "
        "duration. Adaptive duration selection is disabled. The variants compare "
        "Amber SSC2 LJ with a full-VDW midpoint, Amber SSC2 LJ with linear VDWs, "
        "and a Beutler charged handoff. Sample 1 writes start, path-node, and end "
        "PDBs plus three-anchor RMSDs for both directions and environments.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-workflow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dir-name", default="AToM-OpenMM-separated-topology"
    )
    args = parser.parse_args()
    generate(args.base_workflow, args.output, args.source_dir_name)


if __name__ == "__main__":
    main()
