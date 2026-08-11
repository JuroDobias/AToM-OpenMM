#!/usr/bin/env python
"""Generate a shared-node separated-topology CDK2 validation network."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import yaml


NODES = ("1h1q", "1h1s", "21", "32")
EDGES = (
    ("21", "32"),
    ("1h1q", "1h1s"),
    ("1h1s", "32"),
    ("1h1q", "21"),
)
ALIGNMENT_SMARTS = "c1ncc2ncnc2n1"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_ligand(source_cohort, ligand):
    root = Path(source_cohort)
    matches = sorted({
        *root.glob(f"*/alignment_structures/{ligand}-p.sdf"),
        *root.glob(f"*/ligands/{ligand}-p.sdf"),
    })
    if not matches:
        raise FileNotFoundError(f"no aligned benchmark structure found for {ligand}")
    hashes = {_sha256(path) for path in matches}
    if len(hashes) != 1:
        raise ValueError(f"aligned benchmark structures differ for node {ligand}")
    return matches[0]


def _md_step(identifier, ensemble, n_steps, timestep_ps, restrained):
    step = {
        "id": identifier,
        "type": "md",
        "ensemble": ensemble,
        "n_steps": int(n_steps),
        "timestep_ps": float(timestep_ps),
        "thermostat": {"temperature_k": 300.0, "friction_per_ps": 1.0},
        "reporters": {"state": {"interval": 10000}},
    }
    if identifier == "restrained_nvt":
        step["reset_velocities"] = True
    if restrained:
        step["positional_restraints"] = {
            "mask": "!@H= & !:HOH,WAT,NA,CL,K,CA",
            "k_kcal_mol_a2": 2.0,
            "tolerance_a": 0.5,
        }
    return step


def _workflow(pairs, *, node_bank_path, receptor, ligands, workdir):
    return {
        "workflow": {
            "type": "rbfe",
            "chemistry": "noncovalent",
            "alchemy": {
                "model": "separated_topology",
                "cycle": "complex_solvent",
                "node_bank": {
                    "path": node_bank_path,
                    "snapshots": 100,
                    "extensible": True,
                    "extension_max_linear_scale": 1.05,
                    "vacuum_initial_steps": 50000,
                    "vacuum_decorrelation_steps": 100000,
                },
                "frame_restraint": {
                    "translation_k_kcal_mol_a2": 10.0,
                    "orientation_k_kcal_mol": 25.0,
                    "roll_k_kcal_mol": 25.0,
                    "thermalization_steps": 5000,
                },
            },
            "sampling": {"method": "neqti"},
            "workdir": workdir,
            "receptor": receptor,
            "ligands": dict(ligands),
            "pairs": [list(pair) for pair in pairs],
            "alignment": {
                "method": "smarts",
                "smarts": ALIGNMENT_SMARTS,
                "smarts_atom_ids": [2, 5, 7],
            },
            "alignments_out": "separated_alignments.yaml",
            "platform": "CUDA",
            "platform_properties": {"Precision": "mixed"},
            "equilibration": {
                "neqti": {
                    "complex_endpoint": {
                        "steps": [
                            {
                                "id": "restrained_minimization",
                                "type": "minimization",
                                "tolerance_kj_mol_nm": 10.0,
                                "max_iterations": 1000,
                                "positional_restraints": {
                                    "mask": "!@H= & !:HOH,WAT,NA,CL,K,CA",
                                    "k_kcal_mol_a2": 5.0,
                                    "tolerance_a": 0.25,
                                },
                            },
                            _md_step("restrained_nvt", "NVT", 50000, 0.001, True),
                            _md_step("restrained_npt", "NPT", 100000, 0.001, True),
                            _md_step("unrestrained_npt", "NPT", 500000, 0.002, False),
                        ]
                    },
                    "solvent_endpoint": {"mode": "default"},
                }
            },
            "setup": {
                "protein_forcefield": [
                    "openmmforcefields:amber/ff14SB.xml",
                    "openmmforcefields:amber/phosaa14SB.xml",
                ],
                "solvent_forcefield": [
                    "openmmforcefields:amber/tip3p_standard.xml"
                ],
                "solvent_model": "tip3p",
                "solvent_box_shape": "dodecahedron",
                "solvent_padding_a": 10.0,
                "ionic_strength_molar": 0.15,
                "ligand_forcefield": "espaloma-0.3.2",
                "ligand_charge_model": "nn",
            },
            "neqti": {
                "temperature_k": 300.0,
                "pressure_bar": 1.0,
                "timestep_fs": 2.0,
                "endpoint_equilibration": {
                    "minimization_tolerance_kj_mol_nm": 10.0,
                    "minimization_max_iterations": 1000,
                    "nvt_steps": 100000,
                    "nvt_timestep_fs": 1.0,
                    "npt_steps": 250000,
                    "npt_timestep_fs": 2.0,
                },
                "n_snapshots": 100,
                "decorrelation_steps": 100000,
                "interpolation": "softcore_linear",
                "softcore": {
                    "function": "amber_ssc2",
                    "coulomb_function": "amber_ssc2",
                    "stage_interpolation": "linear",
                    "ssc2_alpha_lj": 0.5,
                    "ssc2_beta_coul": 1.0,
                    "ssc2_switch_width_nm": 0.2,
                    "total_steps": 50000,
                    "path": {"mode": "concerted"},
                    "long_range_correction": "dynamic",
                },
                "schedule_optimization": {
                    "enabled": True,
                    "pilot_samples": 20,
                    "segments_per_interval": [20],
                    "min_segment_steps": 250,
                    "max_segment_steps": 15000,
                },
                "adaptive_switching": {
                    "enabled": True,
                    "candidate_times_ps": [100.0, 300.0, 1000.0],
                    "pilot_samples_per_direction": 20,
                    "min_overlap_score_per_leg": 0.08,
                    "max_failed_fraction_per_direction": 0.05,
                    "reuse_selected_pilot_samples": True,
                    "on_exhausted": "use_longest",
                },
                "failed_switch_policy": "count_as_infinite",
                "bootstrap_samples": 500,
                "random_seed": 2026,
                "rest2": {
                    "enabled": True,
                    "effective_temperatures_k": [300.0, 356.8, 424.3, 504.5, 600.0],
                    "exchange_interval_steps": 500,
                    "checkpoint_interval_cycles": 10,
                    "execution": "serial",
                },
            },
        }
    }


def _slurm_script(
    command, job_name, completion_file, source_dir_name, script_name="run.sh"
):
    return f"""#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name={job_name}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-b|gen-d
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH -t 12:00:00
#SBATCH --signal=B:USR1@600

set -euo pipefail
RUN_DIR="${{SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}}"
SCRIPT="$RUN_DIR/{script_name}"
CHAIN_INDEX="${{ATOM_CHAIN_INDEX:-0}}"
CHILD=""
on_timeout() {{
    [[ -z "$CHILD" ]] || kill -TERM "$CHILD" 2>/dev/null || true
    [[ -f "$RUN_DIR/{completion_file}" ]] || (( CHAIN_INDEX >= 9 )) || \
        sbatch --dependency="afterany:${{SLURM_JOB_ID}}" \
        --export=ALL,ATOM_CHAIN_INDEX=$((CHAIN_INDEX + 1)) "$SCRIPT"
    exit 0
}}
trap on_timeout USR1
cd "$RUN_DIR"
[[ -f "{completion_file}" ]] && exit 0
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myatom
export PYTHONPATH="$HOME/myAToM/{source_dir_name}${{PYTHONPATH:+:$PYTHONPATH}}"
export LD_LIBRARY_PATH="$HOME/myAToM/openmm-build-env/lib:$HOME/myAToM/openmm-endpoint-gates-install/lib:${{LD_LIBRARY_PATH:-}}"
export OPENMM_PLUGIN_DIR="$HOME/myAToM/openmm-endpoint-gates-install/lib/plugins"
PYTHON_BIN="${{ATOM_PYTHON:-$HOME/myAToM/atm-gates-venv/bin/python}}"
{command} &
CHILD=$!
wait "$CHILD"
"""


def _runtime(source_dir_name):
    return f"""source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myatom
export PYTHONPATH="$HOME/myAToM/{source_dir_name}${{PYTHONPATH:+:$PYTHONPATH}}"
export LD_LIBRARY_PATH="$HOME/myAToM/openmm-build-env/lib:$HOME/myAToM/openmm-endpoint-gates-install/lib:${{LD_LIBRARY_PATH:-}}"
export OPENMM_PLUGIN_DIR="$HOME/myAToM/openmm-endpoint-gates-install/lib/plugins"
PYTHON_BIN="${{ATOM_PYTHON:-$HOME/myAToM/atm-gates-venv/bin/python}}"
"""


def _bank_initialize_script(source_dir_name):
    return f"""#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name=sep-cdk2-init
#SBATCH --output=slurm-init-%j.out
#SBATCH --error=slurm-init-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH -t 01:00:00

set -euo pipefail
cd "${{SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}}"
{_runtime(source_dir_name)}
"$PYTHON_BIN" -m atom_openmm.rbfe_workflow \
    --initialize-node-bank prepare_nodes.yaml
"""


def _bank_node_script(source_dir_name, node):
    return f"""#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name=sep-node-{node}
#SBATCH --output=slurm-node-{node}-%j.out
#SBATCH --error=slurm-node-{node}-%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-b|gen-d
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH -t 12:00:00
#SBATCH --signal=B:USR1@600

set -euo pipefail
RUN_DIR="${{SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}}"
SCRIPT="$RUN_DIR/prepare_node_{node}.sh"
CHAIN_INDEX="${{ATOM_CHAIN_INDEX:-0}}"
CHILD=""
on_timeout() {{
    [[ -z "$CHILD" ]] || kill -TERM "$CHILD" 2>/dev/null || true
    [[ -f "$RUN_DIR/.node_bank.building/nodes/{node}/manifest.yaml" ]] || \
        (( CHAIN_INDEX >= 19 )) || sbatch --dependency="afterany:${{SLURM_JOB_ID}}" \
        --export=ALL,ATOM_CHAIN_INDEX=$((CHAIN_INDEX + 1)) "$SCRIPT"
    exit 0
}}
trap on_timeout USR1
cd "$RUN_DIR"
NODE="{node}"
COMPLETION=".node_bank.building/nodes/{node}/manifest.yaml"
[[ -f "$COMPLETION" ]] && exit 0
{_runtime(source_dir_name)}
"$PYTHON_BIN" -m atom_openmm.rbfe_workflow \
    --prepare-node "$NODE" prepare_nodes.yaml &
CHILD=$!
wait "$CHILD"
"""


def _bank_finalize_script(source_dir_name):
    return f"""#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name=sep-cdk2-finalize
#SBATCH --output=slurm-finalize-%j.out
#SBATCH --error=slurm-finalize-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 01:00:00

set -euo pipefail
RUN_DIR="${{SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}}"
cd "$RUN_DIR"
[[ -f node_bank/manifest.yaml ]] && exit 0
if [[ ! -f .node_bank.building/nodes/1h1q/manifest.yaml || \
      ! -f .node_bank.building/nodes/1h1s/manifest.yaml || \
      ! -f .node_bank.building/nodes/21/manifest.yaml || \
      ! -f .node_bank.building/nodes/32/manifest.yaml ]]; then
    sbatch --begin=now+20minutes "$RUN_DIR/finalize_nodes.sh"
    exit 0
fi
{_runtime(source_dir_name)}
"$PYTHON_BIN" -m atom_openmm.rbfe_workflow \
    --finalize-node-bank prepare_nodes.yaml
"""


def generate(source_cohort, benchmark_root, output, source_dir_name):
    source_cohort = Path(source_cohort).resolve()
    benchmark_root = Path(benchmark_root).resolve()
    output = Path(output).resolve()
    receptor = benchmark_root / "ATM_Validation/CDK2/receptor/CDK2_new_2_edit.pdb"
    if not receptor.is_file():
        raise FileNotFoundError(receptor)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(receptor, output / "receptor.pdb")
    ligands_dir = output / "ligands"
    ligands_dir.mkdir()
    for node in NODES:
        shutil.copy2(_source_ligand(source_cohort, node), ligands_dir / f"{node}-p.sdf")

    root_ligands = {node: f"ligands/{node}-p.sdf" for node in NODES}
    bank_workflow = _workflow(
        EDGES,
        node_bank_path="node_bank",
        receptor="receptor.pdb",
        ligands=root_ligands,
        workdir="node_bank_work",
    )
    (output / "prepare_nodes.yaml").write_text(
        yaml.safe_dump(bank_workflow, sort_keys=False)
    )
    bank_scripts = {
        "initialize_nodes.sh": _bank_initialize_script(source_dir_name),
        "finalize_nodes.sh": _bank_finalize_script(source_dir_name),
    }
    bank_scripts.update({
        f"prepare_node_{node}.sh": _bank_node_script(source_dir_name, node)
        for node in NODES
    })
    for name, contents in bank_scripts.items():
        script = output / name
        script.write_text(contents)
        script.chmod(0o755)

    for ligand_a, ligand_b in EDGES:
        edge = f"{ligand_a}--{ligand_b}"
        directory = output / edge
        directory.mkdir()
        edge_ligands = {
            ligand_a: f"../ligands/{ligand_a}-p.sdf",
            ligand_b: f"../ligands/{ligand_b}-p.sdf",
        }
        workflow = _workflow(
            [(ligand_a, ligand_b)],
            node_bank_path="../node_bank",
            receptor="../receptor.pdb",
            ligands=edge_ligands,
            workdir="run",
        )
        (directory / "workflow.yaml").write_text(
            yaml.safe_dump(workflow, sort_keys=False)
        )
        run_script = directory / "run.sh"
        result = f"run/receptor-{ligand_a}-{ligand_b}/result.yaml"
        run_script.write_text(_slurm_script(
            '"$PYTHON_BIN" -m atom_openmm.rbfe_workflow workflow.yaml',
            f"sep-{edge}",
            result,
            source_dir_name,
        ))
        run_script.chmod(0o755)

    network = {
        "schema_version": 1,
        "reference_node": "1h1q",
        "target": ["1h1q", "32"],
        "temperature_k": 300.0,
        "node_bootstrap_samples": 500,
        "random_seed": 2026,
        "nodes": list(NODES),
        "edges": [
            {"ligand_a": a, "ligand_b": b, "directory": f"{a}--{b}"}
            for a, b in EDGES
        ],
        "cycles": [{
            "id": "sulfonamide_additivity",
            "terms": [
                {"edge": "1h1q--1h1s", "coefficient": 1},
                {"edge": "1h1s--32", "coefficient": 1},
                {"edge": "21--32", "coefficient": -1},
                {"edge": "1h1q--21", "coefficient": -1},
            ],
        }],
    }
    (output / "network.yaml").write_text(yaml.safe_dump(network, sort_keys=False))
    scripts = {
        "submit_nodes.sh": (
            'init=$(sbatch --parsable initialize_nodes.sh)\n'
            'init=${init%%;*}\n'
            'node_ids=()\n'
            + "".join(
                f'job=$(sbatch --parsable --dependency="afterok:$init" prepare_node_{node}.sh)\n'
                'node_ids+=("${job%%;*}")\n'
                for node in NODES
            )
            + 'deps=$(IFS=:; echo "${node_ids[*]}")\n'
            'final=$(sbatch --parsable --dependency="afterok:$deps" finalize_nodes.sh)\n'
            'final=${final%%;*}\n'
            'printf "initialize=%s nodes=%s finalize=%s\\n" "$init" "$deps" "$final"\n'
        ),
        "submit_pilot.sh": '(cd 21--32 && sbatch run.sh)\n',
        "submit_cycle.sh": "\n".join(
            f"(cd {a}--{b} && sbatch run.sh)"
            for a, b in EDGES if (a, b) != ("21", "32")
        ) + "\n",
    }
    for name, commands in scripts.items():
        script = output / name
        guard = ""
        if name != "submit_nodes.sh":
            guard = (
                "[[ -f node_bank/manifest.yaml ]] || "
                "{ echo 'node bank is not complete' >&2; exit 1; }\n"
            )
        script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            'cd "$(dirname "$0")"\n' + guard + commands
        )
        script.chmod(0o755)
    (output / "README.md").write_text(
        "# CDK2 separated-topology sulfonamide cycle\n\n"
        "The physical complex, solvent, and vacuum ensembles are prepared once per "
        "node and reused by every edge. `21--32` is the first pilot edge. The four "
        "edges close the sulfonamide additivity cycle. Each edge uses 100 paired node "
        "snapshots. The first 20 optimize the SSC2 schedule and probe adaptive "
        "100/300/1000 ps switching; the selected frozen protocol then produces the "
        "100-sample BAR dataset.\n\n"
        "Run `./submit_nodes.sh` to initialize common solvent metadata, prepare the "
        "four nodes as resumable one-GPU jobs, and atomically finalize the "
        "bank. Re-running the script resumes incomplete shards. After it completes, run "
        "`./submit_pilot.sh`. Submit the other three edges with `./submit_cycle.sh` "
        "after accepting the pilot diagnostics. The bank can also be prepared with "
        "`atom-rbfe --prepare-node-bank prepare_nodes.yaml`. Analyze completed edges "
        "with `python -m atom_openmm.rbfe_network network.yaml`.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cohort", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dir-name", default="AToM-OpenMM-separated-topology"
    )
    args = parser.parse_args()
    generate(
        args.source_cohort,
        args.benchmark_root,
        args.output,
        args.source_dir_name,
    )


if __name__ == "__main__":
    main()
