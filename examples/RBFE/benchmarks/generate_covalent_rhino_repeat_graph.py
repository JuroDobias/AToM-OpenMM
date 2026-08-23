#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import yaml


R_KJ_MOL_K = 0.00831446261815324
DUAL_BRANCH_CORE = (
    "O=C[C@H](C[C@@H]1CCNC1=O)NC(=O)"
    "C1(n2ccccc2=O)Cc2ccccc2C1"
)
ACTIVE_CORE = "NC(=O)C1(n2ccccc2=O)Cc2ccccc2C1"
RESTRAINT_MASK = (
    "(!@H= & !:CVL,HOH,WAT,NA,CL,K,CA) | "
    "(:CVL@N,CA,C,O,CB,SG & !@H=) | "
    f'#active:"{ACTIVE_CORE}"'
)
PAIRED_MAPPINGS = {
    "I79DJ_733--I79DJ_734": {
        "method": "paired_smarts_transmutation",
        "ligand_a_smarts": (
            "[c:1]1([Br:2])[c:3]([CH3:4])[o:5][n:6]"
            "[c:7]1[C:8](=[O:9])[N:10]"
        ),
        "ligand_b_smarts": (
            "[c:1]1([Cl:2])[c:3]([CH3:4])[o:5][n:6]"
            "[c:7]1[C:8](=[O:9])[N:10]"
        ),
    },
    "I79DJ_820--I79DJ_866": {
        "method": "paired_smarts_transmutation",
        "ligand_a_smarts": (
            "[CH2:1][c:2]1[c:3]([#1:4])[c:5]([#1:6])"
            "[c:7]([#1:8])[c:9]([#1:10])[c:11]1[CH2:12]"
        ),
        "ligand_b_smarts": (
            "[CH2:1][c:2]1[c:3]([F:4])[c:5]([#1:6])"
            "[c:7]([#1:8])[c:9]([F:10])[c:11]1[CH2:12]"
        ),
    },
}
DUAL_BRANCH_EDGES = {
    "I79DJ_543--I79DJ_733",
    "I79DJ_543--I79DJ_734",
    "I79DJ_543--I79DJ_820",
    "I79DJ_543--I79DJ_736",
    "I79DJ_736--I79DJ_793",
}


def _mapping(edge_id):
    if edge_id in PAIRED_MAPPINGS:
        return PAIRED_MAPPINGS[edge_id]
    if edge_id in DUAL_BRANCH_EDGES:
        return {"method": "mcs_core_smarts", "smarts": DUAL_BRANCH_CORE}
    return {"method": "dataset_core"}


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
            "mask": RESTRAINT_MASK,
            "k_kcal_mol_a2": 2.0,
            "tolerance_a": 0.5,
        }
    return step


def _workflow(dataset_path, ligand_a, ligand_b, experimental_ddg, seed):
    edge_id = f"{ligand_a}--{ligand_b}"
    return {
        "workflow": {
            "type": "rbfe",
            "chemistry": "covalent",
            "alchemy": {
                "model": "hybrid_topology",
                "cycle": "complex_solvent",
                "dummy_core_nonbonded": "off",
                "mapping": _mapping(edge_id),
            },
            "sampling": {"method": "neqti"},
            "dataset": str(dataset_path),
            "workdir": "run",
            "pairs": [{
                "ligand_a": ligand_a,
                "ligand_b": ligand_b,
                "serotype": "B14",
                "experimental_ddg_kj_per_mol": experimental_ddg,
            }],
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
                                    "mask": RESTRAINT_MASK,
                                    "k_kcal_mol_a2": 5.0,
                                    "tolerance_a": 0.25,
                                },
                            },
                            _md_step("restrained_nvt", "NVT", 50000, 0.001, True),
                            _md_step("restrained_npt", "NPT", 100000, 0.001, True),
                            _md_step("unrestrained_npt", "NPT", 1000000, 0.002, False),
                        ]
                    },
                    "solvent_endpoint": {"mode": "default"},
                }
            },
            "setup": {
                "protein_forcefield": "amber19/protein.ff19SB.xml",
                "water_forcefield": "amber19/opc.xml",
                "ligand_forcefield": "openff-2.2.1.offxml",
                "ligand_charge_model": "espaloma_nn",
                "espaloma_model": "espaloma-0.3.2",
                "solvent_padding_a": 10.0,
                "ionic_strength_molar": 0.15,
                "solvation_seed": int(seed),
                "dummy_bonded_scales": {
                    "bond": 1.0,
                    "angle": 1.0,
                    "proper_torsion": 1.0,
                    "junction_angle": 1.0,
                    "junction_proper_torsion": 1.0,
                },
            },
            "neqti": {
                "temperature_k": 300.0,
                "pressure_bar": 1.0,
                "timestep_fs": 2.0,
                "n_snapshots": 100,
                "decorrelation_steps": 100000,
                "write_switch_pdbs": False,
                "interpolation": "softcore_linear",
                "softcore": {
                    "function": "gapsys",
                    "gapsys_scale_linpoint_lj": 0.85,
                    "gapsys_sigma_nm": 0.3,
                    "total_steps": 50000,
                    "long_range_correction": "endpoint_correction",
                    "path": {
                        "nodes": [0.2, 0.8],
                        "vdw_a": [1.0, 1.0, 0.0, 0.0],
                        "charge_a": [1.0, 0.0, 0.0, 0.0],
                    },
                },
                "schedule_optimization": {
                    "enabled": True,
                    "pilot_samples": 10,
                    "segments_per_interval": [6, 18, 6],
                    "min_segment_steps": 250,
                    "max_segment_steps": 15000,
                },
                "adaptive_switching": {
                    "enabled": True,
                    "candidate_times_ps": [100, 300, 1000],
                    "pilot_samples_per_direction": 20,
                    "min_overlap_score_per_leg": 0.08,
                    "max_failed_fraction_per_direction": 0.05,
                    "reuse_selected_pilot_samples": True,
                    "on_exhausted": "use_longest",
                },
                "convergence": {
                    "enabled": True,
                    "reopen_on_settings_change": True,
                    "min_samples_per_direction": 50,
                    "min_overlap_score_per_leg": 0.05,
                    "max_dg_error_kcal_per_mol": 0.5,
                    "check_interval_samples": 5,
                    "consecutive_checks": 3,
                    "max_dg_range_kcal_per_mol": 0.25,
                    "stationarity": {
                        "enabled": True,
                        "discard_fraction": 0.1,
                        "min_discard_samples": 5,
                        "max_discard_first_shift_kcal_per_mol": 0.3,
                        "max_discard_last_shift_kcal_per_mol": 0.2,
                    },
                },
                "failed_switch_policy": "count_as_infinite",
                "bootstrap_samples": 500,
                "random_seed": int(seed),
                "rest2": {
                    "enabled": True,
                    "execution": "process",
                    "device_indices": [0],
                    "effective_temperatures_k": [
                        300.0, 344.6, 395.9, 454.7, 522.3, 600.0
                    ],
                    "exchange_interval_steps": 500,
                    "checkpoint_interval_cycles": 10,
                },
            },
        }
    }


def _run_script(edge_id, source_dir_name):
    return f'''#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name=cov-{edge_id.replace("I79DJ_", "").replace("I79FC_", "")}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-b|gen-d
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH -t 12:00:00
#SBATCH --signal=B:USR1@600

set -euo pipefail
MAX_CHAIN_JOBS="${{ATOM_MAX_CHAIN_JOBS:-10}}"
CHAIN_INDEX="${{ATOM_CHAIN_INDEX:-0}}"
ROOT_JOB_ID="${{ATOM_ROOT_JOB_ID:-${{SLURM_JOB_ID:-manual}}}}"
RUN_DIR="$(dirname "$(readlink -f "$0")")"
RESULT_FILE="$RUN_DIR/run/{edge_id}/result.yaml"
SOURCE_DIR="$HOME/myAToM/{source_dir_name}"
CHILD_PID=""
RESUBMITTING=0

on_timeout() {{
    if (( RESUBMITTING )); then return; fi
    RESUBMITTING=1
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    if [[ -f "$RESULT_FILE" ]] && grep -q '^status: completed$' "$RESULT_FILE"; then
        exit 0
    fi
    if (( CHAIN_INDEX + 1 < MAX_CHAIN_JOBS )); then
        sbatch --dependency="afterany:${{SLURM_JOB_ID}}" \
            --export=ALL,ATOM_CHAIN_INDEX=$((CHAIN_INDEX + 1)),ATOM_ROOT_JOB_ID="$ROOT_JOB_ID",ATOM_MAX_CHAIN_JOBS="$MAX_CHAIN_JOBS" \
            "$RUN_DIR/run.sh"
    fi
    exit 0
}}
trap on_timeout USR1

cd "$RUN_DIR"
if [[ -f "$RESULT_FILE" ]] && grep -q '^status: completed$' "$RESULT_FILE"; then
    exit 0
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myatom
export PYTHONPATH="$SOURCE_DIR${{PYTHONPATH:+:$PYTHONPATH}}"
git -C "$SOURCE_DIR" rev-parse HEAD > source_commit.txt
python -m atom_openmm.rbfe_workflow --validate workflow.yaml
python -m atom_openmm.rbfe_workflow workflow.yaml &
CHILD_PID=$!
set +e
wait "$CHILD_PID"
status=$?
set -e
CHILD_PID=""
exit "$status"
'''


def _b14_values(assay_path):
    values = {}
    with Path(assay_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["serotype"] == "B14" and row["qualifier"] == "=":
                values[row["ligand_id"]] = float(row["ki_value"])
    return values


def generate(graph_path, dataset_path, runtime_dataset_path, output, source_dir_name):
    graph = yaml.safe_load(Path(graph_path).read_text()) or {}
    dataset_path = Path(dataset_path).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    b14 = _b14_values(dataset_path.parent / "assays.csv")
    edges = []
    temperature = float(
        (yaml.safe_load(dataset_path.read_text()) or {}).get(
            "assay_temperature_k", 298.15
        )
    )
    for index, pair in enumerate(graph["simulated_edges"], 1):
        ligand_a, ligand_b = map(str, pair)
        edge_id = f"{ligand_a}--{ligand_b}"
        ddg_kj = R_KJ_MOL_K * temperature * math.log(
            b14[ligand_b] / b14[ligand_a]
        )
        directory = output / edge_id
        directory.mkdir()
        workflow = _workflow(
            runtime_dataset_path, ligand_a, ligand_b, ddg_kj, 20260823 + index
        )
        (directory / "workflow.yaml").write_text(
            yaml.safe_dump(workflow, sort_keys=False)
        )
        run = directory / "run.sh"
        run.write_text(_run_script(edge_id, source_dir_name))
        run.chmod(0o755)
        edges.append({
            "id": edge_id,
            "ligand_a": ligand_a,
            "ligand_b": ligand_b,
            "directory": edge_id,
        })
    experiment = [
        {
            "ligand_a": edge["ligand_a"],
            "ligand_b": edge["ligand_b"],
            "ddg_kcal_per_mol": R_KJ_MOL_K * temperature
            * math.log(b14[edge["ligand_b"]] / b14[edge["ligand_a"]]) / 4.184,
        }
        for edge in edges
    ]
    network = {
        "schema_version": 2,
        "reference_node": graph["reference_node"],
        "nodes": graph["nodes"],
        "edges": edges,
        "targets": edges,
        "auto_cycles": bool(graph.get("auto_cycles", True)),
        "reference_datasets": {
            "experiment_b14": {
                "reference_node": graph["reference_node"],
                "edges": experiment,
            }
        },
    }
    (output / "network.yaml").write_text(yaml.safe_dump(network, sort_keys=False))
    submit = output / "submit_all.sh"
    submit.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(
            f'(cd "$(dirname "$0")/{edge["directory"]}" && sbatch run.sh)'
            for edge in edges
        )
        + "\n"
    )
    submit.chmod(0o755)
    return network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir-name", default="AToM-OpenMM-unified")
    args = parser.parse_args()
    generate(
        args.graph, args.dataset, args.runtime_dataset, args.output,
        args.source_dir_name,
    )


if __name__ == "__main__":
    main()
