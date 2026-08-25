#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


EDGES = (
    ("17", "21"),
    ("17", "22"),
    ("1h1r", "1oi9"),
    ("1h1s", "1oiy"),
    ("1oiu", "26"),
    ("1oiy", "29"),
    ("1oiy", "32"),
    ("20", "1h1q"),
    ("28", "26"),
    ("30", "31"),
)
MAPPING_SMARTS = "c1cc(Nc2nc3c(ncn3)c(OCC3CCCCC3)n2)ccc1"
ACTIVE_SMARTS = "c1ncc2ncnc2n1"
RESTRAINT_MASK = (
    f'(!@H= & !:HYB,HOH,WAT,NA,CL,K,CA) | #active:"{ACTIVE_SMARTS}"'
)


def _md_step(identifier, ensemble, n_steps, timestep_ps, restrained):
    step = {
        "id": identifier,
        "type": "md",
        "ensemble": ensemble,
        "n_steps": n_steps,
        "timestep_ps": timestep_ps,
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


def _workflow(
    ligand_a,
    ligand_b,
    *,
    n_snapshots=100,
    switch_time_ps=100.0,
    adaptive_switching=True,
    convergence=True,
    dummy_core_nonbonded="off",
    schedule_optimization=False,
    unrestrained_npt_steps=500000,
    random_seed=2026,
):
    total_switch_steps = int(round(float(switch_time_ps) * 1000.0 / 2.0))
    charge_steps = total_switch_steps // 4
    sterics_steps = total_switch_steps - 2 * charge_steps
    payload = {
        "workflow": {
            "type": "rbfe",
            "chemistry": "noncovalent",
            "alchemy": {
                "model": "hybrid_topology",
                "cycle": "complex_solvent",
                "dummy_core_nonbonded": str(dummy_core_nonbonded),
                "mapping": {
                    "method": "mcs_core_smarts",
                    "smarts": MAPPING_SMARTS,
                },
            },
            "sampling": {"method": "neqti"},
            "workdir": "run",
            "receptor": "receptor.pdb",
            "ligands": {
                ligand_a: f"ligands/{ligand_a}-p.sdf",
                ligand_b: f"ligands/{ligand_b}-p.sdf",
            },
            "pairs": [[ligand_a, ligand_b]],
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
                            _md_step(
                                "unrestrained_npt", "NPT",
                                int(unrestrained_npt_steps), 0.002, False,
                            ),
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
                "solvent_box_shape": "rectangular",
                "ligand_forcefield": "espaloma-0.3.2",
                "ligand_charge_model": "nn",
                "solvent_padding_a": 10.0,
                "ionic_strength_molar": 0.15,
                "dummy_bonded_scales": {
                    "bond": 1.0,
                    "angle": 1.0,
                    "proper_torsion": 1.0,
                    "junction_angle": 1.0,
                    "junction_proper_torsion": 1.0,
                    "junction_rotatable_torsion": 1.0,
                    "internal_rotatable_torsion": 1.0,
                },
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
                "n_snapshots": int(n_snapshots),
                "decorrelation_steps": 50000,
                "interpolation": "softcore_linear",
                "softcore": {
                    "function": "beutler",
                    "alpha": 0.3,
                    "sigma_nm": 0.25,
                    "power": 1,
                    "charge_steps_per_stage": charge_steps,
                    "sterics_steps": sterics_steps,
                    "long_range_correction": "dynamic",
                },
                "failed_switch_policy": "count_as_infinite",
                "bootstrap_samples": 500,
                "random_seed": int(random_seed),
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
    if adaptive_switching:
        payload["workflow"]["neqti"]["adaptive_switching"] = {
            "enabled": True,
            "candidate_times_ps": [100, 300, 1000],
            "pilot_samples_per_direction": 20,
            "min_overlap_score_per_leg": 0.08,
            "max_failed_fraction_per_direction": 0.05,
            "reuse_selected_pilot_samples": True,
            "on_exhausted": "use_longest",
        }
    if schedule_optimization:
        payload["workflow"]["neqti"]["schedule_optimization"] = {
            "enabled": True,
            "pilot_samples": 10,
            "subdivisions_per_stage": 10,
            "min_segment_steps": 250,
            "max_segment_steps": 15000,
        }
    if convergence:
        payload["workflow"]["neqti"]["convergence"] = {
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
        }
    if {ligand_a, ligand_b} & {"30", "31"}:
        payload["workflow"]["setup"]["allow_undefined_stereo"] = True
    return payload


def _run_script(edge, source_dir_name="AToM-OpenMM-unified"):
    return f"""#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name=hyb-{edge}
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
FORCE_REOPEN="${{ATOM_FORCE_REOPEN:-0}}"
RUN_DIR="$(dirname "$(readlink -f "$0")")"
SCRIPT_PATH="$RUN_DIR/run.sh"
CHAIN_LOG="$RUN_DIR/slurm-chain.log"
SOURCE_DIR="$HOME/myAToM/{source_dir_name}"
RESULT_FILE="$RUN_DIR/run/receptor-{edge.replace('--', '-')}/result.yaml"
CHILD_PID=""
RESUBMITTING=0

log_chain() {{
    printf '%s root=%s job=%s chain_index=%s %s\n' \
        "$(date --iso-8601=seconds)" "$ROOT_JOB_ID" \
        "${{SLURM_JOB_ID:-manual}}" "$CHAIN_INDEX" "$*" >> "$CHAIN_LOG"
}}

result_completed() {{
    [[ -f "$RESULT_FILE" ]] && grep -q '^status: completed$' "$RESULT_FILE"
}}

on_timeout() {{
    if (( RESUBMITTING )); then return; fi
    RESUBMITTING=1
    log_chain "received pre-timeout USR1"
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    if result_completed; then
        log_chain "workflow completed during shutdown; no successor submitted"
    elif (( CHAIN_INDEX + 1 >= MAX_CHAIN_JOBS )); then
        log_chain "automatic resubmission cap reached; no successor submitted"
    else
        submission="$(sbatch --dependency="afterany:${{SLURM_JOB_ID}}" \
            --export=ALL,ATOM_CHAIN_INDEX=$((CHAIN_INDEX + 1)),ATOM_ROOT_JOB_ID="$ROOT_JOB_ID",ATOM_MAX_CHAIN_JOBS="$MAX_CHAIN_JOBS" \
            "$SCRIPT_PATH")"
        log_chain "submitted successor: $submission"
    fi
    exit 0
}}

trap on_timeout USR1
cd "$RUN_DIR"
log_chain "started"
if result_completed && [[ "$FORCE_REOPEN" != "1" ]]; then
    log_chain "workflow already completed; exiting"
    exit 0
fi
if [[ "$FORCE_REOPEN" == "1" ]]; then
    log_chain "forcing convergence reopen with existing production samples"
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myatom
export PYTHONPATH="$SOURCE_DIR${{PYTHONPATH:+:$PYTHONPATH}}"
export LD_LIBRARY_PATH="$HOME/myAToM/openmm-build-env/lib:$HOME/myAToM/openmm-endpoint-gates-install/lib:${{LD_LIBRARY_PATH:-}}"
export OPENMM_PLUGIN_DIR="$HOME/myAToM/openmm-endpoint-gates-install/lib/plugins"
PYTHON_BIN="$HOME/myAToM/atm-gates-venv/bin/python"

git -C "$SOURCE_DIR" rev-parse HEAD > source_commit.txt
git -C "$SOURCE_DIR" diff --binary | sha256sum > source_worktree_diff.sha256
"$PYTHON_BIN" -m atom_openmm.rbfe_workflow --validate workflow.yaml
"$PYTHON_BIN" -m atom_openmm.rbfe_workflow workflow.yaml &
CHILD_PID=$!
set +e
wait "$CHILD_PID"
status=$?
set -e
CHILD_PID=""
if (( status == 0 )); then
    log_chain "workflow exited successfully"
else
    log_chain "workflow failed with exit code $status; no successor submitted"
fi
exit "$status"
"""


def generate(
    source_cohort: Path,
    benchmark_root: Path,
    output: Path,
    *,
    n_snapshots=100,
    switch_time_ps=100.0,
    adaptive_switching=True,
    convergence=True,
    dummy_core_nonbonded="off",
    source_dir_name="AToM-OpenMM-unified",
):
    receptor = benchmark_root / "ATM_Validation/CDK2/receptor/CDK2_new_2_edit.pdb"
    if not receptor.is_file():
        raise FileNotFoundError(receptor)
    output.mkdir(parents=True, exist_ok=False)
    for ligand_a, ligand_b in EDGES:
        edge = f"{ligand_a}--{ligand_b}"
        source = source_cohort / edge / "alignment_structures"
        target = output / edge
        (target / "ligands").mkdir(parents=True)
        shutil.copy2(receptor, target / "receptor.pdb")
        for ligand in (ligand_a, ligand_b):
            shutil.copy2(source / f"{ligand}-p.sdf", target / "ligands")
        (target / "workflow.yaml").write_text(
            yaml.safe_dump(
                _workflow(
                    ligand_a,
                    ligand_b,
                    n_snapshots=n_snapshots,
                    switch_time_ps=switch_time_ps,
                    adaptive_switching=adaptive_switching,
                    convergence=convergence,
                    dummy_core_nonbonded=dummy_core_nonbonded,
                ),
                sort_keys=False,
            )
        )
        script = target / "run.sh"
        script.write_text(_run_script(edge, source_dir_name=source_dir_name))
        script.chmod(0o755)
    submit = output / "submit_all.sh"
    submit.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(f"(cd {a}--{b} && sbatch run.sh)" for a, b in EDGES)
        + "\n"
    )
    submit.chmod(0o755)
    archive = output / "archive_snapshot_bank.sh"
    archive.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "if [[ $# -ne 1 ]]; then echo 'usage: archive_snapshot_bank.sh OUTPUT_DIR' >&2; exit 2; fi\n"
        f'python -m atom_openmm.hybrid_switch_benchmark archive "$(dirname "$0")" "$1" --snapshots {int(n_snapshots)}\n'
    )
    archive.chmod(0o755)
    mode = "adaptive 100/300/1000 ps" if adaptive_switching else f"fixed {float(switch_time_ps):g} ps"
    stopping = "automatic BAR convergence stopping" if convergence else "a fixed sample count"
    (output / "README.md").write_text(
        "# CDK2 hybrid-topology cohort\n\n"
        "Ten published CDK2 benchmark edges are repeated with a noncovalent "
        "hybrid topology, Espaloma NN ligand parameters, REST2 endpoint "
        f"sampling, {mode} staged-linear NEQTI switches, and {stopping}. "
        f"Each endpoint produces {int(n_snapshots)} decorrelated snapshots. "
        f"Inactive dummy-core interactions use `{dummy_core_nonbonded}` mode. "
        "The exact benchmark receptor "
        "is retained and TPO161 is parameterized with ff14SB/phosaa14SB.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cohort", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-snapshots", type=int, default=100)
    parser.add_argument("--switch-time-ps", type=float, default=100.0)
    parser.add_argument("--no-adaptive-switching", action="store_true")
    parser.add_argument("--no-convergence", action="store_true")
    parser.add_argument(
        "--dummy-core-nonbonded", choices=("off", "retain"), default="off"
    )
    parser.add_argument("--source-dir-name", default="AToM-OpenMM-unified")
    args = parser.parse_args()
    if args.n_snapshots < 1 or args.switch_time_ps <= 0.0:
        parser.error("n-snapshots and switch-time-ps must be positive")
    generate(
        args.source_cohort.resolve(),
        args.benchmark_root.resolve(),
        args.output.resolve(),
        n_snapshots=args.n_snapshots,
        switch_time_ps=args.switch_time_ps,
        adaptive_switching=not args.no_adaptive_switching,
        convergence=not args.no_convergence,
        dummy_core_nonbonded=args.dummy_core_nonbonded,
        source_dir_name=args.source_dir_name,
    )


if __name__ == "__main__":
    main()
