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


def _workflow(ligand_a, ligand_b):
    return {
        "workflow": {
            "type": "rbfe",
            "chemistry": "noncovalent",
            "alchemy": {
                "model": "hybrid_topology",
                "cycle": "complex_solvent",
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
                "n_snapshots": 20,
                "decorrelation_steps": 50000,
                "interpolation": "softcore_linear",
                "softcore": {
                    "function": "beutler",
                    "alpha": 0.3,
                    "sigma_nm": 0.25,
                    "power": 1,
                    "charge_steps_per_stage": 12500,
                    "sterics_steps": 25000,
                    "long_range_correction": "dynamic",
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


def _run_script(edge):
    return f"""#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name=hyb-{edge}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:geforce_rtx_3090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH -t 12:00:00

set -euo pipefail
RUN_DIR="${{SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}}"
SOURCE_DIR="$HOME/myAToM/AToM-OpenMM-unified"
cd "$RUN_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myatom
export PYTHONPATH="$SOURCE_DIR${{PYTHONPATH:+:$PYTHONPATH}}"
export LD_LIBRARY_PATH="$HOME/myAToM/openmm-build-env/lib:$HOME/myAToM/openmm-endpoint-gates-install/lib:${{LD_LIBRARY_PATH:-}}"
export OPENMM_PLUGIN_DIR="$HOME/myAToM/openmm-endpoint-gates-install/lib/plugins"
PYTHON_BIN="$HOME/myAToM/atm-gates-venv/bin/python"

git -C "$SOURCE_DIR" rev-parse HEAD > source_commit.txt
"$PYTHON_BIN" -m atom_openmm.rbfe_workflow --validate workflow.yaml
"$PYTHON_BIN" -m atom_openmm.rbfe_workflow workflow.yaml
"""


def generate(source_cohort: Path, benchmark_root: Path, output: Path):
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
            yaml.safe_dump(_workflow(ligand_a, ligand_b), sort_keys=False)
        )
        script = target / "run.sh"
        script.write_text(_run_script(edge))
        script.chmod(0o755)
    submit = output / "submit_all.sh"
    submit.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(f"(cd {a}--{b} && sbatch run.sh)" for a, b in EDGES)
        + "\n"
    )
    submit.chmod(0o755)
    (output / "README.md").write_text(
        "# CDK2 hybrid-topology cohort\n\n"
        "Ten published CDK2 benchmark edges are repeated with a noncovalent "
        "hybrid topology, Espaloma NN ligand parameters, REST2 endpoint "
        "sampling, and 100 ps NEQTI switches. The exact benchmark receptor "
        "is retained and TPO161 is parameterized with ff14SB/phosaa14SB.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cohort", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source_cohort.resolve(), args.benchmark_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
