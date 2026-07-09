#!/usr/bin/env python
"""Generate one AToM-OpenMM RBFE workflow per ATM benchmark ligand pair."""

import argparse
import csv
import os
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


ATOM_SCHEDULE = {
    "TEMPERATURES": [300.0],
    "LAMBDAS": [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ],
    "DIRECTION": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    "INTERMEDIATE": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "LAMBDA1": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "LAMBDA2": [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00],
    "ALPHA": [0.10] * 22,
    "U0": [110.0] * 22,
    "W0COEFF": [0] * 22,
    "WALL_TIME": 10,
    "MAX_SAMPLES": 1,
    "CYCLE_TIME": 10,
    "CHECKPOINT_TIME": 1200,
    "SUBJOBS_BUFFER_SIZE": 1.0,
    "PRODUCTION_STEPS": 10,
    "PRNT_FREQUENCY": 10,
    "TRJ_FREQUENCY": 20,
    "CM_KF": 25.00,
    "CM_TOL": 5,
    "POSRE_FORCE_CONSTANT": 0.0,
    "POSRE_TOLERANCE": 3.5,
    "ALIGN_KF_SEP": 0.0,
    "ALIGN_K_THETA": 25.0,
    "ALIGN_K_PSI": 25.0,
    "UMAX": 200.00,
    "ACORE": 0.062500,
    "UBCORE": 100.0,
    "FRICTION_COEFF": 0.500000,
    "HMASS": 1.5,
    "TIME_STEP": 0.002,
    "VERBOSE": False,
}


BENCHMARK_V1_EQUILIBRATION = {
    "pre_atm": {
        "steps": [
            {
                "id": "benchmark_v1_min_heavy_restrained",
                "type": "minimization",
                "tolerance_kj_mol_nm": 10.0,
                "max_iterations": 200,
                "positional_restraints": {
                    "mask": "!@H= & !:HOH,WAT,NA,CL,K,CA",
                    "k_kcal_mol_a2": 25.0,
                    "tolerance_a": 0.01,
                },
            },
            {
                "id": "benchmark_v1_nvt_short_heavy_restrained",
                "type": "md",
                "ensemble": "NVT",
                "n_steps": 25000,
                "timestep_ps": 0.002,
                "thermostat": {"temperature_k": 300.0, "friction_per_ps": 1.0},
                "positional_restraints": {
                    "mask": "!@H= & !:HOH,WAT,NA,CL,K,CA",
                    "k_kcal_mol_a2": 5.0,
                    "tolerance_a": 0.25,
                },
                "reporters": {"state": {"interval": 5000}},
            },
            {
                "id": "benchmark_v1_npt_short_heavy_restrained",
                "type": "md",
                "ensemble": "NPT",
                "n_steps": 50000,
                "timestep_ps": 0.002,
                "thermostat": {"temperature_k": 300.0, "friction_per_ps": 1.0},
                "positional_restraints": {
                    "mask": "!@H= & !:HOH,WAT,NA,CL,K,CA",
                    "k_kcal_mol_a2": 5.0,
                    "tolerance_a": 0.25,
                },
                "reporters": {"state": {"interval": 5000}},
            },
        ]
    },
    "neqti": {
        "endpoint": {
            "steps": [
                {
                    "id": "benchmark_v1_min_heavy_restrained",
                    "type": "minimization",
                    "tolerance_kj_mol_nm": 10.0,
                    "max_iterations": 200,
                    "positional_restraints": {
                        "mask": "!@H= & !:L1,L2,HOH,WAT,NA,CL,K,CA",
                        "k_kcal_mol_a2": 25.0,
                        "tolerance_a": 0.01,
                    },
                },
                {
                    "id": "benchmark_v1_endpoint_nvt_reheat",
                    "type": "md",
                    "ensemble": "NVT",
                    "n_steps": 10000,
                    "timestep_ps": 0.002,
                    "reset_velocities": True,
                    "thermostat": {"temperature_k": 300.0, "friction_per_ps": 1.0},
                    "positional_restraints": {
                        "mask": "!@H= & !:HOH,WAT,NA,CL,K,CA",
                        "k_kcal_mol_a2": 2.0,
                        "tolerance_a": 0.5,
                    },
                    "reporters": {"state": {"interval": 5000}},
                },
                {
                    "id": "benchmark_v1_endpoint_npt_short",
                    "type": "md",
                    "ensemble": "NPT",
                    "n_steps": 10000,
                    "timestep_ps": 0.002,
                    "thermostat": {"temperature_k": 300.0, "friction_per_ps": 1.0},
                    "positional_restraints": {
                        "mask": "!@H= & !:HOH,WAT,NA,CL,K,CA",
                        "k_kcal_mol_a2": 2.0,
                        "tolerance_a": 0.5,
                    },
                    "reporters": {"state": {"interval": 5000}},
                },
                {
                    "id": "benchmark_v1_endpoint_eq",
                    "type": "md",
                    "ensemble": "NPT",
                    "n_steps": 50000,
                    "timestep_ps": 0.004,
                    "thermostat": {"temperature_k": 300.0, "friction_per_ps": 1.0},
                    "positional_restraints": {
                        "mask": "!@H= & !:L1,L2,HOH,WAT,NA,CL,K,CA",
                        "k_kcal_mol_a2": 2.0,
                        "tolerance_a": 0.5,
                    },
                    "reporters": {"state": {"interval": 5000}},
                },
            ]
        }
    },
}


NEQTI_BENCHMARK_V1 = {
    "initial_equilibration_steps": 100000,
    "n_snapshots": 40,
    "decorrelation_steps": 25000,
    "switch_steps_per_segment": 5000,
    "switch_integrator": "custom",
    "validate_switch_integrator": False,
    "resume": True,
    "bootstrap_samples": 500,
}


@dataclass(frozen=True)
class BenchmarkPair:
    system: str
    ligand_a: str
    ligand_b: str
    reference_ddg: str
    row_index: int
    reference_alignment_atoms: Optional[List[int]] = None
    ligand_a_alignment_atoms: Optional[List[int]] = None
    ligand_b_alignment_atoms: Optional[List[int]] = None
    displacement: Optional[List[float]] = None


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _column(fieldnames, aliases, required=True):
    normalized = {_norm(name): name for name in fieldnames}
    for alias in aliases:
        key = _norm(alias)
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(f"could not identify required CSV column; tried {aliases}")
    return None


def _parse_atoms(value):
    if value is None or str(value).strip() == "":
        return None
    atoms = [int(part) for part in re.split(r"[,;:\s]+", str(value).strip()) if part]
    if len(atoms) != 3:
        raise ValueError(f"alignment atom list must contain exactly three atoms, got {value!r}")
    return atoms


def _parse_displacement(value):
    if value is None or str(value).strip() == "":
        return None
    displacement = [float(part) for part in re.split(r"[,;:\s]+", str(value).strip()) if part]
    if len(displacement) != 3:
        raise ValueError(f"displacement must contain exactly three values, got {value!r}")
    return displacement


def read_benchmark_csv(csv_path, default_alignment_atoms=None, pairs_filter=None):
    pairs_filter = set(pairs_filter or [])
    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"empty benchmark CSV: {csv_path}")
        system_col = _column(reader.fieldnames, ["system", "target", "protein", "receptor", "dataset"])
        lig_a_col = _column(reader.fieldnames, ["ligand_a", "ligand1", "lig1", "ligand_i", "mol_a", "molecule_a"])
        lig_b_col = _column(reader.fieldnames, ["ligand_b", "ligand2", "lig2", "ligand_j", "mol_b", "molecule_b"])
        ddg_col = _column(
            reader.fieldnames,
            ["DDG_ATM_GAFF2", "ddg_atm_gaff2", "ddg", "predicted_ddg", "calculated_ddg", "atm_ddg"],
        )
        atoms_col = _column(
            reader.fieldnames,
            ["reference_alignment_atoms", "align_atoms", "alignment_atoms", "ref_atoms", "refatoms"],
            required=False,
        )
        lig_a_atoms_col = _column(
            reader.fieldnames,
            ["align_ligand1_ref_atoms", "align_ligand_a_ref_atoms", "ligand1_alignment_atoms"],
            required=False,
        )
        lig_b_atoms_col = _column(
            reader.fieldnames,
            ["align_ligand2_ref_atoms", "align_ligand_b_ref_atoms", "ligand2_alignment_atoms"],
            required=False,
        )
        displacement_col = _column(reader.fieldnames, ["displacement"], required=False)
        for index, row in enumerate(reader, start=2):
            reference_atoms = _parse_atoms(row.get(atoms_col)) if atoms_col else default_alignment_atoms
            ligand_a_atoms = _parse_atoms(row.get(lig_a_atoms_col)) if lig_a_atoms_col else None
            ligand_b_atoms = _parse_atoms(row.get(lig_b_atoms_col)) if lig_b_atoms_col else None
            if reference_atoms and ligand_a_atoms is None:
                ligand_a_atoms = reference_atoms
            if reference_atoms and ligand_b_atoms is None:
                ligand_b_atoms = reference_atoms
            pair = BenchmarkPair(
                system=row[system_col].strip(),
                ligand_a=row[lig_a_col].strip(),
                ligand_b=row[lig_b_col].strip(),
                reference_ddg=row[ddg_col].strip(),
                row_index=index,
                reference_alignment_atoms=reference_atoms,
                ligand_a_alignment_atoms=ligand_a_atoms,
                ligand_b_alignment_atoms=ligand_b_atoms,
                displacement=_parse_displacement(row.get(displacement_col)) if displacement_col else None,
            )
            pair_id = f"{pair.system}:{pair.ligand_a}:{pair.ligand_b}"
            reverse_pair_id = f"{pair.system}:{pair.ligand_b}:{pair.ligand_a}"
            simple_pair_id = f"{pair.ligand_a}:{pair.ligand_b}"
            if pairs_filter and not ({pair_id, reverse_pair_id, simple_pair_id} & pairs_filter):
                continue
            if not pair.system or not pair.ligand_a or not pair.ligand_b:
                raise ValueError(f"missing system or ligand name in CSV row {index}")
            rows.append(pair)
    return rows


def _safe_name(value):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "unnamed"


def _unique_file(paths, label):
    existing = []
    seen = set()
    for path in paths:
        if path.exists() and path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                existing.append(path)
                seen.add(resolved)
    if len(existing) == 1:
        return existing[0]
    if not existing:
        raise FileNotFoundError(f"could not find {label}")
    raise FileNotFoundError(f"multiple candidates for {label}: {', '.join(str(p) for p in existing)}")


def discover_system_files(systems_root, pair):
    validation_candidate = systems_root / "ATM_Validation"
    if validation_candidate.is_dir():
        systems_root = validation_candidate

    system_dir = systems_root / pair.system
    if not system_dir.exists():
        system_dir = systems_root / _safe_name(pair.system)
    if not system_dir.exists():
        system_dir = systems_root

    receptor_candidates = [
        system_dir / "receptor.pdb",
        system_dir / "receptor" / f"{pair.system}.pdb",
        system_dir / "receptors" / f"{pair.system}.pdb",
        system_dir / f"{pair.system}.pdb",
    ]
    receptor_matches = list(system_dir.glob("receptor/*.pdb")) + list(system_dir.glob("receptors/*.pdb"))
    receptor = _unique_file(receptor_candidates + receptor_matches, f"receptor for system {pair.system}")

    ligand_a = discover_ligand_file(system_dir, pair.ligand_a)
    ligand_b = discover_ligand_file(system_dir, pair.ligand_b)
    return receptor, ligand_a, ligand_b


def discover_ligand_file(system_dir, ligand):
    candidates = []
    for subdir in ("ligands", "ligand", "molecules", ""):
        base = system_dir / subdir if subdir else system_dir
        candidates.extend(
            [
                base / f"{ligand}-p.mol2",
                base / f"{ligand}.sdf",
                base / f"{ligand}.mol2",
            ]
        )
    existing = [path for path in candidates if path.exists() and path.is_file()]
    if existing:
        return existing[0]
    matches = (
        sorted(system_dir.glob(f"**/{ligand}-p.mol2"))
        + sorted(system_dir.glob(f"**/{ligand}.sdf"))
        + sorted(system_dir.glob(f"**/{ligand}.mol2"))
    )
    matches = [path for path in matches if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"could not find ligand {ligand} under {system_dir}")
    raise FileNotFoundError(f"multiple candidates for ligand {ligand}: {', '.join(str(p) for p in matches)}")


def _materialize_file(source, destination, link_mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if link_mode == "copy":
        shutil.copy2(source, destination)
    elif link_mode == "symlink":
        os.symlink(source.resolve(), destination)
    else:
        raise ValueError("link_mode must be 'symlink' or 'copy'")


def _frcmod_for_ligand(ligand_file):
    frcmod = ligand_file.with_suffix(".frcmod")
    if not frcmod.exists():
        raise FileNotFoundError(f"could not find frcmod file for {ligand_file.name}: {frcmod}")
    return frcmod


def _alignment_dict(pair, ligand_a_name, ligand_b_name):
    if pair.ligand_a_alignment_atoms is None or pair.ligand_b_alignment_atoms is None:
        raise ValueError(
            f"CSV row {pair.row_index} has no reference alignment atoms; pass --reference-alignment-atoms"
        )
    return {
        ligand_a_name: {"align_atom_ids": pair.ligand_a_alignment_atoms},
        ligand_b_name: {"align_atom_ids": pair.ligand_b_alignment_atoms},
    }


def workflow_for_pair(pair, ligand_a_file, ligand_b_file):
    atom_schedule = deepcopy(ATOM_SCHEDULE)
    if pair.displacement is not None:
        atom_schedule["DISPLACEMENT"] = pair.displacement
    return {
        "workflow": {
            "type": "rbfe",
            "mode": "small_molecule",
            "workdir": "run",
            "receptor": "receptor/receptor.pdb",
            "ligands_dir": "ligands",
            "pairs": [[ligand_a_file.name, ligand_b_file.name]],
            "alignments": "alignments.yaml",
            "forcefield_cache": "ff.json",
            "run": True,
            "analyze": True,
            "production_method": "neqti",
            "equilibration": BENCHMARK_V1_EQUILIBRATION,
            "neqti": NEQTI_BENCHMARK_V1,
            "setup": {
                "mode": "ambertools",
                "protein_forcefield": "leaprc.protein.ff14SB",
                "additional_forcefields": ["leaprc.phosaa14SB"],
                "ligand_forcefield": "leaprc.gaff2",
                "water_forcefield": "leaprc.water.tip3p",
                "solvent_box": "TIP3PBOX",
                "solvent_padding_a": 10.0,
                "neutralize": True,
            },
        },
        "atom_options": atom_schedule,
    }


def slurm_script(
    job_name,
    conda_sh="$HOME/miniconda3/etc/profile.d/conda.sh",
    conda_env="myatom",
    gres="gpu:nvidia_L40S:1",
    cpus_per_task=16,
    mem="100G",
    time_limit="12:00:00",
):
    return f"""#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name={job_name}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH -t {time_limit}

set -euo pipefail

source "${{ATOM_CONDA_SH:-{conda_sh}}}"
conda activate "${{ATOM_CONDA_ENV:-{conda_env}}}"

atom-rbfe workflow.yaml
"""


def generate_workflows(
    benchmark_csv,
    systems_root,
    outdir,
    default_alignment_atoms=None,
    pairs_filter=None,
    link_mode="symlink",
    conda_sh="$HOME/miniconda3/etc/profile.d/conda.sh",
    conda_env="myatom",
    slurm_gres="gpu:nvidia_L40S:1",
    slurm_cpus_per_task=16,
    slurm_mem="100G",
    slurm_time="12:00:00",
):
    pairs = read_benchmark_csv(benchmark_csv, default_alignment_atoms, pairs_filter)
    if not pairs:
        raise ValueError("no benchmark pairs selected")

    outdir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for pair in pairs:
        receptor, ligand_a_source, ligand_b_source = discover_system_files(systems_root, pair)
        job_slug = f"{_safe_name(pair.system)}/{_safe_name(pair.ligand_a)}--{_safe_name(pair.ligand_b)}"
        jobdir = outdir / job_slug
        receptor_dest = jobdir / "receptor" / "receptor.pdb"
        ligand_a_dest = jobdir / "ligands" / ligand_a_source.name
        ligand_b_dest = jobdir / "ligands" / ligand_b_source.name
        ligand_a_frcmod = _frcmod_for_ligand(ligand_a_source)
        ligand_b_frcmod = _frcmod_for_ligand(ligand_b_source)

        _materialize_file(receptor, receptor_dest, link_mode)
        _materialize_file(ligand_a_source, ligand_a_dest, link_mode)
        _materialize_file(ligand_b_source, ligand_b_dest, link_mode)
        _materialize_file(ligand_a_frcmod, jobdir / "ligands" / ligand_a_frcmod.name, link_mode)
        _materialize_file(ligand_b_frcmod, jobdir / "ligands" / ligand_b_frcmod.name, link_mode)

        alignments_path = jobdir / "ligands" / "alignments.yaml"
        with open(alignments_path, "w") as handle:
            yaml.dump(
                _alignment_dict(pair, ligand_a_dest.stem, ligand_b_dest.stem),
                handle,
                default_flow_style=None,
                sort_keys=False,
                width=1000000,
            )

        workflow = workflow_for_pair(pair, ligand_a_dest, ligand_b_dest)
        workflow_path = jobdir / "workflow.yaml"
        with open(workflow_path, "w") as handle:
            yaml.dump(workflow, handle, default_flow_style=None, sort_keys=False, width=1000000)

        script_path = jobdir / "run.sh"
        script_path.write_text(
            slurm_script(
                f"atm-{_safe_name(pair.system)}-{_safe_name(pair.ligand_a)}-{_safe_name(pair.ligand_b)}",
                conda_sh=conda_sh,
                conda_env=conda_env,
                gres=slurm_gres,
                cpus_per_task=slurm_cpus_per_task,
                mem=slurm_mem,
                time_limit=slurm_time,
            )
        )
        script_path.chmod(0o755)

        index_rows.append(
            {
                "system": pair.system,
                "ligand_a": pair.ligand_a,
                "ligand_b": pair.ligand_b,
                "reference_ddg": pair.reference_ddg,
                "workflow": str(workflow_path),
                "slurm_script": str(script_path),
                "workdir": str(jobdir / "run"),
            }
        )

    index_path = outdir / "index.csv"
    with open(index_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    return index_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-csv", required=True, type=Path)
    parser.add_argument("--systems-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--reference-alignment-atoms",
        help="fallback comma-separated reference atoms for each ligand_a, e.g. 14,21,18",
    )
    parser.add_argument(
        "--pairs-filter",
        nargs="*",
        help="optional filters like system:ligA:ligB or ligA:ligB",
    )
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--conda-sh", default="$HOME/miniconda3/etc/profile.d/conda.sh")
    parser.add_argument("--conda-env", default="myatom")
    parser.add_argument("--slurm-gres", default="gpu:nvidia_L40S:1")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=16)
    parser.add_argument("--slurm-mem", default="100G")
    parser.add_argument("--slurm-time", default="12:00:00")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = generate_workflows(
        args.benchmark_csv.resolve(),
        args.systems_root.resolve(),
        args.outdir.resolve(),
        default_alignment_atoms=_parse_atoms(args.reference_alignment_atoms),
        pairs_filter=args.pairs_filter,
        link_mode=args.link_mode,
        conda_sh=args.conda_sh,
        conda_env=args.conda_env,
        slurm_gres=args.slurm_gres,
        slurm_cpus_per_task=args.slurm_cpus_per_task,
        slurm_mem=args.slurm_mem,
        slurm_time=args.slurm_time,
    )
    print(f"Generated {len(rows)} benchmark workflows in {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
