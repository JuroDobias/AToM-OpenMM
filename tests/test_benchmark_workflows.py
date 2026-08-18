import csv
import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "examples" / "RBFE" / "benchmarks" / "generate_atm_benchmark_workflows.py"
COLLECTOR = ROOT / "examples" / "RBFE" / "benchmarks" / "collect_results.py"
ENRICHER = ROOT / "examples" / "RBFE" / "benchmarks" / "enrich_atm_benchmark_csv.py"
HYBRID_GENERATOR = (
    ROOT / "examples" / "RBFE" / "benchmarks" / "generate_hybrid_cdk2_cohort.py"
)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _test_hybrid_cdk2_generator_uses_adaptive_production_protocol():
    generator = _load_module(HYBRID_GENERATOR, "generate_hybrid_cdk2_cohort")
    workflow = generator._workflow("1oiy", "32")["workflow"]
    neqti = workflow["neqti"]

    assert neqti["n_snapshots"] == 100
    assert neqti["adaptive_switching"]["candidate_times_ps"] == [100, 300, 1000]
    assert neqti["adaptive_switching"]["pilot_samples_per_direction"] == 20
    assert neqti["adaptive_switching"]["reuse_selected_pilot_samples"]
    assert neqti["convergence"] == {
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
    run_script = generator._run_script("1oiy--32")
    assert 'FORCE_REOPEN="${ATOM_FORCE_REOPEN:-0}"' in run_script
    assert 'result_completed && [[ "$FORCE_REOPEN" != "1" ]]' in run_script

    extended = generator._workflow(
        "1oiy",
        "32",
        schedule_optimization=True,
        unrestrained_npt_steps=1000000,
        random_seed=3031,
    )["workflow"]
    assert extended["neqti"]["schedule_optimization"]["pilot_samples"] == 10
    assert extended["neqti"]["schedule_optimization"]["subdivisions_per_stage"] == 10
    assert extended["neqti"]["random_seed"] == 3031
    assert (
        extended["equilibration"]["neqti"]["complex_endpoint"]["steps"][-1]["n_steps"]
        == 1000000
    )


def _write_synthetic_benchmark(tmp_path):
    systems_root = tmp_path / "systems"
    system_dir = systems_root / "cdk2"
    (system_dir / "receptor").mkdir(parents=True)
    (system_dir / "ligands").mkdir()
    (system_dir / "receptor" / "cdk2.pdb").write_text("RECEPTOR\n")
    for ligand in ("H1Q", "H1R", "H1S"):
        (system_dir / "ligands" / f"{ligand}-p.mol2").write_text(f"{ligand}\n")
        (system_dir / "ligands" / f"{ligand}-p.frcmod").write_text(f"{ligand} frcmod\n")

    csv_path = tmp_path / "DDG_ATM_GAFF2.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Protein",
                "Ligand1",
                "Ligand2",
                "ATM_ddG",
                "align_ligand1_ref_atoms",
                "align_ligand2_ref_atoms",
                "displacement",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Protein": "cdk2",
                "Ligand1": "H1Q",
                "Ligand2": "H1R",
                "ATM_ddG": "0.12",
                "align_ligand1_ref_atoms": "14,21,18",
                "align_ligand2_ref_atoms": "13,20,17",
                "displacement": "20.0, 0.0, -5.0",
            }
        )
        writer.writerow(
            {
                "Protein": "cdk2",
                "Ligand1": "H1Q",
                "Ligand2": "H1S",
                "ATM_ddG": "1.23",
                "align_ligand1_ref_atoms": "14,21,18",
                "align_ligand2_ref_atoms": "14,21,18",
                "displacement": "21.0, 0.0, -5.0",
            }
        )
    return systems_root, csv_path


def _test_generate_workflows_writes_one_job_per_pair(tmp_path):
    generator = _load_module(GENERATOR, "generate_atm_benchmark_workflows")
    systems_root, csv_path = _write_synthetic_benchmark(tmp_path)
    outdir = tmp_path / "jobs"

    rows = generator.generate_workflows(
        csv_path,
        systems_root,
        outdir,
        link_mode="copy",
    )

    assert len(rows) == 2
    workflow_path = outdir / "cdk2" / "H1Q--H1R" / "workflow.yaml"
    run_script = outdir / "cdk2" / "H1Q--H1R" / "run.sh"
    index_csv = outdir / "index.csv"
    assert workflow_path.exists()
    assert run_script.exists()
    assert index_csv.exists()

    workflow = yaml.safe_load(workflow_path.read_text())
    assert workflow["workflow"]["sampling"]["method"] == "neqti"
    assert workflow["workflow"]["alchemy"] == {"model": "atm", "cycle": "transfer"}
    assert workflow["workflow"]["pairs"] == [["H1Q-p.mol2", "H1R-p.mol2"]]
    assert workflow["workflow"]["alignments"] == "alignments.yaml"
    assert workflow["workflow"]["setup"]["mode"] == "ambertools"
    assert workflow["workflow"]["setup"]["protein_forcefield"] == "leaprc.protein.ff14SB"
    assert workflow["workflow"]["setup"]["additional_forcefields"] == ["leaprc.phosaa14SB"]
    assert workflow["workflow"]["setup"]["ligand_forcefield"] == "leaprc.gaff2"
    assert workflow["workflow"]["setup"]["water_forcefield"] == "leaprc.water.tip3p"
    assert workflow["atom_options"]["DISPLACEMENT"] == [20.0, 0.0, -5.0]
    assert workflow["workflow"]["neqti"]["n_snapshots"] == 40
    assert workflow["workflow"]["neqti"]["switch_steps_per_segment"] == 5000
    assert workflow["workflow"]["neqti"]["switch_integrator"] == "custom"
    assert workflow["workflow"]["neqti"]["sampling_order"] == "interleaved"
    assert workflow["workflow"]["equilibration"]["pre_atm"]["steps"][0]["id"].startswith("benchmark_v1")
    run_text = run_script.read_text()
    assert "#SBATCH --gres=gpu:nvidia_L40S:1" in run_text
    assert "#SBATCH --cpus-per-task=16" in run_text
    assert "#SBATCH --mem=100G" in run_text
    assert 'source "${ATOM_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"' in run_text
    assert 'conda activate "${ATOM_CONDA_ENV:-myatom}"' in run_text
    assert "atom-rbfe workflow.yaml" in run_text
    alignments = yaml.safe_load((outdir / "cdk2" / "H1Q--H1R" / "ligands" / "alignments.yaml").read_text())
    assert alignments == {
        "H1Q-p": {"align_atom_ids": [14, 21, 18]},
        "H1R-p": {"align_atom_ids": [13, 20, 17]},
    }
    assert (outdir / "cdk2" / "H1Q--H1R" / "ligands" / "H1Q-p.frcmod").exists()
    assert (outdir / "cdk2" / "H1Q--H1R" / "ligands" / "H1R-p.frcmod").exists()

    with open(index_csv, newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    assert index_rows[0]["system"] == "cdk2"
    assert index_rows[0]["ligand_a"] == "H1Q"
    assert index_rows[0]["ligand_b"] == "H1R"
    assert index_rows[0]["reference_ddg"] == "0.12"


def _test_generate_workflows_can_filter_pairs(tmp_path):
    generator = _load_module(GENERATOR, "generate_atm_benchmark_workflows")
    systems_root, csv_path = _write_synthetic_benchmark(tmp_path)

    rows = generator.generate_workflows(
        csv_path,
        systems_root,
        tmp_path / "jobs",
        pairs_filter=["H1Q:H1S"],
        link_mode="copy",
    )

    assert len(rows) == 1
    assert rows[0]["ligand_b"] == "H1S"


def _test_generate_workflows_can_override_neqti_and_restart_settings(tmp_path):
    generator = _load_module(GENERATOR, "generate_atm_benchmark_workflows")
    systems_root, csv_path = _write_synthetic_benchmark(tmp_path)

    generator.generate_workflows(
        csv_path,
        systems_root,
        tmp_path / "jobs",
        pairs_filter=["H1Q:H1R"],
        link_mode="copy",
        neqti_n_snapshots=40,
        neqti_switch_steps_per_segment=15000,
        neqti_max_switch_attempts_per_direction=80,
        production_restart_attempts=6,
    )

    workflow = yaml.safe_load((tmp_path / "jobs" / "cdk2" / "H1Q--H1R" / "workflow.yaml").read_text())

    assert workflow["workflow"]["neqti"]["n_snapshots"] == 40
    assert workflow["workflow"]["neqti"]["switch_steps_per_segment"] == 15000
    assert workflow["workflow"]["neqti"]["max_switch_attempts_per_direction"] == 80
    assert workflow["workflow"]["production_restarts"] == {"enabled": True, "max_attempts": 6}


def _test_collect_results_handles_completed_and_missing_pairs(tmp_path):
    collector = _load_module(COLLECTOR, "collect_atm_benchmark_results")
    outdir = tmp_path / "jobs"
    completed_workdir = outdir / "cdk2" / "H1Q--H1R" / "run"
    missing_workdir = outdir / "cdk2" / "H1Q--H1S" / "run"
    result_dir = completed_workdir / "cdk2-H1Q-H1R"
    result_dir.mkdir(parents=True)
    missing_workdir.mkdir(parents=True)
    (result_dir / "result.yaml").write_text(
        yaml.dump(
            {
                "status": "completed",
                "result": {"ddg_kcal_per_mol": -0.1, "ddg_error_kcal_per_mol": 0.2},
                "quality": {"overlap_score": 0.5, "warnings": []},
            }
        )
    )

    index_csv = outdir / "index.csv"
    with open(index_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["system", "ligand_a", "ligand_b", "reference_ddg", "workflow", "slurm_script", "workdir"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "system": "cdk2",
                "ligand_a": "H1Q",
                "ligand_b": "H1R",
                "reference_ddg": "0.12",
                "workflow": "workflow.yaml",
                "slurm_script": "run.sh",
                "workdir": str(completed_workdir),
            }
        )
        writer.writerow(
            {
                "system": "cdk2",
                "ligand_a": "H1Q",
                "ligand_b": "H1S",
                "reference_ddg": "1.23",
                "workflow": "workflow.yaml",
                "slurm_script": "run.sh",
                "workdir": str(missing_workdir),
            }
        )

    rows = collector.collect_results(index_csv, outdir / "benchmark_results.csv")

    assert rows[0]["status"] == "completed"
    assert rows[0]["neqti_ddg"] == -0.1
    assert rows[0]["overlap_score"] == 0.5
    assert rows[1]["status"] == "missing"
    assert rows[1]["warnings"] == "result.yaml missing"


def _test_enrich_csv_extracts_alignment_atoms_from_cntl(tmp_path):
    enricher = _load_module(ENRICHER, "enrich_atm_benchmark_csv")
    validation_root = tmp_path / "ATM_Validation"
    pair_dir = validation_root / "CDK2" / "complexes" / "CDK2_new_2_edit-17-1h1q"
    pair_dir.mkdir(parents=True)
    (pair_dir / "CDK2_new_2_edit-17-1h1q_asyncre.cntl").write_text(
        "\n".join(
            [
                "BASENAME = 'CDK2_new_2_edit-17-1h1q'",
                "DISPLACEMENT = ' -25.0, 0.0, -25.0 '",
                "ALIGN_LIGAND1_REF_ATOMS = 12, 19, 16",
                "ALIGN_LIGAND2_REF_ATOMS = 13, 20, 17",
            ]
        )
    )
    input_csv = tmp_path / "DDG_ATM_GAFF2.csv"
    with open(input_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Protein", "Ligand1", "Ligand2", "ATM_ddG"])
        writer.writeheader()
        writer.writerow({"Protein": "CDK2", "Ligand1": "17", "Ligand2": "1h1q", "ATM_ddG": "-1.0"})

    rows = enricher.enrich_csv(input_csv, validation_root, tmp_path / "enriched.csv")

    assert rows[0]["align_ligand1_ref_atoms"] == "12, 19, 16"
    assert rows[0]["align_ligand2_ref_atoms"] == "13, 20, 17"
    assert rows[0]["displacement"] == "-25.0, 0.0, -25.0"


def _test_generator_prefers_prepared_mol2_files(tmp_path):
    generator = _load_module(GENERATOR, "generate_atm_benchmark_workflows")
    systems_root = tmp_path / "ATM_benchmark"
    system_dir = systems_root / "ATM_Validation" / "CDK2"
    (system_dir / "receptor").mkdir(parents=True)
    (system_dir / "ligands").mkdir()
    (system_dir / "receptor" / "CDK2_new_2_edit.pdb").write_text("RECEPTOR\n")
    for ligand in ("17", "1h1q"):
        (system_dir / "ligands" / f"{ligand}.mol2").write_text(f"{ligand} raw\n")
        (system_dir / "ligands" / f"{ligand}-p.mol2").write_text(f"{ligand} prepared\n")
        (system_dir / "ligands" / f"{ligand}-p.frcmod").write_text(f"{ligand} frcmod\n")
    csv_path = tmp_path / "enriched.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Protein",
                "Ligand1",
                "Ligand2",
                "ATM_ddG",
                "align_ligand1_ref_atoms",
                "align_ligand2_ref_atoms",
                "displacement",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Protein": "CDK2",
                "Ligand1": "17",
                "Ligand2": "1h1q",
                "ATM_ddG": "-1.0",
                "align_ligand1_ref_atoms": "12,19,16",
                "align_ligand2_ref_atoms": "12,19,16",
                "displacement": "22,0,-10",
            }
        )

    generator.generate_workflows(csv_path, systems_root, tmp_path / "jobs", link_mode="copy")
    workflow = yaml.safe_load((tmp_path / "jobs" / "CDK2" / "17--1h1q" / "workflow.yaml").read_text())

    assert workflow["workflow"]["pairs"] == [["17-p.mol2", "1h1q-p.mol2"]]
    assert (tmp_path / "jobs" / "CDK2" / "17--1h1q" / "ligands" / "17-p.mol2").read_text() == "17 prepared\n"
    assert (tmp_path / "jobs" / "CDK2" / "17--1h1q" / "ligands" / "17-p.frcmod").read_text() == "17 frcmod\n"
