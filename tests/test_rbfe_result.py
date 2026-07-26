import yaml
import pytest


def _writer(tmp_path, method="async_re", requested_samples=10):
    from atom_openmm.rbfe_result import RBFEResultWriter

    receptor = tmp_path / "receptor.pdb"
    ligand_a = tmp_path / "A.sdf"
    ligand_b = tmp_path / "B.sdf"
    workflow = tmp_path / "workflow.yaml"
    for path in (receptor, ligand_a, ligand_b, workflow):
        path.write_text("test\n")
    workdir = tmp_path / "run" / "pair-A-B"
    return RBFEResultWriter(
        pair_plan={
            "jobname": "pair-A-B",
            "jobdir": workdir,
            "lig1_name": "A",
            "lig2_name": "B",
            "lig1_file": ligand_a,
            "lig2_file": ligand_b,
            "external_metadata": {"edge_id": 44},
        },
        receptor_file=receptor,
        workflow_yaml=workflow,
        method=method,
        requested_samples=requested_samples,
    )


def _read(writer):
    return yaml.safe_load(writer.path.read_text())


def _test_async_result_schema_and_unit_conversion(tmp_path):
    writer = _writer(tmp_path)
    writer.workdir.mkdir(parents=True)
    (writer.workdir / "pair-A-B.pdb").write_text("pdb\n")
    (writer.workdir / "pair-A-B.png").write_text("plot\n")

    writer.update("completed", analysis={"ddg": -1.23, "ddg_std": 0.31, "samples": 8})
    result = _read(writer)

    assert result["schema_version"] == 1
    assert result["tool"] == "atom_openmm_rbfe"
    assert result["status"] == "completed"
    assert result["method"] == "async_re"
    assert result["convention"]["edge_direction"] == "ligand_a_to_ligand_b"
    assert result["convention"]["ddg_definition"] == "G(ligand_b) - G(ligand_a)"
    assert result["external_metadata"] == {"edge_id": 44}
    assert result["result"]["estimator"] == "UWHAM"
    assert result["result"]["ddg_kj_per_mol"] == -1.23 * 4.184
    assert result["result"]["ddg_error_kj_per_mol"] == 0.31 * 4.184
    assert result["result"]["samples_per_replica"] == 8
    assert result["result"]["samples_forward"] is None
    assert result["quality"]["convergence_status"] == "usable"
    assert result["artifacts"]["prepared_complex"] == "pair-A-B.pdb"
    assert "endpoint_a" in result["artifacts"]
    assert result["artifacts"]["endpoint_a"] is None
    assert result["artifacts"]["plot"] == "pair-A-B.png"
    assert result["progress"]["stage"] == "completed"
    assert result["progress"]["current_pair_index"] == 1
    assert result["progress"]["total_pairs"] == 1
    assert not writer.path.with_suffix(".yaml.tmp").exists()


def _test_neqti_partial_result_and_missing_uncertainty(tmp_path):
    writer = _writer(tmp_path, method="neqti", requested_samples=10)
    writer.update(
        "partial",
        analysis={
            "forward_samples": 10,
            "reverse_samples": 3,
            "sample_counts": {
                "leg_a_forward": 4,
                "leg_a_reverse": 2,
                "leg_b_forward": 6,
                "leg_b_reverse": 1,
            },
            "finite_sample_counts": {
                "leg_a_forward": 3,
                "leg_a_reverse": 2,
                "leg_b_forward": 5,
                "leg_b_reverse": 1,
            },
            "counted_infinite_work_counts": {
                "leg_a_forward": 1,
                "leg_a_reverse": 0,
                "leg_b_forward": 1,
                "leg_b_reverse": 0,
            },
            "warnings": ["2 numerical switch failures were included as +infinite protocol work."],
            "analysis": {
                "bar_dg_kcal_per_mol": 2.0,
                "bar_bootstrap_std_kcal_per_mol": None,
            },
        },
    )
    result = _read(writer)

    assert result["result"]["estimator"] == "BAR"
    assert result["result"]["samples_forward"] == 10
    assert result["result"]["samples_reverse"] == 3
    assert result["result"]["samples_per_replica"] is None
    assert result["quality"]["convergence_status"] == "partial"
    assert result["progress"]["forward_samples"] == 10
    assert result["progress"]["reverse_samples"] == 3
    assert result["progress"]["target_forward_samples"] == 20
    assert result["progress"]["target_reverse_samples"] == 20
    assert result["progress"]["sample_counts"] == {
        "leg_a_forward": 4,
        "leg_a_reverse": 2,
        "leg_b_forward": 6,
        "leg_b_reverse": 1,
    }
    assert result["progress"]["target_sample_counts"] == {
        "leg_a_forward": 10,
        "leg_a_reverse": 10,
        "leg_b_forward": 10,
        "leg_b_reverse": 10,
    }
    assert result["quality"]["counted_infinite_work_counts"]["leg_a_forward"] == 1
    assert result["progress"]["finite_sample_counts"]["leg_b_forward"] == 5
    assert "2 numerical switch failures" in " ".join(result["quality"]["warnings"])
    assert result["progress"]["completed_snapshot_cycles"] == 1
    assert result["progress"]["target_snapshot_cycles"] == 10
    assert "Free-energy uncertainty is unavailable." in result["quality"]["warnings"]
    assert "Sampling or overlap quality requirements were not met." in result["quality"]["warnings"]


def _test_neqti_result_surfaces_work_estimator_variants(tmp_path):
    writer = _writer(tmp_path, method="neqti", requested_samples=2)
    exact = {
        "bar_dg_kcal_per_mol": 1.0,
        "bar_dg_kj_per_mol": 4.184,
        "bar_bootstrap_std_kcal_per_mol": 0.2,
        "bar_bootstrap_std_kj_per_mol": 0.8368,
        "overlap_score": 0.3,
    }
    interval = {
        **exact,
        "bar_dg_kcal_per_mol": 1.1,
        "bar_dg_kj_per_mol": 4.6024,
        "difference_from_exact_kcal_per_mol": 0.1,
        "paired_bootstrap_difference_std_kcal_per_mol": 0.03,
    }
    writer.update(
        "completed",
        analysis={
            "forward_samples": 4,
            "reverse_samples": 4,
            "analysis": exact,
            "work_estimator_analyses": {"exact": exact, "interval_50": interval},
        },
    )

    result = _read(writer)["result"]
    assert result["ddg_kcal_per_mol"] == pytest.approx(1.0)
    assert result["estimator_variants"]["exact"]["difference_from_exact_kcal_per_mol"] == 0.0
    assert result["estimator_variants"]["interval_50"]["ddg_kcal_per_mol"] == pytest.approx(1.1)
    assert result["estimator_variants"]["interval_50"]["paired_bootstrap_difference_std_kcal_per_mol"] == pytest.approx(0.03)


def _test_neqti_result_surfaces_adaptive_termination_state(tmp_path):
    writer = _writer(tmp_path, method="neqti", requested_samples=100)
    writer.workdir.mkdir(parents=True)
    (writer.workdir / "neqti_schedule_optimization.yaml").write_text("status: frozen\n")
    (writer.workdir / "neqti_convergence.yaml").write_text("termination_reason: converged\n")
    convergence = {"termination_reason": "converged", "history": [{"thresholds_pass": True}]}
    optimizer = {"status": "frozen", "completed_pilot_cycles": 10}

    writer.update(
        "completed",
        analysis={
            "termination_reason": "converged",
            "forward_samples": 60,
            "reverse_samples": 60,
            "analysis": {
                "bar_dg_kcal_per_mol": 1.2,
                "bar_bootstrap_std_kcal_per_mol": 0.3,
            },
            "convergence": convergence,
            "schedule_optimization": optimizer,
        },
    )
    result = _read(writer)

    assert result["termination_reason"] == "converged"
    assert result["quality"]["convergence"] == convergence
    assert result["quality"]["schedule_optimization"] == optimizer
    assert result["artifacts"]["neqti_schedule_optimization"] == "neqti_schedule_optimization.yaml"
    assert result["artifacts"]["neqti_convergence"] == "neqti_convergence.yaml"


def _test_neqti_progress_handles_staged_two_leg_counts(tmp_path):
    writer = _writer(tmp_path, method="neqti", requested_samples=40)
    writer.update(
        "partial",
        analysis={
            "forward_samples": 17,
            "reverse_samples": 80,
            "sample_counts": {
                "leg_a_forward": 17,
                "leg_a_reverse": 40,
                "leg_b_forward": 0,
                "leg_b_reverse": 40,
            },
            "analysis": None,
        },
        stage="production",
    )
    result = _read(writer)

    assert result["progress"]["stage"] == "production"
    assert result["progress"]["forward_samples"] == 17
    assert result["progress"]["target_forward_samples"] == 80
    assert result["progress"]["reverse_samples"] == 80
    assert result["progress"]["target_reverse_samples"] == 80
    assert result["progress"]["sample_counts"]["leg_a_forward"] == 17
    assert result["progress"]["sample_counts"]["leg_b_forward"] == 0
    assert result["progress"]["completed_snapshot_cycles"] == 0
    assert result["progress"]["target_snapshot_cycles"] == 40


def _test_awh_result_uses_fixed_bias_uwham_and_tracks_progress(tmp_path):
    writer = _writer(tmp_path, method="awh", requested_samples=None)
    writer.workdir.mkdir(parents=True)
    (writer.workdir / "awh_summary.yaml").write_text("status: completed\n")
    (writer.workdir / "awh_state_trace.csv").write_text("state\n")
    (writer.workdir / "awh_diagnostics.yaml").write_text("quality_passed: true\n")
    (writer.workdir / "awh_trajectory.xtc").write_bytes(b"xtc")
    (writer.workdir / "awh_trajectory_topology.pdb").write_text("END\n")
    (writer.workdir / "awh_trajectory_frames.csv").write_text("frame,state\n")
    writer.update(
        "completed",
        analysis={
            "stage": "production",
            "total_steps": 6000000,
            "round_trips": 12,
            "minimum_visits": 105,
            "state_visits": [105, 120, 118],
            "overlap_score": 0.08,
            "rest2": {
                "endpoint_a": {
                    "production": {
                        "complete_physical_hottest_physical_returns": 7
                    }
                }
            },
            "diagnostics": {
                "production": {"moves": 10000},
                "uwham": {
                    "endpoint_effective_samples": {
                        "a_physical": 80,
                        "b_physical": 75,
                    }
                },
                "bias_stability": {"maximum_last_update_kbt": 0.02},
            },
            "analysis": {
                "awh_bias_ddg_kcal_per_mol": 1.3,
                "uwham_ddg_kcal_per_mol": 1.2,
                "uwham_bootstrap_std_kcal_per_mol": 0.2,
            },
        },
    )
    result = _read(writer)
    assert result["method"] == "awh"
    assert result["result"]["estimator"] == "UWHAM"
    assert result["result"]["ddg_kcal_per_mol"] == pytest.approx(1.2)
    assert result["result"]["estimator_variants"]["awh_bias"]["ddg_kcal_per_mol"] == 1.3
    assert result["progress"]["current_awh_stage"] == "production"
    assert result["progress"]["round_trips"] == 12
    assert result["quality"]["overlap_score"] == pytest.approx(0.08)
    assert result["quality"]["rest2"]["endpoint_a"]["production"][
        "complete_physical_hottest_physical_returns"
    ] == 7
    assert result["quality"]["convergence"]["endpoint_effective_samples"][
        "a_physical"
    ] == 80
    assert result["artifacts"]["awh_summary"] == "awh_summary.yaml"
    assert result["artifacts"]["awh_state_trace"] == "awh_state_trace.csv"
    assert result["artifacts"]["awh_diagnostics"] == "awh_diagnostics.yaml"
    assert result["artifacts"]["awh_trajectory"] == "awh_trajectory.xtc"


def _test_failed_result_has_structured_error(tmp_path):
    writer = _writer(tmp_path, method="neqti")
    writer.update(
        "failed",
        error={"type": "OpenMMException", "message": "Particle coordinate is NaN", "stage": "production"},
    )
    result = _read(writer)

    assert result["status"] == "failed"
    assert result["quality"]["convergence_status"] == "failed"
    assert result["error"] == {
        "type": "OpenMMException",
        "message": "Particle coordinate is NaN",
        "stage": "production",
    }
    assert result["inputs"]["receptor"].startswith("/")
    assert all(value is None or not value.startswith("/") for value in result["artifacts"].values())


def _test_run_pair_records_setup_failure_and_reraises(tmp_path):
    from atom_openmm.rbfe_workflow import WorkflowConfigError, run_pair

    receptor = tmp_path / "receptor.pdb"
    ligand_a = tmp_path / "A.sdf"
    ligand_b = tmp_path / "B.sdf"
    workflow_yaml = tmp_path / "workflow.yaml"
    for path in (receptor, ligand_a, ligand_b, workflow_yaml):
        path.write_text("test\n")
    workdir = tmp_path / "pair-A-B"
    pair_plan = {
        "jobname": "pair-A-B",
        "jobdir": workdir,
        "lig1_name": "A",
        "lig2_name": "B",
        "lig1_file": ligand_a,
        "lig2_file": ligand_b,
    }

    with pytest.raises(WorkflowConfigError, match="missing alignment atoms"):
        run_pair(
            pair_plan,
            {"production_method": "async_re"},
            {"MAX_SAMPLES": 10},
            {"ligandforcefield": "gaff-2.2.20"},
            receptor,
            {},
            workflow_yaml,
        )

    result = yaml.safe_load((workdir / "result.yaml").read_text())
    assert result["status"] == "failed"
    assert result["error"]["type"] == "WorkflowConfigError"
    assert result["error"]["stage"] == "setup"
