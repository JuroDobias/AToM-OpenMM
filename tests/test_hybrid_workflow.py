from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem import AllChem


def _write_ligand(path, smiles):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(molecule, randomSeed=2026)
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()


def _workflow(tmp_path, ligand_b="B.sdf"):
    (tmp_path / "receptor.pdb").write_text("END\n")
    ligands = tmp_path / "ligands"
    ligands.mkdir()
    _write_ligand(ligands / "A.sdf", "CCO")
    _write_ligand(ligands / ligand_b, "CCN")
    payload = {
        "workflow": {
            "type": "rbfe",
            "chemistry": "noncovalent",
            "alchemy": {
                "model": "hybrid_topology",
                "cycle": "complex_solvent",
                "mapping": {"method": "mcs"},
            },
            "sampling": {"method": "neqti"},
            "receptor": "receptor.pdb",
            "ligands_dir": "ligands",
            "pairs": [["A.sdf", ligand_b]],
            "workdir": "run",
            "setup": {
                "ligand_forcefield": "espaloma-0.3.2",
                "ligand_charge_model": "nn",
            },
            "neqti": {
                "interpolation": "softcore_linear",
                "n_snapshots": 2,
                "softcore": {
                    "charge_steps_per_stage": 10,
                    "sterics_steps": 20,
                },
                "rest2": {"enabled": False},
            },
        }
    }
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def _test_hybrid_workflow_validates_and_plans_two_environments(tmp_path):
    from atom_openmm.rbfe_workflow import plan_workflow, validate_workflow

    path = _workflow(tmp_path)
    assert validate_workflow(path)
    plan = plan_workflow(path)
    assert plan["alchemy_model"] == "hybrid_topology"
    assert plan["thermodynamic_cycle"] == "complex_solvent"
    assert plan["pairs"][0]["environments"] == ["complex", "solvent"]


def _test_hybrid_dummy_core_nonbonded_defaults_off_and_accepts_retain(tmp_path):
    from atom_openmm.hybrid_workflow import HybridWorkflowError, _validate_settings

    workflow = yaml.safe_load(_workflow(tmp_path).read_text())["workflow"]
    assert _validate_settings(workflow)["dummy_core_nonbonded"] == "off"

    workflow["alchemy"]["dummy_core_nonbonded"] = "retain"
    assert _validate_settings(workflow)["dummy_core_nonbonded"] == "retain"

    workflow["alchemy"]["dummy_core_nonbonded"] = "one_way"
    try:
        _validate_settings(workflow)
    except HybridWorkflowError as exc:
        assert "dummy_core_nonbonded" in str(exc)
    else:
        raise AssertionError("invalid dummy-core nonbonded mode was accepted")


def _test_hybrid_workflow_rejects_charge_change(tmp_path):
    from atom_openmm.hybrid_workflow import HybridWorkflowError
    from atom_openmm.rbfe_workflow import validate_workflow

    path = _workflow(tmp_path, ligand_b="charged.sdf")
    _write_ligand(tmp_path / "ligands" / "charged.sdf", "C[NH3+]")
    try:
        validate_workflow(path)
    except HybridWorkflowError as exc:
        assert "equal endpoint formal charges" in str(exc)
    else:
        raise AssertionError("charge-changing hybrid edge was accepted")


def _test_hybrid_adaptive_settings_validate_duration_and_sample_cap(tmp_path):
    from atom_openmm.hybrid_workflow import HybridWorkflowError, _validate_settings

    workflow = yaml.safe_load(_workflow(tmp_path).read_text())["workflow"]
    workflow["neqti"].update({
        "timestep_fs": 2.0,
        "n_snapshots": 100,
        "adaptive_switching": {
            "enabled": True,
            "candidate_times_ps": [0.08, 0.24, 0.8],
            "pilot_samples_per_direction": 20,
            "reuse_selected_pilot_samples": True,
        },
        "convergence": {
            "enabled": True,
            "min_samples_per_direction": 30,
        },
    })
    config = _validate_settings(workflow)
    assert config["adaptive_switching"]["candidate_total_steps"] == [40, 120, 400]
    assert config["convergence"]["min_samples_per_direction"] == 30

    workflow["neqti"]["adaptive_switching"]["candidate_times_ps"][0] = 0.1
    try:
        _validate_settings(workflow)
    except HybridWorkflowError as exc:
        assert "first adaptive switching candidate" in str(exc)
    else:
        raise AssertionError("mismatched base switching duration was accepted")


def _test_hybrid_stage_interpolation_validation(tmp_path):
    from atom_openmm.hybrid_workflow import HybridWorkflowError, _validate_settings

    workflow = yaml.safe_load(_workflow(tmp_path).read_text())["workflow"]
    workflow["neqti"]["softcore"]["stage_interpolation"] = "smoothstep2"
    config = _validate_settings(workflow)
    assert config["softcore"]["stage_interpolation"] == "smoothstep2"

    workflow["neqti"]["softcore"]["stage_interpolation"] = "cubic"
    try:
        _validate_settings(workflow)
    except HybridWorkflowError as exc:
        assert "stage_interpolation" in str(exc)
    else:
        raise AssertionError("unknown stage interpolation was accepted")


def _test_hybrid_accepts_amber_ssc2_and_rejects_invalid_parameters(tmp_path):
    from atom_openmm.hybrid_workflow import HybridWorkflowError, _validate_settings

    workflow = yaml.safe_load(_workflow(tmp_path).read_text())["workflow"]
    workflow["neqti"]["softcore"].update(
        {
            "function": "amber_ssc2",
            "ssc2_alpha_lj": 0.5,
            "ssc2_switch_width_nm": 0.2,
        }
    )
    config = _validate_settings(workflow)
    assert config["softcore"]["function"] == "amber_ssc2"

    workflow["neqti"]["softcore"]["ssc2_alpha_lj"] = 0.0
    try:
        _validate_settings(workflow)
    except HybridWorkflowError as exc:
        assert "SSC(2)" in str(exc)
    else:
        raise AssertionError("non-positive SSC(2) alpha was accepted")


def _test_hybrid_accepts_only_balanced_concerted_ssc2_coulomb(tmp_path):
    from atom_openmm.hybrid_workflow import HybridWorkflowError, _validate_settings

    workflow = yaml.safe_load(_workflow(tmp_path).read_text())["workflow"]
    workflow["neqti"]["softcore"] = {
        "function": "amber_ssc2",
        "coulomb_function": "amber_ssc2",
        "ssc2_alpha_lj": 0.5,
        "ssc2_beta_coul": 1.0,
        "ssc2_switch_width_nm": 0.2,
        "total_steps": 50,
        "path": {"mode": "concerted"},
    }
    config = _validate_settings(workflow)
    assert config["softcore"]["path_mode"] == "concerted"
    assert config["softcore"]["coulomb_function"] == "amber_ssc2"

    workflow["neqti"]["softcore"]["path"] = {
        "nodes": [],
        "vdw_a": [1.0, 0.0],
        "charge_a": [1.0, 0.0],
    }
    try:
        _validate_settings(workflow)
    except HybridWorkflowError as exc:
        assert "path.mode: concerted" in str(exc)
    else:
        raise AssertionError("SSC(2) Coulomb accepted a non-concerted path")


def _test_hybrid_convergence_uses_matched_prefix_and_truncates_extra_work(tmp_path):
    from atom_openmm.covalent_workflow import _read_work, _rewrite_work
    from atom_openmm.hybrid_workflow import (
        _hybrid_convergence_callback,
        _normalize_converged_work_prefix,
    )

    for environment, samples in (("complex", 32), ("solvent", 100)):
        _rewrite_work(tmp_path / f"{environment}_forward.csv", [1.0] * samples)
        _rewrite_work(tmp_path / f"{environment}_reverse.csv", [-1.0] * samples)
    config = {
        "temperature_k": 300.0,
        "bootstrap_samples": 20,
        "random_seed": 2026,
        "convergence": {
            "enabled": True,
            "min_samples_per_direction": 30,
            "min_overlap_score_per_leg": 0.05,
            "max_ddg_error_kcal_per_mol": 0.5,
            "consecutive_checks": 3,
            "max_ddg_range_kcal_per_mol": 0.25,
        },
    }
    callback, state = _hybrid_convergence_callback(tmp_path, config)
    assert callback("complex", 32, [1.0] * 32, [-1.0] * 32)
    assert state["termination_reason"] == "converged"
    assert [row["sample_count_per_direction"] for row in state["history"]] == [30, 31, 32]
    assert len(_read_work(tmp_path / "solvent_forward.csv")) == 32
    assert len(_read_work(tmp_path / "solvent_reverse.csv")) == 32

    _rewrite_work(tmp_path / "solvent_forward.csv", [1.0] * 40)
    _normalize_converged_work_prefix(tmp_path, state)
    assert len(_read_work(tmp_path / "solvent_forward.csv")) == 32


def _test_hybrid_result_reports_adaptive_selection_and_max_samples(tmp_path):
    from atom_openmm.covalent_workflow import _rewrite_work
    from atom_openmm.hybrid_workflow import _analysis_payload, _result

    for environment in ("complex", "solvent"):
        _rewrite_work(tmp_path / f"{environment}_forward.csv", [1.0] * 20)
        _rewrite_work(tmp_path / f"{environment}_reverse.csv", [-1.0] * 20)
    adaptive = {
        "environments": {
            name: {
                "selected": {
                    "time_ps": duration,
                    "selection_reason": reason,
                    "pilot_samples_reused": True,
                }
            }
            for name, duration, reason in (
                ("complex", 1000, "candidate_list_exhausted"),
                ("solvent", 100, "thresholds_passed"),
            )
        }
    }
    adaptive["environments"]["pending"] = {"selected": None}
    (tmp_path / "neqti_adaptive_switching.yaml").write_text(
        yaml.safe_dump(adaptive)
    )
    convergence = {"termination_reason": "max_samples", "history": []}
    (tmp_path / "neqti_convergence.yaml").write_text(
        yaml.safe_dump(convergence)
    )
    config = {
        "temperature_k": 300.0,
        "bootstrap_samples": 10,
        "random_seed": 2026,
        "n_snapshots": 20,
        "convergence": {"enabled": True},
    }
    work = {
        "leg_a_forward": [1.0] * 20,
        "leg_a_reverse": [-1.0] * 20,
        "leg_b_forward": [1.0] * 20,
        "leg_b_reverse": [-1.0] * 20,
    }
    result = _result(
        {
            "jobname": "test-A-B",
            "lig1_name": "A",
            "lig2_name": "B",
            "lig1_file": tmp_path / "A.sdf",
            "lig2_file": tmp_path / "B.sdf",
        },
        tmp_path / "workflow.yaml",
        tmp_path / "receptor.pdb",
        tmp_path,
        _analysis_payload(work, config),
        {"parameterization": {}},
        config,
        "partial",
    )
    assert result["termination_reason"] == "max_samples"
    assert result["quality"]["convergence_status"] == "partial"
    assert result["quality"]["adaptive_switching"] == adaptive
    assert result["artifacts"]["adaptive_switching"] == "neqti_adaptive_switching.yaml"
    assert len(result["quality"]["warnings"]) == 2


def _test_hybrid_environment_iterators_advance_in_matched_cycles(tmp_path):
    from atom_openmm.covalent_workflow import _append_work, _read_work, _rewrite_work
    from atom_openmm.hybrid_workflow import (
        _hybrid_convergence_callback,
        _run_matched_environment_iterators,
    )

    for environment in ("complex", "solvent"):
        _rewrite_work(tmp_path / f"{environment}_forward.csv", [1.0] * 20)
        _rewrite_work(tmp_path / f"{environment}_reverse.csv", [-1.0] * 20)

    def iterator(environment):
        for sample in range(21, 41):
            _append_work(tmp_path / f"{environment}_forward.csv", sample, 4.184)
            _append_work(tmp_path / f"{environment}_reverse.csv", sample, -4.184)
            yield [1.0] * sample, [-1.0] * sample, {"sample": sample}

    config = {
        "temperature_k": 300.0,
        "bootstrap_samples": 20,
        "random_seed": 2026,
        "convergence": {
            "enabled": True,
            "min_samples_per_direction": 30,
            "min_overlap_score_per_leg": 0.05,
            "max_ddg_error_kcal_per_mol": 0.5,
            "consecutive_checks": 3,
            "max_ddg_range_kcal_per_mol": 0.25,
        },
    }
    callback, state = _hybrid_convergence_callback(tmp_path, config)
    summaries = _run_matched_environment_iterators(
        {"complex": iterator("complex"), "solvent": iterator("solvent")},
        tmp_path,
        callback,
    )
    assert state["termination_reason"] == "converged"
    assert summaries["complex"] == {"sample": 32}
    assert summaries["solvent"] == {"sample": 32}
    assert len(_read_work(tmp_path / "complex_forward.csv")) == 32
    assert len(_read_work(tmp_path / "solvent_forward.csv")) == 32
