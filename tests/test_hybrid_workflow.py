import copy
from pathlib import Path

import pytest
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


def _test_resp_sigma_hole_setup_requires_gaff_and_cache(tmp_path):
    from atom_openmm.hybrid_workflow import HybridWorkflowError, _validate_settings

    workflow = yaml.safe_load(_workflow(tmp_path).read_text())["workflow"]
    workflow["setup"] = {
        "ligand_forcefield": "gaff-2.2.20",
        "ligand_charge_model": "resp-sigma-hole",
        "ligand_parameter_cache": "/shared/ligand-parameters",
        "ligand_parameter_protocol": "gaff2-resp-cl-ep-v1",
    }
    _validate_settings(workflow)

    del workflow["setup"]["ligand_parameter_cache"]
    try:
        _validate_settings(workflow)
    except HybridWorkflowError as exc:
        assert "ligand_parameter_cache" in str(exc)
    else:
        raise AssertionError("RESP setup without a parameter cache was accepted")


def _test_panteva_accepts_espaloma_with_separate_gaff2_c4_typing(tmp_path):
    from atom_openmm.hybrid_workflow import _validate_settings

    workflow = yaml.safe_load(_workflow(tmp_path).read_text())["workflow"]
    workflow["setup"].update({
        "solvent_model": "tip4pew",
        "metal_ions": {
            "model": "panteva_m12_6_4",
            "polarizability_table": "/shared/lj_1264_pol.dat",
        },
    })
    _validate_settings(workflow)


def _test_fixed_sigma_hole_settings_are_validated(tmp_path):
    from atom_openmm.hybrid_workflow import HybridWorkflowError, _validate_settings

    workflow = yaml.safe_load(_workflow(tmp_path).read_text())["workflow"]
    workflow["setup"]["ligand_sigma_holes"] = {
        "model": "fixed",
        "halogens": ["Cl"],
        "charge_e": 0.03,
        "distance_a": 1.64,
        "compensate_on": "halogen",
    }
    _validate_settings(workflow)

    workflow["setup"]["ligand_sigma_holes"]["charge_e"] = 0.0
    with pytest.raises(HybridWorkflowError, match="charge_e"):
        _validate_settings(workflow)

    workflow["setup"]["ligand_sigma_holes"]["charge_e"] = 0.03
    workflow["setup"]["ligand_sigma_holes"]["smarts"] = "[#6]-[#17]"
    with pytest.raises(HybridWorkflowError, match="not both"):
        _validate_settings(workflow)


def _test_fixed_sigma_hole_defaults_validate_without_numeric_fields(tmp_path):
    from atom_openmm.hybrid_workflow import _validate_settings

    workflow = yaml.safe_load(_workflow(tmp_path).read_text())["workflow"]
    workflow["setup"]["ligand_sigma_holes"] = {"halogens": ["Cl", "Br"]}
    _validate_settings(workflow)


def _test_legacy_bond_only_preparation_inputs_match_current_default():
    from atom_openmm.hybrid_workflow import _legacy_preparation_inputs_match

    expected = {
        "schema_version": 3,
        "files": {"ligand_a": {"sha256": "a"}},
        "setup": {"solvent_model": "tip3p"},
        "mapping": {"method": "mcs"},
    }
    observed = copy.deepcopy(expected)
    observed["mapping"]["inactive_bonded_geometry"] = "bond_only"

    assert _legacy_preparation_inputs_match(observed, expected)
    observed["mapping"]["method"] = "explicit_pairs"
    assert not _legacy_preparation_inputs_match(observed, expected)


def _test_legacy_fixed_sigma_hole_fingerprint_resumes_after_normalization():
    from atom_openmm.hybrid_workflow import _legacy_preparation_inputs_match

    observed = {
        "setup": {
            "ligand_sigma_holes": {
                "model": "fixed", "halogens": ["Cl"],
                "charge_e": 0.03, "distance_a": 1.64,
                "compensate_on": "halogen",
            }
        },
        "mapping": {"method": "mcs"},
    }
    expected = copy.deepcopy(observed)
    expected["setup"]["ligand_sigma_holes"] = {
        "halogens": ["Cl"], "charges_e": {"Cl": 0.03},
        "distances_a": {"Cl": 1.64}, "model": "fixed",
        "compensate_on": "halogen",
    }
    assert _legacy_preparation_inputs_match(observed, expected)

    expected["setup"]["ligand_sigma_holes"]["charges_e"]["Cl"] = 0.033
    assert not _legacy_preparation_inputs_match(observed, expected)


def _test_explicit_mapping_requires_single_edge(tmp_path):
    from atom_openmm.hybrid_workflow import HybridWorkflowError
    from atom_openmm.rbfe_workflow import validate_workflow

    path = _workflow(tmp_path)
    payload = yaml.safe_load(path.read_text())
    payload["workflow"]["alchemy"]["mapping"] = {
        "method": "explicit_pairs",
        "pairs_0based": [[0, 0]],
    }
    payload["workflow"]["pairs"].append(["A.sdf", "B.sdf"])
    path.write_text(yaml.safe_dump(payload))

    try:
        validate_workflow(path)
    except HybridWorkflowError as exc:
        assert "exactly one edge" in str(exc)
    else:
        raise AssertionError("multi-edge explicit mapping was accepted")


def _test_element_transmutation_requires_neqti():
    from atom_openmm.hybrid_workflow import (
        HybridWorkflowError,
        _validate_mapping_sampling,
    )

    mapping = {"transmuted_pairs_0based": [[1, 1]]}
    try:
        _validate_mapping_sampling(mapping, {"sampling": {"method": "awh"}})
    except HybridWorkflowError as exc:
        assert "only NEQTI" in str(exc)
    else:
        raise AssertionError("non-NEQTI mapped-atom transmutation was accepted")

    _validate_mapping_sampling(mapping, {"sampling": {"method": "neqti"}})
    _validate_mapping_sampling(
        {"transmuted_pairs_0based": []}, {"sampling": {"method": "awh"}}
    )


def _test_explicit_mapping_does_not_mask_automatic_junction_geometry():
    from atom_openmm.hybrid_workflow import _mapping_settings

    settings = _mapping_settings(
        {
            "alchemy": {
                "mapping": {
                    "method": "explicit_pairs",
                    "pairs_0based": [[0, 0], [1, 1]],
                }
            }
        }
    )

    assert "inactive_bonded_geometry" not in settings


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


def _test_hybrid_convergence_stops_environments_independently(tmp_path):
    from atom_openmm.covalent_workflow import _read_work, _rewrite_work
    from atom_openmm.hybrid_workflow import (
        _hybrid_convergence_callback,
    )

    for environment, samples in (("complex", 32), ("solvent", 40)):
        _rewrite_work(tmp_path / f"{environment}_forward.csv", [1.0] * samples)
        _rewrite_work(tmp_path / f"{environment}_reverse.csv", [-1.0] * samples)
    config = {
        "temperature_k": 300.0,
        "bootstrap_samples": 20,
        "random_seed": 2026,
        "convergence": {
            "enabled": True,
            "reopen_on_settings_change": False,
            "min_samples_per_direction": 30,
            "min_overlap_score_per_leg": 0.05,
            "max_dg_error_kcal_per_mol": 0.5,
            "consecutive_checks": 3,
            "max_dg_range_kcal_per_mol": 0.25,
            "check_interval_samples": 1,
            "stationarity": {
                "enabled": False,
                "discard_fraction": 0.1,
                "min_discard_samples": 5,
                "max_discard_first_shift_kcal_per_mol": 0.3,
                "max_discard_last_shift_kcal_per_mol": 0.2,
            },
        },
    }
    callback, state = _hybrid_convergence_callback(tmp_path, config)
    assert callback("complex", 32, [1.0] * 32, [-1.0] * 32)
    assert state["termination_reason"] is None
    assert callback("solvent", 40, [1.0] * 40, [-1.0] * 40)
    assert state["termination_reason"] == "converged"
    assert [
        row["sample_count_per_direction"]
        for row in state["environments"]["complex"]["history"]
    ] == [30, 31, 32]
    assert [
        row["sample_count_per_direction"]
        for row in state["environments"]["solvent"]["history"]
    ] == [30, 31, 32]
    assert len(_read_work(tmp_path / "complex_forward.csv")) == 32
    assert len(_read_work(tmp_path / "solvent_forward.csv")) == 40


def _test_hybrid_convergence_reopens_with_stationarity_and_spaced_checks(tmp_path):
    from atom_openmm.covalent_workflow import _rewrite_work
    from atom_openmm.hybrid_workflow import _hybrid_convergence_callback

    old_settings = {
        "enabled": True,
        "min_samples_per_direction": 30,
        "min_overlap_score_per_leg": 0.05,
        "max_dg_error_kcal_per_mol": 0.5,
        "consecutive_checks": 3,
        "max_dg_range_kcal_per_mol": 0.25,
    }
    (tmp_path / "neqti_convergence.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "settings": old_settings,
                "environments": {
                    name: {
                        "history": [],
                        "termination_reason": "converged",
                    }
                    for name in ("complex", "solvent")
                },
                "combined_history": [],
                "termination_reason": "converged",
            }
        )
    )
    for environment in ("complex", "solvent"):
        _rewrite_work(tmp_path / f"{environment}_forward.csv", [1.0] * 60)
        _rewrite_work(tmp_path / f"{environment}_reverse.csv", [-1.0] * 60)
    config = {
        "temperature_k": 300.0,
        "bootstrap_samples": 20,
        "random_seed": 2026,
        "convergence": {
            "enabled": True,
            "reopen_on_settings_change": True,
            "min_samples_per_direction": 50,
            "min_overlap_score_per_leg": 0.05,
            "max_dg_error_kcal_per_mol": 0.5,
            "consecutive_checks": 3,
            "max_dg_range_kcal_per_mol": 0.25,
            "check_interval_samples": 5,
            "stationarity": {
                "enabled": True,
                "discard_fraction": 0.1,
                "min_discard_samples": 5,
                "max_discard_first_shift_kcal_per_mol": 0.3,
                "max_discard_last_shift_kcal_per_mol": 0.2,
            },
        },
    }

    callback, state = _hybrid_convergence_callback(tmp_path, config)
    assert state["termination_reason"] is None
    assert state["reopened_from"]["termination_reason"] == "converged"
    assert callback("complex", 60, None, None)
    assert callback("solvent", 60, None, None)
    assert state["termination_reason"] == "converged"
    for environment in ("complex", "solvent"):
        history = state["environments"][environment]["history"]
        assert [row["sample_count_per_direction"] for row in history] == [50, 55, 60]
        assert all(row["stationarity"]["passed"] for row in history)
        assert all(row["stationarity"]["discard_samples"] >= 5 for row in history)


def _test_reopened_convergence_uses_latest_available_checks(tmp_path):
    from atom_openmm.covalent_workflow import _rewrite_work
    from atom_openmm.hybrid_workflow import _hybrid_convergence_callback

    old_settings = {
        "enabled": True,
        "min_samples_per_direction": 30,
    }
    (tmp_path / "neqti_convergence.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "settings": old_settings,
                "environments": {
                    name: {"history": [], "termination_reason": "converged"}
                    for name in ("complex", "solvent")
                },
                "combined_history": [],
                "termination_reason": "converged",
            }
        )
    )
    for environment in ("complex", "solvent"):
        _rewrite_work(tmp_path / f"{environment}_forward.csv", [1.0] * 84)
        _rewrite_work(tmp_path / f"{environment}_reverse.csv", [-1.0] * 84)
    config = {
        "temperature_k": 300.0,
        "bootstrap_samples": 20,
        "random_seed": 2026,
        "convergence": {
            "enabled": True,
            "reopen_on_settings_change": True,
            "min_samples_per_direction": 50,
            "min_overlap_score_per_leg": 0.05,
            "max_dg_error_kcal_per_mol": 0.5,
            "consecutive_checks": 3,
            "max_dg_range_kcal_per_mol": 0.25,
            "check_interval_samples": 5,
            "stationarity": {
                "enabled": False,
                "discard_fraction": 0.1,
                "min_discard_samples": 5,
                "max_discard_first_shift_kcal_per_mol": 0.3,
                "max_discard_last_shift_kcal_per_mol": 0.2,
            },
        },
    }

    callback, state = _hybrid_convergence_callback(tmp_path, config)
    assert callback("complex", 84, None, None)
    assert [
        row["sample_count_per_direction"]
        for row in state["environments"]["complex"]["history"]
    ] == [70, 75, 80]


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


def _test_hybrid_environment_iterators_stop_at_different_counts(tmp_path):
    from atom_openmm.covalent_workflow import _append_work, _read_work, _rewrite_work
    from atom_openmm.hybrid_workflow import (
        _run_independent_environment_iterators,
    )

    for environment in ("complex", "solvent"):
        _rewrite_work(tmp_path / f"{environment}_forward.csv", [1.0] * 20)
        _rewrite_work(tmp_path / f"{environment}_reverse.csv", [-1.0] * 20)

    def iterator(environment):
        for sample in range(21, 41):
            _append_work(tmp_path / f"{environment}_forward.csv", sample, 4.184)
            _append_work(tmp_path / f"{environment}_reverse.csv", sample, -4.184)
            yield [1.0] * sample, [-1.0] * sample, {"sample": sample}

    def callback(environment, _sample, _forward, _reverse):
        count = len(_read_work(tmp_path / f"{environment}_forward.csv"))
        return count >= {"complex": 22, "solvent": 25}[environment]

    summaries = _run_independent_environment_iterators(
        {"complex": iterator("complex"), "solvent": iterator("solvent")},
        callback,
    )
    assert summaries["complex"] == {"sample": 22}
    assert summaries["solvent"] == {"sample": 25}
    assert len(_read_work(tmp_path / "complex_forward.csv")) == 22
    assert len(_read_work(tmp_path / "solvent_forward.csv")) == 25
