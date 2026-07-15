import csv

import pytest
import yaml
import openmm as mm
from openmm.unit import kelvin, kilocalories_per_mole, picosecond


def _atom_options():
    return {
        "BASENAME": "test",
        "TEMPERATURES": [300.0],
        "LAMBDAS": [0.0, 0.5, 0.5, 1.0],
        "DIRECTION": [1, 1, -1, -1],
        "INTERMEDIATE": [0, 1, 1, 0],
        "LAMBDA1": [0.0, 0.5, 0.5, 0.0],
        "LAMBDA2": [0.0, 0.5, 0.5, 0.0],
        "ALPHA": [0.1, 0.1, 0.1, 0.1],
        "U0": [110.0, 110.0, 110.0, 110.0],
        "W0COEFF": [0.0, 0.0, 0.0, 0.0],
        "UMAX": 200.0,
        "UBCORE": 100.0,
        "ACORE": 0.0625,
        "PRODUCTION_STEPS": 10,
        "MAX_SAMPLES": 2,
    }


def _test_normalize_neqti_options_derives_two_leg_paths():
    from atom_openmm.neqti import normalize_neqti_options

    settings = normalize_neqti_options({"neqti": {"switch_steps_per_segment": 7}}, _atom_options())

    assert settings["paths"] == {
        "leg_a_forward": [0, 1], "leg_a_reverse": [1, 0],
        "leg_b_forward": [3, 2], "leg_b_reverse": [2, 3],
    }
    assert settings["hamiltonian"] == "atm_softplus_single_midpoint"
    assert settings["switch_steps_per_segment"] == 7
    assert settings["preparation_annealing_steps_per_segment"] == 0
    assert settings["n_snapshots"] == 2
    assert settings["resume"] is True
    assert settings["switch_integrator"] == "custom"
    assert settings["sampling_order"] == "interleaved"


def _test_normalize_neqti_options_accepts_batched_sampling_order():
    from atom_openmm.neqti import normalize_neqti_options

    settings = normalize_neqti_options({"neqti": {"sampling_order": "batched"}}, _atom_options())

    assert settings["sampling_order"] == "batched"


def _test_normalize_neqti_options_accepts_rest2_sampling():
    from atom_openmm.neqti import normalize_neqti_options

    settings = normalize_neqti_options(
        {"neqti": {
            "initial_equilibration_steps": 1000,
            "decorrelation_steps": 2000,
            "rest2": {
                "enabled": True,
                "effective_temperatures_k": [300, 450, 700],
                "exchange_interval_steps": 500,
            },
        }},
        _atom_options(),
    )

    assert settings["rest2"]["enabled"] is True
    assert settings["rest2"]["solute"] == "both_ligands"
    assert settings["rest2"]["effective_temperatures_k"] == [300.0, 450.0, 700.0]
    assert settings["rest2"]["ensembles"] == ["a", "m", "b"]
    assert settings["endpoint_system"] == "atm"


def _test_normalize_neqti_options_accepts_native_endpoint_rest2():
    from atom_openmm.neqti import normalize_neqti_options

    settings = normalize_neqti_options(
        {"neqti": {
            "endpoint_system": "native",
            "sampling_order": "batched",
            "preparation_annealing_steps_per_segment": 10,
            "initial_equilibration_steps": 1000,
            "decorrelation_steps": 2000,
            "rest2": {
                "enabled": True,
                "ensembles": ["a", "b"],
                "effective_temperatures_k": [300, 450, 600],
                "exchange_interval_steps": 500,
            },
        }},
        _atom_options(),
    )

    assert settings["endpoint_system"] == "native"
    assert settings["sampling_order"] == "batched"
    assert settings["rest2"]["ensembles"] == ["a", "b"]


@pytest.mark.parametrize(
    "neqti, message",
    [
        ({"endpoint_system": "native", "sampling_order": "batched"}, "requires REST2"),
        ({
            "endpoint_system": "native", "sampling_order": "batched", "decorrelation_steps": 500,
            "rest2": {"enabled": True},
        }, "ensembles: \\[a, b\\]"),
        ({
            "endpoint_system": "native", "decorrelation_steps": 500,
            "rest2": {"enabled": True, "ensembles": ["a", "b"]},
        }, "sampling_order: batched"),
    ],
)
def _test_normalize_neqti_options_rejects_incomplete_native_endpoint_mode(neqti, message):
    from atom_openmm.neqti import NEQTIConfigError, normalize_neqti_options

    with pytest.raises(NEQTIConfigError, match=message):
        normalize_neqti_options({"neqti": neqti}, _atom_options())


def _test_normalize_neqti_options_rejects_incompatible_rest2_steps():
    from atom_openmm.neqti import NEQTIConfigError, normalize_neqti_options

    with pytest.raises(NEQTIConfigError, match="decorrelation_steps must be divisible"):
        normalize_neqti_options(
            {"neqti": {
                "decorrelation_steps": 750,
                "rest2": {"enabled": True, "exchange_interval_steps": 500},
            }},
            _atom_options(),
        )


def _test_normalize_neqti_options_rejects_batched_rest2():
    from atom_openmm.neqti import NEQTIConfigError, normalize_neqti_options

    with pytest.raises(NEQTIConfigError, match="sampling_order: interleaved"):
        normalize_neqti_options(
            {"neqti": {"sampling_order": "batched", "rest2": {"enabled": True}}},
            _atom_options(),
        )


def _test_normalize_neqti_options_rejects_invalid_sampling_order():
    from atom_openmm.neqti import NEQTIConfigError, normalize_neqti_options

    with pytest.raises(NEQTIConfigError, match="sampling_order"):
        normalize_neqti_options({"neqti": {"sampling_order": "random"}}, _atom_options())


def _test_failed_switch_policy_preserves_legacy_defaults():
    from atom_openmm.neqti import normalize_neqti_options

    abort = normalize_neqti_options({"neqti": {}}, _atom_options())
    retry = normalize_neqti_options(
        {"neqti": {"tolerate_failed_switches": True}}, _atom_options()
    )
    counted = normalize_neqti_options(
        {"neqti": {"failed_switch_policy": "count_as_infinite"}}, _atom_options()
    )

    assert abort["failed_switch_policy"] == "abort"
    assert retry["failed_switch_policy"] == "retry"
    assert counted["failed_switch_policy"] == "count_as_infinite"
    assert counted["tolerate_failed_switches"] is False


def _test_numerical_switch_failure_classification_is_conservative():
    from atom_openmm.neqti import _is_numerical_switch_failure

    assert _is_numerical_switch_failure(mm.OpenMMException("Particle coordinate is NaN"))
    assert _is_numerical_switch_failure(
        mm.OpenMMException("The constraints could not be satisfied")
    )
    assert not _is_numerical_switch_failure(
        mm.OpenMMException("Error loading CUDA module: CUDA_ERROR_UNSUPPORTED_PTX_VERSION")
    )
    assert not _is_numerical_switch_failure(
        mm.OpenMMException("Requested two different values for random number seed")
    )
    assert not _is_numerical_switch_failure(KeyError("programming error"))


def _test_switch_schedule_interpolates_between_knots():
    from atom_openmm.neqti import build_atm_state_parameters, make_switch_schedule

    states = build_atm_state_parameters(_atom_options())
    schedule = make_switch_schedule(states, [0, 1], 2)

    assert len(schedule) == 2
    assert schedule[0]["lambda1"] == pytest.approx(0.25)
    assert schedule[1]["lambda1"] == pytest.approx(0.5)


def _test_legacy_state_path_is_rejected():
    from atom_openmm.neqti import NEQTIConfigError, normalize_neqti_options

    with pytest.raises(NEQTIConfigError, match="derives its two half paths"):
        normalize_neqti_options(
            {"neqti": {"state_path": [0, 1], "lambda_schedule": [0.0, 1.0]}},
            _atom_options(),
        )


def _test_schedule_without_adjacent_directional_midpoints_is_rejected():
    from atom_openmm.neqti import NEQTIConfigError, normalize_neqti_options

    with pytest.raises(NEQTIConfigError):
        options = _atom_options()
        options["INTERMEDIATE"] = [0, 1, 0, 0]
        normalize_neqti_options({"neqti": {}}, options)


def _test_protocol_manifest_rejects_legacy_artifacts(tmp_path, monkeypatch):
    from atom_openmm.neqti import NEQTIConfigError, _initialize_protocol_manifest

    monkeypatch.chdir(tmp_path)
    (tmp_path / "neqti_leg_a_forward.csv").write_text("legacy\n")
    settings = {"paths": {"leg_a_forward": [0, 1]}, "switch_steps_per_segment": 10}

    with pytest.raises(NEQTIConfigError, match="predate the single-midpoint protocol"):
        _initialize_protocol_manifest(settings, resume=True)


def _test_protocol_manifest_rejects_sampling_order_changes(tmp_path, monkeypatch):
    from atom_openmm.neqti import NEQTIConfigError, _ensure_work_csv, _initialize_protocol_manifest

    monkeypatch.chdir(tmp_path)
    settings = {
        "paths": {"leg_a_forward": [0, 1]},
        "switch_steps_per_segment": 10,
        "sampling_order": "batched",
    }
    _initialize_protocol_manifest(settings, resume=False)
    _ensure_work_csv(tmp_path / "neqti_leg_a_forward.csv")

    settings["sampling_order"] = "interleaved"
    with pytest.raises(NEQTIConfigError, match="sampling order"):
        _initialize_protocol_manifest(settings, resume=True)


def _test_count_as_infinite_changes_protocol_signature_only_for_that_policy():
    from atom_openmm.neqti import _protocol_signature

    settings = {"paths": {"leg_a_forward": [0, 1]}, "switch_steps_per_segment": 10}

    assert _protocol_signature(settings) == _protocol_signature(
        {**settings, "failed_switch_policy": "retry"}
    )
    assert _protocol_signature(settings) == _protocol_signature(
        {**settings, "failed_switch_policy": "abort"}
    )
    assert _protocol_signature(settings) != _protocol_signature(
        {**settings, "failed_switch_policy": "count_as_infinite"}
    )


def _test_disabled_rest2_preserves_existing_protocol_signature():
    from atom_openmm.neqti import _protocol_signature

    settings = {
        "paths": {"leg_a_forward": [0, 1]},
        "switch_steps_per_segment": 10,
    }
    disabled = {**settings, "rest2": {"enabled": False, "effective_temperatures_k": [300, 900]}}
    enabled = {**settings, "rest2": {"enabled": True, "effective_temperatures_k": [300, 900]}}

    assert _protocol_signature(settings) == _protocol_signature(disabled)
    assert _protocol_signature(settings) != _protocol_signature(enabled)


def _test_switch_progress_reports_effective_ns_per_day():
    from atom_openmm.neqti import _run_switch

    class FakeIntegrator:
        def getStepSize(self):
            return 0.002 * picosecond

    class FakeWorker:
        integrator = FakeIntegrator()

        def set_state(self, state):
            self.state = state

        def get_energy(self):
            return {"potential_energy": 0.0 * kilocalories_per_mole}

        def run(self, steps):
            pass

    messages = []

    class FakeLogger:
        def info(self, message, *args):
            messages.append(message % args)

    state = {"lambda1": 0.0}
    _run_switch(FakeWorker(), state, [state, state], 2, [0, 1], FakeLogger(), "forward trajectory 0")

    assert len(messages) == 1
    assert "state 0 -> 1" in messages[0]
    assert "ns/day" in messages[0]


def _test_worker_timestep_resolves_active_compound_integrator():
    from atom_openmm.neqti import _worker_timestep_ps

    compound = mm.CompoundIntegrator()
    compound.addIntegrator(mm.VerletIntegrator(0.001 * picosecond))
    compound.addIntegrator(mm.VerletIntegrator(0.004 * picosecond))
    compound.setCurrentIntegrator(1)
    worker = type("Worker", (), {"integrator": compound})()

    assert _worker_timestep_ps(worker) == pytest.approx(0.004)


def _test_bar_estimator_sign_convention():
    from atom_openmm.neqti import estimate_bar

    assert estimate_bar([2.0, 2.0, 2.0], [-2.0, -2.0, -2.0], 300.0) == pytest.approx(2.0)


def _test_bar_estimator_includes_infinite_work_observations():
    from atom_openmm.neqti import estimate_bar

    assert estimate_bar([2.0] * 4, [-2.0] * 3 + [float("inf")], 300.0) == pytest.approx(
        1.8284951018381972
    )
    assert estimate_bar([2.0] * 3 + [float("inf")], [-2.0] * 4, 300.0) == pytest.approx(
        2.17150489816196
    )
    assert estimate_bar([float("inf")], [-2.0], 300.0) is None


def _test_single_midpoint_analysis_combines_two_legs_without_bridge():
    from atom_openmm.neqti import analyze_two_leg_work

    work = {
        "leg_a_forward": [2.0, 2.0], "leg_a_reverse": [-2.0, -2.0],
        "leg_b_forward": [1.0, 1.0], "leg_b_reverse": [-1.0, -1.0],
    }

    result = analyze_two_leg_work(work, 300.0, bootstrap_samples=0)

    assert result["bar_dg_kcal_per_mol"] == pytest.approx(1.0)
    assert set(result["components"]) == {"leg_a", "leg_b"}
    assert result["overlap_score"] > 0.0


def _test_integrated_work_file_is_pmx_compatible(tmp_path):
    from atom_openmm.neqti import _write_integrated_work

    rows = [
        {"trajectory": "0", "work_kj_per_mol": "4.184"},
        {"trajectory": "1", "work_kj_per_mol": "8.368"},
    ]
    out = tmp_path / "integA.dat"
    _write_integrated_work(out, rows, "forward")

    assert out.read_text().splitlines() == [
        "forward_0 4.184",
        "forward_1 8.368",
    ]


def _test_completed_rows_filter_incomplete_rows(tmp_path):
    from atom_openmm.neqti import _read_completed_rows

    out = tmp_path / "neqti_forward.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trajectory", "status"])
        writer.writeheader()
        writer.writerow({"trajectory": 0, "status": "complete"})
        writer.writerow({"trajectory": 1, "status": "running"})

    assert _read_completed_rows(out) == [{"trajectory": "0", "status": "complete"}]


def _test_sampling_stream_resume_skips_initial_equilibration(tmp_path):
    from atom_openmm.neqti import _initialize_sampling_stream

    class FakeSimulation:
        def __init__(self):
            self.loaded = []

        def loadState(self, path):
            self.loaded.append(path)

    class FakeWorker:
        def __init__(self):
            self.simulation = FakeSimulation()
            self.checkpoints = []
            self.runs = []

        def set_chkpt(self, checkpoint):
            self.checkpoints.append(checkpoint)

        def set_state(self, state):
            pass

        def run(self, steps):
            self.runs.append(steps)

        def get_chkpt(self):
            return b"current"

    class FakeLogger:
        def info(self, *args):
            pass

    checkpoint = tmp_path / "sampling.chk"
    checkpoint.write_bytes(b"saved")
    worker = FakeWorker()
    _initialize_sampling_stream(
        worker,
        direction="forward",
        initial_state_file="endpoint.xml",
        start_state={},
        checkpoint_file=checkpoint,
        completed_count=7,
        initial_equilibration_steps=25000,
        resume=True,
        logger=FakeLogger(),
    )

    assert worker.checkpoints == [b"saved"]
    assert worker.simulation.loaded == []
    assert worker.runs == []


def _test_legacy_resume_without_checkpoint_skips_initial_equilibration(tmp_path, monkeypatch):
    from atom_openmm import neqti

    class FakeSimulation:
        def __init__(self):
            self.loaded = []

        def loadState(self, path):
            self.loaded.append(path)

    class FakeWorker:
        def __init__(self):
            self.simulation = FakeSimulation()
            self.runs = []

        def set_state(self, state):
            pass

        def run(self, steps):
            self.runs.append(steps)

        def get_chkpt(self):
            return b"current"

    class FakeLogger:
        def info(self, *args):
            pass

    monkeypatch.setattr(neqti, "_write_worker_pdb_pair", lambda worker, path: None)
    worker = FakeWorker()
    checkpoint = tmp_path / "sampling.chk"
    neqti._initialize_sampling_stream(
        worker,
        direction="forward",
        initial_state_file="endpoint.xml",
        start_state={},
        checkpoint_file=checkpoint,
        completed_count=7,
        initial_equilibration_steps=25000,
        resume=True,
        logger=FakeLogger(),
    )

    assert worker.simulation.loaded == ["endpoint.xml"]
    assert worker.runs == []
    assert checkpoint.read_bytes() == b"current"


def _test_incompatible_sampling_checkpoint_has_actionable_error(tmp_path):
    from atom_openmm.neqti import NEQTIConfigError, _initialize_sampling_stream

    class FakeWorker:
        def set_chkpt(self, checkpoint):
            raise RuntimeError("incompatible")

    class FakeLogger:
        def info(self, *args):
            pass

    checkpoint = tmp_path / "sampling.chk"
    checkpoint.write_bytes(b"old")
    with pytest.raises(NEQTIConfigError, match="start a clean job"):
        _initialize_sampling_stream(
            FakeWorker(),
            direction="forward",
            initial_state_file="endpoint.xml",
            start_state={},
            checkpoint_file=checkpoint,
            completed_count=1,
            initial_equilibration_steps=10,
            resume=True,
            logger=FakeLogger(),
        )


def _test_switch_validation_restores_snapshot_and_writes_report(tmp_path, monkeypatch):
    from atom_openmm import neqti

    events = []

    class FakeWorker:
        def set_chkpt(self, checkpoint):
            events.append(("checkpoint", checkpoint))

        def set_state(self, state):
            events.append(("state", state))

    class FakeLogger:
        def info(self, *args):
            pass

    monkeypatch.setattr(neqti, "_run_switch_custom", lambda *args, **kwargs: 1.25)
    monkeypatch.setattr(neqti, "_run_switch", lambda *args, **kwargs: 1.5)
    monkeypatch.setattr(neqti, "_effective_ns_per_day", lambda *args, **kwargs: 10.0)
    output = tmp_path / "validation.yaml"
    neqti._validate_switch_implementations(
        FakeWorker(),
        direction="forward",
        snapshot=b"snapshot",
        start_state={"lambda1": 0.0},
        schedule=[{}, {}],
        steps_per_segment=1,
        state_path=[0, 1, 2],
        logger=FakeLogger(),
        output_file=output,
    )

    report = yaml.safe_load(output.read_text())
    assert report["directions"]["forward"]["custom_work_kcal_per_mol"] == 1.25
    assert report["directions"]["forward"]["python_work_kcal_per_mol"] == 1.5
    assert report["directions"]["forward"]["work_difference_kcal_per_mol"] == -0.25
    assert [event for event in events if event[0] == "checkpoint"] == [
        ("checkpoint", b"snapshot"),
        ("checkpoint", b"snapshot"),
        ("checkpoint", b"snapshot"),
    ]


def _test_neqti_loads_physical_initial_state_for_both_directions(tmp_path, monkeypatch):
    from atom_openmm import neqti

    monkeypatch.chdir(tmp_path)
    options = _atom_options()
    options.update(
        {
            "BASENAME": "pair",
            "NEQTI_INITIAL_STATE_FILE": "pair_equil.xml",
        }
    )

    load_calls = []
    worker_options = {}
    events = []

    class FakeSimulation:
        def loadState(self, path):
            load_calls.append(path)

    class FakeWorker:
        def __init__(self, basename, ommsystem, options, node_info=None, compute=True, logger=None):
            worker_options.update(options)
            self.simulation = FakeSimulation()

        def set_state(self, par):
            pass

        def run(self, nsteps):
            pass

        def get_chkpt(self):
            return b"checkpoint"

        def set_chkpt(self, chkpt):
            events.append(("restore", chkpt))

        def get_energy(self):
            return {"potential_energy": 0.0 * kilocalories_per_mole}

        def finish(self):
            pass

    class FakeOMMSystem:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(neqti, "_select_node_info", lambda options, neqti_options: {"node_name": "local"})
    monkeypatch.setattr(neqti, "OMMSystemRBFE", FakeOMMSystem)
    monkeypatch.setattr(neqti, "OMMWorkerATMSync", FakeWorker)
    monkeypatch.setattr(
        neqti,
        "_write_worker_pdb_pair",
        lambda worker, path: events.append(("pdb", str(path))),
    )

    progress = []
    summary = neqti.run_neqti(
        options,
        {
            "initial_equilibration_steps": 0,
            "n_snapshots": 1,
            "decorrelation_steps": 0,
            "switch_steps_per_segment": 1,
            "hamiltonian": "atm_softplus_single_midpoint",
            "paths": {
                "leg_a_forward": [0, 1], "leg_a_reverse": [1, 0],
                "leg_b_forward": [3, 2], "leg_b_reverse": [2, 3],
            },
            "resume": False,
            "bootstrap_samples": 0,
            "random_seed": 1,
            "platform": None,
            "switch_integrator": "python",
            "validate_switch_integrator": False,
            "preparation_annealing_steps_per_segment": 0,
            "tolerate_failed_switches": False,
            "max_switch_attempts_per_direction": 1,
        },
        progress_callback=progress.append,
    )

    assert worker_options["INITIAL_STATE_FILE"] == "pair_equil.xml"
    assert load_calls == ["pair_equil.xml"] * 3
    assert summary["forward_samples"] == 2
    assert summary["reverse_samples"] == 2
    assert summary["analysis"]["bar_dg_kcal_per_mol"] == pytest.approx(0.0)
    assert summary["sample_counts"] == {
        "leg_a_forward": 1,
        "leg_a_reverse": 1,
        "leg_b_forward": 1,
        "leg_b_reverse": 1,
    }


def _run_fake_neqti_for_order(
    tmp_path, monkeypatch, sampling_order, *, switch_implementation=None, settings_overrides=None
):
    from atom_openmm import neqti

    monkeypatch.chdir(tmp_path)
    options = _atom_options()
    options.update({"BASENAME": "pair", "NEQTI_INITIAL_STATE_FILE": "pair_equil.xml"})

    class FakeSimulation:
        def loadState(self, path):
            pass

    class FakeWorker:
        def __init__(self, basename, ommsystem, options, node_info=None, compute=True, logger=None):
            self.simulation = FakeSimulation()

        def set_state(self, par):
            pass

        def run(self, nsteps):
            pass

        def get_chkpt(self):
            return b"checkpoint"

        def set_chkpt(self, chkpt):
            pass

        def get_energy(self):
            return {"potential_energy": 0.0 * kilocalories_per_mole}

        def finish(self):
            pass

    class FakeOMMSystem:
        def __init__(self, *args, **kwargs):
            pass

    labels = []
    progress = []

    def fake_run_switch(worker, start_state, schedule, steps_per_segment, path, logger, label):
        labels.append(label.split()[0])
        if switch_implementation is not None:
            return switch_implementation(label)
        return 0.0

    monkeypatch.setattr(neqti, "_select_node_info", lambda options, neqti_options: {"node_name": "local"})
    monkeypatch.setattr(neqti, "OMMSystemRBFE", FakeOMMSystem)
    monkeypatch.setattr(neqti, "OMMWorkerATMSync", FakeWorker)
    monkeypatch.setattr(neqti, "_write_worker_pdb_pair", lambda worker, path: None)
    monkeypatch.setattr(neqti, "_run_switch", fake_run_switch)

    settings = {
            "initial_equilibration_steps": 0,
            "n_snapshots": 2,
            "decorrelation_steps": 0,
            "switch_steps_per_segment": 1,
            "hamiltonian": "atm_softplus_single_midpoint",
            "paths": {
                "leg_a_forward": [0, 1], "leg_a_reverse": [1, 0],
                "leg_b_forward": [3, 2], "leg_b_reverse": [2, 3],
            },
            "resume": False,
            "bootstrap_samples": 0,
            "random_seed": 1,
            "platform": None,
            "switch_integrator": "python",
            "sampling_order": sampling_order,
            "validate_switch_integrator": False,
            "preparation_annealing_steps_per_segment": 0,
            "tolerate_failed_switches": False,
            "failed_switch_policy": "abort",
            "max_switch_attempts_per_direction": 2,
        }
    settings.update(settings_overrides or {})
    summary = neqti.run_neqti(
        options,
        settings,
        progress_callback=progress.append,
    )
    return labels, progress, summary


def _test_interleaved_sampling_runs_all_legs_per_cycle(tmp_path, monkeypatch):
    labels, progress, summary = _run_fake_neqti_for_order(tmp_path, monkeypatch, "interleaved")

    assert labels == [
        "leg_a_reverse", "leg_b_reverse", "leg_a_forward", "leg_b_forward",
        "leg_a_reverse", "leg_b_reverse", "leg_a_forward", "leg_b_forward",
    ]
    assert progress[-1]["analysis"]["bar_dg_kcal_per_mol"] == pytest.approx(0.0)
    assert summary["sample_counts"] == {
        "leg_a_forward": 2,
        "leg_a_reverse": 2,
        "leg_b_forward": 2,
        "leg_b_reverse": 2,
    }


def _test_batched_sampling_preserves_old_leg_order(tmp_path, monkeypatch):
    labels, progress, summary = _run_fake_neqti_for_order(tmp_path, monkeypatch, "batched")

    assert labels == [
        "leg_a_reverse", "leg_b_reverse", "leg_a_reverse", "leg_b_reverse",
        "leg_a_forward", "leg_a_forward", "leg_b_forward", "leg_b_forward",
    ]
    assert summary["sample_counts"]["leg_b_forward"] == 2


def _test_counted_numerical_failure_consumes_sample_without_replacement(tmp_path, monkeypatch):
    failed = False

    def switch(label):
        nonlocal failed
        if label.startswith("leg_a_reverse") and not failed:
            failed = True
            raise mm.OpenMMException("Particle coordinate is NaN")
        return 0.0

    labels, progress, summary = _run_fake_neqti_for_order(
        tmp_path,
        monkeypatch,
        "interleaved",
        switch_implementation=switch,
        settings_overrides={
            "n_snapshots": 1,
            "failed_switch_policy": "count_as_infinite",
            "max_switch_attempts_per_direction": 2,
        },
    )

    assert labels.count("leg_a_reverse") == 1
    assert summary["sample_counts"]["leg_a_reverse"] == 1
    assert summary["finite_sample_counts"]["leg_a_reverse"] == 0
    assert summary["counted_infinite_work_counts"]["leg_a_reverse"] == 1
    assert summary["failed_switch_counts"]["leg_a_reverse"] == 0
    assert summary["analysis"] is None
    with (tmp_path / "neqti_leg_a_reverse.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "counted_infinite"
    assert rows[0]["work_kcal_per_mol"] == "inf"
    assert progress[-1]["counted_infinite_work_counts"]["leg_a_reverse"] == 1

    resumed_labels, _, resumed = _run_fake_neqti_for_order(
        tmp_path,
        monkeypatch,
        "interleaved",
        switch_implementation=lambda label: pytest.fail(f"unexpected resumed switch: {label}"),
        settings_overrides={
            "n_snapshots": 1,
            "resume": True,
            "failed_switch_policy": "count_as_infinite",
            "max_switch_attempts_per_direction": 2,
        },
    )
    assert resumed_labels == []
    assert resumed["sample_counts"]["leg_a_reverse"] == 1


def _test_retry_policy_replaces_failed_switch(tmp_path, monkeypatch):
    calls = 0

    def switch(label):
        nonlocal calls
        if label.startswith("leg_a_reverse"):
            calls += 1
            if calls == 1:
                raise mm.OpenMMException("Particle coordinate is NaN")
        return 0.0

    labels, _, summary = _run_fake_neqti_for_order(
        tmp_path,
        monkeypatch,
        "interleaved",
        switch_implementation=switch,
        settings_overrides={
            "n_snapshots": 1,
            "failed_switch_policy": "retry",
            "tolerate_failed_switches": True,
            "max_switch_attempts_per_direction": 2,
        },
    )

    assert labels.count("leg_a_reverse") == 2
    assert summary["sample_counts"]["leg_a_reverse"] == 1
    assert summary["failed_switch_counts"]["leg_a_reverse"] == 1
    assert summary["counted_infinite_work_counts"]["leg_a_reverse"] == 0


def _test_count_as_infinite_does_not_hide_infrastructure_failure(tmp_path, monkeypatch):
    def switch(label):
        raise mm.OpenMMException(
            "Error loading CUDA module: CUDA_ERROR_UNSUPPORTED_PTX_VERSION"
        )

    with pytest.raises(mm.OpenMMException, match="UNSUPPORTED_PTX_VERSION"):
        _run_fake_neqti_for_order(
            tmp_path,
            monkeypatch,
            "interleaved",
            switch_implementation=switch,
            settings_overrides={
                "n_snapshots": 1,
                "failed_switch_policy": "count_as_infinite",
                "max_switch_attempts_per_direction": 2,
            },
        )

    with (tmp_path / "neqti_leg_a_reverse.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "failed"
    assert rows[0]["work_kcal_per_mol"] == ""
