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
