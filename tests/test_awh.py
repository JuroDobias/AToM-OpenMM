import numpy as np
import pytest


def _atom_options():
    return {
        "TEMPERATURES": [300.0],
        "LAMBDAS": [0.0, 0.5, 0.5, 1.0],
        "DIRECTION": [1, 1, -1, -1],
        "INTERMEDIATE": [0, 1, 1, 0],
        "LAMBDA1": [0.0, 0.5, 0.5, 0.0],
        "LAMBDA2": [0.0, 0.5, 0.5, 0.0],
        "ALPHA": [0.1] * 4,
        "U0": [110.0] * 4,
        "W0COEFF": [0.0] * 4,
        "UMAX": 200.0,
        "UBCORE": 100.0,
        "ACORE": 0.0625,
    }


def _test_awh_options_and_graph_include_physical_endpoints_once():
    from atom_openmm.awh import build_awh_state_graph, normalize_awh_options
    from atom_openmm.neqti import build_atm_state_parameters

    settings = normalize_awh_options(
        {"awh": {"rest2": {"effective_temperatures_k": [300, 450, 600]}}},
        _atom_options(),
    )
    graph = build_awh_state_graph(build_atm_state_parameters(_atom_options()), settings)
    assert [node["name"] for node in graph] == [
        "a_hot_600",
        "a_hot_450",
        "a_physical",
        "atm_1",
        "atm_2",
        "b_physical",
        "b_hot_450",
        "b_hot_600",
    ]
    assert graph[0]["rest2"] == {"a": 0.5, "b": 1.0}
    assert graph[-1]["rest2"] == {"a": 1.0, "b": 0.5}


def _test_awh_bias_moves_free_energy_against_oversampling():
    from atom_openmm.awh import AWHBias

    bias = AWHBias(3, initial_histogram_size=10)
    before = bias.free_energy.copy()
    bias.update([0, 1], np.array([0.9, 0.1]))
    assert bias.free_energy[0] < before[0] or bias.free_energy[1] > before[1]
    assert np.isfinite(bias.bias).all()


def _test_awh_fixed_learning_rate_does_not_decay():
    from atom_openmm.awh import AWHBias

    bias = AWHBias(3, initial_histogram_size=10)
    bias.update([0, 1], np.array([0.9, 0.1]), learning_rate_kbt=0.1)
    first_change = bias.free_energy[1] - bias.free_energy[0]
    bias.update([0, 1], np.array([0.9, 0.1]), learning_rate_kbt=0.1)
    second_change = bias.free_energy[1] - bias.free_energy[0]
    assert second_change - first_change == pytest.approx(first_change)


def _test_awh_bias_roundtrip_serialization():
    from atom_openmm.awh import AWHBias

    bias = AWHBias(4, initial_histogram_size=20)
    bias.update([1, 2], np.array([0.4, 0.6]))
    bias.visits[2] = 7
    restored = AWHBias.from_dict(bias.to_dict())
    assert np.allclose(restored.free_energy, bias.free_energy)
    assert restored.visits.tolist() == [0, 0, 7, 0]


def _test_mbar_recovers_sampled_two_state_offset():
    from atom_openmm.awh import estimate_mbar

    rng = np.random.default_rng(7)
    states = np.repeat([0, 1], 2000)
    x = np.concatenate([rng.normal(0, 1, 2000), rng.normal(2, 1, 2000)])
    u0 = 0.5 * x**2
    u1 = 0.5 * (x - 2) ** 2 + 1.25
    estimate = estimate_mbar(np.column_stack([u0, u1]), states, 2)
    assert estimate is not None
    assert estimate[1] == pytest.approx(1.25, abs=0.08)


def _test_awh_rejects_nonuniform_target_for_v1():
    from atom_openmm.awh import AWHConfigError, normalize_awh_options

    with pytest.raises(AWHConfigError, match="uniform"):
        normalize_awh_options(
            {"awh": {"target_distribution": "metric"}},
            _atom_options(),
        )


def _test_awh_analysis_defaults_are_normalized():
    from atom_openmm.awh import normalize_awh_options

    settings = normalize_awh_options({"awh": {}}, _atom_options())
    assert settings["analysis"]["trajectory"] == {
        "enabled": True,
        "interval_moves": 10,
        "atom_selection": "!:HOH,WAT,NA,CL,K,CA",
    }
    assert settings["analysis"]["thresholds"]["min_adjacent_overlap"] == 0.03
    assert settings["analysis"]["friction"] == {
        "enabled": False,
        "max_correlation_lag_moves": 50,
        "min_effective_samples": 200,
    }
    assert settings["adaptive"]["learning_rate_kbt"] == pytest.approx(0.1)
    assert settings["adaptive"]["metric_target"]["enabled"] is False
    assert settings["atm_state_count"] == 4


def _test_awh_densifies_both_legs_and_preserves_midpoints():
    from atom_openmm.awh import densify_atm_states
    from atom_openmm.neqti import build_atm_state_parameters, split_two_leg_paths

    original = build_atm_state_parameters(_atom_options())
    dense = densify_atm_states(original, 20)
    paths = split_two_leg_paths(dense)
    assert len(dense) == 20
    assert dense[0]["lambda2"] == pytest.approx(original[0]["lambda2"])
    assert dense[-1]["lambda2"] == pytest.approx(original[-1]["lambda2"])
    assert dense[paths["leg_a_forward"][-1]]["atmdirection"] == 1
    assert dense[paths["leg_b_reverse"][0]]["atmdirection"] == -1
    original_delta = max(
        abs(b["lambda2"] - a["lambda2"]) for a, b in zip(original, original[1:])
    )
    dense_delta = max(
        abs(b["lambda2"] - a["lambda2"]) for a, b in zip(dense, dense[1:])
    )
    assert dense_delta < original_delta


def _test_awh_analysis_output_settings_do_not_change_dynamics_signature():
    from atom_openmm.awh import _protocol_signature, normalize_awh_options

    first = normalize_awh_options({"awh": {}}, _atom_options())
    second = normalize_awh_options(
        {
            "awh": {
                "analysis": {
                    "trajectory": {"enabled": False, "interval_moves": 50},
                    "friction": {
                        "enabled": True,
                        "max_correlation_lag_moves": 20,
                    },
                    "thresholds": {"min_adjacent_overlap": 0.1},
                }
            }
        },
        _atom_options(),
    )
    assert _protocol_signature(_atom_options(), first) == _protocol_signature(
        _atom_options(), second
    )


def _test_awh_metric_target_changes_dynamics_signature():
    from atom_openmm.awh import _protocol_signature, normalize_awh_options

    first = normalize_awh_options({"awh": {}}, _atom_options())
    second = normalize_awh_options(
        {
            "awh": {
                "analysis": {"friction": {"enabled": True}},
                "adaptive": {"metric_target": {"enabled": True}},
            }
        },
        _atom_options(),
    )
    assert _protocol_signature(_atom_options(), first) != _protocol_signature(
        _atom_options(), second
    )


def _test_awh_state_sampling_defaults_preserve_legacy_signature():
    from atom_openmm.awh import _protocol_signature, normalize_awh_options

    omitted = normalize_awh_options({"awh": {}}, _atom_options())
    explicit = normalize_awh_options(
        {"awh": {"state_sampling": {"method": "legacy_local_gibbs"}}},
        _atom_options(),
    )
    global_gibbs = normalize_awh_options(
        {"awh": {"state_sampling": {"method": "hybrid_global_gibbs"}}},
        _atom_options(),
    )
    assert omitted["state_sampling"]["method"] == "legacy_local_gibbs"
    assert _protocol_signature(_atom_options(), omitted) == _protocol_signature(
        _atom_options(), explicit
    )
    assert _protocol_signature(_atom_options(), omitted) != _protocol_signature(
        _atom_options(), global_gibbs
    )


def _test_atm_energy_reconstructs_both_directions():
    from openmm import unit

    from atom_openmm.atm_energy import reconstruct_atm_energies
    from atom_openmm.neqti import build_atm_state_parameters

    states = build_atm_state_parameters(_atom_options())
    energies = reconstruct_atm_energies(
        states,
        reference_total_energy=103.0,
        reference_bias_energy=3.0,
        reference_direction=1,
        u0=20.0,
        u1=25.0,
    )
    assert energies.shape == (4,)
    assert energies[0] == pytest.approx(100.0)
    assert energies[-1] == pytest.approx(105.0)
    assert np.isfinite(energies).all()
    assert states[0]["uoffset"].value_in_unit(unit.kilojoule_per_mole) == 0.0


def _test_atm_softcore_matches_reference_expression():
    from atom_openmm.atm_energy import softcore_perturbation_energy

    energy = 150.0
    maximum = 200.0
    core = 100.0
    exponent = 0.0625
    reduced = (energy - core) / (exponent * (maximum - core))
    zeta = 1.0 + 2.0 * reduced * (reduced + 1.0)
    expected = (
        (maximum - core)
        * (zeta**exponent - 1.0)
        / (zeta**exponent + 1.0)
        + core
    )
    assert softcore_perturbation_energy(
        energy, maximum, core, exponent
    ) == pytest.approx(expected)


def _test_global_gibbs_diagnostics_reports_truncated_mass():
    from atom_openmm.awh import GlobalGibbsDiagnostics

    diagnostics = GlobalGibbsDiagnostics()
    expected = diagnostics.update(
        previous=3,
        selected=5,
        probabilities=np.asarray([0.0, 0.0, 0.1, 0.7, 0.1, 0.1, 0.0]),
    )
    result = diagnostics.to_dict()
    assert expected == pytest.approx(0.4)
    assert result["jump_distance_histogram"] == {"2": 1}
    assert result["truncated_probability_mass"]["1"]["mean"] == pytest.approx(0.9)


def _test_rest2_scan_integrator_preserves_coordinates_and_time():
    import openmm as mm
    from openmm import unit

    from atom_openmm.awh_global_gibbs import AWHREST2EnergyScanIntegrator
    from atom_openmm.neqti import build_atm_state_parameters

    class _ATMForceNames:
        @staticmethod
        def Lambda1():
            return "Lambda1"

        @staticmethod
        def Lambda2():
            return "Lambda2"

        @staticmethod
        def Alpha():
            return "Alpha"

        @staticmethod
        def Uh():
            return "Uh"

        @staticmethod
        def W0():
            return "W0"

        @staticmethod
        def Direction():
            return "Direction"

        @staticmethod
        def Umax():
            return "Umax"

        @staticmethod
        def Ubcore():
            return "Ubcore"

        @staticmethod
        def Acore():
            return "Acore"

    class _REST2:
        parameters = {"a": ("REST_A", "REST_A_SQRT")}

    class _OMMSystem:
        atmforce = _ATMForceNames()
        multisoftplus = False
        rest2_system = _REST2()

    system = mm.System()
    system.addParticle(1.0)
    force = mm.CustomExternalForce("Lambda2 + REST_A + REST_A_SQRT")
    for name in (
        "Lambda1",
        "Lambda2",
        "Alpha",
        "Uh",
        "W0",
        "Direction",
        "Umax",
        "Ubcore",
        "Acore",
        "UOffset",
        "REST_A",
        "REST_A_SQRT",
    ):
        force.addGlobalParameter(name, 0.0)
    force.addParticle(0, [])
    system.addForce(force)
    states = build_atm_state_parameters(_atom_options())
    graph = [
        {
            "kind": "rest2_a",
            "atm_state": 0,
            "rest2": {"a": 0.25},
        },
        {
            "kind": "physical",
            "atm_state": 0,
            "rest2": {"a": 1.0},
        },
    ]
    integrator = AWHREST2EnergyScanIntegrator(_OMMSystem(), states, graph)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.25, 0.0, 0.0]])
    context.setVelocities([[0.5, 0.0, 0.0]])
    before = context.getState(getPositions=True, getVelocities=True)
    integrator.step(1)
    after = context.getState(getPositions=True, getVelocities=True)
    assert integrator.scanned_energies()[0] == pytest.approx(0.75)
    assert integrator.reference_energy() == pytest.approx(2.0)
    assert np.allclose(
        before.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        after.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
    )
    assert np.allclose(
        before.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        ),
        after.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        ),
    )
    assert after.getTime().value_in_unit(unit.picosecond) == pytest.approx(0.0)


def _test_portable_state_load_omits_incompatible_integrator_metadata(tmp_path):
    import openmm as mm
    from openmm import unit
    from openmm.app import Simulation, Topology

    from atom_openmm.ommworker import OMMWorker

    system = mm.System()
    system.addParticle(1.0)
    force = mm.CustomExternalForce("state_parameter")
    force.addGlobalParameter("state_parameter", 0.0)
    force.addParticle(0, [])
    system.addForce(force)
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL", chain)
    topology.addAtom("X", None, residue)

    source_integrator = mm.CustomIntegrator(0.001)
    source_integrator.addGlobalVariable("source_only", 3.0)
    source = Simulation(
        topology,
        system,
        source_integrator,
        mm.Platform.getPlatformByName("Reference"),
    )
    source.context.setPositions([[0.25, 0.0, 0.0]])
    source.context.setVelocities([[0.5, 0.0, 0.0]])
    source.context.setParameter("state_parameter", 7.0)
    source.context.setTime(2.5)
    source.context.setStepCount(17)
    state_path = tmp_path / "source.xml"
    source.saveState(str(state_path))

    compound = mm.CompoundIntegrator()
    compound.addIntegrator(mm.VerletIntegrator(0.001))
    compound.addIntegrator(mm.CustomIntegrator(0.0))
    target = Simulation(
        topology,
        system,
        compound,
        mm.Platform.getPlatformByName("Reference"),
    )
    worker = object.__new__(OMMWorker)
    worker.context = target.context
    worker.simulation = target
    worker.load_state(state_path, ignore_integrator_parameters=True)
    restored = target.context.getState(getPositions=True, getVelocities=True)
    assert np.allclose(
        restored.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        [[0.25, 0.0, 0.0]],
    )
    assert np.allclose(
        restored.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        ),
        [[0.5, 0.0, 0.0]],
    )
    assert target.context.getParameter("state_parameter") == pytest.approx(7.0)
    assert target.context.getTime().value_in_unit(unit.picosecond) == pytest.approx(2.5)
    assert target.context.getStepCount() == 17


def _test_disabled_metric_target_preserves_legacy_signature():
    from copy import deepcopy

    from atom_openmm.awh import _protocol_signature, normalize_awh_options

    current = normalize_awh_options({"awh": {}}, _atom_options())
    legacy = deepcopy(current)
    legacy["adaptive"].pop("metric_target")
    assert _protocol_signature(_atom_options(), current) == _protocol_signature(
        _atom_options(), legacy
    )


def _test_generalized_force_uses_centered_and_endpoint_differences():
    from atom_openmm.awh_friction import generalized_forces

    values = {0: 0.0, 1: 1.0, 2: 4.0, 3: 9.0}
    result = generalized_forces(values, [0, 1, 2, 3], 4)
    assert result == pytest.approx({0: 1.0, 1: 2.0, 2: 4.0, 3: 5.0})


def _test_correlated_force_has_larger_friction_than_white_noise():
    from atom_openmm.awh_friction import FrictionAccumulator

    rng = np.random.default_rng(11)
    white = FrictionAccumulator(1, maximum_lag=40, sample_interval_ps=0.2)
    correlated = FrictionAccumulator(1, maximum_lag=40, sample_interval_ps=0.2)
    value = 0.0
    for _ in range(5000):
        white.update({0: (1.0, rng.normal())})
        value = 0.9 * value + np.sqrt(1.0 - 0.9**2) * rng.normal()
        correlated.update({0: (1.0, value)})
    white_metric = white.records(minimum_effective_samples=200)[0]
    correlated_metric = correlated.records(minimum_effective_samples=200)[0]
    assert correlated_metric["friction_kbt2_ps_per_state2"] > (
        5 * white_metric["friction_kbt2_ps_per_state2"]
    )


def _test_friction_checkpoint_and_summary_are_yaml_serializable():
    import yaml

    from atom_openmm.awh_friction import FrictionAccumulator, friction_summary

    accumulator = FrictionAccumulator(2, maximum_lag=2, sample_interval_ps=0.2)
    for value in (0.0, 1.0, -1.0, 0.5):
        accumulator.update({0: (0.8, value), 1: (0.2, -value)})
    restored = FrictionAccumulator.from_dict(accumulator.to_dict())
    summary = friction_summary(restored, [{"name": "a"}, {"name": "b"}], 1)
    yaml.safe_dump(summary)


def _test_friction_target_is_uniform_or_capped():
    from atom_openmm.awh_friction import friction_target

    uniform = [
        {"friction_kbt2_ps_per_state2": 4.0},
        {"friction_kbt2_ps_per_state2": 4.0},
        {"friction_kbt2_ps_per_state2": 4.0},
    ]
    assert friction_target(uniform) == pytest.approx([1 / 3] * 3)
    uneven = [
        {"friction_kbt2_ps_per_state2": 1.0},
        {"friction_kbt2_ps_per_state2": 10000.0},
        {"friction_kbt2_ps_per_state2": 1.0},
    ]
    target = friction_target(uneven, maximum_relative_weight=2.0)
    assert target[1] > target[0]
    assert target[1] / target[0] <= 4.0 + 1e-12


def _test_friction_samples_reconcile_to_checkpoint(tmp_path):
    import csv

    from atom_openmm.awh_friction import (
        FRICTION_SAMPLE_FIELDS,
        reconcile_friction_samples,
    )

    path = tmp_path / "friction.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRICTION_SAMPLE_FIELDS)
        writer.writeheader()
        for move in (1, 2, 3):
            writer.writerow(
                {
                    "move": move,
                    "steps": move * 100,
                    "stage": "adaptive",
                    "state": 0,
                    "state_name": "a",
                    "probability": 1.0,
                    "force_kbt_per_state": 0.0,
                }
            )
    reconcile_friction_samples(path, 2)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["move"]) for row in rows] == [1, 2]


def _test_awh_diagnostics_report_rest2_returns_overlap_and_ess():
    from atom_openmm.awh import (
        build_awh_state_graph,
        estimate_mbar,
        normalize_awh_options,
    )
    from atom_openmm.awh_analysis import analyze_awh_diagnostics
    from atom_openmm.neqti import build_atm_state_parameters

    settings = normalize_awh_options(
        {"awh": {"rest2": {"effective_temperatures_k": [300, 450, 600]}}},
        _atom_options(),
    )
    graph = build_awh_state_graph(
        build_atm_state_parameters(_atom_options()), settings
    )
    path = [2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5]
    rows = []
    previous = 2
    for move, state in enumerate(path, 1):
        rows.append(
            {
                "move": move,
                "stage": "production",
                "previous_state": previous,
                "state": state,
                "left_probability": 0.2,
                "stay_probability": 0.6,
                "right_probability": 0.2,
            }
        )
        previous = state
    sampled_states = np.repeat(np.arange(len(graph)), 5)
    matrix = np.zeros((len(sampled_states), len(graph)))
    free = estimate_mbar(matrix, sampled_states, len(graph))
    result = analyze_awh_diagnostics(
        graph,
        trace_rows=rows,
        sampled_states=sampled_states,
        reduced_energies=matrix,
        free_energies=free,
        thresholds={
            "min_adjacent_overlap": 0.01,
            "min_endpoint_effective_samples": 1,
            "min_rest2_hot_returns": 1,
            "min_uniform_occupancy_overlap": 0.1,
        },
    )
    assert result["uwham"]["minimum_adjacent_overlap"] == pytest.approx(1 / 8)
    assert result["uwham"]["endpoint_effective_samples"]["a_physical"] == pytest.approx(40)
    assert (
        result["rest2"]["endpoint_a"]["production"][
            "complete_physical_hottest_physical_returns"
        ]
        == 1
    )
    assert (
        result["rest2"]["endpoint_b"]["production"][
            "complete_physical_hottest_physical_returns"
        ]
        == 1
    )
    assert result["edges"][2]["expected_left_to_right_probability"] == pytest.approx(0.2)
    assert result["quality_passed"]


def _test_state_tagged_trajectory_writes_frame_metadata(tmp_path, monkeypatch):
    from atom_openmm import awh_trajectory

    reports = []

    class FakeXTCReporter:
        def __init__(self, *args, **kwargs):
            reports.append((args, kwargs))

        def describeNextReport(self, simulation):
            return {"steps": 1}

        def report(self, simulation, state):
            reports.append(("report", simulation.currentStep))

    monkeypatch.setattr(awh_trajectory.app, "XTCReporter", FakeXTCReporter)
    reporter = awh_trajectory.StateTaggedXTCReporter(
        tmp_path / "trajectory.xtc",
        tmp_path / "frames.csv",
        5000,
        [1, 2, 3],
        lambda: {
            "move": 10,
            "stage": "adaptive",
            "state": 4,
            "state_name": "a_physical",
            "kind": "physical",
            "atm_state": 0,
            "rest2_region": None,
            "rest2_scale": 1.0,
            "effective_temperature_k": 310.0,
        },
        origin_simulation_step=450000,
        origin_awh_steps=5000,
    )

    class FakeSimulation:
        currentStep = 455000

    reporter.report(FakeSimulation(), object())
    rows = list(__import__("csv").DictReader((tmp_path / "frames.csv").open()))
    assert rows[0]["awh_steps"] == "10000"
    assert rows[0]["state_name"] == "a_physical"
    assert reports[-1] == ("report", 455000)


def _test_state_tagged_xtc_trajectory_resumes(tmp_path):
    import csv
    import openmm as mm
    from openmm import app

    from atom_openmm.awh_trajectory import StateTaggedXTCReporter

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL", chain)
    topology.addAtom("C1", app.element.carbon, residue)
    topology.addAtom("C2", app.element.carbon, residue)
    system = mm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)
    metadata = lambda: {
        "move": 1,
        "stage": "adaptive",
        "state": 0,
        "state_name": "a_physical",
        "kind": "physical",
        "atm_state": 0,
        "rest2_region": None,
        "rest2_scale": 1.0,
        "effective_temperature_k": 300.0,
    }
    trajectory = tmp_path / "trajectory.xtc"
    frames = tmp_path / "frames.csv"

    first = app.Simulation(topology, system, mm.VerletIntegrator(0.001))
    first.context.setPositions([[0, 0, 0], [0.1, 0, 0]])
    first.reporters.append(
        StateTaggedXTCReporter(
            trajectory, frames, 1, [0, 1], metadata
        )
    )
    first.step(2)

    second = app.Simulation(topology, system, mm.VerletIntegrator(0.001))
    second.context.setPositions([[0, 0, 0], [0.1, 0, 0]])
    second.currentStep = 2
    second.reporters.append(
        StateTaggedXTCReporter(
            trajectory, frames, 1, [0, 1], metadata, append=True
        )
    )
    second.step(1)

    with frames.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["frame"]) for row in rows] == [0, 1, 2]


def _test_analyze_existing_awh_rebuilds_summary_from_raw_artifacts(
    tmp_path, monkeypatch
):
    import csv
    import yaml

    from atom_openmm import awh
    from atom_openmm.neqti import build_atm_state_parameters

    settings = awh.normalize_awh_options(
        {
            "awh": {
                "production": {"steps": 100, "bootstrap_samples": 0},
                "rest2": {"effective_temperatures_k": [300, 450, 600]},
                "analysis": {
                    "trajectory": {"enabled": False},
                    "thresholds": {
                        "min_adjacent_overlap": 0.01,
                        "min_endpoint_effective_samples": 1,
                        "min_rest2_hot_returns": 0,
                        "min_uniform_occupancy_overlap": 0.1,
                    },
                },
            }
        },
        _atom_options(),
    )
    graph = awh.build_awh_state_graph(
        build_atm_state_parameters(_atom_options()), settings
    )
    (tmp_path / "awh_protocol.yaml").write_text(
        yaml.safe_dump({"graph": graph, "settings": settings})
    )
    (tmp_path / "awh_checkpoint.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": "production",
                "total_steps": 1100,
                "production_steps_completed": 100,
                "round_trips": 1,
                "current_state": 2,
                "transitions": np.zeros(
                    (len(graph), len(graph)), dtype=int
                ).tolist(),
                "awh": {
                    "free_energy": [0.0] * len(graph),
                    "visits": [10] * len(graph),
                },
            }
        )
    )
    with (tmp_path / "awh_state_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["move", "stage", "previous_state", "state"],
        )
        writer.writeheader()
        previous = 0
        for move, state in enumerate(range(len(graph)), 1):
            writer.writerow(
                {
                    "move": move,
                    "stage": "production",
                    "previous_state": previous,
                    "state": state,
                }
            )
            previous = state
    with (tmp_path / "awh_reduced_energies.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sampled_state", *[node["name"] for node in graph]])
        for state in np.repeat(np.arange(len(graph)), 5):
            writer.writerow([state, *([0.0] * len(graph))])
    monkeypatch.chdir(tmp_path)
    result = awh.analyze_existing_awh(_atom_options(), settings)
    assert result["status"] == "completed"
    assert result["overlap_score"] == pytest.approx(1 / len(graph))
    assert (tmp_path / "awh_summary.yaml").exists()
    assert (tmp_path / "awh_diagnostics.yaml").exists()
