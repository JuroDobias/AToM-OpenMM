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
