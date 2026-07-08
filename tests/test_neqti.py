import csv

import pytest
from openmm.unit import kelvin, kilocalories_per_mole


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


def _test_build_atm_state_parameters_uses_existing_schedule():
    from atom_openmm.neqti import build_atm_state_parameters

    states = build_atm_state_parameters(_atom_options())

    assert len(states) == 4
    assert states[0]["lambda1"] == 0.0
    assert states[0]["atmdirection"] == 1.0
    assert states[-1]["atmdirection"] == -1.0
    assert states[0]["temperature"] / kelvin == 300.0
    assert states[1]["uh"] / kilocalories_per_mole == 110.0


def _test_normalize_neqti_options_defaults_to_full_state_path():
    from atom_openmm.neqti import normalize_neqti_options

    settings = normalize_neqti_options({"neqti": {"switch_steps_per_segment": 7}}, _atom_options())

    assert settings["state_path"] == [0, 1, 2, 3]
    assert settings["switch_steps_per_segment"] == 7
    assert settings["n_snapshots"] == 2
    assert settings["resume"] is True


def _test_switch_schedule_interpolates_between_knots():
    from atom_openmm.neqti import build_atm_state_parameters, make_switch_schedule

    states = build_atm_state_parameters(_atom_options())
    schedule = make_switch_schedule(states, [0, 1], 2)

    assert len(schedule) == 2
    assert schedule[0]["lambda1"] == pytest.approx(0.25)
    assert schedule[0]["lambda2"] == pytest.approx(0.25)
    assert schedule[1]["lambda1"] == pytest.approx(0.5)
    assert schedule[1]["lambda2"] == pytest.approx(0.5)


def _test_bar_estimator_sign_convention():
    from atom_openmm.neqti import estimate_bar

    assert estimate_bar([2.0, 2.0, 2.0], [-2.0, -2.0, -2.0], 300.0) == pytest.approx(2.0)


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
