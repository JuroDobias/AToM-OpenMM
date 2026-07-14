from pathlib import Path

import numpy as np
import pytest


def _test_torsion_angle_degrees_distinguishes_rotamers():
    from atom_openmm.rest2_validation import torsion_angle_degrees

    first = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]])
    second = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, -1, 0]])
    assert abs(torsion_angle_degrees(first, (0, 1, 2, 3))) < 1e-10
    assert abs(abs(torsion_angle_degrees(second, (0, 1, 2, 3))) - 180.0) < 1e-10


def _test_exchange_log_acceptance_uses_cross_energies():
    from atom_openmm.rest2_validation import exchange_log_acceptance

    assert exchange_log_acceptance(0.5, 10.0, 14.0, 20.0, 12.0) == pytest.approx(2.0)


def _test_periodic_delta_wraps_at_boundary():
    from atom_openmm.rest2_validation import periodic_delta_degrees

    assert periodic_delta_degrees([179.0, -179.0], [-179.0, 179.0]).tolist() == pytest.approx([-2.0, 2.0])


def _test_periodic_wham_recovers_uniform_distribution_from_unbiased_windows():
    from atom_openmm.rest2_validation import periodic_wham

    samples = np.linspace(-179.5, 179.5, 360)
    centers, probability, pmf, histograms, offsets = periodic_wham(
        [samples, samples], [-90.0, 90.0], 0.0, 300.0, bin_width_deg=30.0
    )
    assert len(centers) == 12
    assert probability == pytest.approx(np.full(12, 1.0 / 12.0))
    assert pmf == pytest.approx(np.zeros(12), abs=1e-10)


def _test_periodic_wham_rejects_empty_window():
    from atom_openmm.rest2_validation import REST2ValidationError, periodic_wham

    with pytest.raises(REST2ValidationError, match="every umbrella window"):
        periodic_wham([[0.0], []], [0.0, 90.0], 50.0, 300.0)


def _test_sdf_parameterization_uses_explicit_gaff2_bcc_charge(monkeypatch, tmp_path):
    from atom_openmm import rest2_validation

    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("test sdf\n")
    workdir = tmp_path / "run"
    config = {
        "_base_dir": tmp_path,
        "_workdir": workdir,
        "system": {
            "ligand_file": "ligand.sdf",
            "residue_name": "UNL",
            "net_charge": 0,
            "parameterization": {"charge_model": "bcc"},
        },
    }
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_text("generated\n")

    monkeypatch.setattr(rest2_validation.subprocess, "run", fake_run)
    mol2, frcmod = rest2_validation._prepare_ligand_parameters(config)

    assert mol2.exists() and frcmod.exists()
    assert commands[0][0] == "antechamber"
    assert commands[0][commands[0].index("-at") + 1] == "gaff2"
    assert commands[0][commands[0].index("-c") + 1] == "bcc"
    assert commands[0][commands[0].index("-nc") + 1] == "0"
    assert commands[1][0] == "parmchk2"


def _test_resume_preserves_existing_preparation(monkeypatch, tmp_path):
    from atom_openmm import rest2_validation

    config = tmp_path / "workflow.yaml"
    config.write_text("workdir: run\n")
    prepared = tmp_path / "run" / "prepared"
    prepared.mkdir(parents=True)
    for name in ("1oiy.prmtop", "1oiy.inpcrd", "equilibrated_state.xml"):
        (prepared / name).write_text("existing\n")

    def fail_prepare(_config):
        raise AssertionError("prepare should not run during a compatible resume")

    monkeypatch.setattr(rest2_validation, "prepare", fail_prepare)
    rest2_validation.run(config, stage="prepare", resume=True)
