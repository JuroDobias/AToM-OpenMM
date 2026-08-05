from pathlib import Path

import pytest
import yaml

from atom_openmm.covalent_switch_duration_scan import (
    CovalentSwitchScanError,
    _normalize_config,
    _softcore_options,
    scale_segment_steps,
)


def _test_scale_segment_steps_preserves_exact_total_and_shape():
    assert scale_segment_steps([1000, 2000, 7000], 25000) == [2500, 5000, 17500]


def _test_scale_segment_steps_handles_rounding():
    scaled = scale_segment_steps([1, 1, 1], 10)
    assert sum(scaled) == 10
    assert all(value >= 1 for value in scaled)


def _test_scale_segment_steps_rejects_impossible_target():
    with pytest.raises(CovalentSwitchScanError, match="at least one step"):
        scale_segment_steps([1, 1, 1], 2)


def _test_normalize_scan_config(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text("workflow: {}\n")
    config = tmp_path / "scan.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "source": {
                    "pair_workdir": "source",
                    "workflow_yaml": "workflow.yaml",
                },
                "output_dir": "run",
                "environment": "protein",
                "n_snapshots": 20,
                "decorrelation_steps": 100000,
                "switch_durations_ps": [500, 1000],
            }
        )
    )

    observed = _normalize_config(config)

    assert observed["source_workdir"] == source.resolve()
    assert observed["source_workflow"] == workflow.resolve()
    assert observed["output_dir"] == (tmp_path / "run").resolve()
    assert observed["switch_durations_ps"] == [500.0, 1000.0]
    assert observed["n_snapshots"] == 20


def _test_duration_scan_preserves_softcore_function_and_interpolation():
    source = {
        "function": "amber_ssc2",
        "stage_interpolation": "linear",
        "alpha": 0.3,
        "sigma_nm": 0.25,
        "power": 1,
        "gapsys_scale_linpoint_lj": 0.85,
        "gapsys_sigma_nm": 0.30,
        "ssc2_alpha_lj": 0.5,
        "ssc2_switch_width_nm": 0.2,
        "path_nodes": [],
        "vdw_a": [1.0, 0.0],
        "charge_a": [1.0, 0.0],
        "segments_per_interval": [10],
    }

    observed = _softcore_options(source, 50000)

    assert observed["function"] == "amber_ssc2"
    assert observed["stage_interpolation"] == "linear"
    assert observed["ssc2_alpha_lj"] == 0.5
    assert observed["total_steps"] == 50000
