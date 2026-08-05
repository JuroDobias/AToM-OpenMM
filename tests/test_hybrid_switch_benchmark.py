import pytest
import yaml

from atom_openmm.hybrid_switch_benchmark import (
    HybridSwitchBenchmarkError,
    _resolve_config,
    archive_snapshot_bank,
)


def _cohort(tmp_path, *, complete=True):
    cohort = tmp_path / "cohort"
    edge = cohort / "A--B"
    pair = edge / "run" / "receptor-A-B"
    prepared = pair / "prepared"
    prepared.mkdir(parents=True)
    (prepared / "manifest.yaml").write_text(
        yaml.safe_dump({"schema_version": 2, "fingerprint": "prepared-123"})
    )
    (prepared / "complex_endpoint_a.xml").write_text("complex-a")
    (pair / "hybrid_mapping.yaml").write_text("schema_version: 1\n")
    (pair / "switch_protocol.yaml").write_text("schema_version: 1\n")
    (edge / "source_commit.txt").write_text("abc123\n")
    for environment in ("complex", "solvent"):
        directory = pair / "neqti_adaptive_switching" / "snapshots" / environment
        directory.mkdir(parents=True)
        for endpoint in ("a", "b"):
            count = 2 if complete or environment != "complex" or endpoint != "b" else 1
            for sample in range(count):
                (directory / f"{endpoint}_{sample:04d}.xml").write_text(
                    f"{environment}-{endpoint}-{sample}"
                )
    return cohort


def _test_archive_snapshot_bank_is_self_contained_and_hashed(tmp_path):
    cohort = _cohort(tmp_path)
    output = tmp_path / "bank"

    manifest = archive_snapshot_bank(cohort, output, snapshots=2)

    assert list(manifest["edges"]) == ["A--B"]
    edge = manifest["edges"]["A--B"]
    assert edge["source_commit"] == "abc123"
    assert edge["prepared_fingerprint"] == "prepared-123"
    assert len(edge["snapshots"]["complex"]["a"]) == 2
    entry = edge["snapshots"]["solvent"]["b"][1]
    archived = output / entry["file"]
    assert archived.read_text() == "solvent-b-1"
    assert entry["size_bytes"] == archived.stat().st_size
    assert len(entry["sha256"]) == 64
    assert (output / "edges" / "A--B" / "prepared" / "manifest.yaml").is_file()


def _test_archive_rejects_incomplete_edge_atomically(tmp_path):
    cohort = _cohort(tmp_path, complete=False)
    output = tmp_path / "bank"

    with pytest.raises(HybridSwitchBenchmarkError, match="snapshots; 2 required"):
        archive_snapshot_bank(cohort, output, snapshots=2)

    assert not output.exists()
    assert not list(tmp_path.glob(".bank.tmp-*"))


def _test_replay_config_normalizes_protocols_and_rejects_unknown_curve(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "snapshot_bank": "bank",
                "output_dir": "output",
                "edges": ["A--B"],
                "protocols": [
                    {"name": "linear", "stage_interpolation": "linear"},
                    {
                        "name": "smooth",
                        "stage_interpolation": "smoothstep2",
                        "duration_ps": 300,
                    },
                ],
            }
        )
    )

    observed = _resolve_config(config)

    assert observed["snapshot_bank"] == bank
    assert observed["output_dir"] == tmp_path / "output"
    assert observed["protocols"][0]["duration_ps"] == 100.0
    assert observed["protocols"][1]["duration_ps"] == 300.0

    payload = yaml.safe_load(config.read_text())
    payload["protocols"][1]["stage_interpolation"] = "cubic"
    config.write_text(yaml.safe_dump(payload))
    with pytest.raises(HybridSwitchBenchmarkError, match="unsupported"):
        _resolve_config(config)
