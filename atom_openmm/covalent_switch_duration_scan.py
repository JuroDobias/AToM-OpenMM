from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import openmm as mm
import yaml
from openmm import unit

from atom_openmm.covalent_softcore import create_softcore_hamiltonian
from atom_openmm.covalent_workflow import (
    KCAL_TO_KJ,
    LRC_CORRECTION_VERSION,
    CovalentWorkflowError,
    _EndpointLRCCorrectionEvaluator,
    _apply_state,
    _assert_fixed_volume_switch,
    _is_numerical_switch_failure,
    _load_prepared_pair_bundle,
    _load_state,
    _platform,
    _precompute_endpoint_lrc_corrections,
    _read_work,
    _reset_softcore_context,
    _run_segmented_protocol,
    _sample_endpoint,
    _softcore_switch_context,
    _state_volume_nm3,
    _write_state,
    _write_yaml_atomic,
)
from atom_openmm.neqti import (
    _bar_overlap_score,
    analyze_neqti_work,
    analyze_two_leg_work,
)


LOGGER = logging.getLogger("atom_openmm.covalent_switch_duration_scan")
SCHEMA_VERSION = 1


class CovalentSwitchScanError(CovalentWorkflowError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scale_segment_steps(segment_steps, target_total_steps):
    source = np.asarray(segment_steps, dtype=int)
    target = int(target_total_steps)
    if source.ndim != 1 or len(source) == 0 or np.any(source <= 0):
        raise CovalentSwitchScanError("source segment steps must be positive")
    if target < len(source):
        raise CovalentSwitchScanError(
            "target switch length must provide at least one step per segment"
        )
    raw = source.astype(float) * target / int(source.sum())
    scaled = np.maximum(1, np.floor(raw).astype(int))
    difference = target - int(scaled.sum())
    if difference > 0:
        order = np.argsort(-(raw - np.floor(raw)), kind="stable")
        for index in range(difference):
            scaled[order[index % len(order)]] += 1
    elif difference < 0:
        order = np.argsort(raw - np.floor(raw), kind="stable")
        for index in order:
            remove = min(scaled[index] - 1, -difference)
            scaled[index] -= remove
            difference += remove
            if difference == 0:
                break
    if int(scaled.sum()) != target or np.any(scaled < 1):
        raise CovalentSwitchScanError("could not scale the switching schedule")
    return scaled.tolist()


def _append_work(path, sample, work_kj):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["sample", "work_kcal_per_mol"])
        writer.writerow([int(sample), float(work_kj) / KCAL_TO_KJ])


def _append_timing(path, duration_ps, direction, sample, steps, elapsed):
    path = Path(path)
    exists = path.exists()
    ns_per_day = float(steps) * 2.0e-6 * 86400.0 / float(elapsed)
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                [
                    "duration_ps",
                    "direction",
                    "sample",
                    "steps",
                    "elapsed_seconds",
                    "ns_per_day",
                ]
            )
        writer.writerow(
            [duration_ps, direction, sample, steps, elapsed, ns_per_day]
        )
    return ns_per_day


def _normalize_config(path):
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    source = raw.get("source") or {}
    source_workdir = Path(source.get("pair_workdir", ""))
    source_workflow = Path(source.get("workflow_yaml", ""))
    if not source_workdir.is_absolute():
        source_workdir = (config_path.parent / source_workdir).resolve()
    if not source_workflow.is_absolute():
        source_workflow = (config_path.parent / source_workflow).resolve()
    output = Path(raw.get("output_dir", "scan"))
    if not output.is_absolute():
        output = (config_path.parent / output).resolve()
    durations = [float(value) for value in raw.get("switch_durations_ps", [])]
    if not source_workdir.is_dir():
        raise CovalentSwitchScanError(
            f"source pair workdir does not exist: {source_workdir}"
        )
    if not source_workflow.is_file():
        raise CovalentSwitchScanError(
            f"source workflow does not exist: {source_workflow}"
        )
    if not durations or any(value <= 0 for value in durations):
        raise CovalentSwitchScanError(
            "switch_durations_ps must contain positive values"
        )
    if len(set(durations)) != len(durations):
        raise CovalentSwitchScanError("switch_durations_ps must be unique")
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "source_workdir": source_workdir,
        "source_workflow": source_workflow,
        "output_dir": output,
        "environment": str(raw.get("environment", "protein")),
        "n_snapshots": int(raw.get("n_snapshots", 20)),
        "decorrelation_steps": int(raw.get("decorrelation_steps", 100000)),
        "switch_durations_ps": durations,
        "bootstrap_samples": int(raw.get("bootstrap_samples", 500)),
        "random_seed": int(raw.get("random_seed", 20260803)),
    }
    if normalized["environment"] != "protein":
        raise CovalentSwitchScanError(
            "the initial duration-scan implementation supports environment: protein"
        )
    if normalized["n_snapshots"] < 1 or normalized["decorrelation_steps"] < 1:
        raise CovalentSwitchScanError(
            "n_snapshots and decorrelation_steps must be positive"
        )
    return normalized


def _source_settings(config):
    workflow_document = yaml.safe_load(config["source_workflow"].read_text()) or {}
    workflow = workflow_document.get("workflow") or {}
    neqti = workflow.get("neqti") or {}
    rest2 = dict(neqti.get("rest2") or {})
    rest2.setdefault("enabled", True)
    rest2.setdefault("effective_temperatures_k", [300.0, 344.6, 395.9, 454.7, 522.3, 600.0])
    rest2.setdefault("exchange_interval_steps", 500)
    rest2.setdefault("checkpoint_interval_cycles", 10)
    rest2.setdefault("execution", "serial")
    protocol = yaml.safe_load(
        (config["source_workdir"] / "switch_protocol.yaml").read_text()
    ) or {}
    optimizer = yaml.safe_load(
        (config["source_workdir"] / "covalent_schedule_optimization.yaml").read_text()
    ) or {}
    try:
        segment_steps = optimizer["environments"]["protein"]["segment_steps"]
        softcore = protocol["softcore"]
    except KeyError as exc:
        raise CovalentSwitchScanError(
            f"source run lacks a frozen protein schedule or softcore protocol: {exc}"
        ) from exc
    if optimizer["environments"]["protein"].get("status") != "frozen":
        raise CovalentSwitchScanError("source protein schedule is not frozen")
    if protocol.get("interpolation") != "softcore_linear":
        raise CovalentSwitchScanError("source run is not softcore_linear")
    return {
        "workflow": workflow,
        "rest2": rest2,
        "protocol": protocol,
        "softcore": softcore,
        "segment_steps": [int(value) for value in segment_steps],
        "temperature_k": float(neqti.get("temperature_k", 300.0)),
        "timestep_fs": float(protocol["timestep_fs"]),
    }


def _stage_source(config):
    source = config["source_workdir"]
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = source / "prepared" / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    source_identity = {
        "pair_workdir": str(source.resolve()),
        "prepared_fingerprint": manifest.get("fingerprint"),
        "switch_protocol_sha256": _sha256(source / "switch_protocol.yaml"),
        "optimized_schedule_sha256": _sha256(
            source / "covalent_schedule_optimization.yaml"
        ),
    }
    staged_manifest = output / "source.yaml"
    if staged_manifest.exists():
        observed = yaml.safe_load(staged_manifest.read_text()) or {}
        if observed != source_identity:
            raise CovalentSwitchScanError(
                "output directory was initialized from a different source run"
            )
    else:
        _write_yaml_atomic(staged_manifest, source_identity)
    staging = output / "staged_endpoint_sampling"
    staging.mkdir(exist_ok=True)
    prepared_target = staging / "prepared"
    if not prepared_target.exists():
        temporary = staging / "prepared.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(source / "prepared", temporary)
        os.replace(temporary, prepared_target)
    for endpoint in ("a", "b"):
        state_target = staging / f"protein_endpoint_{endpoint}_state.xml"
        if not state_target.exists():
            shutil.copy2(
                source / f"protein_endpoint_{endpoint}_state.xml", state_target
            )
        rest2_target = staging / f"protein_rest2_{endpoint}"
        if not rest2_target.exists():
            temporary = staging / f"protein_rest2_{endpoint}.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(source / f"protein_rest2_{endpoint}", temporary)
            os.replace(temporary, rest2_target)
    return manifest


def _snapshot_manifest(config, source_identity):
    path = config["output_dir"] / "snapshot_bank" / "manifest.yaml"
    expected = {
        "schema_version": SCHEMA_VERSION,
        "source": source_identity,
        "environment": "protein",
        "decorrelation_steps": config["decorrelation_steps"],
        "target_snapshots": config["n_snapshots"],
        "snapshots": {"a": [], "b": []},
    }
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_yaml_atomic(path, expected)
        return path, expected
    observed = yaml.safe_load(path.read_text()) or {}
    for key in (
        "schema_version",
        "source",
        "environment",
        "decorrelation_steps",
        "target_snapshots",
    ):
        if observed.get(key) != expected[key]:
            raise CovalentSwitchScanError(
                f"snapshot bank setting changed: {key}"
            )
    return path, observed


def _collect_snapshot_bank(config, settings, prepared, source_identity, platform, properties):
    manifest_path, manifest = _snapshot_manifest(config, source_identity)
    bank = manifest_path.parent
    staging = config["output_dir"] / "staged_endpoint_sampling"
    for sample in range(1, config["n_snapshots"] + 1):
        for endpoint, offset in (("a", 0), ("b", 100)):
            entries = manifest["snapshots"][endpoint]
            if len(entries) >= sample:
                entry = entries[sample - 1]
                path = bank / entry["file"]
                if not path.exists() or _sha256(path) != entry["sha256"]:
                    raise CovalentSwitchScanError(
                        f"snapshot bank entry is missing or corrupt: {path}"
                    )
                continue
            state_file = staging / f"protein_endpoint_{endpoint}_state.xml"
            state, _ = _sample_endpoint(
                prepared.endpoint_a if endpoint == "a" else prepared.endpoint_b,
                prepared.topology,
                state_file,
                prepared.hot_atom_indices,
                ensemble=endpoint,
                steps=config["decorrelation_steps"],
                rest2_config=settings["rest2"],
                output_dir=staging / f"protein_rest2_{endpoint}",
                platform=platform,
                properties=properties,
                temperature_k=settings["temperature_k"],
                timestep_fs=settings["timestep_fs"],
                seed=config["random_seed"] + sample * 1000 + offset,
            )
            filename = f"endpoint_{endpoint}_{sample:03d}.xml"
            snapshot_path = bank / filename
            _write_state(snapshot_path, state)
            entries.append(
                {
                    "sample": sample,
                    "file": filename,
                    "sha256": _sha256(snapshot_path),
                }
            )
            _write_yaml_atomic(manifest_path, manifest)
            LOGGER.info(
                "Snapshot bank endpoint %s sample %d/%d complete",
                endpoint.upper(),
                sample,
                config["n_snapshots"],
            )
    return manifest


def _softcore_options(source_softcore, total_steps):
    options = {
        "function": source_softcore.get("function", "beutler"),
        "coulomb_function": source_softcore.get(
            "coulomb_function", "linear_pme"
        ),
        "stage_interpolation": source_softcore.get(
            "stage_interpolation", "linear"
        ),
        "alpha": source_softcore["alpha"],
        "sigma_nm": source_softcore["sigma_nm"],
        "power": source_softcore["power"],
        "gapsys_scale_linpoint_lj": source_softcore.get(
            "gapsys_scale_linpoint_lj", 0.85
        ),
        "gapsys_sigma_nm": source_softcore.get("gapsys_sigma_nm", 0.30),
        "ssc2_alpha_lj": source_softcore.get("ssc2_alpha_lj", 0.5),
        "ssc2_alpha_coul": source_softcore.get("ssc2_alpha_coul", 1.0),
        "ssc2_switch_width_nm": source_softcore.get(
            "ssc2_switch_width_nm", 0.2
        ),
        "total_steps": int(total_steps),
        "segments_per_interval": source_softcore["segments_per_interval"],
    }
    if "path_mode" in source_softcore:
        options["path_mode"] = source_softcore["path_mode"]
    else:
        options.update(
            {
                "path_nodes": source_softcore["path_nodes"],
                "vdw_a": source_softcore["vdw_a"],
                "charge_a": source_softcore["charge_a"],
            }
        )
    return options


def _duration_name(duration_ps):
    return f"{int(round(float(duration_ps)))}ps"


def _run_duration(config, settings, prepared, bank_manifest, duration_ps, platform, properties):
    timestep_fs = settings["timestep_fs"]
    total_steps_float = float(duration_ps) * 1000.0 / timestep_fs
    total_steps = int(round(total_steps_float))
    if not math.isclose(total_steps_float, total_steps, abs_tol=1.0e-8):
        raise CovalentSwitchScanError(
            f"duration {duration_ps} ps is not divisible by timestep {timestep_fs} fs"
        )
    segment_steps = scale_segment_steps(settings["segment_steps"], total_steps)
    directory = config["output_dir"] / f"duration_{_duration_name(duration_ps)}"
    directory.mkdir(exist_ok=True)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "duration_ps": float(duration_ps),
        "total_steps": total_steps,
        "timestep_fs": timestep_fs,
        "source_segment_steps": settings["segment_steps"],
        "scaled_segment_steps": segment_steps,
        "snapshot_manifest_sha256": _sha256(
            config["output_dir"] / "snapshot_bank" / "manifest.yaml"
        ),
        "softcore": settings["softcore"],
    }
    protocol_path = directory / "protocol.yaml"
    if protocol_path.exists():
        if (yaml.safe_load(protocol_path.read_text()) or {}) != protocol:
            raise CovalentSwitchScanError(
                f"duration protocol changed in {directory}"
            )
    else:
        _write_yaml_atomic(protocol_path, protocol)

    unique_a = prepared.provenance["unique_a_particle_indices"]
    unique_b = prepared.provenance["unique_b_particle_indices"]
    options = _softcore_options(settings["softcore"], total_steps)
    hamiltonian = create_softcore_hamiltonian(
        prepared.endpoint_a,
        prepared.endpoint_b,
        unique_a,
        unique_b,
        use_long_range_correction=False,
        **options,
    )
    _assert_fixed_volume_switch(hamiltonian.system)
    if len(segment_steps) != len(hamiltonian.segment_steps):
        raise CovalentSwitchScanError("scaled schedule has the wrong segment count")
    lrc_hamiltonian = create_softcore_hamiltonian(
        prepared.endpoint_a,
        prepared.endpoint_b,
        unique_a,
        unique_b,
        use_long_range_correction=True,
        **options,
    )
    bank_dir = config["output_dir"] / "snapshot_bank"
    first_a = _load_state(bank_dir / bank_manifest["snapshots"]["a"][0]["file"])
    first_b = _load_state(bank_dir / bank_manifest["snapshots"]["b"][0]["file"])
    lrc = _precompute_endpoint_lrc_corrections(
        _EndpointLRCCorrectionEvaluator(hamiltonian, lrc_hamiltonian),
        first_a,
        first_b,
        hamiltonian.parameter_values,
    )
    contexts = {}
    work_files = {
        "forward": directory / "protein_forward.csv",
        "reverse": directory / "protein_reverse.csv",
    }
    timing_path = directory / "switch_timing.csv"
    for direction, endpoint, offset in (("forward", "a", 0), ("reverse", "b", 100)):
        existing = _read_work(work_files[direction])
        cached = _softcore_switch_context(
            hamiltonian,
            start=endpoint,
            timestep_fs=timestep_fs,
            temperature_k=settings["temperature_k"],
            platform=platform,
            properties=properties,
            seed=config["random_seed"] + int(duration_ps) * 100 + offset,
        )
        contexts[direction] = cached
        context, integrator, parameter_values = cached
        direction_steps = segment_steps if direction == "forward" else list(reversed(segment_steps))
        integrator.set_segment_steps(direction_steps)
        for sample in range(len(existing) + 1, config["n_snapshots"] + 1):
            entry = bank_manifest["snapshots"][endpoint][sample - 1]
            state = _load_state(bank_dir / entry["file"])
            _reset_softcore_context(context, integrator, parameter_values)
            _apply_state(context, state)
            started = time.perf_counter()
            try:
                raw_work, _ = _run_segmented_protocol(integrator, direction_steps)
                work_kj = raw_work + lrc[direction]["final"] - lrc[direction]["initial"]
            except Exception as exc:
                if _is_numerical_switch_failure(exc):
                    work_kj = float("inf")
                    LOGGER.warning(
                        "%s %s sample %d failed numerically and was recorded as +inf: %s",
                        _duration_name(duration_ps),
                        direction,
                        sample,
                        exc,
                    )
                    del context
                    cached = _softcore_switch_context(
                        hamiltonian,
                        start=endpoint,
                        timestep_fs=timestep_fs,
                        temperature_k=settings["temperature_k"],
                        platform=platform,
                        properties=properties,
                        seed=config["random_seed"] + int(duration_ps) * 100 + offset + sample,
                    )
                    contexts[direction] = cached
                    context, integrator, parameter_values = cached
                    integrator.set_segment_steps(direction_steps)
                else:
                    raise
            elapsed = time.perf_counter() - started
            _append_work(work_files[direction], sample, work_kj)
            ns_per_day = _append_timing(
                timing_path, duration_ps, direction, sample, total_steps, elapsed
            )
            LOGGER.info(
                "%s %s sample %d/%d complete: work %.6g kJ/mol, %.3f ns/day",
                _duration_name(duration_ps),
                direction,
                sample,
                config["n_snapshots"],
                work_kj,
                ns_per_day,
            )
    forward = _read_work(work_files["forward"])
    reverse = _read_work(work_files["reverse"])
    analysis = analyze_neqti_work(
        forward,
        reverse,
        settings["temperature_k"],
        config["bootstrap_samples"],
        config["random_seed"] + int(duration_ps),
    )
    if analysis is None:
        raise CovalentSwitchScanError(
            f"BAR failed for duration {duration_ps} ps"
        )
    dg = analysis["bar_dg_kcal_per_mol"]
    overlap = _bar_overlap_score(forward, reverse, dg, settings["temperature_k"])
    finite_forward = np.asarray(forward, dtype=float)
    finite_reverse = np.asarray(reverse, dtype=float)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "environment": "protein",
        "duration_ps": float(duration_ps),
        "total_steps": total_steps,
        "samples": {
            "forward": len(forward),
            "reverse": len(reverse),
            "finite_forward": int(np.isfinite(finite_forward).sum()),
            "finite_reverse": int(np.isfinite(finite_reverse).sum()),
        },
        "protein": {
            "dg_kcal_per_mol": dg,
            "error_kcal_per_mol": analysis["bar_bootstrap_std_kcal_per_mol"],
            "overlap_score": overlap,
            "mean_forward_dissipation_kcal_per_mol": float(
                np.mean(finite_forward[np.isfinite(finite_forward)] - dg)
            ),
            "mean_reverse_dissipation_kcal_per_mol": float(
                np.mean(finite_reverse[np.isfinite(finite_reverse)] + dg)
            ),
        },
    }
    _write_yaml_atomic(directory / "result.yaml", result)
    return result, forward, reverse


def _build_comparison(config, settings, duration_results):
    source_result = yaml.safe_load(
        (config["source_workdir"] / "result.yaml").read_text()
    ) or {}
    source_reference = source_result["result"]["components"]["reference"]
    reference_forward = _read_work(config["source_workdir"] / "reference_forward.csv")
    reference_reverse = _read_work(config["source_workdir"] / "reference_reverse.csv")
    entries = []
    for result, forward, reverse in duration_results:
        complete = analyze_two_leg_work(
            {
                "leg_a_forward": forward,
                "leg_a_reverse": reverse,
                "leg_b_forward": reference_forward,
                "leg_b_reverse": reference_reverse,
            },
            settings["temperature_k"],
            config["bootstrap_samples"],
            config["random_seed"] + int(result["duration_ps"]),
        )
        entries.append(
            {
                **result,
                "diagnostic_complete_ddg": {
                    "ddg_kcal_per_mol": complete["bar_dg_kcal_per_mol"],
                    "error_kcal_per_mol": complete[
                        "bar_bootstrap_std_kcal_per_mol"
                    ],
                    "reference_dg_kcal_per_mol": source_reference[
                        "dg_kcal_per_mol"
                    ],
                    "reference_overlap_score": source_reference[
                        "overlap_score"
                    ],
                    "reference_source": str(config["source_workdir"]),
                },
            }
        )
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "source_300ps": {
            "protein": source_result["result"]["components"]["protein"],
            "reference": source_reference,
            "ddg_kcal_per_mol": source_result["result"]["ddg_kcal_per_mol"],
            "ddg_error_kcal_per_mol": source_result["result"][
                "ddg_error_kcal_per_mol"
            ],
        },
        "duration_scans": entries,
    }
    _write_yaml_atomic(config["output_dir"] / "comparison.yaml", comparison)
    return comparison


def run_duration_scan(config_path):
    config = _normalize_config(config_path)
    settings = _source_settings(config)
    if not math.isclose(settings["timestep_fs"], 2.0):
        raise CovalentSwitchScanError(
            "duration scan timing output currently requires a 2 fs source timestep"
        )
    manifest = _stage_source(config)
    prepared, _, _ = _load_prepared_pair_bundle(
        config["output_dir"] / "staged_endpoint_sampling",
        manifest["fingerprint"],
    )
    platform, properties = _platform(settings["workflow"])
    source_identity = yaml.safe_load(
        (config["output_dir"] / "source.yaml").read_text()
    )
    bank = _collect_snapshot_bank(
        config, settings, prepared, source_identity, platform, properties
    )
    results = [
        _run_duration(
            config, settings, prepared, bank, duration, platform, properties
        )
        for duration in config["switch_durations_ps"]
    ]
    return _build_comparison(config, settings, results)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare covalent NEQTI switching durations from matched snapshots"
    )
    parser.add_argument("config", help="duration scan YAML")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)-8s - %(name)-38s - %(message)s",
    )
    result = run_duration_scan(args.config)
    LOGGER.info(
        "Covalent switching-duration scan completed: %s",
        Path(args.config).resolve(),
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
