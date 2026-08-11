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
import yaml

from atom_openmm.covalent_softcore import create_softcore_hamiltonian
from atom_openmm.covalent_systems import load_prepared_hybrid_bundle
from atom_openmm.covalent_workflow import (
    KCAL_TO_KJ,
    _EndpointLRCCorrectionEvaluator,
    _PhysicalEndpointLRCCorrectionEvaluator,
    _apply_state,
    _assert_fixed_volume_switch,
    _is_numerical_switch_failure,
    _load_state,
    _precompute_endpoint_lrc_corrections,
    _read_work,
    _reset_softcore_context,
    _run_segmented_protocol,
    _scale_segment_steps,
    _softcore_switch_context,
    _write_yaml_atomic,
)
from atom_openmm.neqti import analyze_two_leg_work


LOGGER = logging.getLogger("atom_openmm.hybrid_switch_benchmark")
SCHEMA_VERSION = 1


class HybridSwitchBenchmarkError(ValueError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_or_copy(source, target):
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _single_pair_workdir(edge_dir):
    run_dir = Path(edge_dir) / "run"
    if not run_dir.is_dir():
        raise HybridSwitchBenchmarkError(f"edge lacks a run directory: {run_dir}")
    candidates = [path for path in run_dir.iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise HybridSwitchBenchmarkError(
            f"expected one pair workdir below {edge_dir / 'run'}, found {len(candidates)}"
        )
    return candidates[0]


def _archive_file(source, root, relative):
    target = root / relative
    method = _link_or_copy(source, target)
    return {
        "file": str(relative),
        "sha256": _sha256(target),
        "size_bytes": target.stat().st_size,
        "materialization": method,
    }


def archive_snapshot_bank(cohort_dir, output_dir, snapshots=20, edges=None):
    cohort_dir = Path(cohort_dir).resolve()
    output_dir = Path(output_dir).resolve()
    snapshots = int(snapshots)
    if snapshots < 1:
        raise HybridSwitchBenchmarkError("snapshots must be positive")
    if not cohort_dir.is_dir():
        raise HybridSwitchBenchmarkError(f"cohort directory does not exist: {cohort_dir}")
    edge_names = list(edges or sorted(path.name for path in cohort_dir.iterdir() if path.is_dir()))
    if not edge_names:
        raise HybridSwitchBenchmarkError("no cohort edges were selected")
    if output_dir.exists():
        raise HybridSwitchBenchmarkError(f"snapshot bank already exists: {output_dir}")

    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "noncovalent_hybrid_switch_snapshot_bank",
        "source_cohort": str(cohort_dir),
        "snapshots_per_endpoint": snapshots,
        "edges": {},
    }
    try:
        for edge in edge_names:
            edge_dir = cohort_dir / edge
            if not edge_dir.is_dir():
                raise HybridSwitchBenchmarkError(f"cohort edge does not exist: {edge_dir}")
            source = _single_pair_workdir(edge_dir)
            required = {
                "prepared_manifest": source / "prepared" / "manifest.yaml",
                "hybrid_mapping": source / "hybrid_mapping.yaml",
                "switch_protocol": source / "switch_protocol.yaml",
            }
            for label, path in required.items():
                if not path.is_file():
                    raise HybridSwitchBenchmarkError(f"{edge} lacks {label}: {path}")
            prepared_manifest = yaml.safe_load(required["prepared_manifest"].read_text()) or {}
            edge_root = Path("edges") / edge
            files = {}
            for source_file in sorted((source / "prepared").rglob("*")):
                if source_file.is_file():
                    relative = edge_root / "prepared" / source_file.relative_to(source / "prepared")
                    files[str(relative.relative_to(edge_root))] = _archive_file(
                        source_file, temporary, relative
                    )
            for label, source_file in (
                ("hybrid_mapping", required["hybrid_mapping"]),
                ("switch_protocol", required["switch_protocol"]),
            ):
                relative = edge_root / source_file.name
                files[label] = _archive_file(source_file, temporary, relative)

            snapshot_payload = {}
            source_snapshots = source / "neqti_adaptive_switching" / "snapshots"
            for environment in ("complex", "solvent"):
                snapshot_payload[environment] = {}
                for endpoint in ("a", "b"):
                    selected = sorted((source_snapshots / environment).glob(f"{endpoint}_*.xml"))
                    if len(selected) < snapshots:
                        raise HybridSwitchBenchmarkError(
                            f"{edge} has {len(selected)} {environment} endpoint {endpoint} "
                            f"snapshots; {snapshots} required"
                        )
                    entries = []
                    for sample, source_file in enumerate(selected[:snapshots], 1):
                        relative = (
                            edge_root
                            / "snapshots"
                            / environment
                            / f"endpoint_{endpoint}_{sample:03d}.xml"
                        )
                        entry = _archive_file(source_file, temporary, relative)
                        entry["sample"] = sample
                        entries.append(entry)
                    snapshot_payload[environment][endpoint] = entries

            source_commit = edge_dir / "source_commit.txt"
            manifest["edges"][edge] = {
                "source_workdir": str(source),
                "source_commit": source_commit.read_text().strip() if source_commit.exists() else None,
                "prepared_fingerprint": prepared_manifest.get("fingerprint"),
                "root": str(edge_root),
                "files": files,
                "snapshots": snapshot_payload,
            }
            LOGGER.info("Archived %s with %d snapshots per endpoint", edge, snapshots)
        _write_yaml_atomic(temporary / "manifest.yaml", manifest)
        os.replace(temporary, output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def _resolve_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    bank = Path(raw.get("snapshot_bank", ""))
    output = Path(raw.get("output_dir", "benchmark"))
    if not bank.is_absolute():
        bank = (path.parent / bank).resolve()
    if not output.is_absolute():
        output = (path.parent / output).resolve()
    protocols = []
    names = set()
    for item in raw.get("protocols", []):
        item = dict(item or {})
        name = str(item.get("name", "")).strip()
        if not name or name in names:
            raise HybridSwitchBenchmarkError("protocol names must be non-empty and unique")
        names.add(name)
        interpolation = str(item.get("stage_interpolation", "linear")).lower()
        if interpolation not in {"linear", "smoothstep2"}:
            raise HybridSwitchBenchmarkError(
                f"protocol {name} has unsupported stage_interpolation {interpolation}"
            )
        protocols.append(
            {
                "name": name,
                "stage_interpolation": interpolation,
                "duration_ps": float(item.get("duration_ps", raw.get("duration_ps", 100.0))),
                "softcore": dict(item.get("softcore") or {}),
            }
        )
    if not protocols:
        raise HybridSwitchBenchmarkError("at least one protocol is required")
    environments = [str(value) for value in raw.get("environments", ["complex", "solvent"])]
    if not environments or set(environments) - {"complex", "solvent"}:
        raise HybridSwitchBenchmarkError("environments may contain complex and solvent")
    return {
        "path": path,
        "snapshot_bank": bank,
        "output_dir": output,
        "edges": [str(value) for value in raw.get("edges", [])],
        "environments": environments,
        "n_snapshots": int(raw.get("n_snapshots", 20)),
        "temperature_k": float(raw.get("temperature_k", 300.0)),
        "timestep_fs": float(raw.get("timestep_fs", 2.0)),
        "bootstrap_samples": int(raw.get("bootstrap_samples", 500)),
        "random_seed": int(raw.get("random_seed", 20260805)),
        "protocols": protocols,
        "platform": raw.get("platform"),
    }


def _verify_entry(root, entry):
    path = root / entry["file"]
    if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]):
        raise HybridSwitchBenchmarkError(f"snapshot-bank file is missing or truncated: {path}")
    if _sha256(path) != entry["sha256"]:
        raise HybridSwitchBenchmarkError(f"snapshot-bank checksum differs: {path}")
    return path


def _platform(config):
    import openmm as mm

    name = config.get("platform")
    if name:
        platform = mm.Platform.getPlatformByName(str(name))
    else:
        for candidate in ("CUDA", "OpenCL", "CPU", "Reference"):
            try:
                platform = mm.Platform.getPlatformByName(candidate)
                break
            except Exception:
                continue
    properties = {}
    if platform.getName() in {"CUDA", "OpenCL"}:
        properties["Precision"] = "mixed"
    return platform, properties


def _append_work(path, sample, work_kj):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["sample", "work_kj_per_mol", "work_kcal_per_mol"])
        writer.writerow([sample, work_kj, work_kj / KCAL_TO_KJ])


def _append_timing(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _timing_means(path):
    grouped = {}
    path = Path(path)
    if not path.is_file():
        return grouped
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["environment"], []).append(float(row["ns_per_day"]))
    return {
        environment: float(np.mean(values))
        for environment, values in grouped.items()
        if values
    }


def _sample_seed(base, edge, environment, direction, sample):
    token = f"{base}|{edge}|{environment}|{direction}|{sample}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") % 2147483647 or 1


def _source_softcore(protocol, variant):
    softcore = dict(protocol["softcore"])
    softcore.update(variant["softcore"])
    path = softcore.pop("path", None)
    if path is not None:
        if path.get("mode") is not None:
            softcore["path_mode"] = path["mode"]
        for source, target in (
            ("nodes", "path_nodes"),
            ("vdw_a", "vdw_a"),
            ("charge_a", "charge_a"),
            ("segments_per_interval", "segments_per_interval"),
        ):
            if source in path:
                softcore[target] = path[source]
        if "path_nodes" in softcore and "segments_per_interval" not in softcore:
            softcore["segments_per_interval"] = [1] * (
                len(softcore["path_nodes"]) + 1
            )
    if "path_nodes" in softcore:
        for key in (
            "charge_steps_per_stage",
            "sterics_steps",
            "subdivisions_per_stage",
        ):
            softcore.pop(key, None)
    if "path_mode" in softcore:
        for key in (
            "charge_steps_per_stage",
            "sterics_steps",
            "subdivisions_per_stage",
            "path_nodes",
            "vdw_a",
            "charge_a",
        ):
            softcore.pop(key, None)
        softcore.setdefault("segments_per_interval", [1])
    softcore["stage_interpolation"] = variant["stage_interpolation"]
    softcore.pop("long_range_correction", None)
    supported = {
        "function",
        "coulomb_function",
        "alpha",
        "sigma_nm",
        "power",
        "gapsys_scale_linpoint_lj",
        "gapsys_scale_linpoint_q",
        "gapsys_sigma_nm",
        "ssc2_alpha_lj",
        "ssc2_alpha_coul",
        "ssc2_beta_coul",
        "ssc2_switch_width_nm",
        "charge_steps_per_stage",
        "sterics_steps",
        "subdivisions_per_stage",
        "total_steps",
        "path_nodes",
        "vdw_a",
        "charge_a",
        "segments_per_interval",
        "path_mode",
        "stage_interpolation",
    }
    return {key: value for key, value in softcore.items() if key in supported}


def _long_range_correction_mode(protocol, variant):
    return variant["softcore"].get(
        "long_range_correction",
        protocol["softcore"].get("long_range_correction", "dynamic"),
    )


def _run_variant(config, bank_root, edge, edge_payload, variant, platform, properties):
    edge_root = bank_root / edge_payload["root"]
    prepared_manifest = yaml.safe_load((edge_root / "prepared" / "manifest.yaml").read_text()) or {}
    source_protocol = yaml.safe_load((edge_root / "switch_protocol.yaml").read_text()) or {}
    result_dir = config["output_dir"] / edge / variant["name"]
    result_dir.mkdir(parents=True, exist_ok=True)
    total_float = variant["duration_ps"] * 1000.0 / config["timestep_fs"]
    total_steps = int(round(total_float))
    if not math.isclose(total_float, total_steps, abs_tol=1.0e-9):
        raise HybridSwitchBenchmarkError(
            f"duration {variant['duration_ps']} ps is not divisible by timestep"
        )
    protocol_record = {
        "schema_version": SCHEMA_VERSION,
        "edge": edge,
        "snapshot_bank_manifest_sha256": _sha256(bank_root / "manifest.yaml"),
        "prepared_fingerprint": edge_payload["prepared_fingerprint"],
        "temperature_k": config["temperature_k"],
        "timestep_fs": config["timestep_fs"],
        "duration_ps": variant["duration_ps"],
        "total_steps": total_steps,
        "stage_interpolation": variant["stage_interpolation"],
        "softcore_overrides": variant["softcore"],
        "n_snapshots": config["n_snapshots"],
        "environments": config["environments"],
    }
    protocol_path = result_dir / "protocol.yaml"
    if protocol_path.exists():
        if (yaml.safe_load(protocol_path.read_text()) or {}) != protocol_record:
            raise HybridSwitchBenchmarkError(f"replay protocol changed: {protocol_path}")
    else:
        _write_yaml_atomic(protocol_path, protocol_record)

    work = {}
    timing_path = result_dir / "switch_timing.csv"
    for environment in config["environments"]:
        LOGGER.info(
            "Loading %s %s for protocol %s",
            edge,
            environment,
            variant["name"],
        )
        prepared = load_prepared_hybrid_bundle(
            edge_root / "prepared", prepared_manifest["environments"][environment]
        )
        unique_a = prepared.provenance["unique_a_particle_indices"]
        unique_b = prepared.provenance["unique_b_particle_indices"]
        options = _source_softcore(source_protocol, variant)
        lrc_mode = _long_range_correction_mode(source_protocol, variant)
        hamiltonian = create_softcore_hamiltonian(
            prepared.endpoint_a,
            prepared.endpoint_b,
            unique_a,
            unique_b,
            use_long_range_correction=lrc_mode == "dynamic",
            **options,
        )
        segment_steps = _scale_segment_steps(hamiltonian.segment_steps, total_steps)
        _assert_fixed_volume_switch(hamiltonian.system)
        lrc = None
        if lrc_mode == "endpoint_correction":
            first_a = _load_state(_verify_entry(bank_root, edge_payload["snapshots"][environment]["a"][0]))
            first_b = _load_state(_verify_entry(bank_root, edge_payload["snapshots"][environment]["b"][0]))
            if options.get("coulomb_function") in {
                "amber_ssc2",
                "effective_distance_ssc2",
            }:
                evaluator = _PhysicalEndpointLRCCorrectionEvaluator(
                    prepared.endpoint_a,
                    prepared.endpoint_b,
                )
            else:
                lrc_hamiltonian = create_softcore_hamiltonian(
                    prepared.endpoint_a,
                    prepared.endpoint_b,
                    unique_a,
                    unique_b,
                    use_long_range_correction=True,
                    **options,
                )
                evaluator = _EndpointLRCCorrectionEvaluator(
                    hamiltonian,
                    lrc_hamiltonian,
                )
            lrc = _precompute_endpoint_lrc_corrections(
                evaluator,
                first_a,
                first_b,
                hamiltonian.parameter_values,
            )
        for direction, endpoint in (("forward", "a"), ("reverse", "b")):
            path = result_dir / f"{environment}_{direction}.csv"
            existing = _read_work(path)
            LOGGER.info(
                "%s %s %s %s: resuming at sample %d/%d",
                edge,
                variant["name"],
                environment,
                direction,
                len(existing) + 1,
                config["n_snapshots"],
            )
            context, integrator, parameter_values = _softcore_switch_context(
                hamiltonian,
                start=endpoint,
                timestep_fs=config["timestep_fs"],
                temperature_k=config["temperature_k"],
                platform=platform,
                properties=properties,
                seed=config["random_seed"],
            )
            direction_steps = segment_steps if direction == "forward" else list(reversed(segment_steps))
            integrator.set_segment_steps(direction_steps)
            entries = edge_payload["snapshots"][environment][endpoint]
            if len(entries) < config["n_snapshots"]:
                raise HybridSwitchBenchmarkError(
                    f"{edge} lacks requested {environment} endpoint {endpoint} snapshots"
                )
            for sample in range(len(existing) + 1, config["n_snapshots"] + 1):
                state = _load_state(_verify_entry(bank_root, entries[sample - 1]))
                integrator.setRandomNumberSeed(
                    _sample_seed(config["random_seed"], edge, environment, direction, sample)
                )
                _reset_softcore_context(context, integrator, parameter_values)
                _apply_state(context, state)
                started = time.perf_counter()
                try:
                    work_kj, _ = _run_segmented_protocol(integrator, direction_steps)
                    if lrc is not None:
                        work_kj += lrc[direction]["final"] - lrc[direction]["initial"]
                except Exception as exc:
                    if not _is_numerical_switch_failure(exc):
                        raise
                    work_kj = float("inf")
                    LOGGER.warning(
                        "%s %s %s %s sample %d failed: %s",
                        edge,
                        variant["name"],
                        environment,
                        direction,
                        sample,
                        exc,
                    )
                elapsed = time.perf_counter() - started
                ns_per_day = total_steps * config["timestep_fs"] * 1.0e-6 * 86400.0 / elapsed
                _append_work(path, sample, work_kj)
                row = {
                    "environment": environment,
                    "direction": direction,
                    "sample": sample,
                    "elapsed_seconds": elapsed,
                    "ns_per_day": ns_per_day,
                }
                _append_timing(timing_path, row)
                LOGGER.info(
                    "%s %s %s %s sample %d/%d complete: work %.6g kcal/mol, "
                    "%.3f ns/day",
                    edge,
                    variant["name"],
                    environment,
                    direction,
                    sample,
                    config["n_snapshots"],
                    work_kj / KCAL_TO_KJ,
                    ns_per_day,
                )
            work[f"{environment}_{direction}"] = _read_work(path)

    performance = {
        environment: _timing_means(timing_path).get(environment)
        for environment in config["environments"]
    }

    if set(config["environments"]) == {"complex", "solvent"}:
        analysis = analyze_two_leg_work(
            {
                "leg_a_forward": work["complex_forward"],
                "leg_a_reverse": work["complex_reverse"],
                "leg_b_forward": work["solvent_forward"],
                "leg_b_reverse": work["solvent_reverse"],
            },
            config["temperature_k"],
            config["bootstrap_samples"],
            config["random_seed"],
        )
    else:
        analysis = None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "edge": edge,
        "protocol": variant,
        "analysis": analysis,
        "performance_ns_per_day": performance,
        "finite_samples": {
            name: int(np.isfinite(values).sum()) for name, values in work.items()
        },
    }
    _write_yaml_atomic(result_dir / "result.yaml", payload)
    return payload, work


def run_benchmark(path):
    config = _resolve_config(path)
    bank_root = config["snapshot_bank"]
    manifest_path = bank_root / "manifest.yaml"
    if not manifest_path.is_file():
        raise HybridSwitchBenchmarkError(f"snapshot-bank manifest is missing: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise HybridSwitchBenchmarkError("snapshot-bank schema is unsupported")
    edges = config["edges"] or sorted(manifest.get("edges", {}))
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    platform, properties = _platform(config)
    LOGGER.info("Using OpenMM %s platform", platform.getName())
    summary = {"schema_version": SCHEMA_VERSION, "status": "completed", "edges": {}}
    for edge in edges:
        if edge not in manifest.get("edges", {}):
            raise HybridSwitchBenchmarkError(f"edge is absent from snapshot bank: {edge}")
        variants = {}
        variant_work = {}
        for variant in config["protocols"]:
            LOGGER.info("Running edge %s protocol %s", edge, variant["name"])
            result, work = _run_variant(
                config, bank_root, edge, manifest["edges"][edge], variant, platform, properties
            )
            variants[variant["name"]] = result
            variant_work[variant["name"]] = work
        paired = {}
        names = list(variant_work)
        if len(names) >= 2:
            baseline = names[0]
            for other in names[1:]:
                comparisons = {}
                for key in variant_work[baseline]:
                    left = np.asarray(variant_work[baseline][key], dtype=float)
                    right = np.asarray(variant_work[other][key], dtype=float)
                    finite = np.isfinite(left) & np.isfinite(right)
                    delta = right[finite] - left[finite]
                    comparisons[key] = {
                        "paired_finite_samples": int(finite.sum()),
                        "mean_work_difference_kcal_per_mol": (
                            None if not len(delta) else float(np.mean(delta))
                        ),
                        "rms_work_difference_kcal_per_mol": (
                            None if not len(delta) else float(np.sqrt(np.mean(delta**2)))
                        ),
                    }
                paired[f"{other}_minus_{baseline}"] = comparisons
        summary["edges"][edge] = {"variants": variants, "paired_work": paired}
        _write_yaml_atomic(config["output_dir"] / edge / "comparison.yaml", summary["edges"][edge])
        LOGGER.info("Completed benchmark edge %s", edge)
    _write_yaml_atomic(config["output_dir"] / "result.yaml", summary)
    return summary


def _parser():
    parser = argparse.ArgumentParser(description="Archive and replay hybrid NEQTI switch snapshots")
    subparsers = parser.add_subparsers(dest="command", required=True)
    archive = subparsers.add_parser("archive", help="archive a cohort snapshot bank")
    archive.add_argument("cohort_dir")
    archive.add_argument("output_dir")
    archive.add_argument("--snapshots", type=int, default=20)
    archive.add_argument("--edge", action="append", dest="edges")
    run = subparsers.add_parser("run", help="run a manifest-driven switch benchmark")
    run.add_argument("config")
    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    args = _parser().parse_args(argv)
    if args.command == "archive":
        manifest = archive_snapshot_bank(
            args.cohort_dir, args.output_dir, snapshots=args.snapshots, edges=args.edges
        )
        print(f"Archived {len(manifest['edges'])} edges in {Path(args.output_dir).resolve()}")
    else:
        result = run_benchmark(args.config)
        print(f"Completed {len(result['edges'])} benchmark edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
