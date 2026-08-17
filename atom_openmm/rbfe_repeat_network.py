from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path

import yaml


class RBFERepeatNetworkError(ValueError):
    pass


def _copy_template(source, target):
    ignored = shutil.ignore_patterns(
        "run", "slurm-*.out", "slurm-*.err", "network_result.yaml",
        "edge_replicates.csv", "edge_aggregates.csv", "node_free_energies.csv",
        "targets.csv",
    )
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if destination.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, destination, ignore=ignored)
        elif not any(item.match(pattern) for pattern in ("slurm-*.out", "slurm-*.err")):
            shutil.copy2(item, destination)


def _set_random_seed(value, seed):
    if isinstance(value, dict):
        for key in list(value):
            if key == "random_seed":
                value[key] = int(seed)
            else:
                _set_random_seed(value[key], seed)
    elif isinstance(value, list):
        for item in value:
            _set_random_seed(item, seed)


def _share_prepared_bundles(source, target, network):
    linked = []
    for edge in network["edges"]:
        source_edge = source / edge["directory"]
        bundles = sorted(source_edge.glob("run/*/prepared/manifest.yaml"))
        if not bundles:
            continue
        if len(bundles) != 1:
            raise RBFERepeatNetworkError(
                f"expected at most one prepared bundle below {source_edge}"
            )
        source_bundle = bundles[0].parent
        pair_directory = bundles[0].parent.parent.name
        target_bundle = target / edge["directory"] / "run" / pair_directory / "prepared"
        target_bundle.parent.mkdir(parents=True, exist_ok=True)
        if target_bundle.exists() or target_bundle.is_symlink():
            continue
        target_bundle.symlink_to(
            os.path.relpath(source_bundle, target_bundle.parent),
            target_is_directory=True,
        )
        linked.append(str(target_bundle.relative_to(target)))
    return linked


def _has_finite_result(edge_directory):
    candidates = sorted((edge_directory / "run").glob("*/result.yaml"))
    if len(candidates) != 1:
        return False
    payload = yaml.safe_load(candidates[0].read_text()) or {}
    result = payload.get("result") or {}
    try:
        value = float(result["ddg_kcal_per_mol"])
        error = float(result["ddg_error_kcal_per_mol"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isfinite(error) and error > 0.0


def _reuse_completed_edges(reuse_root, repeat_id, target, network):
    if reuse_root is None:
        return {}
    source_repeat = reuse_root / repeat_id
    network_path = source_repeat / "network.yaml"
    if not network_path.is_file():
        return {}
    source_network = yaml.safe_load(network_path.read_text()) or {}
    source_edges = {
        str(edge.get("id", edge["directory"])): edge
        for edge in source_network.get("edges", [])
    }
    reused = {}
    for edge in network["edges"]:
        edge_id = str(edge.get("id", edge["directory"]))
        source_edge = source_edges.get(edge_id)
        if source_edge is None:
            continue
        if (
            str(source_edge["ligand_a"]) != str(edge["ligand_a"])
            or str(source_edge["ligand_b"]) != str(edge["ligand_b"])
        ):
            raise RBFERepeatNetworkError(
                f"reused edge {edge_id} has a different orientation"
            )
        source_directory = source_repeat / source_edge["directory"]
        if not _has_finite_result(source_directory):
            continue
        destination = target / edge["directory"]
        if destination.exists() or destination.is_symlink():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(
            os.path.relpath(source_directory, destination.parent),
            target_is_directory=True,
        )
        reused[edge_id] = str(source_directory.resolve())
    return reused


def generate_repeats(
    source, output, repeats=3, seed_base=20260817, share_prepared=True,
    reuse_repeats_from=None,
):
    source = Path(source).resolve()
    output = Path(output).resolve()
    reuse_root = (
        Path(reuse_repeats_from).resolve() if reuse_repeats_from is not None else None
    )
    if repeats < 1:
        raise RBFERepeatNetworkError("repeats must be positive")
    if not (source / "network.yaml").is_file():
        raise RBFERepeatNetworkError(f"missing template network.yaml in {source}")
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    source_network = yaml.safe_load((source / "network.yaml").read_text()) or {}
    shared_bundles = {}
    reused_edges = {}
    for repeat_index in range(1, repeats + 1):
        repeat_id = f"replicate_{repeat_index}"
        target = output / repeat_id
        target.mkdir(parents=True, exist_ok=True)
        reused_edges[repeat_id] = _reuse_completed_edges(
            reuse_root, repeat_id, target, source_network
        )
        _copy_template(source, target)
        network = yaml.safe_load(yaml.safe_dump(source_network))
        (target / "network.yaml").write_text(yaml.safe_dump(network, sort_keys=False))
        for edge_index, edge in enumerate(network["edges"], 1):
            edge_id = str(edge.get("id", edge["directory"]))
            if edge_id in reused_edges[repeat_id]:
                continue
            workflow_path = target / edge["directory"] / "workflow.yaml"
            if not workflow_path.is_file():
                continue
            workflow = yaml.safe_load(workflow_path.read_text()) or {}
            seed = int(seed_base) + repeat_index * 100000 + edge_index
            _set_random_seed(workflow, seed)
            workflow_path.write_text(yaml.safe_dump(workflow, sort_keys=False))
        shared_bundles[repeat_id] = (
            _share_prepared_bundles(source, target, network) if share_prepared else []
        )
        pending_edges = [
            edge for edge in network["edges"]
            if str(edge.get("id", edge["directory"])) not in reused_edges[repeat_id]
        ]
        submit = target / "submit_all.sh"
        submit.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + "\n".join(
                f'(cd "$(dirname "$0")/{edge["directory"]}" && sbatch run.sh)'
                for edge in pending_edges
            )
            + ("\n" if pending_edges else 'echo "No pending repeat edges."\n')
        )
        submit.chmod(0o755)
        entries.append({"id": repeat_id, "path": f"{repeat_id}/network.yaml"})
    base = source_network
    aggregate = {
        "schema_version": 2,
        "replicate_networks": entries,
        "prepared_system_policy": {
            "share_immutable_bundles": bool(share_prepared),
            "linked_bundles": shared_bundles,
            "sampled_state_sharing": False,
        },
        "reused_edge_results": reused_edges,
        "new_job_count": sum(
            len(source_network["edges"]) - len(values)
            for values in reused_edges.values()
        ),
    }
    for key in ("targets", "target", "target_references", "reference_datasets", "cycles"):
        if key in base:
            aggregate[key] = base[key]
    (output / "repeat_network.yaml").write_text(
        yaml.safe_dump(aggregate, sort_keys=False)
    )
    (output / "analyze_repeats.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "python -m atom_openmm.rbfe_network \"$(dirname \"$0\")/repeat_network.yaml\"\n"
    )
    (output / "submit_all.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(
            f'(cd "$(dirname "$0")/{entry["id"]}" && ./submit_all.sh)'
            for entry in entries
        )
        + "\n"
    )
    for script in (output / "analyze_repeats.sh", output / "submit_all.sh"):
        script.chmod(0o755)
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate independent RBFE network repeats")
    parser.add_argument("source_cohort")
    parser.add_argument("output")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=20260817)
    parser.add_argument("--no-share-prepared", action="store_true")
    parser.add_argument("--reuse-repeats-from")
    args = parser.parse_args(argv)
    generate_repeats(
        args.source_cohort, args.output, args.repeats, args.seed_base,
        share_prepared=not args.no_share_prepared,
        reuse_repeats_from=args.reuse_repeats_from,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
