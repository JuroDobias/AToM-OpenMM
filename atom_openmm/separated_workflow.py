from __future__ import annotations

import csv
import hashlib
import math
import os
from pathlib import Path

import numpy as np
from openmm import app, unit
import yaml

from atom_openmm.covalent_softcore import create_softcore_hamiltonian
from atom_openmm.covalent_workflow import (
    KCAL_TO_KJ,
    _adaptive_work_statistics,
    _is_numerical_switch_failure,
    _normalized_settings,
    _platform,
    _reset_softcore_context,
    _run_segmented_protocol,
    _scale_segment_steps,
    _softcore_switch_context,
    _write_yaml_atomic,
)
from atom_openmm.hybrid_parameters import parameterize_ligand
from atom_openmm.neqti import analyze_two_leg_work
from atom_openmm.separated_node_bank import (
    load_node_state,
    load_physical_node_environment,
)
from atom_openmm.separated_topology import (
    FrameRestraintSettings,
    InactivePoseThermalizer,
    apply_assembled_state,
    assemble_positions_and_velocities,
    environment_signature,
    prepare_separated_environment,
)


class SeparatedWorkflowError(ValueError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _settings(workflow, plan, config_path):
    config = _normalized_settings(workflow)
    alchemy = workflow.get("alchemy") or {}
    bank = dict(alchemy.get("node_bank") or {})
    path = Path(bank.get("path", plan["workdir"] / "node_bank"))
    if not path.is_absolute():
        path = (Path(config_path).parent / path).resolve()
    restraint = dict(alchemy.get("frame_restraint") or {})
    config["node_bank_path"] = path
    config["frame_restraint"] = FrameRestraintSettings(
        translation_k_kcal_mol_a2=float(
            restraint.get("translation_k_kcal_mol_a2", 10.0)
        ),
        orientation_k_kcal_mol=float(
            restraint.get("orientation_k_kcal_mol", 25.0)
        ),
        roll_k_kcal_mol=float(restraint.get("roll_k_kcal_mol", 25.0)),
        thermalization_steps=int(restraint.get("thermalization_steps", 5000)),
    )
    return config


def _validate_config(workflow, plan, config_path, require_bank=True):
    config = _settings(workflow, plan, config_path)
    if config["interpolation"] != "softcore_linear":
        raise SeparatedWorkflowError(
            "separated topology currently requires neqti.interpolation: softcore_linear"
        )
    if config["softcore"]["long_range_correction"] != "dynamic":
        raise SeparatedWorkflowError(
            "separated topology initially requires softcore.long_range_correction: dynamic"
        )
    if config["schedule_optimization"]["enabled"]:
        raise SeparatedWorkflowError(
            "separated topology does not yet support schedule optimization"
        )
    if config["convergence"]["enabled"]:
        raise SeparatedWorkflowError(
            "separated topology initially uses adaptive duration without early convergence"
        )
    if config["failed_switch_policy"] not in {"abort", "count_as_infinite"}:
        raise SeparatedWorkflowError(
            "neqti.failed_switch_policy must be 'abort' or 'count_as_infinite'"
        )
    if config["frame_restraint"].thermalization_steps < 0:
        raise SeparatedWorkflowError("frame restraint thermalization steps cannot be negative")
    if any(
        value <= 0.0
        for value in (
            config["frame_restraint"].translation_k_kcal_mol_a2,
            config["frame_restraint"].orientation_k_kcal_mol,
            config["frame_restraint"].roll_k_kcal_mol,
        )
    ):
        raise SeparatedWorkflowError("frame restraint force constants must be positive")
    adaptive = config["adaptive_switching"]
    if adaptive["enabled"]:
        if adaptive["candidate_total_steps"][0] != config["switch_steps"]:
            raise SeparatedWorkflowError(
                "the first adaptive duration must equal the configured base softcore path"
            )
        if adaptive["pilot_samples_per_direction"] > config["n_snapshots"]:
            raise SeparatedWorkflowError(
                "adaptive pilot samples cannot exceed neqti.n_snapshots"
            )
        if not adaptive["reuse_selected_pilot_samples"]:
            raise SeparatedWorkflowError(
                "separated topology requires reuse_selected_pilot_samples: true"
            )
    if require_bank and not (config["node_bank_path"] / "manifest.yaml").is_file():
        raise SeparatedWorkflowError(
            "node bank is not prepared; run atom-rbfe --prepare-node-bank workflow.yaml"
        )
    return config


def validate_separated_workflow(path, require_bank=True):
    from atom_openmm.rbfe_workflow import build_small_molecule_plan, load_workflow_config

    loaded = load_workflow_config(path)
    plan = build_small_molecule_plan(loaded)
    config = _validate_config(
        loaded["workflow"], plan, loaded["config_path"], require_bank=require_bank
    )
    if require_bank:
        manifest = yaml.safe_load(
            (config["node_bank_path"] / "manifest.yaml").read_text()
        ) or {}
        if manifest.get("receptor_sha256") != _sha256(plan["receptor_file"]):
            raise SeparatedWorkflowError("node bank receptor differs from workflow.receptor")
        for pair in plan["pairs"]:
            for name, source in (
                (pair["lig1_name"], pair["lig1_file"]),
                (pair["lig2_name"], pair["lig2_file"]),
            ):
                if name not in manifest.get("nodes", {}):
                    raise SeparatedWorkflowError(f"node bank does not contain ligand {name}")
                if manifest["nodes"][name].get("ligand_sha256") != _sha256(source):
                    raise SeparatedWorkflowError(
                        f"node bank ligand {name} differs from workflow input"
                    )
    return True


def plan_separated_workflow(path):
    from atom_openmm.rbfe_workflow import build_small_molecule_plan, load_workflow_config

    loaded = load_workflow_config(path)
    plan = build_small_molecule_plan(loaded)
    config = _validate_config(
        loaded["workflow"], plan, loaded["config_path"], require_bank=False
    )
    return {
        "schema_version": 1,
        "chemistry": "noncovalent",
        "alchemy_model": "separated_topology",
        "thermodynamic_cycle": "complex_solvent",
        "sampling_method": "neqti",
        "node_bank": str(config["node_bank_path"]),
        "unique_nodes": sorted({
            pair[key] for pair in plan["pairs"] for key in ("lig1_name", "lig2_name")
        }),
        "pairs": [
            {
                "ligand_a": pair["lig1_name"],
                "ligand_b": pair["lig2_name"],
                "workdir": str(pair["jobdir"]),
            }
            for pair in plan["pairs"]
        ],
    }


def _parameterize_pair(pair, workflow):
    setup = workflow.get("setup") or {}
    options = {
        "ligand_forcefield": setup.get("ligand_forcefield", "espaloma-0.3.2"),
        "ligand_charge_model": setup.get("ligand_charge_model", "nn"),
        "allow_undefined_stereo": bool(setup.get("allow_undefined_stereo", False)),
    }
    return (
        parameterize_ligand(pair["lig1_file"], **options),
        parameterize_ligand(pair["lig2_file"], **options),
    )


def _prepared_pair(pair, workflow, config, environment, bank_manifest):
    parameters_a, parameters_b = _parameterize_pair(pair, workflow)
    physical_a = load_physical_node_environment(
        config["node_bank_path"], pair["lig1_name"], environment
    )
    physical_b = load_physical_node_environment(
        config["node_bank_path"], pair["lig2_name"], environment
    )
    signature_a = environment_signature(
        physical_a.topology, physical_a.solute_atom_count
    )
    signature_b = environment_signature(
        physical_b.topology, physical_b.solute_atom_count
    )
    if signature_a != signature_b:
        raise SeparatedWorkflowError(
            f"canonical {environment} topology differs between "
            f"{pair['lig1_name']} and {pair['lig2_name']}"
        )
    anchors_a = tuple(bank_manifest["nodes"][pair["lig1_name"]]["anchors"])
    anchors_b = tuple(bank_manifest["nodes"][pair["lig2_name"]]["anchors"])
    prepared = prepare_separated_environment(
        parameters_a,
        parameters_b,
        physical_a,
        anchors_a=anchors_a,
        anchors_b=anchors_b,
        restraint_settings=config["frame_restraint"],
    )
    return prepared, parameters_a, parameters_b, anchors_a, anchors_b


def _read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _append_row(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_selected_work(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "sample", "work_kj_per_mol", "work_kcal_per_mol",
        "active_node", "active_snapshot_id", "inactive_node",
        "inactive_vacuum_snapshot_id", "duration_ps", "status",
    ]
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_diagnostic_pdb(path, topology, assembled, inactive_indices, endpoint):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w") as handle:
        app.PDBFile.writeFile(topology, assembled["positions"], handle, keepIds=True)
    inactive = {int(value) for value in inactive_indices}
    atom_index = 0
    output = [
        f"REMARK 900 SEPARATED TOPOLOGY ENDPOINT {endpoint.upper()}\n",
        "REMARK 900 ACTIVE ATOMS OCCUPANCY 1.00; INACTIVE LIGAND 0.00\n",
    ]
    for line in temporary.read_text().splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")):
            occupancy = 0.0 if atom_index in inactive else 1.0
            line = f"{line[:54]}{occupancy:6.2f}{line[60:]}"
            atom_index += 1
        output.append(line)
    path.write_text("".join(output))
    temporary.unlink()


def _anchor_rmsd_a(positions, anchors_a, anchors_b):
    positions_nm = np.asarray(
        positions.value_in_unit(unit.nanometer), dtype=float
    )
    return float(np.sqrt(np.mean(np.sum(
        (positions_nm[list(anchors_a)] - positions_nm[list(anchors_b)]) ** 2,
        axis=1,
    ))) * 10.0)


def _write_switch_frame(path, topology, positions, metadata):
    path = Path(path)
    with path.open("w") as handle:
        handle.write(
            "REMARK 900 SEPARATED TOPOLOGY SWITCH FRAME "
            f"{metadata['point']}\n"
        )
        handle.write(
            "REMARK 900 PATH FRACTION "
            f"{metadata['path_fraction']:.8f} ANCHOR RMSD "
            f"{metadata['anchor_rmsd_a']:.6f} A\n"
        )
        app.PDBFile.writeFile(topology, positions, handle, keepIds=True)


def _switch_frame_diagnostics(
    *, context, topology, anchors_a, anchors_b, directory, direction, segment_steps
):
    directory = Path(directory)
    frames = []
    total_steps = sum(int(value) for value in segment_steps)

    def capture(point, completed_steps, cumulative_work=None):
        state = context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        progress = completed_steps / float(total_steps)
        path_fraction = progress if direction == "forward" else 1.0 - progress
        metadata = {
            "point": point,
            "completed_steps": int(completed_steps),
            "total_steps": int(total_steps),
            "path_fraction": float(path_fraction),
            "anchor_rmsd_a": _anchor_rmsd_a(
                positions, anchors_a, anchors_b
            ),
        }
        if cumulative_work is not None:
            metadata["cumulative_work_kj_per_mol"] = float(cumulative_work)
        filename = f"{direction}_frame_{len(frames):02d}_{point}.pdb"
        metadata["coordinates"] = filename
        _write_switch_frame(directory / filename, topology, positions, metadata)
        frames.append(metadata)

    capture("start", 0)

    def after_segment(
        *, segment, completed_steps, total_steps, cumulative_work_kj_per_mol
    ):
        point = "end" if completed_steps == total_steps else f"node_{segment + 1:02d}"
        capture(point, completed_steps, cumulative_work_kj_per_mol)

    return frames, after_segment


def _candidate_label(time_ps):
    return f"{time_ps:g}ps"


def _run_environment(
    pair,
    environment,
    prepared,
    parameters_a,
    parameters_b,
    anchors_a,
    anchors_b,
    config,
    workdir,
    bank_manifest,
    platform,
    properties,
    seed,
):
    unique_a = prepared.provenance["unique_a_particle_indices"]
    unique_b = prepared.provenance["unique_b_particle_indices"]
    options = {
        key: value for key, value in config["softcore"].items()
        if key != "long_range_correction"
    }
    softcore = create_softcore_hamiltonian(
        prepared.endpoint_a,
        prepared.endpoint_b,
        unique_a,
        unique_b,
        use_long_range_correction=True,
        **options,
    )
    contexts = {
        "forward": _softcore_switch_context(
            softcore,
            start="a",
            timestep_fs=config["timestep_fs"],
            temperature_k=config["temperature_k"],
            platform=platform,
            properties=properties,
            seed=seed,
        ),
        "reverse": _softcore_switch_context(
            softcore,
            start="b",
            timestep_fs=config["timestep_fs"],
            temperature_k=config["temperature_k"],
            platform=platform,
            properties=properties,
            seed=seed + 100,
        ),
    }
    thermalizers = {
        "a": InactivePoseThermalizer(
            prepared.endpoint_a,
            unique_b,
            platform=platform,
            properties=properties,
            temperature_k=config["temperature_k"],
            timestep_fs=config["timestep_fs"],
            seed=seed + 200,
        ),
        "b": InactivePoseThermalizer(
            prepared.endpoint_b,
            unique_a,
            platform=platform,
            properties=properties,
            temperature_k=config["temperature_k"],
            timestep_fs=config["timestep_fs"],
            seed=seed + 300,
        ),
    }
    adaptive = config["adaptive_switching"]
    candidate_times = (
        adaptive["candidate_times_ps"] if adaptive["enabled"]
        else [config["switch_steps"] * config["timestep_fs"] / 1000.0]
    )
    candidate_steps = (
        adaptive["candidate_total_steps"] if adaptive["enabled"]
        else [config["switch_steps"]]
    )
    pilot_count = (
        adaptive["pilot_samples_per_direction"] if adaptive["enabled"]
        else config["n_snapshots"]
    )
    node_count = int(bank_manifest["snapshot_count"])
    if node_count < config["n_snapshots"]:
        raise SeparatedWorkflowError("node bank has too few snapshots")
    base = workdir / "separated_switching" / environment
    base.mkdir(parents=True, exist_ok=True)
    selected = None
    diagnostics = []

    def assembled(endpoint, sample):
        active_name = pair["lig1_name"] if endpoint == "a" else pair["lig2_name"]
        inactive_name = pair["lig2_name"] if endpoint == "a" else pair["lig1_name"]
        active_state, active_entry = load_node_state(
            config["node_bank_path"], active_name, environment, sample
        )
        # Use one graph-stable permutation for every node, independent of edge
        # orientation, so shared vacuum samples can be bootstrapped coherently.
        vacuum_index = (sample * 104729 + 1) % node_count
        inactive_state, inactive_entry = load_node_state(
            config["node_bank_path"], inactive_name, "vacuum", vacuum_index
        )
        value = assemble_positions_and_velocities(
            endpoint=endpoint,
            active_state=active_state,
            inactive_state=inactive_state,
            ligand_a_count=parameters_a.molecule.n_atoms,
            ligand_b_count=parameters_b.molecule.n_atoms,
            anchors_a=anchors_a,
            anchors_b=anchors_b,
        )
        value = thermalizers[endpoint].sample(
            value,
            config["frame_restraint"].thermalization_steps,
            seed + sample * 1000 + (0 if endpoint == "a" else 500),
        )
        dual_a = list(anchors_a)
        dual_b = [parameters_a.molecule.n_atoms + int(index) for index in anchors_b]
        value["anchor_rmsd_a"] = _anchor_rmsd_a(
            value["positions"], dual_a, dual_b
        )
        value.update({
            "active_name": active_name,
            "inactive_name": inactive_name,
            "active_snapshot_id": active_entry["file"],
            "inactive_snapshot_id": inactive_entry["file"],
        })
        return value

    try:
        for time_ps, total_steps in zip(candidate_times, candidate_steps):
            label = _candidate_label(time_ps)
            candidate_dir = base / label
            candidate_dir.mkdir(parents=True, exist_ok=True)
            segment_forward = _scale_segment_steps(softcore.segment_steps, total_steps)
            segment_reverse = list(reversed(segment_forward))
            for direction, endpoint, segment_steps in (
                ("forward", "a", segment_forward),
                ("reverse", "b", segment_reverse),
            ):
                path = candidate_dir / f"{direction}.csv"
                rows = _read_rows(path)
                completed = {int(row["sample"]) for row in rows}
                context, integrator, values = contexts[direction]
                integrator.set_segment_steps(segment_steps)
                for sample in range(pilot_count):
                    if sample + 1 in completed:
                        continue
                    start = assembled(endpoint, sample)
                    _reset_softcore_context(context, integrator, values)
                    apply_assembled_state(
                        context,
                        start,
                        temperature_k=config["temperature_k"],
                        seed=seed + sample,
                    )
                    frame_rows = None
                    frame_callback = None
                    frame_path = candidate_dir / f"{direction}_frame_diagnostics.yaml"
                    if sample == 0 and not frame_path.exists():
                        frame_rows, frame_callback = _switch_frame_diagnostics(
                            context=context,
                            topology=prepared.topology,
                            anchors_a=dual_a,
                            anchors_b=dual_b,
                            directory=candidate_dir,
                            direction=direction,
                            segment_steps=segment_steps,
                        )
                    status = "finite"
                    try:
                        raw_work, _ = _run_segmented_protocol(
                            integrator,
                            segment_steps,
                            segment_callback=frame_callback,
                        )
                    except Exception as exc:
                        if not _is_numerical_switch_failure(exc):
                            raise
                        if config["failed_switch_policy"] == "abort":
                            raise
                        raw_work = math.inf
                        status = "infinite"
                    if frame_rows is not None:
                        _write_yaml_atomic(frame_path, {
                            "schema_version": 1,
                            "environment": environment,
                            "direction": direction,
                            "sample": sample + 1,
                            "status": status,
                            "frames": frame_rows,
                        })
                    row = {
                        "sample": sample + 1,
                        "work_kj_per_mol": raw_work,
                        "work_kcal_per_mol": raw_work / KCAL_TO_KJ,
                        "active_node": start["active_name"],
                        "active_snapshot_id": start["active_snapshot_id"],
                        "inactive_node": start["inactive_name"],
                        "inactive_vacuum_snapshot_id": start["inactive_snapshot_id"],
                        "duration_ps": time_ps,
                        "status": status,
                    }
                    _append_row(path, row)
                    if sample == 0 and time_ps == candidate_times[0]:
                        inactive_indices = unique_b if endpoint == "a" else unique_a
                        _write_diagnostic_pdb(
                            workdir / f"{environment}_endpoint_{endpoint}_assembled.pdb",
                            prepared.topology,
                            start,
                            inactive_indices,
                            endpoint,
                        )
                        diagnostics.append({
                            "environment": environment,
                            "endpoint": endpoint,
                            "anchor_rmsd_a": start["anchor_rmsd_a"],
                        })
            forward_rows = _read_rows(candidate_dir / "forward.csv")
            reverse_rows = _read_rows(candidate_dir / "reverse.csv")
            statistics = _adaptive_work_statistics(
                [float(row["work_kcal_per_mol"]) for row in forward_rows],
                [float(row["work_kcal_per_mol"]) for row in reverse_rows],
                config,
            )
            if selected is None and (statistics["passed"] or time_ps == candidate_times[-1]):
                selected = {
                    "label": label,
                    "time_ps": float(time_ps),
                    "steps": int(total_steps),
                    "statistics": statistics,
                }
                break
        if selected is None:
            raise SeparatedWorkflowError(f"no switching duration selected for {environment}")
        selected_dir = base / selected["label"]
        segment_forward = _scale_segment_steps(softcore.segment_steps, selected["steps"])
        for direction, endpoint, segment_steps in (
            ("forward", "a", segment_forward),
            ("reverse", "b", list(reversed(segment_forward))),
        ):
            path = selected_dir / f"{direction}.csv"
            rows = _read_rows(path)
            completed = {int(row["sample"]) for row in rows}
            context, integrator, values = contexts[direction]
            integrator.set_segment_steps(segment_steps)
            for sample in range(config["n_snapshots"]):
                if sample + 1 in completed:
                    continue
                start = assembled(endpoint, sample)
                _reset_softcore_context(context, integrator, values)
                apply_assembled_state(
                    context, start, temperature_k=config["temperature_k"], seed=seed + sample
                )
                status = "finite"
                try:
                    raw_work, _ = _run_segmented_protocol(integrator, segment_steps)
                except Exception as exc:
                    if not _is_numerical_switch_failure(exc):
                        raise
                    if config["failed_switch_policy"] == "abort":
                        raise
                    raw_work = math.inf
                    status = "infinite"
                _append_row(path, {
                    "sample": sample + 1,
                    "work_kj_per_mol": raw_work,
                    "work_kcal_per_mol": raw_work / KCAL_TO_KJ,
                    "active_node": start["active_name"],
                    "active_snapshot_id": start["active_snapshot_id"],
                    "inactive_node": start["inactive_name"],
                    "inactive_vacuum_snapshot_id": start["inactive_snapshot_id"],
                    "duration_ps": selected["time_ps"],
                    "status": status,
                })
            final_rows = _read_rows(path)
            _write_selected_work(
                workdir / f"{environment}_{direction}.csv",
                final_rows[: config["n_snapshots"]],
            )
        _write_yaml_atomic(base / "selection.yaml", {
            "schema_version": 1,
            "selected": selected,
            "diagnostics": diagnostics,
        })
        return selected
    finally:
        for thermalizer in thermalizers.values():
            thermalizer.close()
        for context, integrator, _ in contexts.values():
            del context, integrator


def _read_work(path):
    return [float(row["work_kcal_per_mol"]) for row in _read_rows(path)]


def run_separated_workflow(path):
    from atom_openmm.rbfe_workflow import build_small_molecule_plan, load_workflow_config

    validate_separated_workflow(path, require_bank=True)
    loaded = load_workflow_config(path)
    workflow = loaded["workflow"]
    plan = build_small_molecule_plan(loaded)
    config = _settings(workflow, plan, loaded["config_path"])
    bank_manifest = yaml.safe_load(
        (config["node_bank_path"] / "manifest.yaml").read_text()
    ) or {}
    platform, properties = _platform(workflow)
    results = []
    for pair_index, pair in enumerate(plan["pairs"]):
        workdir = pair["jobdir"]
        workdir.mkdir(parents=True, exist_ok=True)
        selections = {}
        for environment_index, environment in enumerate(("complex", "solvent")):
            prepared, parameters_a, parameters_b, anchors_a, anchors_b = _prepared_pair(
                pair, workflow, config, environment, bank_manifest
            )
            selections[environment] = _run_environment(
                pair,
                environment,
                prepared,
                parameters_a,
                parameters_b,
                anchors_a,
                anchors_b,
                config,
                workdir,
                bank_manifest,
                platform,
                properties,
                config["random_seed"] + pair_index * 1000000 + environment_index * 100000,
            )
        work = {
            "leg_a_forward": _read_work(workdir / "complex_forward.csv"),
            "leg_a_reverse": _read_work(workdir / "complex_reverse.csv"),
            "leg_b_forward": _read_work(workdir / "solvent_forward.csv"),
            "leg_b_reverse": _read_work(workdir / "solvent_reverse.csv"),
        }
        analysis = analyze_two_leg_work(
            work, config["temperature_k"], config["bootstrap_samples"], config["random_seed"]
        )
        payload = {
            "schema_version": 1,
            "tool": "atom_openmm_rbfe",
            "jobname": pair["jobname"],
            "status": "completed" if analysis is not None else "partial",
            "method": "neqti",
            "chemistry": "noncovalent",
            "alchemy_model": "separated_topology",
            "thermodynamic_cycle": "complex_solvent",
            "ligand_a": pair["lig1_name"],
            "ligand_b": pair["lig2_name"],
            "workdir": str(workdir.resolve()),
            "node_bank": {
                "path": str(config["node_bank_path"]),
                "manifest_sha256": _sha256(config["node_bank_path"] / "manifest.yaml"),
            },
            "frame_restraint": dict(config["frame_restraint"].__dict__),
            "adaptive_switching": selections,
            "result": None if analysis is None else {
                "ddg_kcal_per_mol": analysis["bar_dg_kcal_per_mol"],
                "ddg_error_kcal_per_mol": analysis["bar_bootstrap_std_kcal_per_mol"],
                "ddg_kj_per_mol": analysis["bar_dg_kcal_per_mol"] * KCAL_TO_KJ,
                "ddg_error_kj_per_mol": analysis["bar_bootstrap_std_kcal_per_mol"] * KCAL_TO_KJ,
                "estimator": "BAR",
                "components": {
                    "complex": analysis["components"]["leg_a"],
                    "solvent": analysis["components"]["leg_b"],
                },
            },
            "quality": {
                "convergence_status": "usable" if analysis is not None else "partial",
                "overlap_score": None if analysis is None else min(
                    analysis["components"]["leg_a"]["overlap_score"],
                    analysis["components"]["leg_b"]["overlap_score"],
                ),
                "warnings": [],
            },
        }
        if analysis is not None:
            payload["ddg"] = analysis["bar_dg_kcal_per_mol"]
            payload["ddg_std"] = analysis["bar_bootstrap_std_kcal_per_mol"]
        _write_yaml_atomic(workdir / "result.yaml", payload)
        results.append(payload)
    return results


def analyze_separated_workflow(path):
    from atom_openmm.rbfe_workflow import build_small_molecule_plan, load_workflow_config

    loaded = load_workflow_config(path)
    plan = build_small_molecule_plan(loaded)
    config = _settings(loaded["workflow"], plan, loaded["config_path"])
    results = []
    for pair in plan["pairs"]:
        workdir = pair["jobdir"]
        work = {
            "leg_a_forward": _read_work(workdir / "complex_forward.csv"),
            "leg_a_reverse": _read_work(workdir / "complex_reverse.csv"),
            "leg_b_forward": _read_work(workdir / "solvent_forward.csv"),
            "leg_b_reverse": _read_work(workdir / "solvent_reverse.csv"),
        }
        results.append(analyze_two_leg_work(
            work, config["temperature_k"], config["bootstrap_samples"], config["random_seed"]
        ))
    return results
