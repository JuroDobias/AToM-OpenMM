from __future__ import annotations

import hashlib
import fcntl
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import openmm as mm
from openmm import app, unit
import yaml

from atom_openmm.covalent_systems import PreparedCovalentSystem
from atom_openmm.covalent_workflow import (
    _equilibrate_endpoint,
    _load_state,
    _normalized_settings,
    _platform,
    _sample_endpoint,
    _write_state,
    _write_yaml_atomic,
)
from atom_openmm.equilibration import neqti_hybrid_endpoint_steps
from atom_openmm.hybrid_parameters import parameterize_ligand
from atom_openmm.hybrid_systems import HybridSystemError, create_physical_ligand_environment


SCHEMA_VERSION = 2
READABLE_SCHEMA_VERSIONS = {1, SCHEMA_VERSION}


class NodeBankError(ValueError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(payload):
    return hashlib.sha256(
        yaml.safe_dump(payload, sort_keys=True).encode()
    ).hexdigest()


def _box_vectors_nm(topology):
    vectors = topology.getPeriodicBoxVectors()
    if vectors is None:
        raise NodeBankError("solvated topology has no periodic box vectors")
    return np.asarray([
        vector.value_in_unit(unit.nanometer) for vector in vectors
    ], dtype=float)


def _box_volume_nm3(vectors):
    return abs(float(np.linalg.det(np.asarray(vectors, dtype=float))))


def _serialize_system(directory, prepared):
    directory.mkdir(parents=True, exist_ok=True)
    system_path = directory / "system.xml"
    topology_path = directory / "topology.cif"
    system_path.write_text(mm.XmlSerializer.serialize(prepared.system))
    with topology_path.open("w") as handle:
        app.PDBxFile.writeFile(
            prepared.topology, prepared.positions, handle, keepIds=False
        )
    return {
        "system": {"file": system_path.name, "sha256": _sha256(system_path)},
        "topology": {"file": topology_path.name, "sha256": _sha256(topology_path)},
        "particle_count": prepared.system.getNumParticles(),
        "ligand_atom_count": int(prepared.solute_atom_count),
        "provenance": prepared.provenance,
    }


def _load_artifact(directory, payload, key):
    artifact = payload[key]
    path = Path(directory) / artifact["file"]
    if not path.is_file() or _sha256(path) != artifact["sha256"]:
        raise NodeBankError(f"node-bank artifact is missing or changed: {path}")
    return path


def load_physical_node_environment(bank, node, environment):
    bank = Path(bank)
    manifest = yaml.safe_load((bank / "manifest.yaml").read_text()) or {}
    payload = manifest["nodes"][node]["environments"][environment]
    directory = bank / "nodes" / node / environment
    system_path = _load_artifact(directory, payload, "system")
    topology_path = _load_artifact(directory, payload, "topology")
    topology_file = app.PDBxFile(str(topology_path))
    system = mm.XmlSerializer.deserialize(system_path.read_text())
    if system.getNumParticles() != topology_file.topology.getNumAtoms():
        raise NodeBankError(f"{node} {environment} topology and system counts differ")
    return PreparedCovalentSystem(
        topology_file.topology,
        topology_file.positions,
        system,
        int(payload["ligand_atom_count"]),
        dict(payload["provenance"]),
    )


def load_node_state(bank, node, environment, sample):
    bank = Path(bank)
    manifest = yaml.safe_load((bank / "manifest.yaml").read_text()) or {}
    if environment == "vacuum":
        entries = manifest["nodes"][node]["vacuum"]["snapshots"]
    else:
        entries = manifest["nodes"][node]["environments"][environment]["snapshots"]
    entry = entries[int(sample)]
    path = bank / entry["file"]
    if not path.is_file() or _sha256(path) != entry["sha256"]:
        raise NodeBankError(f"node snapshot is missing or changed: {path}")
    return _load_state(path), entry


def _ligands_from_plan(plan):
    nodes = {}
    for pair in plan["pairs"]:
        for name_key, file_key in (
            ("lig1_name", "lig1_file"),
            ("lig2_name", "lig2_file"),
        ):
            name = pair[name_key]
            path = Path(pair[file_key]).resolve()
            if name in nodes and nodes[name] != path:
                raise NodeBankError(f"node {name} resolves to multiple ligand files")
            nodes[name] = path
    return nodes


def _node_anchors(plan, alignments):
    from atom_openmm.rbfe_workflow import _alignment_atoms_for_pair

    anchors = {}
    for pair in plan["pairs"]:
        for ligand_key, name_key in (
            ("ligand_a", "lig1_name"),
            ("ligand_b", "lig2_name"),
        ):
            values = tuple(
                int(value) - 1
                for value in _alignment_atoms_for_pair(alignments, pair, ligand_key)
            )
            name = pair[name_key]
            if name in anchors and anchors[name] != values:
                raise NodeBankError(
                    f"node {name} has inconsistent frame atoms across graph edges: "
                    f"{anchors[name]} and {values}"
                )
            anchors[name] = values
    return anchors


def _canonical_box_size(parameters, receptor, padding_a, environment):
    ligand_coordinates = [
        np.asarray(
            bundle.molecule.conformers[0].to_openmm().value_in_unit(unit.nanometer)
        )
        for bundle in parameters.values()
    ]
    extents = []
    if environment == "complex":
        receptor_pdb = app.PDBFile(str(receptor))
        receptor_coordinates = np.asarray([
            value.value_in_unit(unit.nanometer) for value in receptor_pdb.positions
        ])
        extents.extend(
            np.ptp(np.concatenate([coordinates, receptor_coordinates], axis=0), axis=0)
            for coordinates in ligand_coordinates
        )
    else:
        extents.extend(np.ptp(coordinates, axis=0) for coordinates in ligand_coordinates)
    maximum = np.max(np.asarray(extents), axis=0) + 2.0 * float(padding_a) / 10.0
    side = float(np.max(maximum))
    return [side, side, side]


def _write_snapshot(path, state, bank_root):
    _write_state(path, state)
    return {
        "file": str(path.relative_to(bank_root)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _vacuum_snapshots(
    node_dir,
    bank_root,
    bundle,
    *,
    count,
    initial_steps,
    decorrelation_steps,
    config,
    platform,
    properties,
    seed,
):
    directory = node_dir / "vacuum"
    snapshots_dir = directory / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    topology = bundle.molecule.to_topology().to_openmm()
    state_file = directory / "state.xml"
    if not state_file.exists():
        integrator = mm.LangevinMiddleIntegrator(
            config["temperature_k"] * unit.kelvin,
            1.0 / unit.picosecond,
            config["timestep_fs"] * unit.femtosecond,
        )
        integrator.setRandomNumberSeed(int(seed))
        context = mm.Context(bundle.system, integrator, platform, properties)
        context.setPositions(bundle.molecule.conformers[0].to_openmm())
        context.setVelocitiesToTemperature(
            config["temperature_k"] * unit.kelvin, int(seed) + 1
        )
        integrator.step(int(initial_steps))
        _write_state(
            state_file,
            context.getState(getPositions=True, getVelocities=True, getEnergy=True),
        )
        del context, integrator
    entries = []
    for sample in range(int(count)):
        path = snapshots_dir / f"{sample:04d}.xml"
        if not path.exists():
            state, _ = _sample_endpoint(
                bundle.system,
                topology,
                state_file,
                tuple(range(bundle.molecule.n_atoms)),
                ensemble="vacuum",
                steps=decorrelation_steps,
                rest2_config=config["rest2"],
                output_dir=directory / "rest2",
                platform=platform,
                properties=properties,
                temperature_k=config["temperature_k"],
                timestep_fs=config["timestep_fs"],
                seed=seed + sample * 1000,
            )
            _write_state(path, state)
        entries.append(_write_snapshot(path, _load_state(path), bank_root))
    system_path = directory / "system.xml"
    topology_path = directory / "topology.cif"
    system_path.write_text(mm.XmlSerializer.serialize(bundle.system))
    with topology_path.open("w") as handle:
        app.PDBxFile.writeFile(
            topology, bundle.molecule.conformers[0].to_openmm(), handle
        )
    return {
        "system": {"file": str(system_path.relative_to(bank_root)), "sha256": _sha256(system_path)},
        "topology": {"file": str(topology_path.relative_to(bank_root)), "sha256": _sha256(topology_path)},
        "snapshots": entries,
    }


def _node_bank_context(path):
    from atom_openmm.rbfe_workflow import (
        build_small_molecule_plan,
        load_or_generate_alignments,
        load_workflow_config,
    )

    loaded = load_workflow_config(path)
    workflow = loaded["workflow"]
    plan = build_small_molecule_plan(loaded)
    nodes = _ligands_from_plan(plan)
    alignments = load_or_generate_alignments(workflow, plan)
    anchors = _node_anchors(plan, alignments)
    alchemy = workflow.get("alchemy") or {}
    settings = dict(alchemy.get("node_bank") or {})
    setup = dict(workflow.get("setup") or {})
    config = _normalized_settings(workflow)
    bank = Path(settings.get("path", plan["workdir"] / "node_bank"))
    if not bank.is_absolute():
        bank = (loaded["config_path"].parent / bank).resolve()
    fingerprint_settings = {
        key: value for key, value in settings.items() if key != "path"
    }
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "receptor_sha256": _sha256(plan["receptor_file"]),
        "setup": setup,
        "node_bank": fingerprint_settings,
        "sampling": {
            "temperature_k": config["temperature_k"],
            "pressure_bar": config["pressure_bar"],
            "timestep_fs": config["timestep_fs"],
            "endpoint_equilibration": config["endpoint_equilibration"],
            "equilibration_protocol": config["equilibration_protocol"],
            "decorrelation_steps": config["decorrelation_steps"],
            "rest2": config["rest2"],
            "random_seed": config["random_seed"],
        },
    }
    fingerprint = _fingerprint(fingerprint_payload)
    if str(setup.get("solvent_box_shape", "cube")) not in {
        "cube", "rectangular", "dodecahedron"
    }:
        raise NodeBankError(
            "canonical node banks support cube, rectangular, or dodecahedron boxes"
        )
    count = int(settings.get("snapshots", config["n_snapshots"]))
    if count < config["n_snapshots"]:
        raise NodeBankError("node bank must contain at least workflow.neqti.n_snapshots")
    extension_max_linear_scale = float(
        settings.get("extension_max_linear_scale", 1.05)
    )
    if extension_max_linear_scale < 1.0:
        raise NodeBankError("node_bank.extension_max_linear_scale must be at least 1")
    return {
        "loaded": loaded,
        "workflow": workflow,
        "plan": plan,
        "nodes": nodes,
        "anchors": anchors,
        "settings": settings,
        "setup": setup,
        "config": config,
        "bank": bank,
        "staging": bank.with_name(f".{bank.name}.building"),
        "fingerprint_payload": fingerprint_payload,
        "fingerprint": fingerprint,
        "snapshot_count": count,
        "extension_max_linear_scale": extension_max_linear_scale,
    }


def _existing_bank(context):
    manifest_path = context["bank"] / "manifest.yaml"
    if not manifest_path.exists():
        return None
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    version = int(manifest.get("schema_version", 1))
    if version not in READABLE_SCHEMA_VERSIONS:
        raise NodeBankError("existing node bank has an unsupported schema")
    if version == 1:
        return manifest
    if manifest.get("compatibility_fingerprint", manifest.get("fingerprint")) != context["fingerprint"]:
        raise NodeBankError(
            "existing node bank inputs differ from this workflow; use a new bank path"
        )
    return manifest


def _validate_staging_fingerprint(context):
    staging = context["staging"]
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / "fingerprint.yaml"
    if path.exists():
        staged = yaml.safe_load(path.read_text()) or {}
        if staged.get("fingerprint") != context["fingerprint"]:
            raise NodeBankError(
                f"staged node bank inputs differ: {staging}; move it aside or use a new path"
            )
    else:
        _write_yaml_atomic(path, {
            "fingerprint": context["fingerprint"],
            "inputs": context["fingerprint_payload"],
        })
    return staging


def initialize_node_bank(path):
    context = _node_bank_context(path)
    existing = _existing_bank(context)
    if existing is not None:
        return existing
    staging = _validate_staging_fingerprint(context)
    initialization_path = staging / "initialization.yaml"
    if initialization_path.exists():
        initialization = yaml.safe_load(initialization_path.read_text()) or {}
        if initialization.get("fingerprint") != context["fingerprint"]:
            raise NodeBankError("node-bank initialization fingerprint differs")
        return initialization

    setup = context["setup"]
    config = context["config"]
    nodes = context["nodes"]
    allow_stereo = bool(setup.get("allow_undefined_stereo", False))
    bundles = {
        name: parameterize_ligand(
            source,
            ligand_forcefield=setup.get("ligand_forcefield", "espaloma-0.3.2"),
            ligand_charge_model=setup.get("ligand_charge_model", "nn"),
            allow_undefined_stereo=allow_stereo,
        )
        for name, source in nodes.items()
    }
    charges = {round(float(bundle.charges_e.sum()), 6) for bundle in bundles.values()}
    if len(charges) != 1:
        raise NodeBankError("separated-topology node banks require equal ligand charges")
    canonical = {}
    for environment in ("complex", "solvent"):
        candidates = {}
        for index, (name, bundle) in enumerate(sorted(bundles.items())):
            candidates[name] = create_physical_ligand_environment(
                bundle,
                receptor=(
                    context["plan"]["receptor_file"]
                    if environment == "complex" else None
                ),
                setup=dict(setup),
                solvation_seed=(
                    config["random_seed"] + index
                    + (0 if environment == "complex" else 10000)
                ),
            )
        largest = max(
            candidates.values(),
            key=lambda item: _box_volume_nm3(_box_vectors_nm(item.topology)),
        )
        box_vectors = _box_vectors_nm(largest.topology)
        provisional = {}
        for index, (name, bundle) in enumerate(sorted(bundles.items())):
            local_setup = dict(setup)
            local_setup["_canonical_box_vectors_nm"] = box_vectors.tolist()
            provisional[name] = create_physical_ligand_environment(
                bundle,
                receptor=(
                    context["plan"]["receptor_file"]
                    if environment == "complex" else None
                ),
                setup=local_setup,
                solvation_seed=(
                    config["random_seed"] + index
                    + (0 if environment == "complex" else 10000)
                ),
            )
        target_waters = min(
            int(item.provenance["water_count"]) for item in provisional.values()
        )
        ion_signatures = {
            tuple(sorted(item.provenance["ion_counts"].items()))
            for item in provisional.values()
        }
        if len(ion_signatures) != 1:
            raise NodeBankError(
                f"canonical {environment} nodes generated different ion counts"
            )
        canonical[environment] = {
            "box_vectors_nm": box_vectors.tolist(),
            "box_volume_nm3": _box_volume_nm3(box_vectors),
            "water_count": target_waters,
            "ion_counts": dict(next(iter(ion_signatures))),
        }
    initialization = {
        "schema_version": SCHEMA_VERSION,
        "kind": "separated_topology_node_bank_initialization",
        "fingerprint": context["fingerprint"],
        "compatibility_fingerprint": context["fingerprint"],
        "fingerprint_inputs": context["fingerprint_payload"],
        "receptor": str(Path(context["plan"]["receptor_file"]).resolve()),
        "receptor_sha256": _sha256(context["plan"]["receptor_file"]),
        "canonical_environments": canonical,
        "snapshot_count": context["snapshot_count"],
        "node_order": sorted(nodes),
        "nodes": {
            name: {
                "ligand_file": str(nodes[name]),
                "ligand_sha256": _sha256(nodes[name]),
                "charge_e": float(bundle.charges_e.sum()),
                "anchors": list(context["anchors"][name]),
                "parameterization": bundle.provenance,
                "node_fingerprint": _fingerprint({
                    "compatibility_fingerprint": context["fingerprint"],
                    "node_id": name,
                    "ligand_sha256": _sha256(nodes[name]),
                    "anchors": list(context["anchors"][name]),
                    "charge_e": round(float(bundle.charges_e.sum()), 8),
                    "parameterization": bundle.provenance,
                }),
            }
            for name, bundle in sorted(bundles.items())
        },
    }
    _write_yaml_atomic(initialization_path, initialization)
    return initialization


def _load_initialization(context):
    path = context["staging"] / "initialization.yaml"
    if not path.exists():
        raise NodeBankError(
            "node bank is not initialized; run --initialize-node-bank first"
        )
    payload = yaml.safe_load(path.read_text()) or {}
    if payload.get("fingerprint") != context["fingerprint"]:
        raise NodeBankError("node-bank initialization fingerprint differs")
    return payload


def _validate_node_payload(staging, name, payload):
    node_dir = staging / "nodes" / name
    for environment in ("complex", "solvent"):
        environment_payload = payload["environments"][environment]
        directory = node_dir / environment
        _load_artifact(directory, environment_payload, "system")
        _load_artifact(directory, environment_payload, "topology")
        for snapshot in environment_payload["snapshots"]:
            path = staging / snapshot["file"]
            if not path.is_file() or _sha256(path) != snapshot["sha256"]:
                raise NodeBankError(f"node snapshot is missing or changed: {path}")
    for key in ("system", "topology"):
        artifact = payload["vacuum"][key]
        path = staging / artifact["file"]
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise NodeBankError(f"vacuum artifact is missing or changed: {path}")
    for snapshot in payload["vacuum"]["snapshots"]:
        path = staging / snapshot["file"]
        if not path.is_file() or _sha256(path) != snapshot["sha256"]:
            raise NodeBankError(f"vacuum snapshot is missing or changed: {path}")


def _prepare_node_payload(context, initialization, node, root, *, box_vectors=None):
    setup = context["setup"]
    config = context["config"]
    node_order = list(initialization.get("node_order", context["nodes"]))
    node_index = node_order.index(node) if node in node_order else len(node_order)
    bundle = parameterize_ligand(
        context["nodes"][node],
        ligand_forcefield=setup.get("ligand_forcefield", "espaloma-0.3.2"),
        ligand_charge_model=setup.get("ligand_charge_model", "nn"),
        allow_undefined_stereo=bool(setup.get("allow_undefined_stereo", False)),
    )
    expected = initialization["nodes"][node]
    if abs(float(bundle.charges_e.sum()) - float(expected["charge_e"])) > 1.0e-6:
        raise NodeBankError(f"parameterized charge changed for node {node}")
    platform, properties = _platform(context["workflow"])
    node_dir = root / "nodes" / node
    node_payload = {**expected, "environments": {}}
    for environment in ("complex", "solvent"):
        canonical = initialization["canonical_environments"][environment]
        vectors = canonical["box_vectors_nm"] if box_vectors is None else box_vectors[environment]
        local_setup = dict(setup)
        local_setup.update({
            "_canonical_box_vectors_nm": vectors,
            "_target_water_count": canonical["water_count"],
        })
        physical = create_physical_ligand_environment(
            bundle,
            receptor=context["plan"]["receptor_file"] if environment == "complex" else None,
            setup=local_setup,
            solvation_seed=config["random_seed"] + node_index + (0 if environment == "complex" else 10000),
        )
        if physical.provenance["ion_counts"] != canonical["ion_counts"]:
            raise NodeBankError(f"canonical {environment} ion counts changed for {node}")
        directory = node_dir / environment
        payload = _serialize_system(directory, physical)
        state_file = directory / "state.xml"
        selection = {"ligand_a": {
            "structure_file": str(context["nodes"][node]),
            "system_atom_indices": list(range(bundle.molecule.n_atoms)),
        }}
        _equilibrate_endpoint(
            physical.system, physical.positions, state_file,
            protocol=config["endpoint_equilibration"],
            temperature_k=config["temperature_k"], pressure_bar=config["pressure_bar"],
            platform=platform, properties=properties,
            seed=config["random_seed"] + node_index * 100000 + (0 if environment == "complex" else 10000),
            label=f"node {node} {environment}", topology=physical.topology,
            custom_steps=neqti_hybrid_endpoint_steps(config.get("equilibration_protocol"), environment),
            custom_output_dir=directory / "equilibration",
            selection_metadata=selection, selection_endpoint="a",
        )
        snapshots = []
        snapshots_dir = directory / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        for sample in range(context["snapshot_count"]):
            snapshot = snapshots_dir / f"{sample:04d}.xml"
            if not snapshot.exists():
                state, _ = _sample_endpoint(
                    physical.system, physical.topology, state_file,
                    tuple(range(bundle.molecule.n_atoms)), ensemble=node,
                    steps=config["decorrelation_steps"], rest2_config=config["rest2"],
                    output_dir=directory / "rest2", platform=platform, properties=properties,
                    temperature_k=config["temperature_k"], timestep_fs=config["timestep_fs"],
                    seed=config["random_seed"] + node_index * 100000 + sample * 1000,
                )
                _write_state(snapshot, state)
            snapshots.append(_write_snapshot(snapshot, _load_state(snapshot), root))
        payload["snapshots"] = snapshots
        node_payload["environments"][environment] = payload
    node_payload["vacuum"] = _vacuum_snapshots(
        node_dir, root, bundle, count=context["snapshot_count"],
        initial_steps=int(context["settings"].get("vacuum_initial_steps", 50000)),
        decorrelation_steps=int(context["settings"].get("vacuum_decorrelation_steps", 10000)),
        config=config, platform=platform, properties=properties,
        seed=config["random_seed"] + node_index * 100000 + 50000,
    )
    _validate_node_payload(root, node, node_payload)
    return node_payload


def prepare_node_bank_node(path, node):
    context = _node_bank_context(path)
    existing = _existing_bank(context)
    if existing is not None:
        if node not in existing["nodes"]:
            raise NodeBankError(f"node bank does not contain {node}")
        return existing["nodes"][node]
    staging = _validate_staging_fingerprint(context)
    initialization = _load_initialization(context)
    node = str(node)
    if node not in context["nodes"]:
        raise NodeBankError(
            f"unknown node {node}; choose one of {initialization['node_order']}"
        )
    node_dir = staging / "nodes" / node
    node_manifest = node_dir / "manifest.yaml"
    if node_manifest.exists():
        payload = yaml.safe_load(node_manifest.read_text()) or {}
        if payload.get("fingerprint") != context["fingerprint"]:
            raise NodeBankError(f"completed node {node} has a different fingerprint")
        _validate_node_payload(staging, node, payload["node"])
        return payload["node"]

    node_payload = _prepare_node_payload(context, initialization, node, staging)
    _write_yaml_atomic(node_manifest, {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": context["fingerprint"],
        "node_id": node,
        "node": node_payload,
    })
    return node_payload


def finalize_node_bank(path):
    context = _node_bank_context(path)
    existing = _existing_bank(context)
    if existing is not None:
        return existing
    staging = _validate_staging_fingerprint(context)
    initialization = _load_initialization(context)
    nodes = {}
    missing = []
    for name in initialization["node_order"]:
        path = staging / "nodes" / name / "manifest.yaml"
        if not path.exists():
            missing.append(name)
            continue
        shard = yaml.safe_load(path.read_text()) or {}
        if shard.get("fingerprint") != context["fingerprint"]:
            raise NodeBankError(f"node shard {name} has a different fingerprint")
        _validate_node_payload(staging, name, shard["node"])
        nodes[name] = shard["node"]
    if missing:
        raise NodeBankError(f"node bank is incomplete; missing shards: {missing}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "separated_topology_node_bank",
        "fingerprint": context["fingerprint"],
        "compatibility_fingerprint": context["fingerprint"],
        "fingerprint_inputs": context["fingerprint_payload"],
        "receptor": initialization["receptor"],
        "receptor_sha256": initialization["receptor_sha256"],
        "canonical_environments": initialization["canonical_environments"],
        "snapshot_count": initialization["snapshot_count"],
        "extensible": bool(context.get("settings", {}).get("extensible", True)),
        "extension_max_linear_scale": context.get("extension_max_linear_scale", 1.05),
        "nodes": nodes,
    }
    _write_yaml_atomic(staging / "manifest.yaml", manifest)
    os.replace(staging, context["bank"])
    return manifest


def _extension_box_vectors(context, manifest, node, bundle):
    selected = {}
    maximum = float(manifest.get(
        "extension_max_linear_scale", context["extension_max_linear_scale"]
    ))
    for environment in ("complex", "solvent"):
        canonical = manifest["canonical_environments"][environment]
        base = np.asarray(canonical["box_vectors_nm"], dtype=float)
        last_error = None
        for scale in np.linspace(1.0, maximum, 11):
            setup = dict(context["setup"])
            setup.update({
                "_canonical_box_vectors_nm": (base * scale).tolist(),
                "_target_water_count": canonical["water_count"],
            })
            try:
                physical = create_physical_ligand_environment(
                    bundle,
                    receptor=(
                        context["plan"]["receptor_file"]
                        if environment == "complex" else None
                    ),
                    setup=setup,
                    solvation_seed=(
                        context["config"]["random_seed"]
                        + len(manifest["nodes"])
                        + (0 if environment == "complex" else 10000)
                    ),
                )
            except HybridSystemError as exc:
                last_error = exc
                continue
            if physical.provenance["ion_counts"] != canonical["ion_counts"]:
                last_error = NodeBankError(
                    f"{environment} ion counts change when extending with {node}"
                )
                continue
            selected[environment] = (base * scale).tolist()
            break
        else:
            detail = "" if last_error is None else f": {last_error}"
            raise NodeBankError(
                f"node {node} does not fit the canonical {environment} box within "
                f"the {maximum:.3f} linear-scale cap{detail}; create a new bank"
            )
    return selected


def extend_node_bank(path, node):
    """Prepare and atomically publish one node into an existing schema-v2 bank."""
    context = _node_bank_context(path)
    node = str(node)
    if node not in context["nodes"]:
        raise NodeBankError(f"workflow does not define node {node}")
    lock_path = context["bank"].with_name(f".{context['bank'].name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest_path = context["bank"] / "manifest.yaml"
        if not manifest_path.is_file():
            raise NodeBankError("node bank must be finalized before it can be extended")
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
        if int(manifest.get("schema_version", 1)) != SCHEMA_VERSION:
            raise NodeBankError("schema-v1 node banks are readable but cannot be extended")
        if not manifest.get("extensible", False):
            raise NodeBankError("this node bank is not marked extensible")
        if manifest.get("compatibility_fingerprint") != context["fingerprint"]:
            raise NodeBankError("workflow settings are incompatible with this node bank")
        if node in manifest["nodes"]:
            return manifest["nodes"][node]

        setup = context["setup"]
        bundle = parameterize_ligand(
            context["nodes"][node],
            ligand_forcefield=setup.get("ligand_forcefield", "espaloma-0.3.2"),
            ligand_charge_model=setup.get("ligand_charge_model", "nn"),
            allow_undefined_stereo=bool(setup.get("allow_undefined_stereo", False)),
        )
        charge = float(bundle.charges_e.sum())
        existing_charges = {
            round(float(payload["charge_e"]), 6)
            for payload in manifest["nodes"].values()
        }
        if existing_charges and round(charge, 6) not in existing_charges:
            raise NodeBankError("new node charge differs from the existing bank")
        expected = {
            "ligand_file": str(context["nodes"][node]),
            "ligand_sha256": _sha256(context["nodes"][node]),
            "charge_e": charge,
            "anchors": list(context["anchors"][node]),
            "parameterization": bundle.provenance,
        }
        expected["node_fingerprint"] = _fingerprint({
            "compatibility_fingerprint": context["fingerprint"],
            "node_id": node,
            "ligand_sha256": expected["ligand_sha256"],
            "anchors": expected["anchors"],
            "charge_e": round(charge, 8),
            "parameterization": bundle.provenance,
        })
        initialization = {
            "canonical_environments": manifest["canonical_environments"],
            "node_order": [*manifest["nodes"], node],
            "nodes": {node: expected},
        }
        vectors = _extension_box_vectors(context, manifest, node, bundle)
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{context['bank'].name}.{node}.", dir=context["bank"].parent
        ))
        try:
            payload = _prepare_node_payload(
                context, initialization, node, temporary, box_vectors=vectors
            )
            _write_yaml_atomic(temporary / "nodes" / node / "manifest.yaml", {
                "schema_version": SCHEMA_VERSION,
                "compatibility_fingerprint": context["fingerprint"],
                "node_id": node,
                "node": payload,
            })
            destination = context["bank"] / "nodes" / node
            if destination.exists():
                raise NodeBankError(f"orphaned destination already exists: {destination}")
            os.replace(temporary / "nodes" / node, destination)
            manifest["nodes"][node] = payload
            _write_yaml_atomic(manifest_path, manifest)
            return payload
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


def prepare_node_bank(path):
    """Prepare all nodes serially; sharded callers should use the three phase API."""
    initialization = initialize_node_bank(path)
    if initialization.get("kind") == "separated_topology_node_bank":
        return initialization
    for node in initialization["node_order"]:
        prepare_node_bank_node(path, node)
    return finalize_node_bank(path)
