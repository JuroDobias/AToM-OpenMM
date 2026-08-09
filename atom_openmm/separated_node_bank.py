from __future__ import annotations

import hashlib
import os
from pathlib import Path

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
from atom_openmm.hybrid_systems import create_physical_ligand_environment


SCHEMA_VERSION = 1


class NodeBankError(ValueError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def prepare_node_bank(path):
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
    manifest_path = bank / "manifest.yaml"
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "receptor_sha256": _sha256(plan["receptor_file"]),
        "ligands": {name: _sha256(source) for name, source in sorted(nodes.items())},
        "anchors": {name: list(value) for name, value in sorted(anchors.items())},
        "setup": setup,
        "node_bank": settings,
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
    fingerprint = hashlib.sha256(
        yaml.safe_dump(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise NodeBankError("existing node bank has an unsupported schema")
        if manifest.get("fingerprint") != fingerprint:
            raise NodeBankError(
                "existing node bank inputs differ from this workflow; use a new bank path"
            )
        return manifest

    if str(setup.get("solvent_box_shape", "cube")) not in {"cube", "rectangular"}:
        raise NodeBankError(
            "canonical node banks currently support cube or rectangular solvent boxes"
        )
    count = int(settings.get("snapshots", config["n_snapshots"]))
    if count < config["n_snapshots"]:
        raise NodeBankError("node bank must contain at least workflow.neqti.n_snapshots")
    platform, properties = _platform(workflow)
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

    temporary = bank.with_name(f".{bank.name}.building")
    temporary.mkdir(parents=True, exist_ok=True)
    staging_fingerprint = temporary / "fingerprint.yaml"
    if staging_fingerprint.exists():
        staged = yaml.safe_load(staging_fingerprint.read_text()) or {}
        if staged.get("fingerprint") != fingerprint:
            raise NodeBankError(
                f"staged node bank inputs differ: {temporary}; move it aside or use a new path"
            )
    else:
        _write_yaml_atomic(staging_fingerprint, {
            "fingerprint": fingerprint,
            "inputs": fingerprint_payload,
        })
    try:
        prepared = {"complex": {}, "solvent": {}}
        canonical = {}
        for environment in ("complex", "solvent"):
            box = _canonical_box_size(
                bundles,
                plan["receptor_file"],
                float(setup.get("solvent_padding_a", 10.0)),
                environment,
            )
            provisional = {}
            for index, (name, bundle) in enumerate(sorted(bundles.items())):
                local_setup = dict(setup)
                local_setup["_canonical_box_size_nm"] = box
                provisional[name] = create_physical_ligand_environment(
                    bundle,
                    receptor=plan["receptor_file"] if environment == "complex" else None,
                    setup=local_setup,
                    solvation_seed=config["random_seed"] + index + (0 if environment == "complex" else 10000),
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
            for index, (name, bundle) in enumerate(sorted(bundles.items())):
                local_setup = dict(setup)
                local_setup.update({
                    "_canonical_box_size_nm": box,
                    "_target_water_count": target_waters,
                })
                prepared[environment][name] = create_physical_ligand_environment(
                    bundle,
                    receptor=plan["receptor_file"] if environment == "complex" else None,
                    setup=local_setup,
                    solvation_seed=config["random_seed"] + index + (0 if environment == "complex" else 10000),
                )
            canonical[environment] = {
                "box_size_nm": box,
                "water_count": target_waters,
                "ion_counts": dict(next(iter(ion_signatures))),
            }

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "separated_topology_node_bank",
            "fingerprint": fingerprint,
            "fingerprint_inputs": fingerprint_payload,
            "receptor": str(Path(plan["receptor_file"]).resolve()),
            "receptor_sha256": _sha256(plan["receptor_file"]),
            "canonical_environments": canonical,
            "snapshot_count": count,
            "nodes": {},
        }
        for node_index, (name, bundle) in enumerate(sorted(bundles.items())):
            node_dir = temporary / "nodes" / name
            node_payload = {
                "ligand_file": str(nodes[name]),
                "ligand_sha256": _sha256(nodes[name]),
                "charge_e": float(bundle.charges_e.sum()),
                "anchors": list(anchors[name]),
                "parameterization": bundle.provenance,
                "environments": {},
            }
            for environment in ("complex", "solvent"):
                physical = prepared[environment][name]
                directory = node_dir / environment
                payload = _serialize_system(directory, physical)
                state_file = directory / "state.xml"
                selection = {
                    "ligand_a": {
                        "structure_file": str(nodes[name]),
                        "system_atom_indices": list(range(bundle.molecule.n_atoms)),
                    }
                }
                custom_steps = neqti_hybrid_endpoint_steps(
                    config.get("equilibration_protocol"), environment
                )
                _equilibrate_endpoint(
                    physical.system,
                    physical.positions,
                    state_file,
                    protocol=config["endpoint_equilibration"],
                    temperature_k=config["temperature_k"],
                    pressure_bar=config["pressure_bar"],
                    platform=platform,
                    properties=properties,
                    seed=config["random_seed"] + node_index * 100000 + (0 if environment == "complex" else 10000),
                    label=f"node {name} {environment}",
                    topology=physical.topology,
                    custom_steps=custom_steps,
                    custom_output_dir=directory / "equilibration",
                    selection_metadata=selection,
                    selection_endpoint="a",
                )
                snapshots = []
                snapshots_dir = directory / "snapshots"
                snapshots_dir.mkdir(parents=True, exist_ok=True)
                for sample in range(count):
                    snapshot = snapshots_dir / f"{sample:04d}.xml"
                    if not snapshot.exists():
                        state, _ = _sample_endpoint(
                            physical.system,
                            physical.topology,
                            state_file,
                            tuple(range(bundle.molecule.n_atoms)),
                            ensemble=name,
                            steps=config["decorrelation_steps"],
                            rest2_config=config["rest2"],
                            output_dir=directory / "rest2",
                            platform=platform,
                            properties=properties,
                            temperature_k=config["temperature_k"],
                            timestep_fs=config["timestep_fs"],
                            seed=config["random_seed"] + node_index * 100000 + sample * 1000,
                        )
                        _write_state(snapshot, state)
                    snapshots.append(_write_snapshot(snapshot, _load_state(snapshot), temporary))
                payload["snapshots"] = snapshots
                node_payload["environments"][environment] = payload
            node_payload["vacuum"] = _vacuum_snapshots(
                node_dir,
                temporary,
                bundle,
                count=count,
                initial_steps=int(settings.get("vacuum_initial_steps", 50000)),
                decorrelation_steps=int(settings.get("vacuum_decorrelation_steps", 10000)),
                config=config,
                platform=platform,
                properties=properties,
                seed=config["random_seed"] + node_index * 100000 + 50000,
            )
            manifest["nodes"][name] = node_payload
            _write_yaml_atomic(temporary / "manifest.partial.yaml", manifest)
        _write_yaml_atomic(temporary / "manifest.yaml", manifest)
        (temporary / "manifest.partial.yaml").unlink(missing_ok=True)
        os.replace(temporary, bank)
        return manifest
    except Exception:
        # Keep the deterministic staging bank so a later invocation can resume.
        raise
