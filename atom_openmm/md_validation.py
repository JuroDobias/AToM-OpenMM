from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import openmm as mm
from openmm import app, unit
import yaml

from atom_openmm import hybrid_parameters, hybrid_systems, hu2024_atp, metal_ions, receptor_normalization
from atom_openmm.hu2024_atp import apply_hu2024_atp
from atom_openmm.hybrid_parameters import parameterize_ligand
from atom_openmm.hybrid_systems import create_physical_ligand_environment
from atom_openmm.metal_ions import apply_panteva_m1264
from atom_openmm.receptor_normalization import normalize_legacy_pdb


LOGGER = logging.getLogger("atom_openmm.md_validation")


class MDValidationError(ValueError):
    pass


VARIANTS = (
    ("legacy_atp_12_6", False, False),
    ("legacy_atp_panteva_m12_6_4", False, True),
    ("hu2024_atp_12_6", True, False),
    ("hu2024_atp_panteva_m12_6_4", True, True),
)


def _load_config(path):
    path = Path(path).resolve()
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise MDValidationError("validation YAML must contain a mapping")
    payload["_config_path"] = path
    payload["_base_dir"] = path.parent
    return payload


def _resolve(config, value):
    path = Path(value)
    return path if path.is_absolute() else config["_base_dir"] / path


def _workdir(config):
    return _resolve(config, config.get("workdir", "run")).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clone(system):
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))


def _write_yaml(path, payload):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False))
    temporary.replace(path)


def _truncate_metrics(path, maximum_step):
    if not path.is_file():
        return
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0]) if rows else []
    if not fields:
        return
    retained = [row for row in rows if int(row["step"]) <= maximum_step]
    if len(retained) == len(rows):
        return
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(retained)
    temporary.replace(path)


def _prepared_paths(config):
    root = _workdir(config) / "prepared"
    return {
        "root": root,
        "receptor": root / "receptor_normalized.pdb",
        "topology": root / "canonical_topology.cif",
        "system": root / "canonical_system.xml",
        "manifest": root / "manifest.yaml",
    }


def _fingerprint(config):
    source_paths = {
        "receptor": _resolve(config, config["receptor"]),
        "ligand": _resolve(config, config["ligand"]),
        "hu_prepi": _resolve(config, config["hu2024_atp"]["prepi"]),
        "hu_frcmod": _resolve(config, config["hu2024_atp"]["frcmod"]),
        "hu_cmap": _resolve(config, config["hu2024_atp"]["mod_py"]),
        "panteva_polarizabilities": _resolve(config, config["panteva_m1264"]["polarizability_table"]),
    }
    for index, value in enumerate(config.get("setup", {}).get("protein_forcefield", [])):
        candidate = _resolve(config, value)
        if candidate.is_file():
            source_paths[f"protein_forcefield_{index}"] = candidate
    record = {
        "configuration": {key: value for key, value in config.items() if not key.startswith("_")},
        "inputs": {key: _sha256(value) for key, value in source_paths.items()},
        "implementation": {
            **{
                Path(module.__file__).name: _sha256(module.__file__)
                for module in (hybrid_parameters, hybrid_systems, hu2024_atp, metal_ions, receptor_normalization)
            },
            Path(__file__).name: _sha256(__file__),
        },
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def _gaff2_classes(sdf, positions):
    from parmed import load_file

    with tempfile.TemporaryDirectory(prefix="atom-md-gaff2-") as directory:
        mol2 = Path(directory) / "typing.mol2"
        subprocess.run(
            ["antechamber", "-i", str(sdf), "-fi", "sdf", "-o", str(mol2),
             "-fo", "mol2", "-at", "gaff2", "-seq", "n", "-s", "0", "-pf", "y"],
            cwd=directory, check=True, capture_output=True, text=True,
        )
        typed = load_file(str(mol2))
        expected = np.asarray(positions.value_in_unit(unit.angstrom))
        observed = np.asarray(typed.coordinates)
        if observed.shape != expected.shape or np.max(np.abs(observed - expected)) > 0.02:
            raise MDValidationError("GAFF2 typing changed ligand atom order or coordinates")
        return [atom.type for atom in typed.atoms]


def prepare(config):
    paths = _prepared_paths(config)
    fingerprint = _fingerprint(config)
    if paths["manifest"].is_file():
        prior_manifest = yaml.safe_load(paths["manifest"].read_text())
        if prior_manifest.get("input_fingerprint") not in (None, fingerprint):
            tasks_directory = _workdir(config) / "tasks"
            if tasks_directory.exists() and any(tasks_directory.iterdir()):
                raise MDValidationError(
                    "validation inputs or settings changed after tasks started; use a new workdir"
                )
            LOGGER.warning("Canonical inputs changed before tasks started; rebuilding prepared system")
        try:
            topology, _, _, previous_manifest = _load_prepared(config)
            if len(previous_manifest.get("atom_classes", [])) != topology.getNumAtoms() or any(
                atom_class is None for atom_class in previous_manifest["atom_classes"]
            ):
                raise MDValidationError("canonical atom-class assignment is incomplete")
        except (MDValidationError, IndexError, ValueError, KeyError) as exc:
            LOGGER.warning("Existing prepared system cannot be read; rebuilding: %s", exc)
        else:
            LOGGER.info("Canonical system is already prepared: %s", paths["manifest"])
            return yaml.safe_load(paths["manifest"].read_text())
    paths["root"].mkdir(parents=True, exist_ok=True)
    receptor = _resolve(config, config["receptor"])
    normalization = normalize_legacy_pdb(receptor, paths["receptor"])
    ligand = _resolve(config, config["ligand"])
    setup = dict(config.get("setup") or {})
    for key in ("protein_forcefield", "solvent_forcefield"):
        values = setup.get(key)
        if values is None:
            continue
        scalar = isinstance(values, str)
        values = [values] if scalar else list(values)
        resolved = []
        for value in values:
            candidate = _resolve(config, value)
            resolved.append(str(candidate.resolve()) if candidate.is_file() else value)
        setup[key] = resolved[0] if scalar else resolved
    parameters = parameterize_ligand(
        ligand,
        ligand_forcefield=setup.get("ligand_forcefield", "espaloma-0.3.2"),
        ligand_charge_model=setup.get("ligand_charge_model", "nn"),
        allow_undefined_stereo=bool(setup.get("allow_undefined_stereo", False)),
    )
    physical = create_physical_ligand_environment(
        parameters,
        receptor=paths["receptor"],
        setup=setup,
        solvation_seed=int(config.get("random_seed", 20260914)),
        record_atom_classes=True,
    )
    atom_classes = list(physical.provenance.get("amber_atom_classes", []))
    ligand_classes = _gaff2_classes(ligand, physical.positions[:physical.solute_atom_count])
    if len(ligand_classes) != physical.solute_atom_count:
        raise MDValidationError("GAFF2 typing atom count differs from physical ligand")
    atom_classes[:physical.solute_atom_count] = ligand_classes
    with paths["topology"].open("w") as handle:
        app.PDBxFile.writeFile(physical.topology, physical.positions, handle, keepIds=False)
    loaded = app.PDBxFile(str(paths["topology"]))
    if loaded.topology.getNumAtoms() != physical.system.getNumParticles():
        raise MDValidationError("serialized mmCIF does not match canonical system")
    paths["system"].write_text(mm.XmlSerializer.serialize(physical.system))
    residues = {}
    for residue in physical.topology.residues():
        residues[residue.name] = residues.get(residue.name, 0) + 1
    checks = config.get("expected") or {}
    observed = {
        "magnesium": residues.get("MG", 0), "zinc": residues.get("ZN", 0),
        "atp": residues.get("ATP", 0),
        "dna_nucleotides": sum(
            count for name, count in residues.items() if name in {"DA", "DT", "DC", "DG", "DA5", "DT5", "DC5", "DG5", "DA3", "DT3", "DC3", "DG3"}
        ),
        "retained_water_extra_particles": normalization["removed_extra_particles"],
    }
    for key, expected in checks.items():
        if observed.get(key) != int(expected):
            raise MDValidationError(
                f"canonical cGAS composition differs: expected {key}={expected}, observed {observed.get(key)}"
            )
    manifest = {
        "schema_version": 1,
        "input_fingerprint": fingerprint,
        "topology": paths["topology"].name,
        "system": paths["system"].name,
        "particle_count": physical.system.getNumParticles(),
        "residue_counts": residues,
        "composition": observed,
        "atom_classes": atom_classes,
        "normalization": normalization,
        "ligand": str(ligand.resolve()),
        "checksums": {
            "topology": _sha256(paths["topology"]),
            "system": _sha256(paths["system"]),
        },
    }
    _write_yaml(paths["manifest"], manifest)
    LOGGER.info(
        "Prepared canonical cGAS system: %d particles, %d waters",
        physical.system.getNumParticles(), residues.get("HOH", 0) + residues.get("WAT", 0),
    )
    return manifest


def _load_prepared(config):
    paths = _prepared_paths(config)
    if not paths["manifest"].is_file():
        raise MDValidationError("canonical system is not prepared; run --prepare-only first")
    manifest = yaml.safe_load(paths["manifest"].read_text())
    fingerprint = _fingerprint(config)
    if manifest.get("input_fingerprint") != fingerprint:
        raise MDValidationError(
            "prepared system does not match validation inputs or settings; use a new workdir"
        )
    if _sha256(paths["topology"]) != manifest["checksums"]["topology"]:
        raise MDValidationError("canonical topology checksum changed")
    if _sha256(paths["system"]) != manifest["checksums"]["system"]:
        raise MDValidationError("canonical system checksum changed")
    topology_file = app.PDBxFile(str(paths["topology"]))
    system = mm.XmlSerializer.deserialize(paths["system"].read_text())
    return topology_file.topology, topology_file.positions, system, manifest


def task_spec(config, task_index):
    replicates = int(config.get("replicates", 3))
    count = len(VARIANTS) * replicates
    if task_index < 0 or task_index >= count:
        raise MDValidationError(f"task index must be between 0 and {count - 1}")
    variant_index, replicate_index = divmod(task_index, replicates)
    name, hu, panteva = VARIANTS[variant_index]
    return {
        "task_index": task_index, "variant": name, "replicate": replicate_index + 1,
        "hu2024_atp": hu, "panteva_m12_6_4": panteva,
    }


def _platform(config):
    name = str(config.get("platform", "CUDA"))
    platform = mm.Platform.getPlatformByName(name)
    properties = dict(config.get("platform_properties") or {})
    if name == "CUDA":
        properties.setdefault("Precision", "mixed")
    return platform, properties


def _restraint_indices(topology, selection):
    solvent = {"HOH", "WAT", "TIP3", "TIP4", "OPC", "NA", "CL", "K", "CA"}
    ligand_residue = next(topology.residues())
    amino = set("ALA ARG ASN ASP CYS GLU GLN GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
    indices = []
    for atom in topology.atoms():
        name = atom.residue.name.upper()
        if atom.element is None or atom.element.symbol == "H" or name in solvent:
            continue
        if selection == "solute_heavy":
            include = True
        elif selection == "environment_heavy":
            include = atom.residue != ligand_residue
        elif selection == "protein_dna_heavy":
            include = name in amino or name.startswith("D")
        else:
            raise MDValidationError(f"unknown positional restraint selection: {selection}")
        if include:
            indices.append(atom.index)
    return indices


def _add_restraints(system, topology, positions, strength, selection="solute_heavy"):
    force = mm.CustomExternalForce(
        "0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2"
    )
    force.setName("MD validation solute-heavy positional restraints")
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    force.addGlobalParameter(
        "k", float(strength) * 418.4 * unit.kilojoule_per_mole / unit.nanometer**2
    )
    for index in _restraint_indices(topology, selection):
        xyz = positions[index].value_in_unit(unit.nanometer)
        force.addParticle(index, [float(value) for value in xyz])
    system.addForce(force)


def _add_ligand_metal_restraint(system, topology, positions, settings, strength):
    if not settings or strength is None or float(strength) <= 0:
        return None
    ligand_residue = next(topology.residues())
    ligand_atoms = list(ligand_residue.atoms())
    ligand_index = int(settings["ligand_atom_index_0based"])
    if ligand_index < 0 or ligand_index >= len(ligand_atoms):
        raise MDValidationError("ligand metal restraint atom index is outside the ligand")
    ligand_atom = ligand_atoms[ligand_index]
    metals = [
        atom for atom in topology.atoms()
        if atom.element is not None and atom.element.symbol == str(settings.get("metal", "Mg"))
    ]
    if not metals:
        raise MDValidationError("ligand metal restraint did not find the requested metal")
    coordinates = np.asarray(positions.value_in_unit(unit.nanometer))
    metal_atom = min(
        metals, key=lambda atom: np.linalg.norm(coordinates[atom.index] - coordinates[ligand_atom.index])
    )
    lower = float(settings.get("lower_bound_a", 1.8)) / 10.0
    upper = float(settings.get("upper_bound_a", 3.0)) / 10.0
    if lower < 0 or upper <= lower:
        raise MDValidationError("ligand metal restraint bounds must satisfy 0 <= lower < upper")
    force = mm.CustomBondForce(
        "0.5*k*(max(0, r-upper)^2 + max(0, lower-r)^2)"
    )
    force.setName("MD validation ligand-metal flat-bottom restraint")
    force.addGlobalParameter(
        "k", float(strength) * 418.4 * unit.kilojoule_per_mole / unit.nanometer**2
    )
    force.addGlobalParameter("lower", lower * unit.nanometer)
    force.addGlobalParameter("upper", upper * unit.nanometer)
    force.addBond(ligand_atom.index, metal_atom.index)
    force.setUsesPeriodicBoundaryConditions(True)
    system.addForce(force)
    return {
        "ligand_atom_index": ligand_atom.index,
        "ligand_atom_name": ligand_atom.name,
        "metal_atom_index": metal_atom.index,
        "metal_atom_name": metal_atom.name,
    }


def _set_barostat(system, pressure_bar, temperature_k, enabled):
    for index in reversed(range(system.getNumForces())):
        if isinstance(system.getForce(index), mm.MonteCarloBarostat):
            system.removeForce(index)
    if enabled:
        system.addForce(mm.MonteCarloBarostat(
            pressure_bar * unit.bar, temperature_k * unit.kelvin, 25
        ))


def _phase_protocol(config):
    protocol = config.get("protocol") or {}
    if "steps" in protocol:
        phases = protocol["steps"]
        if not isinstance(phases, list) or not phases:
            raise MDValidationError("protocol.steps must be a non-empty list")
        normalized = []
        seen = set()
        for raw in phases:
            if not isinstance(raw, dict):
                raise MDValidationError("each protocol step must be a mapping")
            phase = dict(raw)
            identifier = str(phase.get("id", "")).strip()
            kind = str(phase.get("kind", "")).strip()
            if not identifier or identifier in seen:
                raise MDValidationError("protocol step ids must be non-empty and unique")
            if kind not in {"min", "md"}:
                raise MDValidationError(f"protocol step {identifier} kind must be min or md")
            seen.add(identifier)
            phase["id"] = identifier
            phase["kind"] = kind
            if kind == "min":
                phase["max_iterations"] = int(phase.get("max_iterations", 500))
            else:
                phase["steps"] = int(phase.get("steps", 0))
                if phase["steps"] < 1:
                    raise MDValidationError(f"protocol step {identifier} requires positive steps")
            normalized.append(phase)
        return normalized
    return [
        {"id": "minimized", "kind": "min", "restrained": True,
         "max_iterations": int(protocol.get("minimization_max_iterations", 500))},
        {"id": "restrained_nvt", "kind": "md", "npt": False, "restrained": True,
         "reset_velocities": True,
         "steps": int(protocol.get("restrained_nvt_steps", 50000))},
        {"id": "restrained_npt", "kind": "md", "npt": True, "restrained": True,
         "steps": int(protocol.get("restrained_npt_steps", 250000))},
        {"id": "unrestrained_npt", "kind": "md", "npt": True, "restrained": False,
         "steps": int(protocol.get("unrestrained_npt_steps", 250000))},
        {"id": "production", "kind": "md", "npt": True, "restrained": False,
         "steps": int(protocol.get("production_steps", 10000000)), "production": True},
    ]


def _distance_nm(position_a, position_b, vectors):
    delta = position_b - position_a
    if vectors is not None:
        inverse = np.linalg.inv(vectors)
        fractional = delta @ inverse
        delta = (fractional - np.rint(fractional)) @ vectors
    return float(np.linalg.norm(delta))


def _metric_context(topology, initial_positions):
    atoms = list(topology.atoms())
    initial = np.asarray(initial_positions.value_in_unit(unit.nanometer))
    mg = [atom.index for atom in atoms if atom.element and atom.element.symbol == "Mg"]
    donors = [atom.index for atom in atoms if atom.element and atom.element.symbol in {"O", "N"}]
    initial_donors = {
        metal: [index for index in donors if _distance_nm(initial[metal], initial[index], None) < 0.30]
        for metal in mg
    }
    groups = {
        "atp": [a.index for a in atoms if a.residue.name == "ATP" and a.element and a.element.symbol != "H"],
        "ligand": [a.index for a in atoms if a.residue.index == 0 and a.element and a.element.symbol != "H"],
        "dna": [a.index for a in atoms if a.residue.name.startswith("D") and a.element and a.element.symbol != "H"],
    }
    amino = set("ALA ARG ASN ASP CYS GLU GLN GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
    protein = [a.index for a in atoms if a.residue.name in amino and a.element and a.element.symbol != "H"]
    if not protein:
        raise MDValidationError("cannot compute protein-aligned metrics without a protein")
    atp_center = initial[groups["atp"]].mean(axis=0)
    site = [i for i in protein if np.linalg.norm(initial[i] - atp_center) < 0.9]
    return {
        "mg": mg, "donors": np.asarray(donors), "initial_donors": initial_donors,
        "groups": groups, "protein": protein, "site": site,
        "initial": initial, "atoms": atoms,
        "zn": [a.index for a in atoms if a.element and a.element.symbol == "Zn"],
    }


def _dihedral_deg(coordinates):
    p0, p1, p2, p3 = coordinates
    axis = p2 - p1
    axis /= np.linalg.norm(axis)
    first = p0 - p1
    last = p3 - p2
    first -= np.dot(first, axis) * axis
    last -= np.dot(last, axis) * axis
    return float(np.degrees(np.arctan2(np.dot(np.cross(axis, first), last), np.dot(first, last))))


def _metrics_row(state, metric_context, step, time_ps):
    positions = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    vectors = np.asarray(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
    row = {"step": step, "time_ps": time_ps}
    inverse = np.linalg.inv(vectors)
    donors = metric_context["donors"]
    for ordinal, metal in enumerate(metric_context["mg"], 1):
        delta = positions[donors] - positions[metal]
        delta -= np.rint(delta @ inverse) @ vectors
        distances = np.linalg.norm(delta, axis=1)
        neighbors = donors[distances < 0.25]
        row[f"mg{ordinal}_coordination_0.25nm"] = len(neighbors)
        for label, names in {
            "atp": {"ATP"}, "water": {"HOH", "WAT"},
            "ligand": {"UNK"},
        }.items():
            row[f"mg{ordinal}_{label}_donors_0.25nm"] = sum(
                metric_context["atoms"][index].residue.name in names for index in neighbors
            )
        initial = metric_context["initial_donors"][metal]
        row[f"mg{ordinal}_reference_donor_mean_nm"] = (
            float(np.mean(distances[np.isin(donors, initial)]))
            if initial else math.nan
        )
    if len(metric_context["mg"]) == 2:
        row["mg_mg_nm"] = _distance_nm(
            positions[metric_context["mg"][0]], positions[metric_context["mg"][1]], vectors
        )
    reference = metric_context["initial"]
    protein = metric_context["protein"]
    current_protein_center = positions[protein].mean(axis=0)
    ref_protein_center = reference[protein].mean(axis=0)
    x = positions[protein] - current_protein_center
    y = reference[protein] - ref_protein_center
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.diag([1, 1, np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    row["protein_heavy_rmsd_nm"] = float(np.sqrt(np.mean(np.sum((x @ rotation - y) ** 2, axis=1))))
    for name, indices in {**metric_context["groups"], "active_site": metric_context["site"]}.items():
        if not indices:
            continue
        current = positions[indices] - current_protein_center
        current -= np.rint(current @ inverse) @ vectors
        initial = reference[indices] - ref_protein_center
        initial -= np.rint(initial @ inverse) @ vectors
        row[f"{name}_protein_aligned_rmsd_nm"] = float(
            np.sqrt(np.mean(np.sum((current @ rotation - initial) ** 2, axis=1)))
        )
    sulfur = [a.index for a in metric_context["atoms"] if a.element and a.element.symbol == "S"]
    for ordinal, zinc in enumerate(metric_context["zn"], 1):
        index = np.concatenate((metric_context["donors"], sulfur)).astype(int)
        delta = positions[index] - positions[zinc]
        delta -= np.rint(delta @ inverse) @ vectors
        row[f"zn{ordinal}_coordination_0.25nm"] = int(np.sum(np.linalg.norm(delta, axis=1) < 0.25))
    atp_atoms = {metric_context["atoms"][index].name: index for index in metric_context["groups"]["atp"]}
    for label, names in {
        "atp_beta_alpha": ("O3B", "PB", "O3A", "PA"),
        "atp_alpha_ribose": ("PB", "O3A", "PA", "O5*"),
    }.items():
        if all(name in atp_atoms for name in names):
            row[f"{label}_torsion_deg"] = _dihedral_deg(
                positions[[atp_atoms[name] for name in names]].copy()
            )
    return row


def _run_md_phase(config, topology, base_system, positions, state, phase, output, seed, metric_context):
    system = _clone(base_system)
    temperature = float(config.get("temperature_k", 300.0))
    pressure = float(config.get("pressure_bar", 1.0))
    restraint_strength = phase.get("restraint_k_kcal_mol_a2")
    if restraint_strength is None and phase.get("restrained"):
        restraint_strength = float(config.get("restraint_k_kcal_mol_a2", 5.0))
    if restraint_strength is not None and float(restraint_strength) > 0:
        _add_restraints(
            system, topology, positions, float(restraint_strength),
            str(phase.get("restraint_selection", "solute_heavy")),
        )
    _add_ligand_metal_restraint(
        system, topology, positions,
        (config.get("protocol") or {}).get("ligand_metal_restraint"),
        phase.get("ligand_metal_restraint_k_kcal_mol_a2"),
    )
    _set_barostat(system, pressure, temperature, bool(phase.get("npt")))
    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin, float(config.get("friction_per_ps", 1.0)) / unit.picosecond,
        float(phase.get("timestep_fs", config.get("timestep_fs", 2.0))) * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(seed)
    platform, properties = _platform(config)
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)
    progress_path = output / f"{phase['id']}_progress.yaml"
    running_state_path = output / f"{phase['id']}_running_state.xml"
    completed = 0
    report_interval = int(config.get("report_interval_steps", 5000))
    checkpoint_interval = int(config.get("checkpoint_interval_steps", report_interval))
    if report_interval < 1 or checkpoint_interval < 1 or checkpoint_interval % report_interval:
        raise MDValidationError("checkpoint_interval_steps must be a positive multiple of report_interval_steps")
    if running_state_path.is_file() and progress_path.is_file():
        state = mm.XmlSerializer.deserialize(running_state_path.read_text())
        completed = int(yaml.safe_load(progress_path.read_text()).get("completed_steps", 0))
        LOGGER.info("Resuming phase %s at step %d/%d", phase["id"], completed, phase.get("steps", 0))
    resumed = completed > 0
    if state is None:
        simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin, seed)
    else:
        simulation.context.setState(state)
    if phase.get("reset_velocities") and not resumed:
        simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin, seed)
    simulation.currentStep = completed
    if phase["kind"] == "min":
        simulation.minimizeEnergy(
            tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=phase["max_iterations"],
        )
    else:
        simulation.reporters.append(app.StateDataReporter(
            sys.stdout, report_interval, step=True, potentialEnergy=True,
            temperature=True, volume=bool(phase.get("npt")), speed=True,
            totalSteps=phase["steps"], remainingTime=True,
        ))
        if phase.get("production"):
            trajectory = (
                output / "trajectory.dcd" if checkpoint_interval == report_interval
                else output / f"trajectory_from_{completed:09d}.dcd"
            )
            simulation.reporters.append(app.DCDReporter(
                str(trajectory), report_interval,
                append=trajectory.is_file() and completed > 0
                and checkpoint_interval == report_interval,
            ))
        metrics_path = output / "metrics.csv"
        if phase.get("production"):
            _truncate_metrics(metrics_path, completed)
        fields = None
        if metrics_path.is_file() and metrics_path.stat().st_size:
            with metrics_path.open(newline="") as handle:
                fields = next(csv.reader(handle))
        start = time.perf_counter()
        while completed < phase["steps"]:
            block = min(report_interval, phase["steps"] - completed)
            simulation.step(block)
            completed += block
            if phase.get("production"):
                current = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
                row = _metrics_row(
                    current, metric_context, completed,
                    completed * float(phase.get("timestep_fs", config.get("timestep_fs", 2.0))) / 1000.0,
                )
                if fields is None:
                    fields = list(row)
                    with metrics_path.open("w", newline="") as handle:
                        csv.DictWriter(handle, fields).writeheader()
                with metrics_path.open("a", newline="") as handle:
                    csv.DictWriter(handle, fields).writerow(row)
            if completed % checkpoint_interval == 0 or completed == phase["steps"]:
                checkpoint_state = simulation.context.getState(
                    getPositions=True, getVelocities=True, getEnergy=True,
                    enforcePeriodicBox=True,
                )
                temporary_state_path = Path(str(running_state_path) + ".tmp")
                temporary_state_path.write_text(mm.XmlSerializer.serialize(checkpoint_state))
                temporary_state_path.replace(running_state_path)
                _write_yaml(progress_path, {"completed_steps": completed, "target_steps": phase["steps"]})
                if phase.get("production") and checkpoint_interval > report_interval:
                    segment_manifest = output / "trajectory_segments.yaml"
                    segment_payload = (
                        yaml.safe_load(segment_manifest.read_text())
                        if segment_manifest.is_file() else {"segments": []}
                    )
                    entry = next(
                        (item for item in segment_payload["segments"] if item["file"] == trajectory.name), None
                    )
                    if entry is None:
                        entry = {"file": trajectory.name, "start_step": int(trajectory.stem.rsplit("_", 1)[1])}
                        segment_payload["segments"].append(entry)
                    entry["last_committed_step"] = completed
                    _write_yaml(segment_manifest, segment_payload)
        LOGGER.info("Phase %s completed in %.1f s", phase["id"], time.perf_counter() - start)
    final_state = simulation.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True,
        enforcePeriodicBox=True,
    )
    return final_state


def run_task(config, task_index):
    topology, positions, base_system, manifest = _load_prepared(config)
    spec = task_spec(config, task_index)
    output = _workdir(config) / "tasks" / f"{task_index:02d}_{spec['variant']}_rep{spec['replicate']}"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.yaml"
    task_manifest = output / "task.yaml"
    previous_task = yaml.safe_load(task_manifest.read_text()) if task_manifest.is_file() else None
    if previous_task is not None and (
        previous_task.get("input_fingerprint") != manifest["input_fingerprint"]
        or previous_task.get("canonical_system_sha256") != manifest["checksums"]["system"]
    ):
        raise MDValidationError("task input or canonical system changed; use a new workdir")
    if previous_task is None:
        _write_yaml(task_manifest, {
            **spec, "input_fingerprint": manifest["input_fingerprint"],
            "canonical_system_sha256": manifest["checksums"]["system"],
        })
    if result_path.is_file() and yaml.safe_load(result_path.read_text()).get("status") == "completed":
        LOGGER.info("Task already complete: %s", output)
        return
    system = _clone(base_system)
    overlays = []
    if spec["panteva_m12_6_4"]:
        overlays.append(apply_panteva_m1264(
            system, topology, atom_classes=manifest.get("atom_classes"),
            polarizability_table=_resolve(config, config["panteva_m1264"]["polarizability_table"]),
            water_model=(config.get("setup") or {}).get("solvent_model", "tip3p"),
        ))
    if spec["hu2024_atp"]:
        hu = config.get("hu2024_atp") or {}
        overlays.append(apply_hu2024_atp(
            system, topology,
            prepi=_resolve(config, hu["prepi"]), frcmod=_resolve(config, hu["frcmod"]),
            mod_py=_resolve(config, hu["mod_py"]),
            atom_classes=manifest.get("atom_classes"),
        ))
    if spec["hu2024_atp"] and spec["panteva_m12_6_4"]:
        overlays.append({"warning": "experimental combination of independently published ATP and Mg models"})
    metric_context = _metric_context(topology, positions)
    seed = int(config.get("random_seed", 20260914)) + task_index * 1009
    state = None
    completed = []
    for phase in _phase_protocol(config):
        state_path = output / f"{phase['id']}_state.xml"
        if state_path.is_file():
            state = mm.XmlSerializer.deserialize(state_path.read_text())
            completed.append(phase["id"])
            LOGGER.info("Resuming after completed phase %s", phase["id"])
            continue
        LOGGER.info("Starting phase %s for %s replicate %d", phase["id"], spec["variant"], spec["replicate"])
        state = _run_md_phase(
            config, topology, system, positions, state, phase, output, seed, metric_context
        )
        temporary_state_path = Path(str(state_path) + ".tmp")
        temporary_state_path.write_text(mm.XmlSerializer.serialize(state))
        temporary_state_path.replace(state_path)
        for suffix in ("_running_state.xml", "_progress.yaml"):
            running = output / f"{phase['id']}{suffix}"
            if running.exists():
                running.unlink()
        completed.append(phase["id"])
    with (output / "final.cif").open("w") as handle:
        app.PDBxFile.writeFile(topology, state.getPositions(), handle, keepIds=False)
    _write_yaml(result_path, {
        "schema_version": 1, "status": "completed", **spec,
        "overlays": overlays, "completed_phases": completed,
        "canonical_system_sha256": manifest["checksums"]["system"],
        "metrics_csv": "metrics.csv",
        "trajectory": (
            "trajectory.dcd" if int(config.get("checkpoint_interval_steps", config.get("report_interval_steps", 5000)))
            == int(config.get("report_interval_steps", 5000)) else "trajectory_segments.yaml"
        ),
    })


def analyze(config):
    rows = []
    replicates = int(config.get("replicates", 3))
    for index in range(len(VARIANTS) * replicates):
        spec = task_spec(config, index)
        directory = _workdir(config) / "tasks" / f"{index:02d}_{spec['variant']}_rep{spec['replicate']}"
        metrics = directory / "metrics.csv"
        if not metrics.is_file():
            continue
        with metrics.open(newline="") as handle:
            values = list(csv.DictReader(handle))
        summary = {**spec, "frames": len(values), "status": (
            yaml.safe_load((directory / "result.yaml").read_text()).get("status")
            if (directory / "result.yaml").is_file() else "partial"
        )}
        if values:
            for key in values[0]:
                if key in {"step", "time_ps"}:
                    continue
                numeric = np.asarray([float(row[key]) for row in values], dtype=float)
                recent = numeric[len(numeric) // 2:]
                if key.endswith("_torsion_deg"):
                    angles = np.radians(recent)
                    summary[f"{key}_recent_circular_mean_deg"] = float(
                        np.degrees(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
                    )
                else:
                    summary[f"{key}_recent_mean"] = float(np.nanmean(recent))
                    summary[f"{key}_recent_max"] = float(np.nanmax(recent))
        rows.append(summary)
    variants = []
    for name, _, _ in VARIANTS:
        subset = [row for row in rows if row["variant"] == name and row["status"] == "completed"]
        record = {"variant": name, "completed_replicates": len(subset)}
        if subset:
            for key in subset[0]:
                if not key.endswith("_recent_mean"):
                    continue
                numbers = [row[key] for row in subset]
                record[key] = float(np.mean(numbers))
                record[key.replace("_recent_mean", "_between_replica_sd")] = (
                    float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
                )
        variants.append(record)
    payload = {"schema_version": 1, "variants": variants, "tasks": rows}
    _write_yaml(_workdir(config) / "analysis.yaml", payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run reproducible molecular-dynamics validation matrices")
    parser.add_argument("config")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--task", type=int)
    group.add_argument("--analyze", action="store_true")
    group.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    config = _load_config(args.config)
    if args.prepare_only:
        prepare(config)
    elif args.task is not None:
        run_task(config, args.task)
    elif args.analyze:
        print(yaml.safe_dump(analyze(config), sort_keys=False))
    else:
        for index in range(len(VARIANTS) * int(config.get("replicates", 3))):
            print(yaml.safe_dump(task_spec(config, index), sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
