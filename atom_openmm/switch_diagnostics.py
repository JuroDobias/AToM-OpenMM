"""Geometry and energy diagnostics for segmented nonequilibrium switches."""

from __future__ import annotations

import csv
import gzip
import math
import os
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit
import yaml


FRAME_FIELDS = (
    "frame", "step", "time_ps", "segment", "stage", "protocol_fraction",
    "physical_lambda", "potential_energy_kj_per_mol",
    "window_work_kj_per_mol", "cumulative_work_kj_per_mol",
)
GEOMETRY_FIELDS = (
    "frame", "step", "term_id", "endpoint_parameter_set", "term_type",
    "term_class", "geometry", "geometry_unit", "endpoint_form_energy_kj_per_mol",
)


def _float(value, target):
    return float(value.value_in_unit(target))


def _topology_bonds(topology):
    return {
        tuple(sorted((bond[0].index, bond[1].index)))
        for bond in topology.bonds()
    }


def build_bonded_manifest(prepared, mapping=None):
    """Describe endpoint bonded terms that touch any endpoint-unique particle."""
    unique_a = set(map(int, prepared.provenance["unique_a_particle_indices"]))
    unique_b = set(map(int, prepared.provenance["unique_b_particle_indices"]))
    unique = unique_a | unique_b
    topology_bonds = _topology_bonds(prepared.topology)
    soft_pairs = {
        tuple(sorted(map(int, pair)))
        for pair in prepared.provenance.get("soft_bond_system_pairs", [])
    }
    atoms = list(prepared.topology.atoms())
    terms = []

    def classification(indices, term_type):
        touched_a = bool(set(indices) & unique_a)
        touched_b = bool(set(indices) & unique_b)
        if term_type == "bond" and tuple(sorted(indices)) in soft_pairs:
            return "alchemical_bond"
        if (touched_a or touched_b) and not set(indices) <= unique:
            return "junction"
        if touched_a and touched_b:
            return "cross_branch"
        return "internal_unique"

    for endpoint in ("a", "b"):
        system = getattr(prepared, f"endpoint_{endpoint}")
        counters = {"bond": 0, "angle": 0, "proper": 0, "improper": 0}
        for force in system.getForces():
            if isinstance(force, mm.HarmonicBondForce):
                for index in range(force.getNumBonds()):
                    atom1, atom2, length, k = force.getBondParameters(index)
                    indices = (int(atom1), int(atom2))
                    if not set(indices) & unique:
                        continue
                    number = counters["bond"]
                    counters["bond"] += 1
                    terms.append({
                        "term_id": f"{endpoint}_bond_{number:04d}",
                        "endpoint_parameter_set": endpoint,
                        "term_type": "bond",
                        "term_class": classification(indices, "bond"),
                        "atoms_0based": list(indices),
                        "equilibrium_nm": _float(length, unit.nanometer),
                        "k_kj_mol_nm2": _float(
                            k, unit.kilojoule_per_mole / unit.nanometer**2
                        ),
                    })
            elif isinstance(force, mm.HarmonicAngleForce):
                for index in range(force.getNumAngles()):
                    atom1, atom2, atom3, theta, k = force.getAngleParameters(index)
                    indices = tuple(map(int, (atom1, atom2, atom3)))
                    if not set(indices) & unique:
                        continue
                    number = counters["angle"]
                    counters["angle"] += 1
                    terms.append({
                        "term_id": f"{endpoint}_angle_{number:04d}",
                        "endpoint_parameter_set": endpoint,
                        "term_type": "angle",
                        "term_class": classification(indices, "angle"),
                        "atoms_0based": list(indices),
                        "equilibrium_radians": _float(theta, unit.radian),
                        "k_kj_mol_rad2": _float(
                            k, unit.kilojoule_per_mole / unit.radian**2
                        ),
                    })
            elif isinstance(force, mm.PeriodicTorsionForce):
                for index in range(force.getNumTorsions()):
                    atom1, atom2, atom3, atom4, periodicity, phase, k = (
                        force.getTorsionParameters(index)
                    )
                    indices = tuple(map(int, (atom1, atom2, atom3, atom4)))
                    if not set(indices) & unique:
                        continue
                    kind = (
                        "proper"
                        if all(
                            tuple(sorted((indices[position], indices[position + 1])))
                            in topology_bonds
                            for position in range(3)
                        )
                        else "improper"
                    )
                    number = counters[kind]
                    counters[kind] += 1
                    terms.append({
                        "term_id": f"{endpoint}_{kind}_{number:04d}",
                        "endpoint_parameter_set": endpoint,
                        "term_type": kind,
                        "term_class": classification(indices, kind),
                        "atoms_0based": list(indices),
                        "periodicity": int(periodicity),
                        "phase_radians": _float(phase, unit.radian),
                        "k_kj_mol": _float(k, unit.kilojoule_per_mole),
                    })
    atom_rows = [
        {
            "system_atom_0based": atom.index,
            "name": atom.name,
            "element": None if atom.element is None else atom.element.symbol,
            "residue": atom.residue.name,
            "unique_a": atom.index in unique_a,
            "unique_b": atom.index in unique_b,
        }
        for atom in atoms[: int(prepared.provenance["hybrid_solute_atom_count"])]
    ]
    ligand_maps = {
        endpoint: list(map(int, prepared.provenance[f"ligand_{endpoint}_system_atom_indices"]))
        for endpoint in ("a", "b")
    }
    z_matrix_roots_local = {
        endpoint: [] if mapping is None else mapping.get(
            f"inactive_z_matrix_root_atoms_{endpoint}_0based", []
        )
        for endpoint in ("a", "b")
    }
    junctions = []
    if mapping is not None:
        for entry in mapping.get("resolved_junction_bonds", []):
            endpoint = entry["endpoint"]
            ligand_map = ligand_maps[endpoint]
            junctions.append({
                **entry,
                "boundary_system_atoms_0based": [
                    ligand_map[index] for index in entry["boundary_atoms_0based"]
                ],
                "branch_system_atoms_0based": [
                    ligand_map[index] for index in entry["branch_atoms_0based"]
                ],
            })
    return {
        "schema_version": 1,
        "scope": "all bonded terms containing at least one endpoint-unique atom",
        "unique_a_system_atoms_0based": sorted(unique_a),
        "unique_b_system_atoms_0based": sorted(unique_b),
        "trajectory_system_atoms_0based": list(range(len(atom_rows))),
        "atoms": atom_rows,
        "z_matrix_roots_ligand_0based": z_matrix_roots_local,
        "z_matrix_roots_system_0based": {
            endpoint: [ligand_maps[endpoint][index] for index in roots]
            for endpoint, roots in z_matrix_roots_local.items()
        },
        "junctions": junctions,
        "terms": terms,
    }


def _minimum_image(vector, box):
    if box is None:
        return vector
    fractional = vector @ np.linalg.inv(box)
    return vector - np.round(fractional) @ box


def _geometry(term, positions, box):
    atoms = term["atoms_0based"]
    if term["term_type"] == "bond":
        delta = _minimum_image(positions[atoms[1]] - positions[atoms[0]], box)
        return float(np.linalg.norm(delta)), "nm"
    if term["term_type"] == "angle":
        left = _minimum_image(positions[atoms[0]] - positions[atoms[1]], box)
        right = _minimum_image(positions[atoms[2]] - positions[atoms[1]], box)
        cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))), "degree"
    points = [np.zeros(3)]
    for first, second in zip(atoms, atoms[1:]):
        points.append(points[-1] + _minimum_image(
            positions[second] - positions[first], box
        ))
    p0, p1, p2, p3 = points
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    angle = np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))
    return float(angle), "degree"


def _term_energy(term, geometry):
    if term["term_type"] == "bond":
        return 0.5 * term["k_kj_mol_nm2"] * (geometry - term["equilibrium_nm"])**2
    radians = math.radians(geometry)
    if term["term_type"] == "angle":
        return 0.5 * term["k_kj_mol_rad2"] * (
            radians - term["equilibrium_radians"]
        )**2
    return term["k_kj_mol"] * (
        1.0 + math.cos(term["periodicity"] * radians - term["phase_radians"])
    )


def _subset_topology(topology, positions, selected):
    selected = set(selected)
    modeller = app.Modeller(topology, positions)
    modeller.delete([atom for atom in modeller.topology.atoms() if atom.index not in selected])
    return modeller.topology, modeller.positions


class SwitchDiagnosticWriter:
    """Stream one switch's diagnostic tables and ligand trajectory atomically."""

    def __init__(self, prefix, prepared, manifest, parameter_names, timestep_fs=2.0):
        self.prefix = Path(prefix)
        self.manifest = manifest
        self.parameter_names = tuple(sorted(parameter_names))
        self.timestep_fs = float(timestep_fs)
        self.frame = 0
        self.previous_work = 0.0
        self.paths = {
            "frames": self.prefix.with_suffix(".frames.csv.gz"),
            "bonded": self.prefix.with_suffix(".bonded.csv.gz"),
            "trajectory": self.prefix.with_suffix(".ligand.xtc"),
        }
        self.temporary = {
            key: path.with_name(path.name + ".tmp") for key, path in self.paths.items()
        }
        for path in self.temporary.values():
            path.unlink(missing_ok=True)
        self.frame_handle = gzip.open(self.temporary["frames"], "wt", newline="")
        self.geometry_handle = gzip.open(self.temporary["bonded"], "wt", newline="")
        self.frame_writer = csv.DictWriter(
            self.frame_handle, fieldnames=[*FRAME_FIELDS, *self.parameter_names]
        )
        self.geometry_writer = csv.DictWriter(
            self.geometry_handle, fieldnames=GEOMETRY_FIELDS
        )
        self.frame_writer.writeheader()
        self.geometry_writer.writeheader()
        selected = manifest["trajectory_system_atoms_0based"]
        topology, positions = _subset_topology(
            prepared.topology, prepared.positions, selected
        )
        topology_path = self.prefix.parent / "diagnostic_ligand_topology.pdb"
        if not topology_path.exists():
            with topology_path.open("w") as handle:
                app.PDBFile.writeFile(topology, positions, handle)
        self.selected = selected
        self.xtc = app.XTCFile(
            str(self.temporary["trajectory"]), topology,
            self.timestep_fs * unit.femtosecond, interval=1,
        )

    def observe(self, context, *, step, segment, stage, physical_lambda,
                cumulative_work_kj_per_mol, parameters):
        state = context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=False)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        vectors = state.getPeriodicBoxVectors(asNumpy=True)
        box = None if vectors is None else vectors.value_in_unit(unit.nanometer)
        potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        row = {
            "frame": self.frame,
            "step": int(step),
            "time_ps": int(step) * self.timestep_fs / 1000.0,
            "segment": int(segment) + 1,
            "stage": stage,
            "protocol_fraction": float(step) / parameters.pop("_total_steps"),
            "physical_lambda": physical_lambda,
            "potential_energy_kj_per_mol": float(potential),
            "window_work_kj_per_mol": float(cumulative_work_kj_per_mol - self.previous_work),
            "cumulative_work_kj_per_mol": float(cumulative_work_kj_per_mol),
            **parameters,
        }
        self.frame_writer.writerow(row)
        for term in self.manifest["terms"]:
            geometry, geometry_unit = _geometry(term, positions, box)
            self.geometry_writer.writerow({
                "frame": self.frame,
                "step": int(step),
                "term_id": term["term_id"],
                "endpoint_parameter_set": term["endpoint_parameter_set"],
                "term_type": term["term_type"],
                "term_class": term["term_class"],
                "geometry": geometry,
                "geometry_unit": geometry_unit,
                "endpoint_form_energy_kj_per_mol": _term_energy(term, geometry),
            })
        self.xtc.writeModel(
            positions[self.selected] * unit.nanometer,
            periodicBoxVectors=state.getPeriodicBoxVectors(),
        )
        self.previous_work = float(cumulative_work_kj_per_mol)
        self.frame += 1

    def close(self, success):
        self.frame_handle.close()
        self.geometry_handle.close()
        self.xtc = None
        if success:
            for key, path in self.paths.items():
                os.replace(self.temporary[key], path)


def write_manifest(path, manifest):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(manifest, sort_keys=False))
    os.replace(temporary, path)


def analyze_diagnostics(directory):
    """Rank geometry/strain excursions and write compact diagnostic plots."""
    directory = Path(directory)
    manifest = yaml.safe_load(
        (directory.parent / "switch_diagnostic_terms.yaml").read_text()
    )
    term_metadata = {term["term_id"]: term for term in manifest["terms"]}
    records = {term_id: {"geometry": [], "energy": [], "work": [], "lambda": []}
               for term_id in term_metadata}
    frame_series = []
    for frame_path in sorted(directory.glob("*.frames.csv.gz")):
        stem = frame_path.name.removesuffix(".frames.csv.gz")
        bonded_path = directory / f"{stem}.bonded.csv.gz"
        with gzip.open(frame_path, "rt", newline="") as handle:
            frames = {int(row["frame"]): row for row in csv.DictReader(handle)}
        frame_series.append((stem, list(frames.values())))
        with gzip.open(bonded_path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                frame = frames[int(row["frame"])]
                bucket = records[row["term_id"]]
                bucket["geometry"].append(float(row["geometry"]))
                bucket["energy"].append(float(row["endpoint_form_energy_kj_per_mol"]))
                bucket["work"].append(abs(float(frame["window_work_kj_per_mol"])))
                bucket["lambda"].append(float(frame["physical_lambda"]))
    summaries = []
    for term_id, values in records.items():
        if not values["geometry"]:
            continue
        geometry = np.asarray(values["geometry"])
        energy = np.asarray(values["energy"])
        work = np.asarray(values["work"])
        correlation = 0.0
        if len(energy) > 1 and np.std(energy) > 0 and np.std(work) > 0:
            correlation = float(np.corrcoef(energy, work)[0, 1])
        summaries.append({
            "term_id": term_id,
            "term_type": term_metadata[term_id]["term_type"],
            "term_class": term_metadata[term_id]["term_class"],
            "atoms_0based": term_metadata[term_id]["atoms_0based"],
            "geometry_min": float(np.min(geometry)),
            "geometry_max": float(np.max(geometry)),
            "geometry_range": float(np.max(geometry) - np.min(geometry)),
            "energy_max_kj_per_mol": float(np.max(energy)),
            "energy_p99_kj_per_mol": float(np.quantile(energy, 0.99)),
            "work_energy_correlation": correlation,
        })
    ranked = sorted(
        summaries,
        key=lambda row: (row["energy_p99_kj_per_mol"], abs(row["work_energy_correlation"])),
        reverse=True,
    )
    result = {
        "schema_version": 1,
        "switches": len(frame_series),
        "frames": sum(len(rows) for _, rows in frame_series),
        "terms": len(summaries),
        "top_terms_by_strain": ranked[:25],
        "top_terms_by_work_correlation": sorted(
            summaries,
            key=lambda row: abs(row["work_energy_correlation"]),
            reverse=True,
        )[:25],
    }
    write_manifest(directory / "summary.yaml", result)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return result
    if not frame_series:
        return result
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for label, rows in frame_series:
        lam = [float(row["physical_lambda"]) for row in rows]
        axes[0].plot(lam, [float(row["cumulative_work_kj_per_mol"]) for row in rows],
                     alpha=0.55, linewidth=0.7, label=label)
        axes[1].plot(lam, [float(row["potential_energy_kj_per_mol"]) for row in rows],
                     alpha=0.4, linewidth=0.6)
    axes[0].set_ylabel("Cumulative work (kJ/mol)")
    axes[1].set_ylabel("Potential energy (kJ/mol)")
    axes[1].set_xlabel("Physical lambda (A=0, B=1)")
    axes[0].legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(directory / "work_and_potential.png", dpi=180)
    plt.close(figure)

    top = ranked[:12]
    figure, axes = plt.subplots(4, 3, figsize=(14, 12), squeeze=False)
    for axis, summary in zip(axes.flat, top):
        values = records[summary["term_id"]]
        axis.scatter(values["lambda"], values["energy"], s=1, alpha=0.25)
        axis.set_title(summary["term_id"], fontsize=8)
        axis.set_xlabel("lambda")
        axis.set_ylabel("endpoint-form kJ/mol")
    for axis in axes.flat[len(top):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(directory / "top_bonded_strain.png", dpi=180)
    plt.close(figure)
    return result
