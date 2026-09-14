"""Recompute protein-aligned DNA RMSD from OpenMM validation DCD trajectories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import struct

import numpy as np
from openmm import app, unit
import yaml


def _record(handle):
    header = handle.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise ValueError("truncated DCD record header")
    size = struct.unpack("<i", header)[0]
    data = handle.read(size)
    trailer = handle.read(4)
    if len(data) != size or len(trailer) != 4 or struct.unpack("<i", trailer)[0] != size:
        raise ValueError("truncated or invalid DCD record")
    return data


def _box_vectors(cell):
    a, cos_gamma, b, cos_beta, cos_alpha, c = np.asarray(cell) * [0.1, 1, 0.1, 1, 1, 0.1]
    sin_gamma = np.sqrt(max(0.0, 1.0 - cos_gamma**2))
    if sin_gamma < 1e-8:
        raise ValueError("invalid DCD unit cell")
    c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    c_z = np.sqrt(max(0.0, c**2 - (c * cos_beta)**2 - c_y**2))
    return np.array([
        [a, 0.0, 0.0],
        [b * cos_gamma, b * sin_gamma, 0.0],
        [c * cos_beta, c_y, c_z],
    ])


def openmm_dcd_frames(path, expected_atoms):
    with Path(path).open("rb") as handle:
        header = _record(handle)
        if header is None or len(header) != 84 or header[:4] != b"CORD":
            raise ValueError("unsupported DCD header")
        cell_present = bool(struct.unpack_from("<i", header, 44)[0])
        _record(handle)
        atom_record = _record(handle)
        count = struct.unpack("<i", atom_record)[0]
        if count != expected_atoms:
            raise ValueError(f"DCD particle count {count} differs from topology {expected_atoms}")
        while True:
            first = _record(handle)
            if first is None:
                return
            cell = _box_vectors(struct.unpack("<6d", first)) if cell_present else None
            coordinates = [
                np.frombuffer(_record(handle), dtype="<f4") * 0.1
                for _ in range(3)
            ]
            if any(len(axis) != count for axis in coordinates):
                raise ValueError("DCD frame has incorrect coordinate count")
            yield np.stack(coordinates, axis=1), cell


def protein_aligned_dna_rmsd(positions, reference, protein, dna, vectors):
    current_center = positions[protein].mean(axis=0)
    reference_center = reference[protein].mean(axis=0)
    current_protein = positions[protein] - current_center
    reference_protein = reference[protein] - reference_center
    u, _, vt = np.linalg.svd(current_protein.T @ reference_protein)
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    displacement = (
        (positions[dna] - current_center) @ rotation
        - (reference[dna] - reference_center)
    )
    if vectors is not None:
        aligned_vectors = vectors @ rotation
        fractional = displacement @ np.linalg.inv(aligned_vectors)
        displacement -= np.rint(fractional) @ aligned_vectors
    return float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))))


def correct_task(topology_path, task_dir, report_interval_steps=5000):
    topology_file = app.PDBxFile(str(topology_path))
    atoms = list(topology_file.topology.atoms())
    amino = set("ALA ARG ASN ASP CYS GLU GLN GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
    protein = [
        atom.index for atom in atoms if atom.residue.name in amino
        and atom.element and atom.element.symbol != "H"
    ]
    dna = [
        atom.index for atom in atoms if atom.residue.name in {"DA", "DT", "DC", "DG"}
        and atom.element and atom.element.symbol != "H"
    ]
    if not protein or not dna:
        raise ValueError("topology must contain protein and DNA heavy atoms")
    reference = np.asarray(topology_file.positions.value_in_unit(unit.nanometer))
    with (task_dir / "metrics.csv").open(newline="") as handle:
        original = {int(row["step"]): row for row in csv.DictReader(handle)}
    segments = yaml.safe_load((task_dir / "trajectory_segments.yaml").read_text())["segments"]
    corrected = {}
    for segment in segments:
        start = int(segment["start_step"])
        committed = int(segment["last_committed_step"])
        for frame_index, (positions, vectors) in enumerate(
            openmm_dcd_frames(task_dir / segment["file"], len(atoms)), start=1
        ):
            step = start + frame_index * report_interval_steps
            if step > committed:
                break
            if step not in original:
                continue
            corrected[step] = {
                "step": step,
                "time_ps": original[step]["time_ps"],
                "dna_protein_aligned_rmsd_nm_original": original[step]["dna_protein_aligned_rmsd_nm"],
                "dna_protein_aligned_rmsd_nm_corrected": protein_aligned_dna_rmsd(
                    positions, reference, protein, dna, vectors,
                ),
            }
    if not corrected:
        raise ValueError(f"no committed DNA frames found in {task_dir}")
    output = task_dir / "dna_rmsd_corrected.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(corrected.values()))))
        writer.writeheader()
        writer.writerows(corrected[step] for step in sorted(corrected))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topology", type=Path)
    parser.add_argument("task_dirs", nargs="+", type=Path)
    parser.add_argument("--interval", type=int, default=5000)
    args = parser.parse_args(argv)
    summaries = []
    for task in args.task_dirs:
        output = correct_task(args.topology, task, args.interval)
        print(output)
        with output.open(newline="") as handle:
            values = np.asarray([
                float(row["dna_protein_aligned_rmsd_nm_corrected"])
                for row in csv.DictReader(handle)
            ])
        metadata = yaml.safe_load((task / "task.yaml").read_text())
        status_path = task / "result.yaml"
        status = yaml.safe_load(status_path.read_text())["status"] if status_path.is_file() else "partial"
        summaries.append({
            "variant": metadata["variant"], "replicate": metadata["replicate"],
            "status": status, "frames": len(values),
            "dna_rmsd_recent_mean_nm": float(np.mean(values[len(values) // 2:])),
        })
    if len({task.parent for task in args.task_dirs}) == 1:
        variants = []
        for variant in sorted({item["variant"] for item in summaries}):
            completed = [
                item["dna_rmsd_recent_mean_nm"] for item in summaries
                if item["variant"] == variant and item["status"] == "completed"
            ]
            variants.append({
                "variant": variant,
                "completed_replicates": len(completed),
                "dna_rmsd_nm": float(np.mean(completed)) if completed else None,
                "between_replica_sd_nm": (
                    float(np.std(completed, ddof=1)) if len(completed) > 1 else None
                ),
            })
        summary = args.task_dirs[0].parent.parent / "dna_analysis_corrected.yaml"
        summary.write_text(yaml.safe_dump({
            "schema_version": 1, "variants": variants, "tasks": summaries,
        }, sort_keys=False))
        print(summary)


if __name__ == "__main__":
    main()
