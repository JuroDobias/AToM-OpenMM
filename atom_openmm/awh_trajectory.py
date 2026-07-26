"""State-tagged structural trajectory output for AWH simulations."""

from __future__ import annotations

import csv
from pathlib import Path

from openmm import app
from openmm.app.internal.xtc_utils import get_xtc_nframes


FRAME_FIELDS = [
    "frame",
    "simulation_step",
    "awh_steps",
    "move",
    "stage",
    "state",
    "state_name",
    "kind",
    "atm_state",
    "rest2_region",
    "rest2_scale",
    "effective_temperature_k",
]


def _read_frame_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_frame_rows(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in FRAME_FIELDS} for row in rows
        )
    temporary.replace(path)


def _reconcile_resume_files(trajectory_path, frame_path):
    if not trajectory_path.exists():
        if frame_path.exists():
            frame_path.unlink()
        return 0
    trajectory_frames = int(
        get_xtc_nframes(str(trajectory_path).encode("utf-8"))
    )
    rows = _read_frame_rows(frame_path)
    if len(rows) > trajectory_frames:
        rows = rows[:trajectory_frames]
        _write_frame_rows(frame_path, rows)
    elif len(rows) < trajectory_frames:
        for frame in range(len(rows), trajectory_frames):
            rows.append(
                {
                    "frame": frame,
                    "stage": "unknown",
                    "state_name": "unknown_after_interrupted_write",
                }
            )
        _write_frame_rows(frame_path, rows)
    return trajectory_frames


def write_subset_topology(topology, positions, selected_atoms, path):
    selected = set(int(index) for index in selected_atoms)
    modeller = app.Modeller(topology, positions)
    modeller.delete(
        [atom for atom in modeller.topology.atoms() if atom.index not in selected]
    )
    with Path(path).open("w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle)


class StateTaggedXTCReporter:
    """XTC reporter with a sidecar row identifying the active AWH Hamiltonian."""

    def __init__(
        self,
        trajectory_path,
        frame_path,
        report_interval,
        atom_subset,
        metadata_provider,
        *,
        append=False,
        origin_simulation_step=0,
        origin_awh_steps=0,
    ):
        trajectory_path = Path(trajectory_path)
        frame_path = Path(frame_path)
        existing_frames = (
            _reconcile_resume_files(trajectory_path, frame_path) if append else 0
        )
        self._delegate = app.XTCReporter(
            str(trajectory_path),
            int(report_interval),
            append=bool(append and trajectory_path.exists()),
            enforcePeriodicBox=False,
            atomSubset=list(atom_subset),
        )
        self._frame_path = frame_path
        self._metadata_provider = metadata_provider
        self._origin_simulation_step = int(origin_simulation_step)
        self._origin_awh_steps = int(origin_awh_steps)
        self._frame = existing_frames

    def describeNextReport(self, simulation):
        return self._delegate.describeNextReport(simulation)

    def report(self, simulation, state):
        self._delegate.report(simulation, state)
        metadata = dict(self._metadata_provider())
        simulation_step = int(simulation.currentStep)
        row = {
            "frame": self._frame,
            "simulation_step": simulation_step,
            "awh_steps": self._origin_awh_steps
            + simulation_step
            - self._origin_simulation_step,
            **metadata,
        }
        with self._frame_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({field: row.get(field) for field in FRAME_FIELDS})
        self._frame += 1
