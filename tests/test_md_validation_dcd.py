import numpy as np
import openmm as mm
from openmm import app, unit
import pytest

from atom_openmm.md_validation_dcd import (
    _reimage_components, openmm_dcd_frames, protein_aligned_dna_rmsd,
)


def _test_openmm_dcd_dna_rmsd_across_periodic_boundary(tmp_path):
    topology = app.Topology()
    protein = topology.addResidue("ALA", topology.addChain())
    topology.addAtom("CA", app.element.carbon, protein)
    dna = topology.addResidue("DA", topology.addChain())
    topology.addAtom("P", app.element.phosphorus, dna)
    vectors = (
        mm.Vec3(9, 0, 0), mm.Vec3(0, 9, 0), mm.Vec3(0, 0, 9)
    ) * unit.nanometer
    topology.setPeriodicBoxVectors(vectors)
    reference = np.asarray([[0, 0, 0], [4.4, 0, 0]])
    current = np.asarray([[0, 0, 0], [4.6, 0, 0]])
    trajectory = tmp_path / "test.dcd"
    with trajectory.open("wb") as handle:
        writer = app.DCDFile(handle, topology, 0.002 * unit.picosecond)
        writer.writeModel(current * unit.nanometer, periodicBoxVectors=vectors)
    frames = list(openmm_dcd_frames(trajectory, topology.getNumAtoms()))
    assert len(frames) == 1
    positions, cell = frames[0]
    assert np.allclose(positions, current, atol=1e-6)
    assert np.allclose(cell, np.eye(3) * 9, atol=1e-6)
    assert protein_aligned_dna_rmsd(positions, reference, [0], [1], cell) == pytest.approx(0.2)


def _test_reimage_components_rejoins_split_protein_chains():
    reference = np.asarray([
        [0.0, 0.0, 0.0], [0.2, 0.0, 0.0],
        [1.0, 0.0, 0.0], [1.2, 0.0, 0.0],
    ])
    positions = reference.copy()
    positions[2:] += [9.0, 0.0, 0.0]
    corrected = _reimage_components(
        positions, reference, [[0, 1], [2, 3]], [0, 1], np.eye(3) * 9.0,
    )
    assert corrected == pytest.approx(reference)
