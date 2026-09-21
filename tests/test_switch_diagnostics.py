from types import SimpleNamespace

import numpy as np
import openmm as mm
from openmm import app, unit

from atom_openmm.switch_diagnostics import (
    SwitchDiagnosticWriter,
    _geometry,
    _term_energy,
    build_bonded_manifest,
)
from atom_openmm.covalent_workflow import _run_segmented_protocol


def _prepared_system():
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HYB", chain)
    atoms = [
        topology.addAtom(f"C{index}", app.element.carbon, residue)
        for index in range(4)
    ]
    for left, right in zip(atoms, atoms[1:]):
        topology.addBond(left, right)
    system = mm.System()
    for _ in atoms:
        system.addParticle(12.0)
    bonds = mm.HarmonicBondForce()
    bonds.addBond(1, 2, 0.15, 200.0)
    system.addForce(bonds)
    angles = mm.HarmonicAngleForce()
    angles.addAngle(0, 1, 2, 2.0, 40.0)
    system.addForce(angles)
    torsions = mm.PeriodicTorsionForce()
    torsions.addTorsion(0, 1, 2, 3, 3, 0.2, 5.0)
    torsions.addTorsion(1, 0, 2, 3, 2, 0.1, 4.0)
    system.addForce(torsions)
    return SimpleNamespace(
        topology=topology,
        positions=np.asarray([
            [0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
            [0.1, 0.1, 0.0], [0.1, 0.1, 0.1],
        ]) * unit.nanometer,
        endpoint_a=system,
        endpoint_b=mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system)),
        provenance={
            "unique_a_particle_indices": [2],
            "unique_b_particle_indices": [3],
            "hybrid_solute_atom_count": 4,
            "soft_bond_system_pairs": [[1, 2]],
            "ligand_a_system_atom_indices": [0, 1, 2, 3],
            "ligand_b_system_atom_indices": [0, 1, 2, 3],
        },
    )


def _test_manifest_includes_all_unique_bonded_term_types():
    manifest = build_bonded_manifest(_prepared_system())
    types = {term["term_type"] for term in manifest["terms"]}

    assert types == {"bond", "angle", "proper", "improper"}
    assert any(term["term_class"] == "alchemical_bond" for term in manifest["terms"])
    assert all(
        set(term["atoms_0based"]) & {2, 3}
        for term in manifest["terms"]
    )


def _test_internal_coordinate_values_and_endpoint_form_energies():
    positions = np.asarray([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.1, 0.1, 0.0],
        [0.1, 0.1, 0.1],
    ])
    bond = {
        "term_type": "bond", "atoms_0based": [0, 1],
        "equilibrium_nm": 0.08, "k_kj_mol_nm2": 200.0,
    }
    angle = {
        "term_type": "angle", "atoms_0based": [0, 1, 2],
        "equilibrium_radians": np.pi / 2, "k_kj_mol_rad2": 40.0,
    }
    torsion = {
        "term_type": "proper", "atoms_0based": [0, 1, 2, 3],
        "periodicity": 1, "phase_radians": 0.0, "k_kj_mol": 5.0,
    }

    bond_value, _ = _geometry(bond, positions, None)
    angle_value, _ = _geometry(angle, positions, None)
    torsion_value, _ = _geometry(torsion, positions, None)

    assert np.isclose(bond_value, 0.1)
    assert np.isclose(angle_value, 90.0)
    assert np.isclose(abs(torsion_value), 90.0)
    assert np.isclose(_term_energy(bond, bond_value), 0.04)
    assert np.isclose(_term_energy(angle, angle_value), 0.0)
    assert np.isclose(_term_energy(torsion, torsion_value), 5.0)


def _test_internal_coordinates_use_minimum_image_vectors():
    positions = np.asarray([[0.95, 0.0, 0.0], [0.05, 0.0, 0.0]])
    term = {"term_type": "bond", "atoms_0based": [0, 1]}

    value, _ = _geometry(term, positions, np.eye(3))

    assert np.isclose(value, 0.1)


def _test_writer_creates_atomic_tables_and_ligand_trajectory(tmp_path):
    prepared = _prepared_system()
    manifest = build_bonded_manifest(prepared)
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(prepared.endpoint_a, integrator)
    context.setPositions(prepared.positions)
    writer = SwitchDiagnosticWriter(
        tmp_path / "evaluation_010_a",
        prepared,
        manifest,
        ["TEST_PARAMETER"],
    )

    writer.observe(
        context,
        step=0,
        segment=0,
        stage="physical_a",
        physical_lambda=0.0,
        cumulative_work_kj_per_mol=0.0,
        parameters={"TEST_PARAMETER": 1.0, "_total_steps": 100},
    )
    writer.close(True)
    del context, integrator

    assert (tmp_path / "evaluation_010_a.frames.csv.gz").is_file()
    assert (tmp_path / "evaluation_010_a.bonded.csv.gz").is_file()
    assert (tmp_path / "evaluation_010_a.ligand.xtc").is_file()
    assert not list(tmp_path.glob("*.tmp"))


def _test_segmented_protocol_observes_fixed_intervals_and_final_step():
    class Integrator:
        def __init__(self):
            self.steps = 0

        def step(self, count):
            self.steps += int(count)

        def get_protocol_work(self):
            return float(self.steps) * unit.kilojoule_per_mole

        def get_segments_per_stage(self):
            return [2]

        def get_stage_interpolation(self):
            return "linear"

    observed = []
    total, segments = _run_segmented_protocol(
        Integrator(),
        [7, 8],
        observation_interval_steps=4,
        observation_callback=lambda **row: observed.append(row),
    )

    assert [row["completed_steps"] for row in observed] == [4, 8, 12, 15]
    assert observed[-1]["segment"] == 1
    assert total == 15.0
    assert segments == [7.0, 8.0]
