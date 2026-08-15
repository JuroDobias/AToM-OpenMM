import numpy as np
import openmm as mm
from openmm import unit
from openff.toolkit import ForceField, Molecule
from openff.units import unit as offunit
import fcntl
import pytest
from types import SimpleNamespace
import yaml

from atom_openmm.covalent_parameters import CovalentParameterBundle
from atom_openmm.separated_topology import (
    assemble_positions_and_velocities,
    build_separated_ligands,
)


def _bundle(smiles):
    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=1)
    charges = np.zeros(molecule.n_atoms)
    molecule.partial_charges = charges * offunit.elementary_charge
    system = ForceField("openff-2.2.1.offxml").create_openmm_system(
        molecule.to_topology(), charge_from_molecules=[molecule]
    )
    return CovalentParameterBundle(molecule, system, charges, smiles, {})


def _state(positions):
    system = mm.System()
    for _ in positions:
        system.addParticle(12.0)
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(np.asarray(positions) * unit.nanometer)
    context.setVelocities(np.zeros_like(positions) * unit.nanometer / unit.picosecond)
    state = context.getState(getPositions=True, getVelocities=True)
    del context, integrator
    return state


def _test_separated_ligands_have_no_common_atoms_and_full_particle_sets():
    left = _bundle("CCO")
    right = _bundle("CCN")
    dual = build_separated_ligands(left, right)

    assert dual.map_a_to_b == {}
    assert len(dual.unique_a) == left.molecule.n_atoms
    assert len(dual.unique_b) == right.molecule.n_atoms
    assert dual.topology.getNumAtoms() == left.molecule.n_atoms + right.molecule.n_atoms
    assert dual.endpoint_a.getNumParticles() == dual.endpoint_b.getNumParticles()

    for system in (dual.endpoint_a, dual.endpoint_b):
        nonbonded = next(
            force for force in system.getForces()
            if isinstance(force, mm.NonbondedForce)
        )
        exceptions = {
            tuple(sorted(map(int, nonbonded.getExceptionParameters(index)[:2])))
            for index in range(nonbonded.getNumExceptions())
        }
        cross = {
            tuple(sorted((dual.map_a_to_hybrid[a], dual.map_b_to_hybrid[b])))
            for a in range(left.molecule.n_atoms)
            for b in range(right.molecule.n_atoms)
        }
        assert cross <= exceptions

    positions = dual.positions
    energies = []
    for system in (dual.endpoint_a, dual.endpoint_b):
        integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
        context = mm.Context(
            system, integrator, mm.Platform.getPlatformByName("Reference")
        )
        context.setPositions(positions)
        energies.append(
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        del context, integrator
    assert np.isclose(energies[0], energies[1], atol=1.0e-6)


def _test_assembled_state_aligns_inactive_frame_and_preserves_environment():
    active_positions = np.asarray([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [2.0, 2.0, 2.0],
    ])
    inactive_positions = np.asarray([
        [1.0, 1.0, 1.0],
        [1.0, 1.1, 1.0],
        [0.9, 1.0, 1.0],
    ])
    assembled = assemble_positions_and_velocities(
        endpoint="a",
        active_state=_state(active_positions),
        inactive_state=_state(inactive_positions),
        ligand_a_count=3,
        ligand_b_count=3,
        anchors_a=(0, 1, 2),
        anchors_b=(0, 1, 2),
    )
    observed = assembled["positions"].value_in_unit(unit.nanometer)

    assert observed.shape == (7, 3)
    assert assembled["anchor_rmsd_a"] < 1.0e-6
    assert np.allclose(observed[-1], active_positions[-1])


def _test_anchor_rmsd_reports_direct_frame_separation():
    from atom_openmm.separated_workflow import _anchor_rmsd_a

    positions = np.asarray([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.1, 0.1, 0.0],
    ]) * unit.nanometer

    assert np.isclose(_anchor_rmsd_a(positions, (0, 1, 2), (3, 4, 5)), 1.0)


def _test_workflow_schema_accepts_separated_neqti_only():
    from atom_openmm.workflow_schema import normalize_workflow_axes

    axes = normalize_workflow_axes({
        "chemistry": "noncovalent",
        "alchemy": {"model": "separated_topology", "cycle": "complex_solvent"},
        "sampling": {"method": "neqti"},
    })
    assert axes.alchemy_model == "separated_topology"


def _test_adaptive_duration_statistics_use_only_paired_pilot_prefix(monkeypatch):
    from atom_openmm import separated_workflow as module

    observed = {}

    def statistics(forward, reverse, config):
        observed["forward"] = forward
        observed["reverse"] = reverse
        observed["config"] = config
        return {"passed": True}

    monkeypatch.setattr(module, "_adaptive_work_statistics", statistics)
    forward = [
        {"work_kcal_per_mol": str(value)}
        for value in (1, 2, 3, 100, 200)
    ]
    reverse = [
        {"work_kcal_per_mol": str(value)}
        for value in (-1, -2, -3, -100)
    ]

    result = module._pilot_adaptive_statistics(
        forward, reverse, {"name": "config"}, 3
    )

    assert result == {"passed": True}
    assert observed == {
        "forward": [1.0, 2.0, 3.0],
        "reverse": [-1.0, -2.0, -3.0],
        "config": {"name": "config"},
    }


def _test_separated_workflow_plan_exposes_node_bank(tmp_path):
    from atom_openmm.rbfe_workflow import plan_workflow

    (tmp_path / "receptor.pdb").write_text("END\n")
    ligands = tmp_path / "ligands"
    ligands.mkdir()
    for name, smiles in (("A", "CCO"), ("B", "CCN")):
        molecule = Molecule.from_smiles(smiles)
        molecule.generate_conformers(n_conformers=1)
        molecule.to_file(str(ligands / f"{name}.sdf"), file_format="SDF")
    payload = {
        "workflow": {
            "type": "rbfe",
            "chemistry": "noncovalent",
            "alchemy": {
                "model": "separated_topology",
                "cycle": "complex_solvent",
                "node_bank": {"path": "bank", "snapshots": 2},
            },
            "sampling": {"method": "neqti"},
            "receptor": "receptor.pdb",
            "ligands_dir": "ligands",
            "pairs": [["A", "B"]],
            "workdir": "run",
            "neqti": {
                "interpolation": "softcore_linear",
                "n_snapshots": 2,
                "softcore": {
                    "charge_steps_per_stage": 10,
                    "sterics_steps": 20,
                    "long_range_correction": "dynamic",
                },
                "rest2": {"enabled": False},
            },
        }
    }
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(payload))

    plan = plan_workflow(path)
    assert plan["alchemy_model"] == "separated_topology"
    assert plan["unique_nodes"] == ["A", "B"]
    assert plan["node_bank"] == str((tmp_path / "bank").resolve())


def _test_parallel_node_bank_finalization_is_atomic(tmp_path, monkeypatch):
    from atom_openmm import separated_node_bank as module

    bank = tmp_path / "bank"
    staging = tmp_path / ".bank.building"
    staging.mkdir()
    context = {
        "bank": bank,
        "staging": staging,
        "fingerprint": "fingerprint",
        "fingerprint_payload": {"input": "value"},
    }
    monkeypatch.setattr(module, "_node_bank_context", lambda _: context)
    (staging / "fingerprint.yaml").write_text(
        yaml.safe_dump({"fingerprint": "fingerprint"})
    )
    (staging / "initialization.yaml").write_text(yaml.safe_dump({
        "fingerprint": "fingerprint",
        "receptor": "/receptor.pdb",
        "receptor_sha256": "receptor-hash",
        "canonical_environments": {},
        "snapshot_count": 1,
        "node_order": ["A", "B"],
    }))

    with pytest.raises(module.NodeBankError, match="missing shards"):
        module.finalize_node_bank("workflow.yaml")
    assert not bank.exists()

    def artifact(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
        return {"file": path.name, "sha256": module._sha256(path)}

    for node in ("A", "B"):
        node_dir = staging / "nodes" / node
        environments = {}
        for environment in ("complex", "solvent"):
            directory = node_dir / environment
            snapshot = directory / "snapshots" / "0000.xml"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_text("state")
            environments[environment] = {
                "system": artifact(directory / "system.xml"),
                "topology": artifact(directory / "topology.cif"),
                "snapshots": [{
                    "file": str(snapshot.relative_to(staging)),
                    "sha256": module._sha256(snapshot),
                }],
            }
        vacuum = node_dir / "vacuum"
        vacuum_snapshot = vacuum / "snapshots" / "0000.xml"
        vacuum_snapshot.parent.mkdir(parents=True, exist_ok=True)
        vacuum_snapshot.write_text("vacuum-state")
        payload = {
            "environments": environments,
            "vacuum": {
                "system": {
                    **artifact(vacuum / "system.xml"),
                    "file": str((vacuum / "system.xml").relative_to(staging)),
                },
                "topology": {
                    **artifact(vacuum / "topology.cif"),
                    "file": str((vacuum / "topology.cif").relative_to(staging)),
                },
                "snapshots": [{
                    "file": str(vacuum_snapshot.relative_to(staging)),
                    "sha256": module._sha256(vacuum_snapshot),
                }],
            },
        }
        (node_dir / "manifest.yaml").write_text(yaml.safe_dump({
            "fingerprint": "fingerprint",
            "node": payload,
        }))

    manifest = module.finalize_node_bank("workflow.yaml")

    assert sorted(manifest["nodes"]) == ["A", "B"]
    assert (bank / "manifest.yaml").is_file()
    assert not staging.exists()


def _test_schema_v1_node_bank_is_readable_but_not_extensible(tmp_path, monkeypatch):
    from atom_openmm import separated_node_bank as module

    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "manifest.yaml").write_text(yaml.safe_dump({
        "schema_version": 1,
        "nodes": {},
    }))
    context = {
        "bank": bank,
        "nodes": {"C": tmp_path / "C.sdf"},
        "fingerprint": "new-contract",
    }
    monkeypatch.setattr(module, "_node_bank_context", lambda _: context)

    assert module._existing_bank(context)["schema_version"] == 1
    with pytest.raises(module.NodeBankError, match="schema-v1"):
        module.extend_node_bank("workflow.yaml", "C")


def _test_parallel_extension_prepares_without_bank_lock_and_reloads_manifest(
    tmp_path, monkeypatch
):
    from atom_openmm import separated_node_bank as module

    bank = tmp_path / "bank"
    (bank / "nodes").mkdir(parents=True)
    ligand = tmp_path / "C.sdf"
    ligand.write_text("ligand")
    manifest_path = bank / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({
        "schema_version": module.SCHEMA_VERSION,
        "compatibility_fingerprint": "fingerprint",
        "extensible": True,
        "extension_max_linear_scale": 1.05,
        "canonical_environments": {},
        "nodes": {"A": {"charge_e": 0.0, "node_fingerprint": "A"}},
    }))
    context = {
        "bank": bank,
        "nodes": {"C": ligand},
        "anchors": {"C": (0, 1, 2)},
        "fingerprint": "fingerprint",
        "setup": {},
        "config": {"random_seed": 7},
        "extension_max_linear_scale": 1.05,
    }
    bundle = SimpleNamespace(
        charges_e=np.asarray([0.0]),
        provenance={"charge_model": "test"},
    )
    monkeypatch.setattr(module, "_node_bank_context", lambda _: context)
    monkeypatch.setattr(module, "parameterize_ligand", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        module, "_extension_box_vectors", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(module, "_validate_node_payload", lambda *args: None)

    def prepare(context, initialization, node, root, **kwargs):
        # A second process must be able to acquire the global publication lock
        # while this expensive preparation callback is active.
        lock_path = bank.with_name(".bank.lock")
        with lock_path.open("a+") as probe:
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        node_dir = root / "nodes" / node
        node_dir.mkdir(parents=True)
        current = yaml.safe_load(manifest_path.read_text())
        current["nodes"]["D"] = {
            "charge_e": 0.0,
            "node_fingerprint": "D",
        }
        manifest_path.write_text(yaml.safe_dump(current))
        return dict(initialization["nodes"][node])

    monkeypatch.setattr(module, "_prepare_node_payload", prepare)

    payload = module.extend_node_bank("workflow.yaml", "C")
    published = yaml.safe_load(manifest_path.read_text())

    assert payload["node_fingerprint"] == published["nodes"]["C"]["node_fingerprint"]
    assert sorted(published["nodes"]) == ["A", "C", "D"]
    assert (bank / "nodes" / "C" / "manifest.yaml").is_file()
    assert not (tmp_path / ".bank.extensions" / "C").exists()


def _test_triclinic_box_volume_uses_determinant():
    from atom_openmm.separated_node_bank import (
        _box_heights_nm,
        _box_volume_nm3,
        _cutoff_safe_box_vectors,
    )

    vectors = np.asarray([
        [3.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.5, 0.5, 1.5],
    ])
    assert np.isclose(_box_volume_nm3(vectors), 9.0)
    scaled = _cutoff_safe_box_vectors(vectors, 0.9, 0.2)
    assert np.min(_box_heights_nm(scaled)) >= 2.0 - 1.0e-12
    nonzero = vectors != 0.0
    scale = scaled[nonzero][0] / vectors[nonzero][0]
    assert np.allclose(scaled, vectors * scale)


def _test_custom_equilibration_pdb_never_aliases_state_xml(tmp_path):
    from atom_openmm.covalent_workflow import _equilibrated_pdb_path

    plain = tmp_path / "state.xml"
    named = tmp_path / "complex_endpoint_a_state.xml"

    assert _equilibrated_pdb_path(plain) == tmp_path / "state_equilibrated.pdb"
    assert _equilibrated_pdb_path(named) == (
        tmp_path / "complex_endpoint_a_equilibrated.pdb"
    )
