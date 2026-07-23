from pathlib import Path

import openmm as mm
from openmm import app, unit
from rdkit import Chem
from rdkit.Chem import AllChem
import yaml

from atom_openmm.covalent_workflow import (
    CovalentWorkflowError,
    CovalentResumeError,
    _normalized_settings,
    _apply_state,
    plan_covalent_workflow,
    validate_covalent_workflow,
    _dummy_particles,
    _write_switch_pdb,
    _ensure_switch_protocol,
    _constrained_ligand_atom_map,
    _mapping_settings,
    _precompute_endpoint_lrc_corrections,
    _switch_protocol,
    _validate_softcore_endpoint_charge,
    _load_prepared_pair_bundle,
    _validate_state_system_compatibility,
    _write_prepared_pair_bundle,
)
from atom_openmm.covalent_systems import PreparedCovalentHybrid


class _BoxState:
    def __init__(self, length_nm):
        self.vectors = unit.Quantity(
            [
                [length_nm, 0.0, 0.0],
                [0.0, length_nm, 0.0],
                [0.0, 0.0, length_nm],
            ],
            unit.nanometer,
        )

    def getPeriodicBoxVectors(self, asNumpy=False):
        if asNumpy:
            return self.vectors
        values = self.vectors.value_in_unit(unit.nanometer)
        return tuple(mm.Vec3(*row) * unit.nanometer for row in values)


class _RecordingLRCEvaluator:
    platform_name = "CPU"

    def __init__(self):
        self.calls = []
        self.closed = False

    def correction(self, state, parameter_values, node):
        self.calls.append((state, parameter_values, node))
        return float(len(self.calls))

    def close(self):
        self.closed = True


def _write_3d_sdf(path, smiles, seed):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=seed) == 0
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()


def test_apply_state_accepts_positions_without_velocities():
    system = mm.System()
    system.addParticle(12.0)
    source = mm.Context(system, mm.VerletIntegrator(0.001))
    source.setPositions([[0.1, 0.2, 0.3]] * unit.nanometer)
    positions_only = source.getState(getPositions=True)
    target = mm.Context(system, mm.VerletIntegrator(0.001))

    _apply_state(target, positions_only)

    observed = target.getState(getPositions=True).getPositions(asNumpy=True)
    assert observed[0].x == positions_only.getPositions(asNumpy=True)[0].x


def _prepared_fixture():
    topology = app.Topology()
    chain = topology.addChain("A")
    residue = topology.addResidue("MOL", chain, "1")
    first = topology.addAtom("C1", app.Element.getBySymbol("C"), residue)
    second = topology.addAtom("C2", app.Element.getBySymbol("C"), residue)
    topology.addBond(first, second)
    systems = []
    for distance in (0.15, 0.16):
        system = mm.System()
        system.addParticle(12.0)
        system.addParticle(12.0)
        bonds = mm.HarmonicBondForce()
        bonds.addBond(0, 1, distance, 100.0)
        system.addForce(bonds)
        nonbonded = mm.NonbondedForce()
        nonbonded.addParticle(0.0, 0.3, 0.1)
        nonbonded.addParticle(0.0, 0.3, 0.1)
        system.addForce(nonbonded)
        systems.append(system)
    return PreparedCovalentHybrid(
        topology,
        [[0, 0, 0], [0.15, 0, 0]] * unit.nanometer,
        systems[0],
        systems[1],
        2,
        (0, 1),
        {"environment": "test", "unique_a_particle_indices": [0]},
    )


def _test_prepared_pair_bundle_round_trips_and_rejects_changed_fingerprint(tmp_path):
    prepared = _prepared_fixture()
    _write_prepared_pair_bundle(
        tmp_path,
        prepared,
        prepared,
        fingerprint="expected",
        fingerprint_inputs={"schema_version": 1},
        mapping_payload={"schema_version": 1},
        parameterization={"ligand_a": {}, "ligand_b": {}},
    )

    protein, reference, manifest = _load_prepared_pair_bundle(
        tmp_path, "expected"
    )

    assert protein.endpoint_a.getNumParticles() == 2
    assert reference.topology.getNumAtoms() == 2
    assert manifest["fingerprint"] == "expected"
    try:
        _load_prepared_pair_bundle(tmp_path, "changed")
    except CovalentResumeError as exc:
        assert "fingerprint differs" in str(exc)
    else:
        raise AssertionError("changed preparation fingerprint was accepted")


def _test_prepared_pair_bundle_rejects_corrupt_artifact(tmp_path):
    prepared = _prepared_fixture()
    _write_prepared_pair_bundle(
        tmp_path,
        prepared,
        prepared,
        fingerprint="expected",
        fingerprint_inputs={},
        mapping_payload={},
        parameterization={},
    )
    (tmp_path / "prepared" / "protein_endpoint_a.xml").write_text("corrupt")

    try:
        _load_prepared_pair_bundle(tmp_path, "expected")
    except CovalentResumeError as exc:
        assert "checksum differs" in str(exc)
    else:
        raise AssertionError("corrupt prepared artifact was accepted")


def _test_state_compatibility_reports_particle_count_mismatch():
    source = mm.System()
    source.addParticle(12.0)
    context = mm.Context(source, mm.VerletIntegrator(0.001))
    context.setPositions([[0, 0, 0]] * unit.nanometer)
    state = context.getState(getPositions=True)
    target = mm.System()
    target.addParticle(12.0)
    target.addParticle(12.0)

    try:
        _validate_state_system_compatibility(state, target, "endpoint A")
    except CovalentResumeError as exc:
        assert "1 positions" in str(exc)
        assert "2 particles" in str(exc)
    else:
        raise AssertionError("incompatible state particle count was accepted")


def test_covalent_mapping_settings_support_default_and_pair_override():
    workflow = {
        "mapping": {
            "method": "mcs_core_smarts",
            "smarts": "c1ccccc1",
        }
    }

    inherited = _mapping_settings(workflow, {})
    overridden = _mapping_settings(
        workflow,
        {"mapping": {"smarts": "c1ncccc1"}},
    )
    disabled = _mapping_settings(
        workflow,
        {"mapping": {"method": "dataset_core"}},
    )

    assert inherited == {"method": "mcs_core_smarts", "smarts": "c1ccccc1"}
    assert overridden == {"method": "mcs_core_smarts", "smarts": "c1ncccc1"}
    assert disabled == {"method": "dataset_core"}


def test_constrained_ligand_mcs_is_capped_by_smarts_and_reports_rmsd(tmp_path):
    ligand_a = tmp_path / "a.sdf"
    ligand_b = tmp_path / "b.sdf"
    _write_3d_sdf(ligand_a, "Cc1ccccc1", 11)
    _write_3d_sdf(ligand_b, "Oc1ccccc1", 22)
    inputs = {
        "ligand_a": {"name": "A", "aldehyde": ligand_a},
        "ligand_b": {"name": "B", "aldehyde": ligand_b},
    }

    mapping, provenance = _constrained_ligand_atom_map(
        inputs, "c1ccccc1"
    )

    assert len(mapping) == 6
    assert provenance["mcs_heavy_atom_count"] == 6
    assert provenance["candidate_matches_a"] > 1
    assert provenance["candidate_matches_b"] > 1
    assert provenance["selected_direct_rmsd_angstrom"] >= 0.0


def test_endpoint_lrc_corrections_are_precomputed_for_both_fixed_boxes():
    state_a = _BoxState(2.0)
    state_b = _BoxState(3.0)
    values = {"lambda_sterics_a": [1.0, 0.0]}
    evaluator = _RecordingLRCEvaluator()

    corrections = _precompute_endpoint_lrc_corrections(
        evaluator, state_a, state_b, values
    )

    assert corrections == {
        "forward": {"initial": 1.0, "final": 2.0, "volume_nm3": 8.0},
        "reverse": {"initial": 3.0, "final": 4.0, "volume_nm3": 27.0},
    }
    assert [(call[0], call[2]) for call in evaluator.calls] == [
        (state_a, 0),
        (state_a, -1),
        (state_b, -1),
        (state_b, 0),
    ]
    assert evaluator.closed


def test_switch_pdb_marks_dummy_atoms_with_zero_occupancy(tmp_path):
    topology = app.Topology()
    chain = topology.addChain("L")
    residue = topology.addResidue("HYB", chain, "1")
    topology.addAtom("C1", app.Element.getBySymbol("C"), residue)
    topology.addAtom("N1", app.Element.getBySymbol("N"), residue)
    system = mm.System()
    system.addParticle(12.0)
    system.addParticle(14.0)
    context = mm.Context(system, mm.VerletIntegrator(0.001))
    context.setPositions([[0, 0, 0], [0.1, 0, 0]] * unit.nanometer)
    state = context.getState(getPositions=True)
    output = tmp_path / "switch.pdb"

    _write_switch_pdb(
        output,
        topology,
        state,
        endpoint="a",
        dummy_atom_indices=[1],
    )

    text = output.read_text()
    atoms = [line for line in text.splitlines() if line.startswith("ATOM")]
    assert "ENDPOINT A" in text
    assert "INDICES (1-BASED): 2" in text
    assert float(atoms[0][54:60]) == 1.0
    assert float(atoms[1][54:60]) == 0.0


def test_dummy_particles_selects_branch_inactive_at_endpoint():
    prepared = type("Prepared", (), {
        "provenance": {
            "unique_a_particle_indices": [4, 5],
            "unique_b_particle_indices": [8, 9],
        }
    })()

    assert _dummy_particles(prepared, "a") == (8, 9)
    assert _dummy_particles(prepared, "b") == (4, 5)


def test_covalent_endpoint_equilibration_defaults_and_legacy_npt_alias():
    defaults = _normalized_settings({})["endpoint_equilibration"]
    assert defaults == {
        "minimization_tolerance_kj_mol_nm": 10.0,
        "minimization_max_iterations": 2000,
        "nvt_steps": 50000,
        "nvt_timestep_fs": 1.0,
        "npt_steps": 50000,
        "npt_timestep_fs": 2.0,
    }
    configured = _normalized_settings(
        {
            "neqti": {
                "initial_equilibration_steps": 250000,
                "endpoint_equilibration": {"nvt_steps": 25000},
            }
        }
    )["endpoint_equilibration"]
    assert configured["nvt_steps"] == 25000
    assert configured["npt_steps"] == 250000


def test_softcore_settings_derive_total_switch_steps():
    config = _normalized_settings(
        {
            "neqti": {
                "interpolation": "softcore_linear",
                "softcore": {
                    "alpha": 0.3,
                    "sigma_nm": 0.25,
                    "power": 1,
                    "charge_steps_per_stage": 1200,
                    "sterics_steps": 7600,
                },
            }
        }
    )
    assert config["switch_steps"] == 10000
    assert config["softcore"]["charge_steps_per_stage"] == 1200
    assert config["softcore"]["sterics_steps"] == 7600
    assert config["softcore"]["long_range_correction"] == "dynamic"


def _test_covalent_schedule_optimization_settings_enable_subdivision():
    config = _normalized_settings(
        {
            "neqti": {
                "interpolation": "softcore_linear",
                "softcore": {
                    "charge_steps_per_stage": 30000,
                    "sterics_steps": 90000,
                },
                "schedule_optimization": {
                    "enabled": True,
                    "pilot_samples": 10,
                    "subdivisions_per_stage": 10,
                },
            }
        }
    )

    assert config["schedule_optimization"]["enabled"]
    assert config["schedule_optimization"]["pilot_samples"] == 10
    assert config["softcore"]["subdivisions_per_stage"] == 10
    assert config["switch_steps"] == 150000


def test_softcore_resume_rejects_changed_protocol(tmp_path):
    first = _normalized_settings(
        {"neqti": {"interpolation": "softcore_linear", "softcore": {"sterics_steps": 30}}}
    )
    _ensure_switch_protocol(tmp_path, first)
    changed = _normalized_settings(
        {"neqti": {"interpolation": "softcore_linear", "softcore": {"sterics_steps": 31}}}
    )
    try:
        _ensure_switch_protocol(tmp_path, changed)
    except CovalentWorkflowError as exc:
        assert "different switching protocol" in str(exc)
    else:
        raise AssertionError("changed softcore protocol was accepted for resume")


def test_softcore_resume_rejects_changed_lrc_mode(tmp_path):
    first = _normalized_settings(
        {"neqti": {"interpolation": "softcore_linear"}}
    )
    _ensure_switch_protocol(tmp_path, first)
    changed = _normalized_settings(
        {
            "neqti": {
                "interpolation": "softcore_linear",
                "softcore": {"long_range_correction": "endpoint_correction"},
            }
        }
    )
    try:
        _ensure_switch_protocol(tmp_path, changed)
    except CovalentWorkflowError as exc:
        assert "different switching protocol" in str(exc)
    else:
        raise AssertionError("changed LRC mode was accepted for resume")


def test_endpoint_lrc_protocol_records_stable_correction_version():
    config = _normalized_settings(
        {
            "neqti": {
                "interpolation": "softcore_linear",
                "softcore": {"long_range_correction": "endpoint_correction"},
            }
        }
    )
    correction = _switch_protocol(config)["softcore_endpoint_correction"]
    assert correction["version"] == 2
    assert correction["evaluation_platform"] == "CPU"
    assert correction["evaluation"] == "precomputed_per_endpoint_and_switch_volume"


def test_softcore_rejects_different_endpoint_total_charge():
    config = _normalized_settings({"neqti": {"interpolation": "softcore_linear"}})
    left = type("Parameters", (), {"charges_e": [0.2, -0.2]})()
    right = type("Parameters", (), {"charges_e": [0.2, 0.8]})()
    try:
        _validate_softcore_endpoint_charge(config, left, right)
    except CovalentWorkflowError as exc:
        assert "equal endpoint total charge" in str(exc)
    else:
        raise AssertionError("charge-changing softcore edge was accepted")
from atom_openmm.rbfe_workflow import plan_workflow, validate_workflow


def _write_fixture(tmp_path, *, decorrelation_steps=1000):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("END\n")
    ligand_dir = tmp_path / "ligands" / "A"
    ligand_dir.mkdir(parents=True)
    (ligand_dir / "product.sdf").write_text("test\n")
    dataset = {
        "schema_version": 1,
        "receptor": "receptor.pdb",
        "assay_temperature_k": 298.15,
        "covalent_residue": {"name": "CYS", "id": 147, "sulfur_atom": "SG"},
        "ligands": [
            {"ligand_id": "A", "capped_product_sdf": "ligands/A/product.sdf"},
            {"ligand_id": "B", "capped_product_sdf": "ligands/A/product.sdf"},
        ],
    }
    (tmp_path / "dataset.yaml").write_text(yaml.safe_dump(dataset))
    workflow = {
        "workflow": {
            "type": "rbfe",
            "mode": "covalent",
            "dataset": "dataset.yaml",
            "workdir": "run",
            "pairs": [{"ligand_a": "A", "ligand_b": "B"}],
            "neqti": {
                "decorrelation_steps": decorrelation_steps,
                "rest2": {
                    "enabled": True,
                    "effective_temperatures_k": [300.0, 400.0],
                    "exchange_interval_steps": 500,
                },
            },
        }
    }
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(workflow))
    return path


def _test_covalent_mode_routes_through_atom_rbfe(tmp_path):
    path = _write_fixture(tmp_path)
    assert validate_covalent_workflow(path)
    assert validate_workflow(path)
    direct = plan_covalent_workflow(path)
    routed = plan_workflow(path)
    assert routed == direct
    assert routed["mode"] == "covalent"
    assert routed["pairs"][0]["ligand_a"] == "A"


def _test_covalent_validation_rejects_incompatible_rest2_steps(tmp_path):
    path = _write_fixture(tmp_path, decorrelation_steps=750)
    try:
        validate_covalent_workflow(path)
    except CovalentWorkflowError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("incompatible REST2 steps were accepted")
