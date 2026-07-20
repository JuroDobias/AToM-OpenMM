from pathlib import Path

import openmm as mm
from openmm import app, unit
import yaml

from atom_openmm.covalent_workflow import (
    CovalentWorkflowError,
    _normalized_settings,
    plan_covalent_workflow,
    validate_covalent_workflow,
    _dummy_particles,
    _write_switch_pdb,
    _ensure_switch_protocol,
    _validate_softcore_endpoint_charge,
)


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
