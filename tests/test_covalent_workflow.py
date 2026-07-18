from pathlib import Path

import yaml

from atom_openmm.covalent_workflow import (
    CovalentWorkflowError,
    plan_covalent_workflow,
    validate_covalent_workflow,
)
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
