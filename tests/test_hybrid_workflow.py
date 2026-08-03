from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem import AllChem


def _write_ligand(path, smiles):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(molecule, randomSeed=2026)
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()


def _workflow(tmp_path, ligand_b="B.sdf"):
    (tmp_path / "receptor.pdb").write_text("END\n")
    ligands = tmp_path / "ligands"
    ligands.mkdir()
    _write_ligand(ligands / "A.sdf", "CCO")
    _write_ligand(ligands / ligand_b, "CCN")
    payload = {
        "workflow": {
            "type": "rbfe",
            "chemistry": "noncovalent",
            "alchemy": {
                "model": "hybrid_topology",
                "cycle": "complex_solvent",
                "mapping": {"method": "mcs"},
            },
            "sampling": {"method": "neqti"},
            "receptor": "receptor.pdb",
            "ligands_dir": "ligands",
            "pairs": [["A.sdf", ligand_b]],
            "workdir": "run",
            "setup": {
                "ligand_forcefield": "espaloma-0.3.2",
                "ligand_charge_model": "nn",
            },
            "neqti": {
                "interpolation": "softcore_linear",
                "n_snapshots": 2,
                "softcore": {
                    "charge_steps_per_stage": 10,
                    "sterics_steps": 20,
                },
                "rest2": {"enabled": False},
            },
        }
    }
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def _test_hybrid_workflow_validates_and_plans_two_environments(tmp_path):
    from atom_openmm.rbfe_workflow import plan_workflow, validate_workflow

    path = _workflow(tmp_path)
    assert validate_workflow(path)
    plan = plan_workflow(path)
    assert plan["alchemy_model"] == "hybrid_topology"
    assert plan["thermodynamic_cycle"] == "complex_solvent"
    assert plan["pairs"][0]["environments"] == ["complex", "solvent"]


def _test_hybrid_workflow_rejects_charge_change(tmp_path):
    from atom_openmm.hybrid_workflow import HybridWorkflowError
    from atom_openmm.rbfe_workflow import validate_workflow

    path = _workflow(tmp_path, ligand_b="charged.sdf")
    _write_ligand(tmp_path / "ligands" / "charged.sdf", "C[NH3+]")
    try:
        validate_workflow(path)
    except HybridWorkflowError as exc:
        assert "equal endpoint formal charges" in str(exc)
    else:
        raise AssertionError("charge-changing hybrid edge was accepted")
