import pytest
import yaml
from pathlib import Path


def _write_minimal_workflow(tmp_path):
    receptor_dir = tmp_path / "receptor"
    ligands_dir = tmp_path / "ligands"
    receptor_dir.mkdir()
    ligands_dir.mkdir()
    receptor = receptor_dir / "cdk2.pdb"
    lig1 = ligands_dir / "H1Q.sdf"
    lig2 = ligands_dir / "H1R.sdf"
    receptor.write_text("RECEPTOR\n")
    lig1.write_text("LIG1\n")
    lig2.write_text("LIG2\n")

    workflow = {
        "workflow": {
            "type": "rbfe",
            "mode": "small_molecule",
            "workdir": "complexes",
            "receptor": "receptor/cdk2.pdb",
            "ligands_dir": "ligands",
            "pairs": [["H1Q", "H1R"]],
            "reference_ligand": "H1Q",
            "reference_alignment_atoms": [1, 2, 3],
            "alignments_out": "alignments.yaml",
            "setup": {
                "protein_forcefield": ["amber14-all.xml"],
                "solvent_forcefield": ["amber14/tip3p.xml"],
                "ligand_forcefield": "espaloma-0.3.2",
                "ligand_charge_model": "nn",
            },
            "prepare_only": True,
        },
        "atom_options": {
            "TEMPERATURES": [310.0],
            "LAMBDAS": [0.0, 0.5, 1.0],
            "DIRECTION": [1, 1, -1],
            "INTERMEDIATE": [0, 1, 0],
            "LAMBDA1": [0.0, 0.5, 0.0],
            "LAMBDA2": [0.0, 0.5, 0.0],
            "ALPHA": [0.1, 0.1, 0.1],
            "U0": [110.0, 110.0, 110.0],
            "W0COEFF": [0, 0, 0],
            "MAX_SAMPLES": 1,
        },
    }
    config_file = tmp_path / "workflow.yaml"
    config_file.write_text(yaml.dump(workflow))
    return config_file


def _test_load_workflow_config_validates_shape(tmp_path):
    from atom_openmm.rbfe_workflow import build_small_molecule_plan, load_workflow_config

    config_file = _write_minimal_workflow(tmp_path)
    config = load_workflow_config(config_file)
    plan = build_small_molecule_plan(config)

    assert plan["receptor_file"] == (tmp_path / "receptor" / "cdk2.pdb").resolve()
    assert plan["workdir"] == (tmp_path / "complexes").resolve()
    assert plan["pairs"][0]["jobname"] == "cdk2-H1Q-H1R"
    assert plan["pairs"][0]["lig1_file"] == (tmp_path / "ligands" / "H1Q.sdf").resolve()
    assert plan["pairs"][0]["lig2_file"] == (tmp_path / "ligands" / "H1R.sdf").resolve()


def _test_prepare_only_writes_final_pair_yaml(tmp_path, monkeypatch):
    from atom_openmm import rbfe_workflow

    config_file = _write_minimal_workflow(tmp_path)

    monkeypatch.setattr(
        rbfe_workflow,
        "get_alignment_atoms",
        lambda ref_lig_file, ref_atoms, lig_files: {
            "H1Q": {"align_atom_ids": [1, 2, 3], "N_atoms": 3},
            "H1R": {"align_atom_ids": [1, 2, 3], "N_atoms": 3},
        },
    )
    monkeypatch.setattr(rbfe_workflow, "calc_displ_vec", lambda receptor, ligand: [22.0, 22.0, 22.0])

    def fake_setup(receptor_file, lig1_file, lig2_file, ff_json_file, options, setup_options):
        assert receptor_file.name == "cdk2.pdb"
        assert lig1_file.name == "H1Q.sdf"
        assert lig2_file.name == "H1R.sdf"
        assert ff_json_file.name == "ff.json"
        assert setup_options == {
            "proteinforcefield": ["amber14-all.xml"],
            "solventforcefield": ["amber14/tip3p.xml"],
            "ligandforcefield": "espaloma-0.3.2",
            "template_generator_kwargs": {"charge_method": "nn"},
        }
        open(options["BASENAME"] + ".pdb", "w").write("generated\n")

    def fake_derive(options):
        options.update(
            {
                "LIGAND1_ATOMS": [0, 1, 2],
                "LIGAND2_ATOMS": [3, 4, 5],
                "LIGAND1_VAR_ATOMS": [0, 1, 2],
                "LIGAND2_VAR_ATOMS": [3, 4, 5],
                "LIGAND1_ATTACH_ATOM": 0,
                "LIGAND2_ATTACH_ATOM": 3,
                "LIGAND1_CM_ATOMS": [0],
                "LIGAND2_CM_ATOMS": [3],
                "RCPT_CM_ATOMS": [10, 11, 12],
                "RCPT_FRAME_ATOMS_O": [10, 11, 12],
                "RCPT_FRAME_ATOMS_Z": [13, 14, 15],
                "RCPT_FRAME_ATOMS_Y": [16, 17, 18],
                "LIGOFFSET": [1.0, 2.0, 3.0],
                "POS_RESTRAINED_ATOMS": [10, 11, 12],
                "EXCLUSION_POT_MOL1_INDEXES": [20, 21],
                "EXCLUSION_POT_MOL2_INDEXES": [22, 23],
            }
        )

    monkeypatch.setattr(rbfe_workflow, "setup_small_molecule_system", fake_setup)
    monkeypatch.setattr(rbfe_workflow, "derive_small_molecule_options", fake_derive)

    results = rbfe_workflow.run_rbfe_workflow(config_file)

    jobdir = tmp_path / "complexes" / "cdk2-H1Q-H1R"
    alignments_yaml = tmp_path / "complexes" / "alignments.yaml"
    final_yaml = jobdir / "cdk2-H1Q-H1R.yaml"
    assert results == [
        {
            "jobname": "cdk2-H1Q-H1R",
            "status": "prepared",
            "workdir": str(jobdir.resolve()),
        }
    ]
    assert alignments_yaml.exists()
    assert not (tmp_path / "ligands" / "alignments.yaml").exists()
    assert final_yaml.exists()

    options = yaml.safe_load(final_yaml.read_text())
    assert options["BASENAME"] == "cdk2-H1Q-H1R"
    assert options["WORKDIR"] == str(jobdir.resolve())
    assert options["LIGAND_FORCE_FIELD"] == "espaloma-0.3.2"
    assert options["ALIGN_LIGAND1_REF_ATOMS"] == [0, 1, 2]
    assert options["ALIGN_LIGAND2_REF_ATOMS"] == [0, 1, 2]
    assert options["DISPLACEMENT"] == [22.0, 22.0, 22.0]
    assert options["LIGAND1_ATOMS"] == [0, 1, 2]


def _test_normalize_setup_options_maps_charge_models():
    from atom_openmm.rbfe_workflow import normalize_setup_options

    workflow = {
        "setup": {
            "ligand_forcefield": "espaloma-0.3.2",
            "ligand_charge_model": "am1-bcc",
        }
    }
    setup = normalize_setup_options(workflow, {})
    assert setup["ligandforcefield"] == "espaloma-0.3.2"
    assert setup["template_generator_kwargs"] == {"charge_method": "am1-bcc"}

    workflow = {
        "setup": {
            "ligand_forcefield": "gaff-2.2.20",
            "ligand_charge_model": "am1-bcc",
        }
    }
    setup = normalize_setup_options(workflow, {})
    assert setup["ligandforcefield"] == "gaff-2.2.20"
    assert setup["template_generator_kwargs"] is None

    workflow = {
        "setup": {
            "solvent_forcefield": ["amber19/opc.xml"],
            "solvent_model": "tip4pew",
        }
    }
    setup = normalize_setup_options(workflow, {})
    assert setup["solventforcefield"] == ["amber19/opc.xml"]
    assert setup["solvent_model"] == "tip4pew"


def _test_normalize_setup_options_rejects_unsupported_charge_models():
    from atom_openmm.rbfe_workflow import WorkflowConfigError, normalize_setup_options

    with pytest.raises(WorkflowConfigError, match="requires an espaloma"):
        normalize_setup_options(
            {"setup": {"ligand_forcefield": "gaff-2.2.20", "ligand_charge_model": "nn"}},
            {},
        )

    with pytest.raises(WorkflowConfigError, match="not exposed independently for OpenFF"):
        normalize_setup_options(
            {"setup": {"ligand_forcefield": "openff-2.3.0", "ligand_charge_model": "am1-bcc"}},
            {},
        )


def _test_explicit_solvent_adds_extra_particles_before_create_system():
    source = Path("atom_openmm/make_atm_system_from_rcpt_lig.py").read_text()
    add_solvent = source.index("modeller.addSolvent")
    first_add_extra_particles = source.index("modeller.addExtraParticles")
    second_add_extra_particles = source.index("modeller.addExtraParticles", add_solvent)
    create_system = source.index("forcefield.createSystem", second_add_extra_particles)

    assert "model=solvent_model" in source
    assert first_add_extra_particles < add_solvent < second_add_extra_particles < create_system


def _test_solvent_model_inference_uses_four_site_packing_for_opc():
    from atom_openmm.make_atm_system_from_rcpt_lig import _infer_solvent_model

    assert _infer_solvent_model(["amber19/opc.xml"]) == "tip4pew"
    assert _infer_solvent_model(["amber14/tip3p.xml"]) == "tip3p"
