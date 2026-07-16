import pytest
import yaml
from pathlib import Path


def _write_carbon_chain_sdf(path, x_positions):
    from rdkit import Chem

    mol = Chem.RWMol()
    for _ in x_positions:
        mol.AddAtom(Chem.Atom("C"))
    for index in range(len(x_positions) - 1):
        mol.AddBond(index, index + 1, Chem.BondType.SINGLE)
    mol = mol.GetMol()
    conformer = Chem.Conformer(len(x_positions))
    for index, x in enumerate(x_positions):
        conformer.SetAtomPosition(index, (float(x), 0.0, 0.0))
    mol.AddConformer(conformer)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


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


def _test_receptor_exclusion_atoms_fall_back_when_tleap_rewrites_chain_id():
    from openmm.app import Element, Topology
    from atom_openmm.rbfe_workflow import _receptor_heavy_atom_indices

    topology = Topology()
    chain = topology.addChain("1")
    protein = topology.addResidue("ALA", chain)
    topology.addAtom("CA", Element.getBySymbol("C"), protein)
    topology.addAtom("HA", Element.getBySymbol("H"), protein)
    water = topology.addResidue("WAT", chain)
    topology.addAtom("O", Element.getBySymbol("O"), water)
    ligand = topology.addResidue("L1", chain)
    topology.addAtom("C1", Element.getBySymbol("C"), ligand)
    sodium = topology.addResidue("Na+", chain)
    topology.addAtom("Na+", Element.getBySymbol("Na"), sodium)

    assert _receptor_heavy_atom_indices(topology, ["A"]) == [0]
    assert _receptor_heavy_atom_indices(topology, ["1"]) == [0]


def _test_generate_smarts_alignments_selects_lowest_direct_rmsd_pair(tmp_path):
    from atom_openmm.rbfe_workflow import generate_smarts_alignments

    ligands_dir = tmp_path / "ligands"
    ligands_dir.mkdir()
    lig_a = ligands_dir / "A.sdf"
    lig_b = ligands_dir / "B.sdf"
    _write_carbon_chain_sdf(lig_a, [0.0, 1.0, 2.0])
    _write_carbon_chain_sdf(lig_b, [2.0, 1.0, 0.0])
    plan = {
        "pairs": [
            {
                "jobname": "test-A-B",
                "lig1_name": "A",
                "lig2_name": "B",
                "lig1_file": lig_a,
                "lig2_file": lig_b,
            }
        ]
    }

    alignments = generate_smarts_alignments(
        {"method": "smarts", "smarts": "[#6]-[#6]-[#6]", "smarts_atom_ids": [1, 2, 3]},
        plan,
    )

    pair = alignments["pairs"]["test-A-B"]
    assert alignments["schema_version"] == 2
    assert pair["ligand_a"]["align_atom_ids"] == [1, 2, 3]
    assert pair["ligand_b"]["align_atom_ids"] == [3, 2, 1]
    assert pair["selected_rmsd_a"] == pytest.approx(0.0)


def _test_generate_smarts_alignments_supports_separate_structure_files(tmp_path):
    from atom_openmm.rbfe_workflow import generate_smarts_alignments

    lig_a = tmp_path / "A-parameters.mol2"
    lig_b = tmp_path / "B-parameters.mol2"
    lig_a.write_text("parameterized input is intentionally not parsed")
    lig_b.write_text("parameterized input is intentionally not parsed")
    align_a = tmp_path / "A-alignment.sdf"
    align_b = tmp_path / "B-alignment.sdf"
    _write_carbon_chain_sdf(align_a, [0.0, 1.0, 2.0])
    _write_carbon_chain_sdf(align_b, [0.0, 1.0, 2.0])
    plan = {
        "base_dir": tmp_path,
        "pairs": [{
            "jobname": "test-A-B",
            "lig1_name": "A",
            "lig2_name": "B",
            "lig1_file": lig_a,
            "lig2_file": lig_b,
        }],
    }

    alignments = generate_smarts_alignments(
        {
            "method": "smarts",
            "smarts": "[#6]-[#6]-[#6]",
            "smarts_atom_ids": [1, 2, 3],
            "structures": {"A": "A-alignment.sdf", "B": "B-alignment.sdf"},
        },
        plan,
    )

    pair = alignments["pairs"]["test-A-B"]
    assert pair["ligand_a"]["align_atom_ids"] == [1, 2, 3]
    assert pair["ligand_b"]["align_atom_ids"] == [1, 2, 3]


def _test_pair_specific_alignments_are_converted_to_zero_based_options():
    from atom_openmm.rbfe_workflow import _alignment_atoms_for_pair

    pair_plan = {"jobname": "test-A-B", "lig1_name": "A", "lig2_name": "B"}
    alignments = {
        "schema_version": 2,
        "pairs": {
            "test-A-B": {
                "ligand_a": {"name": "A", "align_atom_ids": [4, 5, 6]},
                "ligand_b": {"name": "B", "align_atom_ids": [7, 8, 9]},
            }
        },
    }

    assert [int(i) - 1 for i in _alignment_atoms_for_pair(alignments, pair_plan, "ligand_a")] == [3, 4, 5]
    assert [int(i) - 1 for i in _alignment_atoms_for_pair(alignments, pair_plan, "ligand_b")] == [6, 7, 8]


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


def _test_object_pairs_external_metadata_and_ligand_mapping(tmp_path):
    from atom_openmm.rbfe_workflow import build_small_molecule_plan, load_workflow_config

    receptor = tmp_path / "receptor.pdb"
    mapped_dir = tmp_path / "mapped"
    mapped_dir.mkdir()
    ligand_a = mapped_dir / "A.sdf"
    ligand_b = mapped_dir / "B.sdf"
    receptor.write_text("RECEPTOR\n")
    ligand_a.write_text("A\n")
    ligand_b.write_text("B\n")
    workflow = {
        "workflow": {
            "type": "rbfe",
            "receptor": "receptor.pdb",
            "workdir": "run",
            "external_metadata": {"graph_id": 12, "edge_id": 1},
            "ligands": {"A": "mapped/A.sdf", "B": str(ligand_b)},
            "pairs": [
                {
                    "ligands": ["A", "B"],
                    "external_metadata": {"edge_id": 44, "source_microstate_a_id": 101},
                }
            ],
            "alignments": "alignments.yaml",
        },
        "atom_options": {"MAX_SAMPLES": 1},
    }
    (tmp_path / "alignments.yaml").write_text("A:\n  align_atom_ids: [1, 2, 3]\nB:\n  align_atom_ids: [1, 2, 3]\n")
    config_file = tmp_path / "workflow.yaml"
    config_file.write_text(yaml.dump(workflow))

    plan = build_small_molecule_plan(load_workflow_config(config_file))
    pair = plan["pairs"][0]

    assert pair["lig1_file"] == ligand_a.resolve()
    assert pair["lig2_file"] == ligand_b.resolve()
    assert pair["external_metadata"] == {
        "graph_id": 12,
        "edge_id": 44,
        "source_microstate_a_id": 101,
    }


def _test_validate_and_plan_only_do_not_create_workdirs(tmp_path, capsys):
    from atom_openmm import rbfe_workflow

    config_file = _write_minimal_workflow(tmp_path)
    monkeypatch_target = {
        "H1Q": {"align_atom_ids": [1, 2, 3], "N_atoms": 3},
        "H1R": {"align_atom_ids": [1, 2, 3], "N_atoms": 3},
    }

    original = rbfe_workflow.get_alignment_atoms
    rbfe_workflow.get_alignment_atoms = lambda ref_lig_file, ref_atoms, lig_files: monkeypatch_target
    try:
        assert rbfe_workflow.main(["--validate", str(config_file)]) == 0
        capsys.readouterr()
        assert not (tmp_path / "complexes").exists()
        assert not (tmp_path / "ligands" / "alignments.yaml").exists()

        assert rbfe_workflow.main(["--plan-only", str(config_file)]) == 0
        output = capsys.readouterr().out
        plan = yaml.safe_load(output)
    finally:
        rbfe_workflow.get_alignment_atoms = original

    assert plan["schema_version"] == 1
    assert plan["tool"] == "atom_openmm_rbfe"
    assert plan["pairs"][0]["jobname"] == "cdk2-H1Q-H1R"
    assert plan["pairs"][0]["expected_result"].endswith("/complexes/cdk2-H1Q-H1R/result.yaml")
    assert not (tmp_path / "complexes").exists()


def _test_analyze_only_missing_outputs_writes_failed_result(tmp_path):
    from atom_openmm import rbfe_workflow

    config_file = _write_minimal_workflow(tmp_path)

    assert rbfe_workflow.main(["--analyze-only", str(config_file)]) == 1

    result_path = tmp_path / "complexes" / "cdk2-H1Q-H1R" / "result.yaml"
    result = yaml.safe_load(result_path.read_text())
    assert result["status"] == "failed"
    assert result["error"]["stage"] == "analysis"
    assert result["error"]["type"] == "WorkflowConfigError"
    assert "prepared pair YAML does not exist" in result["error"]["message"]


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
    machine_result = yaml.safe_load((jobdir / "result.yaml").read_text())
    assert machine_result["schema_version"] == 1
    assert machine_result["status"] == "prepared"
    assert machine_result["method"] == "async_re"
    assert machine_result["convention"]["edge_direction"] == "ligand_a_to_ligand_b"
    assert machine_result["external_metadata"] == {}
    assert machine_result["progress"]["stage"] == "prepared"
    assert machine_result["inputs"]["workflow_yaml"] == str(config_file.resolve())
    assert machine_result["artifacts"]["prepared_complex"] == "cdk2-H1Q-H1R.pdb"

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


def _test_normalize_setup_options_maps_ambertools_mode():
    from atom_openmm.rbfe_workflow import normalize_setup_options

    setup = normalize_setup_options(
        {
            "setup": {
                "mode": "ambertools",
                "protein_forcefield": "leaprc.protein.ff19SB",
                "additional_forcefields": ["leaprc.phosaa14SB"],
                "ligand_forcefield": "leaprc.gaff2",
                "water_forcefield": "leaprc.water.opc",
                "solvent_box": "OPCBOX",
                "solvent_padding_a": 12,
                "neutralize": False,
            }
        },
        {},
    )

    assert setup["setup_mode"] == "ambertools"
    assert setup["ligandforcefield"] == "leaprc.gaff2"
    assert setup["ambertools"] == {
        "protein_forcefield": "leaprc.protein.ff19SB",
        "additional_forcefields": ["leaprc.phosaa14SB"],
        "ligand_forcefield": "leaprc.gaff2",
        "ligand_parameterization": "preparameterized",
        "ligand_charge_model": "bcc",
        "ligand_net_charge": 0,
        "ligand_net_charges": {},
        "water_forcefield": "leaprc.water.opc",
        "solvent_box": "OPCBOX",
        "solvent_padding_a": 12.0,
        "neutralize": False,
    }


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


def _test_write_ambertools_tleap_input_renames_ligand_residues(tmp_path):
    from atom_openmm.rbfe_workflow import write_ambertools_tleap_input

    receptor = tmp_path / "receptor.pdb"
    lig1 = tmp_path / "lig1-p.mol2"
    lig2 = tmp_path / "lig2-p.mol2"
    receptor.write_text("RECEPTOR\n")
    mol2 = "\n".join(
        [
            "@<TRIPOS>MOLECULE",
            "LIG",
            " 1 0 1 0 0",
            "SMALL",
            "bcc",
            "",
            "@<TRIPOS>ATOM",
            "      1 C1 0.0000 1.0000 2.0000 c3 1 UNL -0.100000",
            "@<TRIPOS>BOND",
        ]
    )
    lig1.write_text(mol2 + "\n")
    lig2.write_text(mol2 + "\n")
    lig1.with_suffix(".frcmod").write_text("frcmod1\n")
    lig2.with_suffix(".frcmod").write_text("frcmod2\n")

    tleap_file = tmp_path / "tleap.cmd"
    write_ambertools_tleap_input(
        receptor,
        lig1,
        lig2,
        {"BASENAME": "cdk2-lig1-lig2", "DISPLACEMENT": [1.0, 2.0, 3.0]},
        {
            "protein_forcefield": "leaprc.protein.ff14SB",
            "additional_forcefields": ["leaprc.phosaa14SB"],
            "ligand_forcefield": "leaprc.gaff2",
            "water_forcefield": "leaprc.water.tip3p",
            "solvent_box": "TIP3PBOX",
            "solvent_padding_a": 10.0,
            "neutralize": True,
        },
        tleap_file,
    )

    tleap = tleap_file.read_text()
    assert "source leaprc.protein.ff14SB" in tleap
    assert "source leaprc.phosaa14SB" in tleap
    assert "translate LIG2 { 1.000000 2.000000 3.000000 }" in tleap
    assert "addions2 MOL Na+ 0" in tleap
    assert "saveamberparm MOL cdk2-lig1-lig2.prmtop cdk2-lig1-lig2.inpcrd" in tleap
    assert " L1 " in (tmp_path / "ambertools_inputs" / "L1.mol2").read_text()
    assert " L2 " in (tmp_path / "ambertools_inputs" / "L2.mol2").read_text()
    assert (tmp_path / "ambertools_inputs" / "L1.frcmod").read_text() == "frcmod1\n"
    assert (tmp_path / "ambertools_inputs" / "L2.frcmod").read_text() == "frcmod2\n"


def _test_write_ambertools_tleap_input_can_parameterize_sdf_ligands(tmp_path, monkeypatch):
    from atom_openmm import rbfe_workflow

    receptor = tmp_path / "receptor.pdb"
    lig1 = tmp_path / "ms_1.sdf"
    lig2 = tmp_path / "ms_2.sdf"
    receptor.write_text("RECEPTOR\n")
    lig1.write_text("lig1\n")
    lig2.write_text("lig2\n")
    commands = []

    def fake_run(command, check):
        commands.append(command)
        if command[0] == "antechamber":
            output_file = Path(command[command.index("-o") + 1])
            residue_name = command[command.index("-rn") + 1]
            output_file.write_text(
                "\n".join(
                    [
                        "@<TRIPOS>MOLECULE",
                        "LIG",
                        " 1 0 1 0 0",
                        "SMALL",
                        "bcc",
                        "",
                        "@<TRIPOS>ATOM",
                        f"      1 C1 0.0000 1.0000 2.0000 c3 1 {residue_name} -0.100000",
                        "@<TRIPOS>BOND",
                    ]
                )
                + "\n"
            )
        elif command[0] == "parmchk2":
            output_file = Path(command[command.index("-o") + 1])
            output_file.write_text("frcmod\n")

    monkeypatch.setattr(rbfe_workflow.subprocess, "run", fake_run)

    tleap_file = tmp_path / "tleap.cmd"
    rbfe_workflow.write_ambertools_tleap_input(
        receptor,
        lig1,
        lig2,
        {"BASENAME": "job", "DISPLACEMENT": [1.0, 2.0, 3.0]},
        {
            "protein_forcefield": "leaprc.protein.ff14SB",
            "additional_forcefields": ["leaprc.phosaa14SB"],
            "ligand_forcefield": "leaprc.gaff2",
            "ligand_parameterization": "antechamber",
            "ligand_charge_model": "bcc",
            "ligand_net_charge": 0,
            "ligand_net_charges": {"ms_2": -1},
            "water_forcefield": "leaprc.water.tip3p",
            "solvent_box": "TIP3PBOX",
            "solvent_padding_a": 10.0,
            "neutralize": True,
        },
        tleap_file,
    )

    antechamber_commands = [command for command in commands if command[0] == "antechamber"]
    parmchk_commands = [command for command in commands if command[0] == "parmchk2"]
    assert len(antechamber_commands) == 2
    assert len(parmchk_commands) == 2
    assert antechamber_commands[0][antechamber_commands[0].index("-fi") + 1] == "sdf"
    assert antechamber_commands[0][antechamber_commands[0].index("-c") + 1] == "bcc"
    assert antechamber_commands[0][antechamber_commands[0].index("-nc") + 1] == "0"
    assert antechamber_commands[1][antechamber_commands[1].index("-nc") + 1] == "-1"
    assert "loadamberparams" in tleap_file.read_text()


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


def _test_run_production_routes_neqti(monkeypatch):
    from atom_openmm import neqti, rbfe_workflow

    options = {
        "BASENAME": "cdk2-H1Q-H1R",
        "TEMPERATURES": [310.0],
        "LAMBDAS": [0.0, 0.5, 0.5, 1.0],
        "DIRECTION": [1, 1, -1, -1],
        "INTERMEDIATE": [0, 1, 1, 0],
        "LAMBDA1": [0.0, 0.5, 0.5, 0.0],
        "LAMBDA2": [0.0, 0.5, 0.5, 0.0],
        "ALPHA": [0.1, 0.1, 0.1, 0.1],
        "U0": [110.0, 110.0, 110.0, 110.0],
        "W0COEFF": [0, 0, 0, 0],
        "UMAX": 200.0,
        "UBCORE": 100.0,
        "ACORE": 0.0625,
        "PRODUCTION_STEPS": 5,
        "MAX_SAMPLES": 1,
    }
    workflow = {
        "production_method": "neqti",
        "neqti": {
            "n_snapshots": 1,
            "switch_steps_per_segment": 2,
        },
    }

    def fake_run_neqti(received_options, received_neqti_options):
        assert received_options is options
        assert received_neqti_options["n_snapshots"] == 1
        assert received_neqti_options["switch_steps_per_segment"] == 2
        return {"jobname": "cdk2-H1Q-H1R", "status": "completed", "analysis": {"bar_dg_kcal_per_mol": 1.0}}

    monkeypatch.setattr(neqti, "run_neqti", fake_run_neqti)

    assert rbfe_workflow.run_production(options, workflow)["analysis"]["bar_dg_kcal_per_mol"] == 1.0


def _test_neqti_workflow_uses_physical_only_structprep(tmp_path, monkeypatch):
    from atom_openmm import rbfe_workflow

    config_file = _write_minimal_workflow(tmp_path)
    config = yaml.safe_load(config_file.read_text())
    config["workflow"]["prepare_only"] = False
    config["workflow"]["production_method"] = "neqti"
    config["atom_options"]["TIME_STEP"] = 0.002
    config["workflow"]["equilibration"] = {
        "pre_atm": {
            "steps": [
                {
                    "id": "min",
                    "type": "minimization",
                    "tolerance_kj_mol_nm": 10.0,
                    "max_iterations": 1,
                }
            ]
        },
        "neqti": {
            "endpoint": {
                "steps": [
                    {
                        "id": "nvt",
                        "type": "md",
                        "ensemble": "NVT",
                        "n_steps": 1,
                    }
                ]
            }
        },
    }
    config_file.write_text(yaml.dump(config))

    monkeypatch.setattr(
        rbfe_workflow,
        "get_alignment_atoms",
        lambda ref_lig_file, ref_atoms, lig_files: {
            "H1Q": {"align_atom_ids": [1, 2, 3], "N_atoms": 3},
            "H1R": {"align_atom_ids": [1, 2, 3], "N_atoms": 3},
        },
    )
    monkeypatch.setattr(rbfe_workflow, "calc_displ_vec", lambda receptor, ligand: [22.0, 22.0, 22.0])
    monkeypatch.setattr(
        rbfe_workflow,
        "setup_small_molecule_system",
        lambda receptor_file, lig1_file, lig2_file, ff_json_file, options, setup_options: Path(
            options["BASENAME"] + ".pdb"
        ).write_text("generated\n"),
    )
    monkeypatch.setattr(
        rbfe_workflow,
        "derive_small_molecule_options",
        lambda options: options.update(
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
        ),
    )

    received = {}

    def fake_structprep(config_file=None, options=None):
        received["structprep_mode"] = options["STRUCTPREP_MODE"]
        received["initial"] = options["NEQTI_INITIAL_STATE_FILE"]
        received["equilibration"] = options["EQUILIBRATION_PROTOCOL"]
        options["TIME_STEP"] = 0.001
        Path(options["BASENAME"] + "_equil.xml").write_text("<state/>")

    def fake_run_production(options, workflow, progress_callback=None):
        received["production_initial"] = options["NEQTI_INITIAL_STATE_FILE"]
        received["production_time_step"] = options["TIME_STEP"]
        return {"jobname": options["BASENAME"], "status": "completed"}

    monkeypatch.setattr(rbfe_workflow, "rbfe_structprep", fake_structprep)
    monkeypatch.setattr(rbfe_workflow, "run_production", fake_run_production)

    results = rbfe_workflow.run_rbfe_workflow(config_file)

    assert results[0]["status"] == "completed"
    assert received["structprep_mode"] == "physical_only"
    assert received["initial"] == "cdk2-H1Q-H1R_equil.xml"
    assert received["production_initial"] == "cdk2-H1Q-H1R_equil.xml"
    assert received["production_time_step"] == 0.002
    assert received["equilibration"]["neqti"]["endpoint"]["steps"][0]["id"] == "nvt"


def _test_production_restarts_retry_failed_production(monkeypatch):
    from atom_openmm import rbfe_workflow

    calls = []
    updates = []

    class FakeResultWriter:
        def update(self, *args, **kwargs):
            updates.append((args, kwargs))

    def fake_run_production(options, workflow, progress_callback=None):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("transient NaN")
        return {"status": "completed", "analysis": {"bar_dg_kcal_per_mol": 0.0}}

    monkeypatch.setattr(rbfe_workflow, "run_production", fake_run_production)

    result = rbfe_workflow.run_production_with_restarts(
        {},
        {"production_restarts": {"enabled": True, "max_attempts": 3}},
        FakeResultWriter(),
        "production",
    )

    assert len(calls) == 2
    assert result["status"] == "completed"
    assert updates[0][0] == ("partial",)
    assert updates[0][1]["error"]["restart_attempt"] == 1
    assert "retrying with resume state" in updates[0][1]["warning"]


def _test_production_restarts_stop_at_cap(monkeypatch):
    from atom_openmm import rbfe_workflow

    calls = []

    class FakeResultWriter:
        def update(self, *args, **kwargs):
            pass

    def fake_run_production(options, workflow, progress_callback=None):
        calls.append(1)
        raise ValueError("persistent NaN")

    monkeypatch.setattr(rbfe_workflow, "run_production", fake_run_production)

    with pytest.raises(rbfe_workflow.ProductionRestartExhaustedError, match="persistent NaN") as excinfo:
        rbfe_workflow.run_production_with_restarts(
            {},
            {"production_restarts": {"enabled": True, "max_attempts": 2}},
            FakeResultWriter(),
            "production",
        )

    assert len(calls) == 2
    assert isinstance(excinfo.value.original, ValueError)
    assert excinfo.value.attempts == 2


def _test_production_restarts_default_disabled(monkeypatch):
    from atom_openmm import rbfe_workflow

    calls = []

    class FakeResultWriter:
        def update(self, *args, **kwargs):
            pass

    def fake_run_production(options, workflow, progress_callback=None):
        calls.append(1)
        raise ValueError("no retry")

    monkeypatch.setattr(rbfe_workflow, "run_production", fake_run_production)

    with pytest.raises(ValueError, match="no retry"):
        rbfe_workflow.run_production_with_restarts({}, {}, FakeResultWriter(), "production")

    assert len(calls) == 1


def _test_state_xml_sanitizer_removes_transient_context_parameters(tmp_path):
    from atom_openmm.equilibration import _strip_integrator_parameters

    state = tmp_path / "state.xml"
    state.write_text(
        """<?xml version='1.0' encoding='UTF-8'?>
<State>
  <Parameters Lambda1=".5" k="836.8" tol=".05" MonteCarloPressure="1"/>
  <Positions/>
  <IntegratorParameters version="1">
    <Parameter name="unused" value="1"/>
  </IntegratorParameters>
</State>
"""
    )

    _strip_integrator_parameters(state)

    text = state.read_text()
    assert "IntegratorParameters" not in text
    assert 'k="' not in text
    assert 'tol="' not in text
    assert 'Lambda1="' not in text
    assert "<Positions" in text
