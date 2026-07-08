import argparse
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml
from openmm import Vec3
from openmm.app import PDBFile
from openmm.unit import angstrom, nanometer

from atom_openmm.rbfe_production import rbfe_production
from atom_openmm.rbfe_structprep import rbfe_structprep
from atom_openmm.equilibration import normalize_equilibration_protocol
from atom_openmm.utils.AtomUtils import (
    calc_displ_vec,
    cm_from_indexes,
    get_alignment_atoms,
    get_indexes_from_query,
    get_indexes_from_residue,
    get_selected_principal_groups,
)


class WorkflowConfigError(ValueError):
    pass


@contextmanager
def _pushd(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolve_path(path, base_dir):
    p = Path(path)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def _require_mapping(value, name):
    if not isinstance(value, dict):
        raise WorkflowConfigError(f"{name} must be a mapping")
    return value


def _as_list(value, name):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise WorkflowConfigError(f"{name} must be a string or list of strings")


def _ligand_forcefield_family(ligand_forcefield):
    if ligand_forcefield.startswith("gaff"):
        return "gaff"
    if ligand_forcefield.startswith("openff"):
        return "openff"
    if ligand_forcefield.startswith("espaloma"):
        return "espaloma"
    raise WorkflowConfigError(f"unsupported ligand force field: {ligand_forcefield}")


def normalize_setup_options(workflow, atom_options):
    setup = workflow.get("setup", {})
    if setup is None:
        setup = {}
    _require_mapping(setup, "workflow.setup")

    ligand_forcefield = setup.get("ligand_forcefield", atom_options.get("LIGAND_FORCE_FIELD", "openff-2.0.0"))
    if not isinstance(ligand_forcefield, str):
        raise WorkflowConfigError("workflow.setup.ligand_forcefield must be a string")
    ligand_family = _ligand_forcefield_family(ligand_forcefield)

    template_generator_kwargs = deepcopy(setup.get("template_generator_kwargs", {}))
    if template_generator_kwargs is None:
        template_generator_kwargs = {}
    _require_mapping(template_generator_kwargs, "workflow.setup.template_generator_kwargs")

    charge_model = setup.get("ligand_charge_model")
    if charge_model is not None:
        if not isinstance(charge_model, str):
            raise WorkflowConfigError("workflow.setup.ligand_charge_model must be a string")
        if charge_model == "nn":
            if ligand_family != "espaloma":
                raise WorkflowConfigError("workflow.setup.ligand_charge_model='nn' requires an espaloma ligand_forcefield")
            template_generator_kwargs["charge_method"] = "nn"
        elif charge_model == "am1-bcc":
            if ligand_family == "openff":
                raise WorkflowConfigError(
                    "workflow.setup.ligand_charge_model='am1-bcc' is not exposed independently for OpenFF"
                )
            if ligand_family == "espaloma":
                template_generator_kwargs["charge_method"] = "am1-bcc"
        else:
            raise WorkflowConfigError("workflow.setup.ligand_charge_model must be 'nn' or 'am1-bcc'")

    normalized = {
        "ligandforcefield": ligand_forcefield,
        "template_generator_kwargs": template_generator_kwargs or None,
    }
    if "protein_forcefield" in setup:
        normalized["proteinforcefield"] = _as_list(setup["protein_forcefield"], "workflow.setup.protein_forcefield")
    if "solvent_forcefield" in setup:
        normalized["solventforcefield"] = _as_list(setup["solvent_forcefield"], "workflow.setup.solvent_forcefield")
    if "solvent_model" in setup:
        solvent_model = setup["solvent_model"]
        if not isinstance(solvent_model, str):
            raise WorkflowConfigError("workflow.setup.solvent_model must be a string")
        normalized["solvent_model"] = solvent_model
    return normalized


def load_workflow_config(config_file):
    config_path = Path(config_file).resolve()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _require_mapping(config, "workflow config")
    workflow = _require_mapping(config.get("workflow"), "workflow")
    atom_options = _require_mapping(config.get("atom_options"), "atom_options")

    if workflow.get("type") != "rbfe":
        raise WorkflowConfigError("workflow.type must be 'rbfe'")
    if workflow.get("mode", "small_molecule") != "small_molecule":
        raise WorkflowConfigError("only workflow.mode='small_molecule' is supported")

    if "receptor" not in workflow:
        raise WorkflowConfigError("workflow.receptor is required")
    if "pairs" not in workflow:
        raise WorkflowConfigError("workflow.pairs is required")

    pairs = workflow["pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise WorkflowConfigError("workflow.pairs must be a non-empty list")
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise WorkflowConfigError("each workflow.pairs entry must contain two ligand names")

    return {
        "config_path": config_path,
        "base_dir": config_path.parent,
        "workflow": workflow,
        "atom_options": atom_options,
    }


def _ligand_path(ligand, ligands_dir):
    candidate = Path(ligand)
    if candidate.suffix:
        return _resolve_path(candidate, ligands_dir)
    return (ligands_dir / f"{ligand}.sdf").resolve()


def _ligand_name(ligand):
    return Path(str(ligand)).stem


def _validate_file(path, label):
    if not path.exists():
        raise WorkflowConfigError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise WorkflowConfigError(f"{label} is not a file: {path}")


def build_small_molecule_plan(config):
    base_dir = config["base_dir"]
    workflow = config["workflow"]

    receptor_file = _resolve_path(workflow["receptor"], base_dir)
    _validate_file(receptor_file, "workflow.receptor")

    ligands_dir = _resolve_path(workflow.get("ligands_dir", "."), base_dir)
    if not ligands_dir.is_dir():
        raise WorkflowConfigError(f"workflow.ligands_dir is not a directory: {ligands_dir}")

    workdir = _resolve_path(workflow.get("workdir", "complexes"), base_dir)
    job_prefix = workflow.get("job_prefix", receptor_file.stem)

    pair_plans = []
    for pair in workflow["pairs"]:
        lig1_name = _ligand_name(pair[0])
        lig2_name = _ligand_name(pair[1])
        lig1_file = _ligand_path(pair[0], ligands_dir)
        lig2_file = _ligand_path(pair[1], ligands_dir)
        _validate_file(lig1_file, f"ligand {lig1_name}")
        _validate_file(lig2_file, f"ligand {lig2_name}")
        jobname = f"{job_prefix}-{lig1_name}-{lig2_name}"
        pair_plans.append(
            {
                "jobname": jobname,
                "lig1_name": lig1_name,
                "lig2_name": lig2_name,
                "lig1_file": lig1_file,
                "lig2_file": lig2_file,
                "jobdir": workdir / jobname,
            }
        )

    return {
        "receptor_file": receptor_file,
        "ligands_dir": ligands_dir,
        "workdir": workdir,
        "pairs": pair_plans,
    }


def load_or_generate_alignments(workflow, plan):
    if workflow.get("alignments"):
        alignments_file = _resolve_path(workflow["alignments"], plan["ligands_dir"])
        _validate_file(alignments_file, "workflow.alignments")
        with open(alignments_file, "r") as f:
            return yaml.safe_load(f)

    ref_ligand = workflow.get("reference_ligand")
    ref_atoms = workflow.get("reference_alignment_atoms")
    if ref_ligand is None or ref_atoms is None:
        raise WorkflowConfigError(
            "workflow.reference_ligand and workflow.reference_alignment_atoms are required "
            "when workflow.alignments is not provided"
        )
    if not isinstance(ref_atoms, list) or len(ref_atoms) != 3:
        raise WorkflowConfigError("workflow.reference_alignment_atoms must contain three atom ids")

    ref_lig_file = _ligand_path(ref_ligand, plan["ligands_dir"])
    _validate_file(ref_lig_file, "workflow.reference_ligand")
    lig_files = sorted({p["lig1_file"] for p in plan["pairs"]} | {p["lig2_file"] for p in plan["pairs"]})
    alignments = get_alignment_atoms(str(ref_lig_file), [int(i) for i in ref_atoms], [str(p) for p in lig_files])

    alignments_out = workflow.get("alignments_out")
    if alignments_out:
        alignments_out = _resolve_path(alignments_out, plan["workdir"])
        alignments_out.parent.mkdir(parents=True, exist_ok=True)
        with open(alignments_out, "w") as f:
            yaml.dump(alignments, f, default_flow_style=None, width=1000000, sort_keys=False)

    return alignments


def setup_small_molecule_system(receptor_file, lig1_file, lig2_file, ff_json_file, options, setup_options=None):
    from atom_openmm.make_atm_system_from_rcpt_lig import make_system

    if setup_options is None:
        setup_options = {}
    basename = options["BASENAME"]
    setup = {
        "receptorfile": str(receptor_file),
        "lig1file": str(lig1_file),
        "lig2file": str(lig2_file),
        "displacement": options["DISPLACEMENT"],
        "xmloutfile": basename + "_sys.xml",
        "pdboutfile": basename + ".pdb",
        "ffcachefile": str(ff_json_file) if ff_json_file else None,
    }
    if "HMASS" in options:
        setup["hmass"] = options["HMASS"]
    setup.update(setup_options)
    make_system(**setup)


def derive_small_molecule_options(options):
    basename = options["BASENAME"]
    pdb = PDBFile(basename + ".pdb")
    topology = pdb.topology
    positions = pdb.positions

    res1 = None
    for chain in topology.chains():
        for residue in chain.residues():
            if residue.name == "L1":
                res1 = residue
                break
    if res1 is None:
        raise WorkflowConfigError("could not find ligand residue L1 in generated system")
    ligand1_atoms = get_indexes_from_residue(res1)
    options["LIGAND1_ATOMS"] = ligand1_atoms

    res2 = None
    for chain in topology.chains():
        for residue in chain.residues():
            if residue.name == "L2":
                res2 = residue
                break
    if res2 is None:
        raise WorkflowConfigError("could not find ligand residue L2 in generated system")
    ligand2_atoms = get_indexes_from_residue(res2)
    options["LIGAND2_ATOMS"] = ligand2_atoms

    options["LIGAND1_VAR_ATOMS"] = options["LIGAND1_ATOMS"]
    options["LIGAND2_VAR_ATOMS"] = options["LIGAND2_ATOMS"]

    ligand1_ref_atoms = options["ALIGN_LIGAND1_REF_ATOMS"]
    ligand2_ref_atoms = options["ALIGN_LIGAND2_REF_ATOMS"]
    options["LIGAND1_ATTACH_ATOM"] = ligand1_atoms[ligand1_ref_atoms[0]]
    options["LIGAND2_ATTACH_ATOM"] = ligand2_atoms[ligand2_ref_atoms[0]]
    options["LIGAND1_CM_ATOMS"] = [ligand1_atoms[ligand1_ref_atoms[0]]]
    options["LIGAND2_CM_ATOMS"] = [ligand2_atoms[ligand2_ref_atoms[0]]]

    lig1cm_pos = cm_from_indexes(topology, positions, options["LIGAND1_CM_ATOMS"])
    lig2cm_pos = cm_from_indexes(topology, positions, options["LIGAND2_CM_ATOMS"])
    displ = (lig2cm_pos - lig1cm_pos).value_in_unit(angstrom)
    options["DISPLACEMENT"] = [displ.x, displ.y, displ.z]

    rcpt_chain_names = options.get("RCPT_CHAIN_NAMES", ["A"])
    rcpt_chain_query = f"atom.residue.chain.id in {rcpt_chain_names}"
    rcpt_frame_query = rcpt_chain_query + ' and atom.name == "CA"'
    rcpt_frame_indexes = get_indexes_from_query(topology, rcpt_frame_query)
    rcpt_frame = get_selected_principal_groups(topology, positions, rcpt_frame_indexes)

    options["RCPT_CM_ATOMS"] = rcpt_frame["origin"]["indices"]
    options["RCPT_FRAME_ATOMS_O"] = options["RCPT_CM_ATOMS"]
    options["RCPT_FRAME_ATOMS_Z"] = rcpt_frame["z_axis"]["indices"]
    options["RCPT_FRAME_ATOMS_Y"] = rcpt_frame["y_axis"]["indices"]

    rcpt_cm_pos = Vec3(
        float(rcpt_frame["origin"]["com"][0]),
        float(rcpt_frame["origin"]["com"][1]),
        float(rcpt_frame["origin"]["com"][2]),
    ) * nanometer
    offset = (lig1cm_pos - rcpt_cm_pos).value_in_unit(angstrom)
    options["LIGOFFSET"] = [offset.x, offset.y, offset.z]
    options["POS_RESTRAINED_ATOMS"] = options["RCPT_CM_ATOMS"]
    options["EXCLUSION_POT_MOL1_INDEXES"] = get_indexes_from_query(
        topology, f"( {rcpt_chain_query} ) and (atom.element.atomic_number != 1)"
    )
    options["EXCLUSION_POT_MOL2_INDEXES"] = get_indexes_from_residue(
        res2, query="(atom.element.atomic_number != 1)"
    )


def production_sample_count(options):
    rundir = options["WORKDIR"]
    jobname = options["BASENAME"]
    nstates = len(options["LAMBDAS"])
    datafiles = [os.path.join(rundir, f"r{i}", f"{jobname}.out") for i in range(nstates)]
    line_counts = {}
    for file_path in datafiles:
        try:
            count = 0
            with pd.read_csv(file_path, chunksize=10000, header=None, sep=None, engine="python") as reader:
                for chunk in reader:
                    count += len(chunk)
            line_counts[file_path] = count
        except Exception:
            pass
    if not line_counts:
        return None
    return min(line_counts.values())


def create_vmd_infile(options):
    template_path = Path("vmd_template.in")
    if not template_path.exists() or Path("vmd.in").exists():
        return
    replacements = {
        "<LIG1ATTACHATOM>": options["LIGAND1_ATTACH_ATOM"],
        "<LIG2ATTACHATOM>": options["LIGAND2_ATTACH_ATOM"],
    }
    content = template_path.read_text()
    for key, value in replacements.items():
        content = content.replace(key, str(value))
    Path("vmd.in").write_text(content)


def write_options_yaml(options):
    out = Path(options["BASENAME"] + ".yaml")
    with open(out, "w") as f:
        yaml.dump(options, f, default_flow_style=None, width=1000000, sort_keys=False)


def should_run_production(options):
    max_samples = options.get("MAX_SAMPLES")
    nsamples = production_sample_count(options)
    if max_samples is None or nsamples is None:
        return True
    try:
        return nsamples < int(max_samples)
    except (TypeError, ValueError):
        return True


def analyze_pair(options, workflow):
    from atom_openmm.uwham import calculate_uwham_from_rundir, create_quality_assessment_plot

    nsamples_per_replica = production_sample_count(options)
    if not nsamples_per_replica:
        return None

    discard = int(nsamples_per_replica / 3)
    ddg, ddg_std, uwham_data = calculate_uwham_from_rundir(
        options["WORKDIR"], options["BASENAME"], mintimeid=discard
    )

    df1 = uwham_data["df_leg1"]
    n1 = len(uwham_data["uwham_out_leg1"]["W"][:, 0])
    df1["W"] = uwham_data["uwham_out_leg1"]["W"][:, 0] / float(n1)
    df2 = uwham_data["df_leg2"]
    n2 = len(uwham_data["uwham_out_leg2"]["W"][:, 0])
    df2["W"] = uwham_data["uwham_out_leg2"]["W"][:, 0] / float(n2)

    if workflow.get("write_analysis_csv", False):
        df1.to_csv(options["BASENAME"] + "_leg1.csv", index=False)
        df2.to_csv(options["BASENAME"] + "_leg2.csv", index=False)
    if workflow.get("plot", True):
        fig = create_quality_assessment_plot(df1, df2)
        fig.savefig(options["BASENAME"] + ".png")

    return {
        "jobname": options["BASENAME"],
        "ddg": ddg,
        "ddg_std": ddg_std,
        "dg_leg1": uwham_data["dg_leg1"],
        "dg_leg1_std": uwham_data["dg_stderr_leg1"],
        "dg_leg2": uwham_data["dg_leg2"],
        "dg_leg2_std": uwham_data["dg_stderr_leg2"],
        "samples": uwham_data["nsamples"],
        "discarded": discard,
    }


def run_production(options, workflow):
    production_method = workflow.get("production_method", "async_re")
    if production_method == "async_re":
        if should_run_production(options):
            rbfe_production(config_file=None, options=options)
        return None
    if production_method == "neqti":
        from atom_openmm.neqti import normalize_neqti_options, run_neqti

        neqti_options = normalize_neqti_options(workflow, options)
        return run_neqti(options, neqti_options)
    raise WorkflowConfigError("workflow.production_method must be 'async_re' or 'neqti'")


def run_pair(pair_plan, workflow, atom_options, setup_options, receptor_file, alignments):
    jobdir = pair_plan["jobdir"]
    jobdir.mkdir(parents=True, exist_ok=True)

    options = deepcopy(atom_options)
    options["BASENAME"] = pair_plan["jobname"]
    options["WORKDIR"] = str(jobdir.resolve())
    options["LIGAND_FORCE_FIELD"] = setup_options["ligandforcefield"]
    production_method = workflow.get("production_method", "async_re")
    if production_method == "neqti":
        options["STRUCTPREP_MODE"] = "physical_only"
        options["NEQTI_INITIAL_STATE_FILE"] = options["BASENAME"] + "_equil.xml"
    else:
        options["STRUCTPREP_MODE"] = "async_re"
    equilibration_protocol = normalize_equilibration_protocol(workflow)
    if equilibration_protocol is not None:
        options["EQUILIBRATION_PROTOCOL"] = equilibration_protocol

    for lig_name, key in (
        (pair_plan["lig1_name"], "ALIGN_LIGAND1_REF_ATOMS"),
        (pair_plan["lig2_name"], "ALIGN_LIGAND2_REF_ATOMS"),
    ):
        if lig_name not in alignments:
            raise WorkflowConfigError(f"missing alignment atoms for ligand {lig_name}")
        options[key] = [int(i) - 1 for i in alignments[lig_name]["align_atom_ids"]]

    if not options.get("DISPLACEMENT"):
        options["DISPLACEMENT"] = list(calc_displ_vec(str(receptor_file), str(pair_plan["lig2_file"])))

    forcefield_cache = workflow.get("forcefield_cache", "ff.json")
    ff_json_file = Path(forcefield_cache)
    if not ff_json_file.is_absolute():
        ff_json_file = jobdir / ff_json_file

    with _pushd(jobdir):
        if not Path(options["BASENAME"] + ".pdb").exists():
            setup_small_molecule_system(
                receptor_file,
                pair_plan["lig1_file"],
                pair_plan["lig2_file"],
                ff_json_file,
                options,
                setup_options,
            )

        derive_small_molecule_options(options)
        write_options_yaml(options)
        create_vmd_infile(options)

        if workflow.get("prepare_only", False) or not workflow.get("run", True):
            return {"jobname": options["BASENAME"], "status": "prepared", "workdir": options["WORKDIR"]}

        prep_state = options["BASENAME"] + ("_equil.xml" if production_method == "neqti" else "_0.xml")
        if not Path(prep_state).exists():
            rbfe_structprep(config_file=None, options=options)

        production_result = run_production(options, workflow)
        if production_result is not None:
            return {"workdir": options["WORKDIR"], **production_result}

        if workflow.get("production_method", "async_re") == "async_re" and workflow.get("analyze", True):
            analysis = analyze_pair(options, workflow)
            if analysis is not None:
                return {"status": "analyzed", "workdir": options["WORKDIR"], **analysis}

        return {"jobname": options["BASENAME"], "status": "completed", "workdir": options["WORKDIR"]}


def run_rbfe_workflow(config_file):
    config = load_workflow_config(config_file)
    plan = build_small_molecule_plan(config)
    setup_options = normalize_setup_options(config["workflow"], config["atom_options"])
    alignments = load_or_generate_alignments(config["workflow"], plan)
    results = []
    for pair_plan in plan["pairs"]:
        results.append(
            run_pair(
                pair_plan,
                config["workflow"],
                config["atom_options"],
                setup_options,
                plan["receptor_file"],
                alignments,
            )
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run an AToM-OpenMM RBFE workflow from one YAML file.")
    parser.add_argument("workflow_yaml", help="High-level RBFE workflow YAML file")
    args = parser.parse_args(argv)
    results = run_rbfe_workflow(args.workflow_yaml)
    for result in results:
        line = f"{result['jobname']}: {result['status']} in {result['workdir']}"
        if result.get("ddg") is not None:
            line += f" DG = {result['ddg']:8.3f} +/- {result['ddg_std']:8.3f} kcal/mol"
        print(line)


if __name__ == "__main__":
    main()
