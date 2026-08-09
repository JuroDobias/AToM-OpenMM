import argparse
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml
from openmm import Vec3
from openmm.app import PDBFile
from openmm.unit import angstrom, nanometer
from rdkit import Chem

from atom_openmm.rbfe_production import rbfe_production
from atom_openmm.rbfe_result import RBFEResultWriter
from atom_openmm.rbfe_structprep import rbfe_structprep
from atom_openmm.workflow_schema import WorkflowAxesError, normalize_workflow_axes
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


class ProductionRestartExhaustedError(RuntimeError):
    def __init__(self, original, attempts):
        self.original = original
        self.attempts = attempts
        super().__init__(str(original))


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


def _workflow_axes(config_file):
    config_path = Path(config_file).resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("workflow"), dict):
        raise WorkflowConfigError("workflow config must contain a workflow mapping")
    try:
        return normalize_workflow_axes(config["workflow"])
    except WorkflowAxesError as exc:
        raise WorkflowConfigError(str(exc)) from exc


def _require_mapping(value, name):
    if not isinstance(value, dict):
        raise WorkflowConfigError(f"{name} must be a mapping")
    return value


def normalize_production_restarts(workflow):
    cfg = workflow.get("production_restarts") or {}
    if not isinstance(cfg, dict):
        raise WorkflowConfigError("workflow.production_restarts must be a mapping")
    enabled = bool(cfg.get("enabled", False))
    max_attempts = int(cfg.get("max_attempts", 1 if not enabled else 2))
    if max_attempts < 1:
        raise WorkflowConfigError("workflow.production_restarts.max_attempts must be positive")
    return {"enabled": enabled, "max_attempts": max_attempts}


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

    setup_mode = setup.get("mode", "openmmforcefields")
    if not isinstance(setup_mode, str):
        raise WorkflowConfigError("workflow.setup.mode must be a string")
    if setup_mode == "ambertools":
        ambertools = {
            "protein_forcefield": setup.get("protein_forcefield", "leaprc.protein.ff14SB"),
            "additional_forcefields": setup.get("additional_forcefields", []),
            "ligand_forcefield": setup.get("ligand_forcefield", "leaprc.gaff2"),
            "ligand_parameterization": setup.get("ligand_parameterization", "preparameterized"),
            "ligand_charge_model": setup.get("ligand_charge_model", "bcc"),
            "ligand_net_charge": setup.get("ligand_net_charge", 0),
            "ligand_net_charges": setup.get("ligand_net_charges", {}),
            "water_forcefield": setup.get("water_forcefield", "leaprc.water.tip3p"),
            "solvent_box": setup.get("solvent_box", "TIP3PBOX"),
            "solvent_padding_a": setup.get("solvent_padding_a", 10.0),
            "neutralize": setup.get("neutralize", True),
        }
        for key in (
            "protein_forcefield",
            "ligand_forcefield",
            "ligand_parameterization",
            "ligand_charge_model",
            "water_forcefield",
            "solvent_box",
        ):
            if not isinstance(ambertools[key], str):
                raise WorkflowConfigError(f"workflow.setup.{key} must be a string for setup.mode='ambertools'")
        if ambertools["ligand_parameterization"] not in ("preparameterized", "antechamber"):
            raise WorkflowConfigError(
                "workflow.setup.ligand_parameterization must be 'preparameterized' or 'antechamber'"
            )
        if ambertools["ligand_charge_model"] not in ("bcc", "gas"):
            raise WorkflowConfigError("workflow.setup.ligand_charge_model must be 'bcc' or 'gas' for AmberTools setup")
        ambertools["additional_forcefields"] = _as_list(
            ambertools["additional_forcefields"],
            "workflow.setup.additional_forcefields",
        )
        if ambertools["ligand_net_charges"] is None:
            ambertools["ligand_net_charges"] = {}
        _require_mapping(ambertools["ligand_net_charges"], "workflow.setup.ligand_net_charges")
        try:
            ambertools["ligand_net_charge"] = int(ambertools["ligand_net_charge"])
            ambertools["ligand_net_charges"] = {
                str(name): int(charge) for name, charge in ambertools["ligand_net_charges"].items()
            }
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError("workflow.setup ligand net charges must be integers") from exc
        try:
            ambertools["solvent_padding_a"] = float(ambertools["solvent_padding_a"])
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError("workflow.setup.solvent_padding_a must be a number") from exc
        if not isinstance(ambertools["neutralize"], bool):
            raise WorkflowConfigError("workflow.setup.neutralize must be a boolean")
        return {
            "setup_mode": "ambertools",
            "ligandforcefield": ambertools["ligand_forcefield"],
            "ambertools": ambertools,
        }
    if setup_mode != "openmmforcefields":
        raise WorkflowConfigError("workflow.setup.mode must be 'openmmforcefields' or 'ambertools'")

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
    atom_options = _require_mapping(config.get("atom_options", {}), "atom_options")

    if workflow.get("type") != "rbfe":
        raise WorkflowConfigError("workflow.type must be 'rbfe'")
    try:
        axes = normalize_workflow_axes(workflow)
    except WorkflowAxesError as exc:
        raise WorkflowConfigError(str(exc)) from exc
    if axes.chemistry != "noncovalent":
        raise WorkflowConfigError("small-molecule loader requires noncovalent chemistry")

    if "receptor" not in workflow:
        raise WorkflowConfigError("workflow.receptor is required")
    if "pairs" not in workflow:
        raise WorkflowConfigError("workflow.pairs is required")

    pairs = workflow["pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise WorkflowConfigError("workflow.pairs must be a non-empty list")
    for pair in pairs:
        _normalize_pair_entry(pair)

    if "external_metadata" in workflow and not isinstance(workflow["external_metadata"], dict):
        raise WorkflowConfigError("workflow.external_metadata must be a mapping")
    if "ligands" in workflow and not isinstance(workflow["ligands"], dict):
        raise WorkflowConfigError("workflow.ligands must be a mapping")

    return {
        "config_path": config_path,
        "base_dir": config_path.parent,
        "workflow": workflow,
        "atom_options": atom_options,
    }


def _normalize_pair_entry(pair):
    if isinstance(pair, dict):
        ligands = pair.get("ligands")
        if not isinstance(ligands, (list, tuple)) or len(ligands) != 2:
            raise WorkflowConfigError("object workflow.pairs entries must contain ligands: [ligand_a, ligand_b]")
        metadata = pair.get("external_metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise WorkflowConfigError("pair external_metadata must be a mapping")
        return ligands[0], ligands[1], metadata
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return pair[0], pair[1], {}
    raise WorkflowConfigError("each workflow.pairs entry must contain two ligand names")


def _ligand_path(ligand, ligands_dir, ligand_map=None, base_dir=None):
    candidate = Path(str(ligand))
    if candidate.suffix:
        return _resolve_path(candidate, ligands_dir)
    if ligand_map and str(ligand) in ligand_map:
        return _resolve_path(ligand_map[str(ligand)], base_dir or ligands_dir)
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
    ligand_map = workflow.get("ligands", {}) or {}
    global_metadata = workflow.get("external_metadata", {}) or {}

    pair_plans = []
    total_pairs = len(workflow["pairs"])
    for index, pair in enumerate(workflow["pairs"], start=1):
        lig1, lig2, pair_metadata = _normalize_pair_entry(pair)
        lig1_name = _ligand_name(lig1)
        lig2_name = _ligand_name(lig2)
        lig1_file = _ligand_path(lig1, ligands_dir, ligand_map, base_dir)
        lig2_file = _ligand_path(lig2, ligands_dir, ligand_map, base_dir)
        _validate_file(lig1_file, f"ligand {lig1_name}")
        _validate_file(lig2_file, f"ligand {lig2_name}")
        jobname = f"{job_prefix}-{lig1_name}-{lig2_name}"
        external_metadata = deepcopy(global_metadata)
        external_metadata.update(pair_metadata)
        pair_plans.append(
            {
                "jobname": jobname,
                "lig1_name": lig1_name,
                "lig2_name": lig2_name,
                "lig1_file": lig1_file,
                "lig2_file": lig2_file,
                "jobdir": workdir / jobname,
                "external_metadata": external_metadata,
                "pair_index": index,
                "total_pairs": total_pairs,
            }
        )

    return {
        "receptor_file": receptor_file,
        "ligands_dir": ligands_dir,
        "ligand_map": ligand_map,
        "base_dir": base_dir,
        "workdir": workdir,
        "pairs": pair_plans,
    }


def build_execution_plan(config):
    plan = build_small_molecule_plan(config)
    return {
        "schema_version": 1,
        "tool": "atom_openmm_rbfe",
        "workflow_yaml": str(config["config_path"]),
        "workdir": str(plan["workdir"]),
        "pairs": [
            {
                "jobname": pair["jobname"],
                "ligand_a": pair["lig1_name"],
                "ligand_b": pair["lig2_name"],
                "ligand_a_file": str(pair["lig1_file"]),
                "ligand_b_file": str(pair["lig2_file"]),
                "pair_workdir": str(pair["jobdir"].resolve()),
                "expected_result": str((pair["jobdir"] / "result.yaml").resolve()),
                "external_metadata": pair.get("external_metadata") or {},
            }
            for pair in plan["pairs"]
        ],
    }


def load_or_generate_alignments(workflow, plan, write_generated=True):
    if workflow.get("alignments"):
        alignments_file = _resolve_path(workflow["alignments"], plan["ligands_dir"])
        _validate_file(alignments_file, "workflow.alignments")
        with open(alignments_file, "r") as f:
            return yaml.safe_load(f)

    if workflow.get("alignment"):
        alignments = generate_smarts_alignments(workflow["alignment"], plan)
        alignments_out = workflow.get("alignments_out")
        if write_generated and alignments_out:
            alignments_out = _resolve_path(alignments_out, plan["workdir"])
            alignments_out.parent.mkdir(parents=True, exist_ok=True)
            with open(alignments_out, "w") as f:
                yaml.dump(alignments, f, default_flow_style=None, width=1000000, sort_keys=False)
        return alignments

    ref_ligand = workflow.get("reference_ligand")
    ref_atoms = workflow.get("reference_alignment_atoms")
    if ref_ligand is None or ref_atoms is None:
        raise WorkflowConfigError(
            "workflow.reference_ligand and workflow.reference_alignment_atoms are required "
            "when workflow.alignments is not provided"
        )
    if not isinstance(ref_atoms, list) or len(ref_atoms) != 3:
        raise WorkflowConfigError("workflow.reference_alignment_atoms must contain three atom ids")

    ref_lig_file = _ligand_path(ref_ligand, plan["ligands_dir"], plan.get("ligand_map"), plan.get("base_dir"))
    _validate_file(ref_lig_file, "workflow.reference_ligand")
    lig_files = sorted({p["lig1_file"] for p in plan["pairs"]} | {p["lig2_file"] for p in plan["pairs"]})
    alignments = get_alignment_atoms(str(ref_lig_file), [int(i) for i in ref_atoms], [str(p) for p in lig_files])

    alignments_out = workflow.get("alignments_out")
    if write_generated and alignments_out:
        alignments_out = _resolve_path(alignments_out, plan["workdir"])
        alignments_out.parent.mkdir(parents=True, exist_ok=True)
        with open(alignments_out, "w") as f:
            yaml.dump(alignments, f, default_flow_style=None, width=1000000, sort_keys=False)

    return alignments


def _load_alignment_mol(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mol = supplier[0] if supplier and len(supplier) else None
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), removeHs=False)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False)
    else:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mol = supplier[0] if supplier and len(supplier) else None
    if mol is None:
        raise WorkflowConfigError(f"could not read ligand for SMARTS alignment: {path}")
    if mol.GetNumConformers() == 0:
        raise WorkflowConfigError(f"ligand has no coordinates for SMARTS alignment: {path}")
    return mol


def _smarts_match_atom_ids(mol, query, smarts_atom_ids, ligand_name):
    matches = mol.GetSubstructMatches(query, uniquify=False)
    if not matches:
        raise WorkflowConfigError(f"SMARTS alignment pattern did not match ligand {ligand_name}")
    max_position = max(smarts_atom_ids)
    if max_position > query.GetNumAtoms():
        raise WorkflowConfigError(
            "workflow.alignment.smarts_atom_ids contains an atom position outside the SMARTS pattern"
        )
    selected = []
    for match in matches:
        selected.append([int(match[position - 1]) for position in smarts_atom_ids])
    return selected


def _direct_alignment_rmsd(mol_a, atom_ids_a, mol_b, atom_ids_b):
    conf_a = mol_a.GetConformer()
    conf_b = mol_b.GetConformer()
    squared = 0.0
    for atom_a, atom_b in zip(atom_ids_a, atom_ids_b):
        pos_a = conf_a.GetAtomPosition(atom_a)
        pos_b = conf_b.GetAtomPosition(atom_b)
        squared += (pos_a.x - pos_b.x) ** 2 + (pos_a.y - pos_b.y) ** 2 + (pos_a.z - pos_b.z) ** 2
    return math.sqrt(squared / len(atom_ids_a))


def _best_smarts_pair_alignment(lig_a_file, lig_a_name, lig_b_file, lig_b_name, query, smarts_atom_ids):
    mol_a = _load_alignment_mol(lig_a_file)
    mol_b = _load_alignment_mol(lig_b_file)
    matches_a = _smarts_match_atom_ids(mol_a, query, smarts_atom_ids, lig_a_name)
    matches_b = _smarts_match_atom_ids(mol_b, query, smarts_atom_ids, lig_b_name)

    best = None
    for index_a, atom_ids_a in enumerate(matches_a, start=1):
        for index_b, atom_ids_b in enumerate(matches_b, start=1):
            rmsd = _direct_alignment_rmsd(mol_a, atom_ids_a, mol_b, atom_ids_b)
            if best is None or rmsd < best["selected_rmsd_a"]:
                best = {
                    "ligand_a": {
                        "name": lig_a_name,
                        "align_atom_ids": [atom_id + 1 for atom_id in atom_ids_a],
                        "match_index": index_a,
                    },
                    "ligand_b": {
                        "name": lig_b_name,
                        "align_atom_ids": [atom_id + 1 for atom_id in atom_ids_b],
                        "match_index": index_b,
                    },
                    "selected_rmsd_a": float(rmsd),
                }
    return best


def generate_smarts_alignments(alignment, plan):
    if not isinstance(alignment, dict):
        raise WorkflowConfigError("workflow.alignment must be a mapping")
    method = alignment.get("method")
    if method != "smarts":
        raise WorkflowConfigError("workflow.alignment.method must be 'smarts'")
    smarts = alignment.get("smarts")
    if not isinstance(smarts, str) or not smarts.strip():
        raise WorkflowConfigError("workflow.alignment.smarts must be a non-empty string")
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise WorkflowConfigError("workflow.alignment.smarts is not a valid SMARTS pattern")

    smarts_atom_ids = alignment.get("smarts_atom_ids")
    if not isinstance(smarts_atom_ids, list) or len(smarts_atom_ids) != 3:
        raise WorkflowConfigError("workflow.alignment.smarts_atom_ids must contain three 1-based atom ids")
    try:
        smarts_atom_ids = [int(atom_id) for atom_id in smarts_atom_ids]
    except (TypeError, ValueError) as exc:
        raise WorkflowConfigError("workflow.alignment.smarts_atom_ids must contain integers") from exc
    if any(atom_id < 1 for atom_id in smarts_atom_ids):
        raise WorkflowConfigError("workflow.alignment.smarts_atom_ids must be 1-based positive atom ids")

    structures = alignment.get("structures", {}) or {}
    if not isinstance(structures, dict):
        raise WorkflowConfigError("workflow.alignment.structures must be a mapping")

    def alignment_structure(pair, ligand_key):
        name_key = "lig1_name" if ligand_key == "ligand_a" else "lig2_name"
        file_key = "lig1_file" if ligand_key == "ligand_a" else "lig2_file"
        ligand_name = pair[name_key]
        configured = structures.get(ligand_name)
        if configured is None:
            return pair[file_key]
        path = _resolve_path(configured, plan.get("base_dir", Path.cwd()))
        _validate_file(path, f"workflow.alignment.structures.{ligand_name}")
        return path

    pair_alignments = {}
    for pair in plan["pairs"]:
        pair_alignments[pair["jobname"]] = _best_smarts_pair_alignment(
            alignment_structure(pair, "ligand_a"),
            pair["lig1_name"],
            alignment_structure(pair, "ligand_b"),
            pair["lig2_name"],
            query,
            smarts_atom_ids,
        )

    return {
        "schema_version": 2,
        "method": "smarts",
        "smarts": smarts,
        "smarts_atom_ids": smarts_atom_ids,
        "pairs": pair_alignments,
    }


def setup_small_molecule_system(receptor_file, lig1_file, lig2_file, ff_json_file, options, setup_options=None):
    from atom_openmm.make_atm_system_from_rcpt_lig import make_system

    if setup_options is None:
        setup_options = {}
    if setup_options.get("setup_mode") == "ambertools":
        setup_small_molecule_system_ambertools(receptor_file, lig1_file, lig2_file, options, setup_options["ambertools"])
        return
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


def _alignment_atoms_for_pair(alignments, pair_plan, ligand_key):
    pair_alignments = alignments.get("pairs") if isinstance(alignments, dict) else None
    if isinstance(pair_alignments, dict):
        pair_alignment = pair_alignments.get(pair_plan["jobname"])
        if pair_alignment is None:
            raise WorkflowConfigError(f"missing pair-specific alignment atoms for pair {pair_plan['jobname']}")
        ligand_alignment = pair_alignment.get(ligand_key)
        if not ligand_alignment or "align_atom_ids" not in ligand_alignment:
            raise WorkflowConfigError(
                f"missing pair-specific alignment atoms for {ligand_key} in pair {pair_plan['jobname']}"
            )
        return ligand_alignment["align_atom_ids"]

    lig_name = pair_plan["lig1_name"] if ligand_key == "ligand_a" else pair_plan["lig2_name"]
    if lig_name not in alignments:
        raise WorkflowConfigError(f"missing alignment atoms for ligand {lig_name}")
    return alignments[lig_name]["align_atom_ids"]


def _inferred_frcmod_path(mol2_file):
    frcmod = mol2_file.with_suffix(".frcmod")
    if not frcmod.exists():
        raise WorkflowConfigError(f"missing Amber frcmod file for {mol2_file.name}: {frcmod}")
    return frcmod


def _write_mol2_with_residue_name(source, destination, residue_name):
    in_atom_section = False
    in_substructure_section = False
    lines = []
    for line in source.read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            in_atom_section = line.strip().upper() == "@<TRIPOS>ATOM"
            in_substructure_section = line.strip().upper() == "@<TRIPOS>SUBSTRUCTURE"
            lines.append(line)
            continue
        if in_atom_section and line.strip():
            parts = line.split()
            if len(parts) >= 9:
                parts[7] = residue_name
                line = (
                    f"{int(parts[0]):7d} {parts[1]:<8s}"
                    f" {float(parts[2]):10.4f} {float(parts[3]):10.4f} {float(parts[4]):10.4f}"
                    f" {parts[5]:<8s} {int(parts[6]):5d} {parts[7]:<8s} {float(parts[8]):12.6f}"
                )
        elif in_substructure_section and line.strip():
            parts = line.split()
            if len(parts) >= 2:
                parts[1] = residue_name
                line = " ".join(parts)
        lines.append(line)
    destination.write_text("\n".join(lines) + "\n")


def _antechamber_input_format(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".sdf":
        return "sdf"
    if suffix == ".mol2":
        return "mol2"
    if suffix == ".pdb":
        return "pdb"
    raise WorkflowConfigError(f"antechamber ligand input must be SDF, MOL2, or PDB: {path}")


def _ambertools_atom_type(ligand_forcefield):
    if "gaff2" in ligand_forcefield.lower():
        return "gaff2"
    return "gaff"


def _ligand_net_charge(ligand_file, ambertools_options):
    ligand_name = Path(ligand_file).stem
    charges = ambertools_options.get("ligand_net_charges", {})
    if ligand_name in charges:
        return int(charges[ligand_name])
    return int(ambertools_options.get("ligand_net_charge", 0))


def _prepare_ambertools_ligand(ligand_file, mol2_file, frcmod_file, residue_name, ambertools_options):
    ligand_file = Path(ligand_file)
    parameterization = ambertools_options.get("ligand_parameterization", "preparameterized")
    if parameterization == "preparameterized":
        _write_mol2_with_residue_name(ligand_file, mol2_file, residue_name)
        source_frcmod = _inferred_frcmod_path(ligand_file)
        frcmod_file.write_text(source_frcmod.read_text())
        return

    if parameterization != "antechamber":
        raise WorkflowConfigError(f"unsupported AmberTools ligand parameterization: {parameterization}")

    raw_mol2 = mol2_file.with_name(mol2_file.stem + "_antechamber.mol2")
    atom_type = _ambertools_atom_type(ambertools_options["ligand_forcefield"])
    input_format = _antechamber_input_format(ligand_file)
    charge_model = ambertools_options.get("ligand_charge_model", "bcc")
    net_charge = _ligand_net_charge(ligand_file, ambertools_options)

    subprocess.run(
        [
            "antechamber",
            "-i",
            str(ligand_file.resolve()),
            "-fi",
            input_format,
            "-o",
            str(raw_mol2.resolve()),
            "-fo",
            "mol2",
            "-at",
            atom_type,
            "-c",
            charge_model,
            "-nc",
            str(net_charge),
            "-rn",
            residue_name,
        ],
        check=True,
    )
    _write_mol2_with_residue_name(raw_mol2, mol2_file, residue_name)
    subprocess.run(
        [
            "parmchk2",
            "-i",
            str(mol2_file.resolve()),
            "-f",
            "mol2",
            "-o",
            str(frcmod_file.resolve()),
            "-s",
            atom_type,
        ],
        check=True,
    )


def _format_displacement(displacement):
    if isinstance(displacement, str):
        values = [float(part) for part in displacement.replace(",", " ").split()]
    else:
        values = [float(part) for part in displacement]
    if len(values) != 3:
        raise WorkflowConfigError("DISPLACEMENT must contain three values")
    return values


def write_ambertools_tleap_input(
    receptor_file,
    lig1_file,
    lig2_file,
    options,
    ambertools_options,
    tleap_file,
):
    basename = options["BASENAME"]
    displacement = _format_displacement(options["DISPLACEMENT"])
    inputs_dir = tleap_file.parent / "ambertools_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    lig1_mol2 = inputs_dir / "L1.mol2"
    lig2_mol2 = inputs_dir / "L2.mol2"
    lig1_frcmod = inputs_dir / "L1.frcmod"
    lig2_frcmod = inputs_dir / "L2.frcmod"
    _prepare_ambertools_ligand(Path(lig1_file), lig1_mol2, lig1_frcmod, "L1", ambertools_options)
    _prepare_ambertools_ligand(Path(lig2_file), lig2_mol2, lig2_frcmod, "L2", ambertools_options)

    commands = [
        f"source {ambertools_options['protein_forcefield']}",
    ]
    commands.extend(f"source {forcefield}" for forcefield in ambertools_options.get("additional_forcefields", []))
    commands.extend(
        [
            f"source {ambertools_options['ligand_forcefield']}",
            f"source {ambertools_options['water_forcefield']}",
            f'RCPT = loadpdb "{Path(receptor_file).resolve()}"',
            f'LIG1 = loadmol2 "{lig1_mol2.resolve()}"',
            f'loadamberparams "{lig1_frcmod.resolve()}"',
            f'LIG2 = loadmol2 "{lig2_mol2.resolve()}"',
            f'loadamberparams "{lig2_frcmod.resolve()}"',
            f"translate LIG2 {{ {displacement[0]:.6f} {displacement[1]:.6f} {displacement[2]:.6f} }}",
            "MOL = combine {RCPT LIG1 LIG2}",
        ]
    )
    if ambertools_options.get("neutralize", True):
        commands.extend(["addions2 MOL Na+ 0", "addions2 MOL Cl- 0"])
    commands.extend(
        [
            f"solvateBox MOL {ambertools_options['solvent_box']} {ambertools_options['solvent_padding_a']:.6f}",
            f"saveamberparm MOL {basename}.prmtop {basename}.inpcrd",
            "quit",
        ]
    )
    tleap_file.write_text("\n".join(commands) + "\n")
    return tleap_file


def setup_small_molecule_system_ambertools(receptor_file, lig1_file, lig2_file, options, ambertools_options):
    from atom_openmm.make_atm_system_from_amber import make_system as make_amber_system

    basename = options["BASENAME"]
    tleap_file = Path("tleap.cmd")
    write_ambertools_tleap_input(receptor_file, lig1_file, lig2_file, options, ambertools_options, tleap_file)
    subprocess.run(["tleap", "-f", str(tleap_file)], check=True)
    make_amber_system(
        prmtopfile=basename + ".prmtop",
        crdfile=basename + ".inpcrd",
        xmloutfile=basename + "_sys.xml",
        pdboutfile=basename + ".pdb",
        hmass=float(options.get("HMASS", 1.0)),
        nonbondedCutoff=float(options.get("NONBONDED_CUTOFF", 0.9)),
        switchDistance=float(options.get("SWITCH_DISTANCE", 0.0)),
    )


def _receptor_heavy_atom_indices(topology, chain_names):
    chain_names = {str(value) for value in chain_names}
    excluded_residues = {
        "L1", "L2", "HOH", "WAT", "SOL",
        "NA", "NA+", "CL", "CL-", "K", "K+", "CA", "CA2", "MG", "MG2", "ZN", "ZN2",
    }
    selected = [
        atom.index
        for atom in topology.atoms()
        if atom.residue.chain.id in chain_names
        and atom.residue.name.upper() not in excluded_residues
        and atom.element is not None
        and atom.element.atomic_number != 1
    ]
    if selected:
        return selected

    selected = [
        atom.index
        for atom in topology.atoms()
        if atom.residue.name.upper() not in excluded_residues
        and atom.element is not None
        and atom.element.atomic_number != 1
    ]
    if not selected:
        raise WorkflowConfigError(
            "could not identify receptor heavy atoms for the ligand exclusion potential"
        )
    return selected


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
    if not rcpt_frame_indexes:
        rcpt_frame_query = 'atom.name == "CA" and atom.residue.name not in ["L1", "L2", "WAT", "HOH"]'
        rcpt_frame_indexes = get_indexes_from_query(topology, rcpt_frame_query)
    if not rcpt_frame_indexes:
        raise WorkflowConfigError("could not identify receptor CA atoms for the ATM reference frame")
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
    options["EXCLUSION_POT_MOL1_INDEXES"] = _receptor_heavy_atom_indices(
        topology, rcpt_chain_names
    )
    options["EXCLUSION_POT_MOL2_INDEXES"] = get_indexes_from_residue(
        res2, query="(atom.element.atomic_number != 1)"
    )


def _contains_role_smarts(value):
    if isinstance(value, str):
        return any(f"#{role}:\"" in value for role in ("ligand", "ligand_a", "ligand_b", "bound", "unbound"))
    if isinstance(value, dict):
        return any(_contains_role_smarts(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_role_smarts(item) for item in value)
    return False


def _selection_structure_file(workflow, pair_plan, identity, base_dir):
    file_key = "lig1_file" if identity == "ligand_a" else "lig2_file"
    name_key = "lig1_name" if identity == "ligand_a" else "lig2_name"
    structures = ((workflow.get("alignment") or {}).get("structures") or {})
    configured = structures.get(pair_plan[name_key])
    if configured is None:
        return Path(pair_plan[file_key]).resolve()
    path = _resolve_path(configured, base_dir)
    _validate_file(path, f"workflow.alignment.structures.{pair_plan[name_key]}")
    return path


def prepare_selection_metadata(options, workflow, pair_plan, *, base_dir=None):
    """Persist canonical ligand graphs and their prepared-system atom mappings."""
    if not _contains_role_smarts(workflow):
        return
    pdb = PDBFile(options["BASENAME"] + ".pdb")
    topology_atoms = list(pdb.topology.atoms())
    metadata = {}
    for identity, atom_key in (
        ("ligand_a", "LIGAND1_ATOMS"),
        ("ligand_b", "LIGAND2_ATOMS"),
    ):
        source = _selection_structure_file(
            workflow, pair_plan, identity, Path(base_dir or Path.cwd())
        )
        molecule = _load_alignment_mol(source)
        system_indices = [int(value) for value in options[atom_key]]
        expected = [topology_atoms[index].element.symbol for index in system_indices]
        observed = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        if observed != expected:
            raise WorkflowConfigError(
                f"{identity} SMARTS graph does not match the prepared ligand atom order: "
                f"expected {len(expected)} atoms with prepared element sequence, got {len(observed)} atoms. "
                "Use workflow.alignment.structures to provide a graph with matching explicit hydrogens and atom order."
            )
        filename = f"selection_{identity}.sdf"
        writer = Chem.SDWriter(filename)
        writer.write(molecule)
        writer.close()
        metadata[identity] = {
            "structure_file": filename,
            "system_atom_indices": system_indices,
        }
    options["SELECTION_METADATA"] = metadata


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


def run_production(options, workflow, progress_callback=None):
    production_method = normalize_workflow_axes(workflow).sampling_method
    if production_method == "async_re":
        if should_run_production(options):
            rbfe_production(config_file=None, options=options)
        return None
    if production_method == "neqti":
        from atom_openmm.neqti import normalize_neqti_options, run_neqti

        neqti_options = normalize_neqti_options(workflow, options)
        if progress_callback is None:
            return run_neqti(options, neqti_options)
        return run_neqti(options, neqti_options, progress_callback=progress_callback)
    if production_method == "awh":
        from atom_openmm.awh import normalize_awh_options, run_awh

        awh_options = normalize_awh_options(workflow, options)
        if progress_callback is None:
            return run_awh(options, awh_options)
        return run_awh(options, awh_options, progress_callback=progress_callback)
    raise WorkflowConfigError(
        "workflow.sampling.method must be 'async_re', 'neqti', or 'awh'"
    )


def run_production_with_restarts(options, workflow, result_writer, stage, progress_callback=None):
    restart_cfg = normalize_production_restarts(workflow)
    max_attempts = restart_cfg["max_attempts"] if restart_cfg["enabled"] else 1
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return run_production(options, workflow, progress_callback=progress_callback)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                if restart_cfg["enabled"]:
                    raise ProductionRestartExhaustedError(exc, max_attempts) from exc
                raise
            warning = (
                f"Production attempt {attempt}/{max_attempts} failed with "
                f"{type(exc).__name__}: {exc}; retrying with resume state."
            )
            result_writer.update(
                "partial",
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "stage": stage,
                    "restart_attempt": attempt,
                    "restart_attempts_allowed": max_attempts,
                },
                warning=warning,
                stage=stage,
            )
    raise ProductionRestartExhaustedError(last_exc, max_attempts)


def run_pair(pair_plan, workflow, atom_options, setup_options, receptor_file, alignments, workflow_yaml=None):
    jobdir = pair_plan["jobdir"]
    jobdir.mkdir(parents=True, exist_ok=True)

    options = deepcopy(atom_options)
    options["BASENAME"] = pair_plan["jobname"]
    options["WORKDIR"] = str(jobdir.resolve())
    options["LIGAND_FORCE_FIELD"] = setup_options["ligandforcefield"]
    axes = normalize_workflow_axes(workflow)
    production_method = axes.sampling_method
    requested_samples = (
        (workflow.get("neqti") or {}).get("n_snapshots", atom_options.get("MAX_SAMPLES", 1))
        if production_method == "neqti"
        else None if production_method == "awh" else atom_options.get("MAX_SAMPLES")
    )
    result_writer = RBFEResultWriter(
        pair_plan=pair_plan,
        receptor_file=receptor_file,
        workflow_yaml=workflow_yaml or (Path.cwd() / "workflow.yaml"),
        method=production_method,
        chemistry=axes.chemistry,
        alchemy_model=axes.alchemy_model,
        thermodynamic_cycle=axes.thermodynamic_cycle,
        requested_samples=requested_samples,
        pair_index=pair_plan.get("pair_index", 1),
        total_pairs=pair_plan.get("total_pairs", 1),
    )
    stage = "setup"
    result_writer.update("running", stage=stage)
    try:
        if production_method in ("neqti", "awh"):
            options["STRUCTPREP_MODE"] = "physical_only"
            options["NEQTI_INITIAL_STATE_FILE"] = options["BASENAME"] + "_equil.xml"
        else:
            options["STRUCTPREP_MODE"] = "async_re"
        equilibration_protocol = normalize_equilibration_protocol(workflow)
        if equilibration_protocol is not None:
            options["EQUILIBRATION_PROTOCOL"] = equilibration_protocol

        for ligand_key, option_key in (
            ("ligand_a", "ALIGN_LIGAND1_REF_ATOMS"),
            ("ligand_b", "ALIGN_LIGAND2_REF_ATOMS"),
        ):
            options[option_key] = [int(i) - 1 for i in _alignment_atoms_for_pair(alignments, pair_plan, ligand_key)]

        if not options.get("DISPLACEMENT"):
            options["DISPLACEMENT"] = list(calc_displ_vec(str(receptor_file), str(pair_plan["lig2_file"])))

        forcefield_cache = workflow.get("forcefield_cache", "ff.json")
        ff_json_file = Path(forcefield_cache)
        if not ff_json_file.is_absolute():
            ff_json_file = jobdir / ff_json_file
    except (Exception, KeyboardInterrupt) as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "stage": stage}
        if isinstance(exc, ProductionRestartExhaustedError):
            error.update(
                {
                    "type": type(exc.original).__name__,
                    "message": str(exc.original),
                    "restart_attempts_used": exc.attempts,
                }
            )
        result_writer.update(
            "failed",
            error=error,
            stage=stage,
        )
        raise

    try:
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
            prepare_selection_metadata(
                options,
                workflow,
                pair_plan,
                base_dir=Path(workflow_yaml).resolve().parent if workflow_yaml else Path.cwd(),
            )
            write_options_yaml(options)
            create_vmd_infile(options)

            if workflow.get("prepare_only", False) or not workflow.get("run", True):
                result_writer.update("prepared", stage="prepared")
                return {"jobname": options["BASENAME"], "status": "prepared", "workdir": options["WORKDIR"]}

            stage = "preparation"
            prep_state = options["BASENAME"] + (
                "_equil.xml" if production_method in ("neqti", "awh") else "_0.xml"
            )
            if not Path(prep_state).exists():
                result_writer.update("running", stage=stage)
                rbfe_structprep(config_file=None, options=deepcopy(options))
            result_writer.update("prepared", stage=stage)

            stage = "production"
            result_writer.update("running", stage=stage)

            def record_production_progress(summary):
                result_writer.update("partial", analysis=summary, stage=stage)

            production_result = run_production_with_restarts(
                options,
                workflow,
                result_writer,
                stage,
                progress_callback=record_production_progress,
            )
            if production_result is not None:
                final_status = production_result.get("status", "completed")
                method_label = "AWH" if production_method == "awh" else "NEQTI"
                warning = (
                    None
                    if production_result.get("analysis")
                    else f"No finite {method_label} estimate is available."
                )
                result_writer.update(final_status, analysis=production_result, warning=warning, stage=stage)
                return {
                    **production_result,
                    "jobname": options["BASENAME"],
                    "workdir": options["WORKDIR"],
                }

            if production_method == "async_re" and workflow.get("analyze", True):
                stage = "analysis"
                result_writer.update("running", stage=stage)
                analysis = analyze_pair(options, workflow)
                raw_samples = production_sample_count(options)
                reached_target = (
                    raw_samples is not None
                    and (requested_samples is None or raw_samples >= int(requested_samples))
                )
                final_status = "completed" if reached_target else "partial"
                warning = None if analysis is not None else "No finite UWHAM estimate is available."
                result_writer.update(final_status, analysis=analysis, warning=warning, stage=stage)
                if analysis is not None:
                    return {"status": "analyzed", "workdir": options["WORKDIR"], **analysis}
            else:
                raw_samples = production_sample_count(options) if production_method == "async_re" else None
                reached_target = raw_samples is not None and (
                    requested_samples is None or raw_samples >= int(requested_samples)
                )
                result_writer.update(
                    "completed" if reached_target else "partial",
                    warning="Analysis is disabled; no free-energy estimate is available.",
                    stage=stage,
                )

            return {"jobname": options["BASENAME"], "status": "completed", "workdir": options["WORKDIR"]}
    except (Exception, KeyboardInterrupt) as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "stage": stage}
        if isinstance(exc, ProductionRestartExhaustedError):
            error.update(
                {
                    "type": type(exc.original).__name__,
                    "message": str(exc.original),
                    "restart_attempts_used": exc.attempts,
                }
            )
        result_writer.update(
            "failed",
            error=error,
            stage=stage,
        )
        raise


def _prepare_run_context(config_file):
    config = load_workflow_config(config_file)
    plan = build_small_molecule_plan(config)
    setup_options = normalize_setup_options(config["workflow"], config["atom_options"])
    alignments = load_or_generate_alignments(config["workflow"], plan)
    return config, plan, setup_options, alignments


def validate_workflow(config_file):
    axes = _workflow_axes(config_file)
    if axes.chemistry == "covalent":
        from atom_openmm.covalent_workflow import validate_covalent_workflow

        return validate_covalent_workflow(config_file)
    if axes.alchemy_model == "hybrid_topology":
        from atom_openmm.hybrid_workflow import validate_noncovalent_hybrid_workflow

        return validate_noncovalent_hybrid_workflow(config_file)
    if axes.alchemy_model == "separated_topology":
        from atom_openmm.separated_workflow import validate_separated_workflow

        return validate_separated_workflow(config_file)
    config = load_workflow_config(config_file)
    plan = build_small_molecule_plan(config)
    normalize_setup_options(config["workflow"], config["atom_options"])
    normalize_equilibration_protocol(config["workflow"])
    normalize_production_restarts(config["workflow"])
    load_or_generate_alignments(config["workflow"], plan, write_generated=False)
    return True


def plan_workflow(config_file):
    axes = _workflow_axes(config_file)
    if axes.chemistry == "covalent":
        from atom_openmm.covalent_workflow import plan_covalent_workflow

        validate_workflow(config_file)
        return plan_covalent_workflow(config_file)
    if axes.alchemy_model == "hybrid_topology":
        from atom_openmm.hybrid_workflow import plan_noncovalent_hybrid_workflow

        return plan_noncovalent_hybrid_workflow(config_file)
    if axes.alchemy_model == "separated_topology":
        from atom_openmm.separated_workflow import plan_separated_workflow

        return plan_separated_workflow(config_file)
    config = load_workflow_config(config_file)
    # Build alignments too so --plan-only catches missing alignment inputs.
    plan = build_small_molecule_plan(config)
    load_or_generate_alignments(config["workflow"], plan, write_generated=False)
    normalize_setup_options(config["workflow"], config["atom_options"])
    normalize_equilibration_protocol(config["workflow"])
    normalize_production_restarts(config["workflow"])
    return build_execution_plan(config)


def _load_pair_options(pair_plan):
    pair_yaml = pair_plan["jobdir"] / f"{pair_plan['jobname']}.yaml"
    if not pair_yaml.exists():
        raise WorkflowConfigError(f"prepared pair YAML does not exist: {pair_yaml}")
    with open(pair_yaml) as handle:
        return yaml.safe_load(handle)


def analyze_neqti_existing(options, workflow):
    from atom_openmm.neqti import analyze_existing_neqti, normalize_neqti_options

    neqti_options = normalize_neqti_options(workflow, options)
    return analyze_existing_neqti(options, neqti_options)


def analyze_awh_existing(options, workflow):
    from atom_openmm.awh import analyze_existing_awh, normalize_awh_options

    awh_options = normalize_awh_options(workflow, options)
    return analyze_existing_awh(options, awh_options)


def analyze_pair_existing(pair_plan, workflow, atom_options, receptor_file, workflow_yaml=None):
    axes = normalize_workflow_axes(workflow)
    production_method = axes.sampling_method
    requested_samples = (
        (workflow.get("neqti") or {}).get("n_snapshots", atom_options.get("MAX_SAMPLES", 1))
        if production_method == "neqti"
        else None if production_method == "awh" else atom_options.get("MAX_SAMPLES")
    )
    result_writer = RBFEResultWriter(
        pair_plan=pair_plan,
        receptor_file=receptor_file,
        workflow_yaml=workflow_yaml or (Path.cwd() / "workflow.yaml"),
        method=production_method,
        chemistry=axes.chemistry,
        alchemy_model=axes.alchemy_model,
        thermodynamic_cycle=axes.thermodynamic_cycle,
        requested_samples=requested_samples,
        pair_index=pair_plan.get("pair_index", 1),
        total_pairs=pair_plan.get("total_pairs", 1),
    )
    stage = "analysis"
    result_writer.update("running", stage=stage)
    try:
        with _pushd(pair_plan["jobdir"]):
            options = _load_pair_options(pair_plan)
            if production_method == "neqti":
                analysis = analyze_neqti_existing(options, workflow)
                status = analysis.get("status", "completed")
                result_writer.update(status, analysis=analysis, stage=stage)
                return {"workdir": options["WORKDIR"], **analysis}
            if production_method == "awh":
                analysis = analyze_awh_existing(options, workflow)
                status = analysis.get("status", "completed")
                result_writer.update(status, analysis=analysis, stage=stage)
                return {
                    **analysis,
                    "jobname": options["BASENAME"],
                    "workdir": options["WORKDIR"],
                }
            analysis = analyze_pair(options, workflow)
            if analysis is None:
                raise WorkflowConfigError("No async-RE production samples are available for analysis")
            raw_samples = production_sample_count(options)
            reached_target = raw_samples is not None and (
                requested_samples is None or raw_samples >= int(requested_samples)
            )
            final_status = "completed" if reached_target else "partial"
            result_writer.update(final_status, analysis=analysis, stage=stage)
            return {"status": final_status, "workdir": options["WORKDIR"], **analysis}
    except (Exception, KeyboardInterrupt) as exc:
        result_writer.update(
            "failed",
            error={"type": type(exc).__name__, "message": str(exc), "stage": stage},
            stage=stage,
        )
        raise


def run_rbfe_workflow(config_file):
    axes = _workflow_axes(config_file)
    if axes.chemistry == "covalent":
        from atom_openmm.covalent_workflow import run_covalent_workflow

        return run_covalent_workflow(config_file)
    if axes.alchemy_model == "hybrid_topology":
        from atom_openmm.hybrid_workflow import run_noncovalent_hybrid_workflow

        return run_noncovalent_hybrid_workflow(config_file)
    if axes.alchemy_model == "separated_topology":
        from atom_openmm.separated_workflow import run_separated_workflow

        return run_separated_workflow(config_file)
    config, plan, setup_options, alignments = _prepare_run_context(config_file)
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
                config["config_path"],
            )
        )
    return results


def analyze_existing_workflow(config_file):
    axes = _workflow_axes(config_file)
    if axes.chemistry == "covalent":
        from atom_openmm.covalent_workflow import analyze_covalent_workflow

        return analyze_covalent_workflow(config_file)
    if axes.alchemy_model == "hybrid_topology":
        from atom_openmm.hybrid_workflow import analyze_noncovalent_hybrid_workflow

        return analyze_noncovalent_hybrid_workflow(config_file)
    if axes.alchemy_model == "separated_topology":
        from atom_openmm.separated_workflow import analyze_separated_workflow

        return analyze_separated_workflow(config_file)
    config = load_workflow_config(config_file)
    plan = build_small_molecule_plan(config)
    results = []
    for pair_plan in plan["pairs"]:
        results.append(
            analyze_pair_existing(
                pair_plan,
                config["workflow"],
                config["atom_options"],
                plan["receptor_file"],
                config["config_path"],
            )
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run an AToM-OpenMM RBFE workflow from one YAML file.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate", action="store_true", help="validate workflow inputs without creating outputs")
    mode.add_argument("--plan-only", action="store_true", help="print a machine-readable execution plan without creating outputs")
    mode.add_argument("--analyze-only", action="store_true", help="reanalyze existing pair outputs without running setup or simulation")
    mode.add_argument(
        "--prepare-node-bank",
        action="store_true",
        help="prepare reusable separated-topology node ensembles",
    )
    parser.add_argument("workflow_yaml", help="High-level RBFE workflow YAML file")
    args = parser.parse_args(argv)
    try:
        if args.prepare_node_bank:
            axes = _workflow_axes(args.workflow_yaml)
            if axes.alchemy_model != "separated_topology":
                raise WorkflowConfigError(
                    "--prepare-node-bank requires alchemy.model: separated_topology"
                )
            from atom_openmm.separated_node_bank import prepare_node_bank

            result = prepare_node_bank(args.workflow_yaml)
            print(yaml.safe_dump(result, sort_keys=False))
            return 0
        if args.validate:
            validate_workflow(args.workflow_yaml)
            print("valid")
            return 0
        if args.plan_only:
            yaml.safe_dump(plan_workflow(args.workflow_yaml), sys.stdout, sort_keys=False)
            return 0
        if args.analyze_only:
            results = analyze_existing_workflow(args.workflow_yaml)
        else:
            results = run_rbfe_workflow(args.workflow_yaml)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for result in results:
        line = f"{result['jobname']}: {result['status']} in {result['workdir']}"
        if result.get("ddg") is not None:
            line += f" DG = {result['ddg']:8.3f}"
            if result.get("ddg_std") is not None:
                line += f" +/- {result['ddg_std']:8.3f}"
            line += " kcal/mol"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
