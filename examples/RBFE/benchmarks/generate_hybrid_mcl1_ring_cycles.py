#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import shutil
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem import rdFMCS

from atom_openmm.rbfe_repeat_network import generate_repeats


PRIMARY_EDGES = (
    ("27", "45", "soft_closure"),
    ("45", "46", "mcs"),
    ("27", "46", "soft_closure"),
    ("27", "43", "soft_closure"),
    ("43", "47", "topology_transmutation"),
    ("27", "47", "soft_closure"),
)
CONTROL_EDGE = ("27", "43")


def _molecules(path):
    return {
        molecule.GetProp("_Name"): molecule
        for molecule in Chem.SDMolSupplier(str(path), removeHs=False)
        if molecule is not None
    }


def _heavy_with_original_indices(molecule):
    editable = Chem.Mol(molecule)
    for atom in editable.GetAtoms():
        atom.SetIntProp("original_index", atom.GetIdx())
    return Chem.RemoveHs(editable)


def _embedded_base_map(base, target):
    base_heavy = _heavy_with_original_indices(base)
    target_heavy = _heavy_with_original_indices(target)
    matches = target_heavy.GetSubstructMatches(
        base_heavy, uniquify=False, useChirality=True, maxMatches=1000
    )
    if not matches:
        raise ValueError("the ligand-27 heavy graph is not embedded in the target")
    base_xyz = base_heavy.GetConformer()
    target_xyz = target_heavy.GetConformer()

    def direct_rmsd(match):
        squared = []
        for index_a, index_b in enumerate(match):
            delta = (
                np.asarray(base_xyz.GetAtomPosition(index_a))
                - np.asarray(target_xyz.GetAtomPosition(index_b))
            )
            squared.append(float(np.dot(delta, delta)))
        return float(np.sqrt(np.mean(squared)))

    match = min(matches, key=direct_rmsd)
    pairs = [
        [
            base_heavy.GetAtomWithIdx(index_a).GetIntProp("original_index"),
            target_heavy.GetAtomWithIdx(index_b).GetIntProp("original_index"),
        ]
        for index_a, index_b in enumerate(match)
    ]
    mapped_target = set(match)
    boundary = []
    for bond in target_heavy.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (first in mapped_target) == (second in mapped_target):
            continue
        original = sorted((
            target_heavy.GetAtomWithIdx(first).GetIntProp("original_index"),
            target_heavy.GetAtomWithIdx(second).GetIntProp("original_index"),
        ))
        boundary.append(original)
    if len(boundary) != 2:
        raise ValueError(
            f"expected an annulated branch with two boundary bonds, found {boundary}"
        )
    # Keep one junction as the branch anchor and alchemically close the other.
    soft_bond = max(boundary, key=lambda pair: (max(pair), min(pair)))
    return pairs, soft_bond


def _closure_mapping(base, target):
    pairs, soft_bond = _embedded_base_map(base, target)
    return {
        "method": "explicit_pairs",
        "pairs_0based": pairs,
        "alchemical_bonds": {
            "ligand_b": [{"atoms_0based": soft_bond, "mode": "soft_bond"}]
        },
    }


def _whole_ring_control_mapping():
    # Atoms 0-15 are the common scaffold through the ether oxygen. The complete
    # terminal phenyl and naphthyl systems are endpoint-specific.
    return {
        "method": "explicit_pairs",
        "pairs_0based": [[index, index] for index in range(16)],
    }


def _topology_transmutation_mapping(molecule_a, molecule_b):
    heavy_a = _heavy_with_original_indices(molecule_a)
    heavy_b = _heavy_with_original_indices(molecule_b)
    result = rdFMCS.FindMCS(
        [heavy_a, heavy_b],
        atomCompare=rdFMCS.AtomCompare.CompareAny,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=30,
    )
    query = Chem.MolFromSmarts(result.smartsString)
    if query is None or result.numAtoms != heavy_a.GetNumAtoms():
        raise ValueError("topology-transmutation edge does not preserve ligand A")
    matches_a = heavy_a.GetSubstructMatches(query, uniquify=False, maxMatches=1000)
    matches_b = heavy_b.GetSubstructMatches(query, uniquify=False, maxMatches=1000)
    coordinates_a = heavy_a.GetConformer()
    coordinates_b = heavy_b.GetConformer()

    def score(pair):
        match_a, match_b = pair
        squared = []
        for query_index in range(query.GetNumAtoms()):
            delta = (
                np.asarray(coordinates_a.GetAtomPosition(match_a[query_index]))
                - np.asarray(coordinates_b.GetAtomPosition(match_b[query_index]))
            )
            squared.append(float(np.dot(delta, delta)))
        return float(np.sqrt(np.mean(squared)))

    match_a, match_b = min(
        ((a, b) for a in matches_a for b in matches_b), key=score
    )
    mapping = sorted(
        (
            heavy_a.GetAtomWithIdx(match_a[index]).GetIntProp("original_index"),
            heavy_b.GetAtomWithIdx(match_b[index]).GetIntProp("original_index"),
        )
        for index in range(query.GetNumAtoms())
    )
    junctions = {"ligand_a": [], "ligand_b": []}
    for atom_a, atom_b in mapping:
        source_a = molecule_a.GetAtomWithIdx(atom_a)
        source_b = molecule_b.GetAtomWithIdx(atom_b)
        if source_a.GetAtomicNum() == source_b.GetAtomicNum():
            continue
        hydrogens_a = sorted(
            atom.GetIdx() for atom in source_a.GetNeighbors()
            if atom.GetAtomicNum() == 1
        )
        hydrogens_b = sorted(
            atom.GetIdx() for atom in source_b.GetNeighbors()
            if atom.GetAtomicNum() == 1
        )
        if hydrogens_a and not hydrogens_b:
            junctions["ligand_a"].extend(
                {
                    "atoms_0based": [atom_a, hydrogen],
                    "inactive_geometry": "terminal_z_matrix",
                }
                for hydrogen in hydrogens_a
            )
        elif hydrogens_b and not hydrogens_a:
            junctions["ligand_b"].extend(
                {
                    "atoms_0based": [atom_b, hydrogen],
                    "inactive_geometry": "terminal_z_matrix",
                }
                for hydrogen in hydrogens_b
            )
    settings = {
        "method": "explicit_pairs",
        "pairs_0based": [list(pair) for pair in mapping],
    }
    if any(junctions.values()):
        settings["junction_bonds"] = junctions
    return settings


def _reference_datasets(table):
    rows = [
        row for row in csv.DictReader(Path(table).open())
        if row["Protein"] == "MCL1"
    ]
    return {
        "experiment": {
            "reference_node": "27",
            "edges": [
                {
                    "ligand_a": row["Ligand1"],
                    "ligand_b": row["Ligand2"],
                    "ddg_kcal_per_mol": float(row["experimental_ddG"]),
                }
                for row in rows
            ],
        },
        "published_atm_gaff2": {
            "reference_node": "27",
            "edges": [
                {
                    "ligand_a": row["Ligand1"],
                    "ligand_b": row["Ligand2"],
                    "ddg_kcal_per_mol": float(row["ATM_ddG"]),
                    "ddg_error_kcal_per_mol": float(row["ATM_error"]),
                }
                for row in rows
            ],
        },
    }


def _run_script(edge_id, source_dir_name):
    return f"""#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name=mcl1-{edge_id}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-b
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH -t 12:00:00
#SBATCH --signal=B:USR1@600

set -euo pipefail
MAX_CHAIN_JOBS="${{ATOM_MAX_CHAIN_JOBS:-10}}"
CHAIN_INDEX="${{ATOM_CHAIN_INDEX:-0}}"
ROOT_JOB_ID="${{ATOM_ROOT_JOB_ID:-${{SLURM_JOB_ID:-manual}}}}"
RUN_DIR="${{SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}}"
RESULT_GLOB="$RUN_DIR/run/*/result.yaml"
CHILD_PID=""
RESUBMITTING=0

completed() {{ grep -qs '^status: completed$' $RESULT_GLOB 2>/dev/null; }}
on_timeout() {{
    (( RESUBMITTING )) && return
    RESUBMITTING=1
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    if ! completed && (( CHAIN_INDEX + 1 < MAX_CHAIN_JOBS )); then
        sbatch --dependency="afterany:${{SLURM_JOB_ID}}" \
          --export=ALL,ATOM_CHAIN_INDEX=$((CHAIN_INDEX+1)),ATOM_ROOT_JOB_ID="$ROOT_JOB_ID",ATOM_MAX_CHAIN_JOBS="$MAX_CHAIN_JOBS" \
          "$RUN_DIR/run.sh"
    fi
    exit 0
}}
trap on_timeout USR1
cd "$RUN_DIR"
completed && exit 0
unset OPENMM_PLUGIN_DIR
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myatom_openmm86
export PYTHONPATH="$HOME/myAToM/{source_dir_name}${{PYTHONPATH:+:$PYTHONPATH}}"
python -m atom_openmm.rbfe_workflow --validate workflow.yaml
python -m atom_openmm.rbfe_workflow workflow.yaml &
CHILD_PID=$!
wait "$CHILD_PID"
"""


def _write_template(output, base_workflow, molecules, receptor, table, edges, control=False):
    output.mkdir(parents=True, exist_ok=True)
    ligand_ids = sorted({ligand for edge in edges for ligand in edge[:2]})
    for edge_index, (ligand_a, ligand_b, mode) in enumerate(edges, 1):
        edge_id = f"{ligand_a}--{ligand_b}"
        edge_dir = output / edge_id
        (edge_dir / "inputs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(receptor, edge_dir / "inputs/receptor.pdb")
        for ligand in (ligand_a, ligand_b):
            writer = Chem.SDWriter(str(edge_dir / f"inputs/{ligand}.sdf"))
            writer.write(molecules[ligand])
            writer.close()
        workflow = copy.deepcopy(base_workflow)
        workflow["workflow"]["workdir"] = "run"
        workflow["workflow"]["receptor"] = "inputs/receptor.pdb"
        workflow["workflow"]["ligands"] = {
            ligand_a: f"inputs/{ligand_a}.sdf",
            ligand_b: f"inputs/{ligand_b}.sdf",
        }
        workflow["workflow"]["pairs"] = [[ligand_a, ligand_b]]
        workflow["workflow"]["neqti"]["random_seed"] = 20260828 + edge_index
        if mode == "soft_closure":
            mapping = _closure_mapping(molecules[ligand_a], molecules[ligand_b])
        elif mode == "whole_ring":
            mapping = _whole_ring_control_mapping()
        elif mode == "topology_transmutation":
            mapping = _topology_transmutation_mapping(
                molecules[ligand_a], molecules[ligand_b]
            )
        elif mode == "mcs":
            mapping = {"method": "mcs"}
        else:
            raise ValueError(f"unknown mapping mode {mode}")
        workflow["workflow"]["alchemy"]["mapping"] = mapping
        (edge_dir / "workflow.yaml").write_text(
            yaml.safe_dump(workflow, sort_keys=False)
        )
        script = edge_dir / "run.sh"
        script.write_text(_run_script(edge_id, "AToM-OpenMM-covalent-repeat"))
        script.chmod(0o755)

    network_edges = [
        {
            "id": f"{a}--{b}", "ligand_a": a, "ligand_b": b,
            "directory": f"{a}--{b}",
        }
        for a, b, _ in edges
    ]
    network = {
        "schema_version": 2,
        "reference_node": "27",
        "nodes": [{"id": ligand} for ligand in ligand_ids],
        "edges": network_edges,
        "auto_cycles": not control,
        "node_bootstrap_samples": 2000,
        "reference_datasets": _reference_datasets(table),
    }
    (output / "network.yaml").write_text(yaml.safe_dump(network, sort_keys=False))


def _reuse_existing_pilot(output, existing_run):
    if existing_run is None:
        return False
    existing_run = Path(existing_run).resolve()
    candidates = sorted(existing_run.rglob("result.yaml"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one result.yaml below existing pilot {existing_run}"
        )
    result = yaml.safe_load(candidates[0].read_text()) or {}
    if str(result.get("ligand_a")) != "27" or str(result.get("ligand_b")) != "46":
        raise ValueError("existing pilot is not the 27->46 edge")
    target = output / "primary_cycles/replicate_1/27--46/run"
    if target.exists() or target.is_symlink():
        raise ValueError(f"repeat run destination already exists: {target}")
    target.symlink_to(existing_run, target_is_directory=True)
    network = yaml.safe_load(
        (output / "primary_cycles/replicate_1/network.yaml").read_text()
    )
    pending = [edge for edge in network["edges"] if edge["id"] != "27--46"]
    submit = output / "primary_cycles/replicate_1/submit_all.sh"
    submit.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(
            f'(cd "$(dirname "$0")/{edge["directory"]}" && sbatch run.sh)'
            for edge in pending
        )
        + "\n"
    )
    submit.chmod(0o755)
    aggregate_path = output / "primary_cycles/repeat_network.yaml"
    aggregate = yaml.safe_load(aggregate_path.read_text())
    aggregate["reused_edge_results"].setdefault("replicate_1", {})["27--46"] = str(
        existing_run
    )
    aggregate["new_job_count"] = int(aggregate["new_job_count"]) - 1
    aggregate_path.write_text(yaml.safe_dump(aggregate, sort_keys=False))
    return True


def generate(benchmark_root, pilot_workflow, output, repeats=3, existing_27_46=None):
    benchmark_root = Path(benchmark_root).resolve()
    output = Path(output).resolve()
    ligand_file = benchmark_root / "ATM_Validation/MCL-1/ligands/MCL1_ligands.sdf"
    receptor = benchmark_root / "ATM_Validation/MCL-1/receptor/MCL1_new_2.pdb"
    table = benchmark_root / "ATM_Validation/DDG_ATM_GAFF2.csv"
    molecules = _molecules(ligand_file)
    base_workflow = yaml.safe_load(Path(pilot_workflow).read_text())
    if base_workflow.get("atom_options") is None:
        base_workflow.pop("atom_options", None)
    primary_template = output / "templates/primary_cycles"
    control_template = output / "templates/whole_ring_control"
    _write_template(
        primary_template, base_workflow, molecules, receptor, table, PRIMARY_EDGES
    )
    _write_template(
        control_template, base_workflow, molecules, receptor, table,
        [(CONTROL_EDGE[0], CONTROL_EDGE[1], "whole_ring")], control=True,
    )
    generate_repeats(
        primary_template, output / "primary_cycles", repeats=repeats,
        seed_base=20260828, share_prepared=False,
    )
    generate_repeats(
        control_template, output / "whole_ring_control", repeats=repeats,
        seed_base=20270828, share_prepared=False,
    )
    _reuse_existing_pilot(output, existing_27_46)
    submit = output / "submit_all.sh"
    submit.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        '"$(dirname "$0")/primary_cycles/submit_all.sh"\n'
        '"$(dirname "$0")/whole_ring_control/submit_all.sh"\n'
    )
    submit.chmod(0o755)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--pilot-workflow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--existing-27-46",
        type=Path,
        help="existing 27->46 pilot run to reuse as primary replicate 1",
    )
    args = parser.parse_args()
    generate(
        args.benchmark_root, args.pilot_workflow, args.output, args.repeats,
        args.existing_27_46,
    )


if __name__ == "__main__":
    main()
