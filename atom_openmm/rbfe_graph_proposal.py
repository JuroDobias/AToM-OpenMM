from __future__ import annotations

import argparse
import heapq
import math
from pathlib import Path

import yaml
from rdkit import Chem

from atom_openmm.covalent_hybrid import (
    find_covalent_atom_map,
    normalize_mapping_aromaticity,
)
from atom_openmm.hybrid_mapping import _direct_rmsd


class RBFEGraphProposalError(ValueError):
    pass


def _load_molecule(path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() != 1:
        raise RBFEGraphProposalError(f"could not read one 3D molecule from {path}")
    return normalize_mapping_aromaticity(molecule)


def discover_ligand_file(source, node, graph_root):
    if node.get("file"):
        path = (Path(graph_root) / node["file"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    source = Path(source)
    matches = sorted({
        *source.rglob(f"alignment_structures/{node['id']}-p.sdf"),
        *source.rglob(f"ligands/{node['id']}-p.sdf"),
    })
    if not matches:
        raise FileNotFoundError(f"could not locate {node['id']}-p.sdf below {source}")
    identities = {
        Chem.MolToSmiles(Chem.RemoveHs(_load_molecule(path)), isomericSmiles=True)
        for path in matches
    }
    if len(identities) != 1:
        raise ValueError(f"source structures disagree chemically for node {node['id']}")
    return matches[0]


def _heavy_atoms(molecule):
    return {
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
    }


def _formal_charge(molecule):
    return int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()))


def _attachment_sites(molecule_a, molecule_b, mapping):
    heavy_a = _heavy_atoms(molecule_a)
    heavy_b = _heavy_atoms(molecule_b)
    mapped_a = set(mapping) & heavy_a
    mapped_b = set(mapping.values()) & heavy_b
    inverse = {atom_b: atom_a for atom_a, atom_b in mapping.items()}
    sites = set()
    for atom_a in heavy_a - mapped_a:
        for neighbor in molecule_a.GetAtomWithIdx(atom_a).GetNeighbors():
            if neighbor.GetIdx() in mapped_a:
                sites.add((neighbor.GetIdx(), mapping[neighbor.GetIdx()]))
    for atom_b in heavy_b - mapped_b:
        for neighbor in molecule_b.GetAtomWithIdx(atom_b).GetNeighbors():
            if neighbor.GetIdx() in mapped_b:
                sites.add((inverse[neighbor.GetIdx()], neighbor.GetIdx()))
    return sites, heavy_a - mapped_a, heavy_b - mapped_b


def score_pair(node_a, node_b, timeout_seconds=30):
    molecule_a = node_a["molecule"]
    molecule_b = node_b["molecule"]
    result = {
        "ligand_a": node_a["id"],
        "ligand_b": node_b["id"],
        "accepted": False,
        "reasons": [],
    }
    charge_a = _formal_charge(molecule_a)
    charge_b = _formal_charge(molecule_b)
    result["formal_charge_a"] = charge_a
    result["formal_charge_b"] = charge_b
    if charge_a != charge_b:
        result["reasons"].append("net_charge_change")
    try:
        mapping = find_covalent_atom_map(
            molecule_a, molecule_b, timeout_seconds=timeout_seconds
        )
    except Exception as exc:
        result["reasons"].append(f"mapping_failed: {exc}")
        return result
    sites, unmatched_a, unmatched_b = _attachment_sites(
        molecule_a, molecule_b, mapping
    )
    heavy_a = _heavy_atoms(molecule_a)
    heavy_b = _heavy_atoms(molecule_b)
    mapped_heavy = len(set(mapping) & heavy_a)
    mapped_fraction = mapped_heavy / max(len(heavy_a), len(heavy_b))
    rmsd = _direct_rmsd(molecule_a, molecule_b, mapping)
    result.update({
        "mapped_heavy_atom_count": mapped_heavy,
        "heavy_atom_count_a": len(heavy_a),
        "heavy_atom_count_b": len(heavy_b),
        "mapped_heavy_fraction": float(mapped_fraction),
        "changed_heavy_atoms_a": len(unmatched_a),
        "changed_heavy_atoms_b": len(unmatched_b),
        "attachment_site_count": len(sites),
        "attachment_atom_pairs_1based": [
            [atom_a + 1, atom_b + 1] for atom_a, atom_b in sorted(sites)
        ],
        "mapped_direct_rmsd_angstrom": float(rmsd),
    })
    if not unmatched_a and not unmatched_b:
        result["reasons"].append("no_heavy_atom_change")
    if len(sites) != 1:
        result["reasons"].append("not_one_attachment_site")
    if mapped_fraction < 0.5:
        result["reasons"].append("mapped_fraction_below_0.5")
    result["accepted"] = not result["reasons"]
    result["score"] = float(
        len(unmatched_a) + len(unmatched_b) + 5.0 * (1.0 - mapped_fraction) + rmsd
    )
    return result


def _shortest_path(adjacency, start, end):
    queue = [(0.0, start, [])]
    best = {}
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in best and best[node] <= cost:
            continue
        best[node] = cost
        if node == end:
            return cost, path
        for neighbor, edge_id, weight in adjacency.get(node, []):
            heapq.heappush(queue, (cost + weight, neighbor, path + [edge_id]))
    return None


def propose_graph(config_path, output_path=None):
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = yaml.safe_load(config_path.read_text()) or {}
    nodes = []
    for item in config["nodes"]:
        identifier = str(item["id"])
        path = (root / item["file"]).resolve()
        nodes.append({**item, "id": identifier, "file": str(path), "molecule": _load_molecule(path)})
    identifiers = [node["id"] for node in nodes]
    if len(set(identifiers)) != len(identifiers):
        raise RBFEGraphProposalError("node ids must be unique")
    candidates = []
    for index, node_a in enumerate(nodes):
        for node_b in nodes[index + 1:]:
            candidates.append(score_pair(node_a, node_b))
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    adjacency = {node: [] for node in identifiers}
    by_id = {}
    for candidate in accepted:
        edge_id = f'{candidate["ligand_a"]}--{candidate["ligand_b"]}'
        by_id[edge_id] = candidate
        adjacency[candidate["ligand_a"]].append(
            (candidate["ligand_b"], edge_id, candidate["score"])
        )
        adjacency[candidate["ligand_b"]].append(
            (candidate["ligand_a"], edge_id, candidate["score"])
        )
    proposed = set()
    targets = []
    for item in config["targets"]:
        ligand_a = str(item["ligand_a"] if isinstance(item, dict) else item[0])
        ligand_b = str(item["ligand_b"] if isinstance(item, dict) else item[1])
        path = _shortest_path(adjacency, ligand_a, ligand_b)
        target = {
            "id": str(item.get("id", f"{ligand_a}--{ligand_b}")) if isinstance(item, dict) else f"{ligand_a}--{ligand_b}",
            "ligand_a": ligand_a,
            "ligand_b": ligand_b,
            "status": "connected" if path else "requires_intermediate",
            "path_score": None if path is None else float(path[0]),
            "path_edges": [] if path is None else path[1],
        }
        proposed.update(target["path_edges"])
        targets.append(target)
    selected_nodes = {
        node for edge_id in proposed
        for node in (by_id[edge_id]["ligand_a"], by_id[edge_id]["ligand_b"])
    }
    cycle_candidates = sorted(
        (
            {"id": edge_id, **candidate}
            for edge_id, candidate in by_id.items()
            if edge_id not in proposed
            and candidate["ligand_a"] in selected_nodes
            and candidate["ligand_b"] in selected_nodes
        ),
        key=lambda item: item["score"],
    )
    payload = {
        "schema_version": 1,
        "source": str(config_path),
        "nodes": [
            {key: value for key, value in node.items() if key != "molecule"}
            for node in nodes
        ],
        "targets": targets,
        "suggested_path_edges": [
            {"id": edge_id, **by_id[edge_id]} for edge_id in sorted(proposed)
        ],
        "cycle_candidates": cycle_candidates,
        "accepted_candidates": sorted(accepted, key=lambda item: item["score"]),
        "rejected_candidates": [candidate for candidate in candidates if not candidate["accepted"]],
        "requires_curation": True,
    }
    output = Path(output_path).resolve() if output_path else root / "graph_proposal.yaml"
    output.write_text(yaml.safe_dump(payload, sort_keys=False))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Propose auditable single-site RBFE graph edges")
    parser.add_argument("config_yaml")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = propose_graph(args.config_yaml, args.output)
    connected = sum(target["status"] == "connected" for target in result["targets"])
    print(f"Connected {connected}/{len(result['targets'])} targets with single-site candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
