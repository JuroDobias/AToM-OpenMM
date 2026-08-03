from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS

from atom_openmm.covalent_alchemy import CovalentAlchemyError
from atom_openmm.covalent_hybrid import (
    complete_covalent_atom_map,
    find_covalent_atom_map,
)


HybridMappingError = CovalentAlchemyError


def _direct_rmsd(molecule_a, molecule_b, mapping):
    conformer_a = molecule_a.GetConformer()
    conformer_b = molecule_b.GetConformer()
    differences = [
        np.asarray(conformer_a.GetAtomPosition(atom_a))
        - np.asarray(conformer_b.GetAtomPosition(atom_b))
        for atom_a, atom_b in mapping.items()
        if molecule_a.GetAtomWithIdx(atom_a).GetAtomicNum() != 1
    ]
    if not differences:
        return 0.0
    return float(np.sqrt(np.mean([difference @ difference for difference in differences])))


def _smarts_constrained_map(molecule_a, molecule_b, smarts, timeout_seconds=30):
    core = Chem.MolFromSmarts(smarts)
    if core is None:
        raise HybridMappingError("workflow.alchemy.mapping.smarts is invalid")
    result = rdFMCS.FindMCS(
        [molecule_a, molecule_b, core],
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        timeout=int(timeout_seconds),
    )
    if result.canceled or result.numAtoms == 0:
        raise HybridMappingError("SMARTS-constrained hybrid MCS failed or timed out")
    query = result.queryMol
    matches_a = molecule_a.GetSubstructMatches(query, uniquify=False, useChirality=True)
    matches_b = molecule_b.GetSubstructMatches(query, uniquify=False, useChirality=True)
    if not matches_a or not matches_b:
        raise HybridMappingError("SMARTS-constrained MCS does not match both ligands")
    candidates = []
    for match_a in matches_a:
        for match_b in matches_b:
            mapping = dict(zip(match_a, match_b))
            candidates.append((_direct_rmsd(molecule_a, molecule_b, mapping), mapping))
    _, mapping = min(candidates, key=lambda item: item[0])
    return complete_covalent_atom_map(molecule_a, molecule_b, mapping)


def build_hybrid_atom_map(parameters_a, parameters_b, settings):
    molecule_a = parameters_a.molecule.to_rdkit()
    molecule_b = parameters_b.molecule.to_rdkit()
    method = settings.get("method", "mcs")
    if method == "mcs":
        mapping = find_covalent_atom_map(molecule_a, molecule_b)
    elif method == "mcs_core_smarts":
        smarts = settings.get("smarts")
        if not isinstance(smarts, str) or not smarts.strip():
            raise HybridMappingError(
                "mcs_core_smarts mapping requires workflow.alchemy.mapping.smarts"
            )
        mapping = _smarts_constrained_map(molecule_a, molecule_b, smarts.strip())
    else:
        raise HybridMappingError(
            "workflow.alchemy.mapping.method must be 'mcs' or 'mcs_core_smarts'"
        )
    rmsd = _direct_rmsd(molecule_a, molecule_b, mapping)
    maximum = settings.get("max_mapped_rmsd_a")
    if maximum is not None and rmsd > float(maximum):
        raise HybridMappingError(
            f"mapped ligand RMSD {rmsd:.3f} A exceeds max_mapped_rmsd_a {float(maximum):.3f} A"
        )
    return mapping, {
        "schema_version": 1,
        "method": method,
        "smarts": settings.get("smarts") if method == "mcs_core_smarts" else None,
        "mapped_atom_count": len(mapping),
        "mapped_heavy_atom_count": sum(
            molecule_a.GetAtomWithIdx(atom).GetAtomicNum() != 1 for atom in mapping
        ),
        "mapped_direct_rmsd_angstrom": rmsd,
        "map_a_to_b_0based": {
            int(atom_a): int(atom_b) for atom_a, atom_b in sorted(mapping.items())
        },
    }
