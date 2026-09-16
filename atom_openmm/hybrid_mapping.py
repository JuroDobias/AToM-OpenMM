from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS

from atom_openmm.covalent_alchemy import CovalentAlchemyError
from atom_openmm.covalent_hybrid import (
    _inactive_branch_components,
    complete_covalent_atom_map,
    find_covalent_atom_map,
    normalize_mapping_aromaticity,
)


HybridMappingError = CovalentAlchemyError
INACTIVE_GEOMETRIES = {"bond_only", "terminal_z_matrix"}
_TETRAHEDRAL_CHIRAL_SIGNS = {
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: -1,
}


def _strict_nonnegative_indices(raw, field, *, allow_empty=True):
    if not isinstance(raw, list) or (not raw and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise HybridMappingError(f"{field} must be {qualifier}")
    indices = []
    for position, value in enumerate(raw, start=1):
        if type(value) is not int or value < 0:
            raise HybridMappingError(
                f"{field} entry {position} must be a non-negative integer"
            )
        indices.append(value)
    if len(set(indices)) != len(indices):
        raise HybridMappingError(f"{field} must not contain duplicate indices")
    return indices


def _strict_explicit_pairs(raw_pairs):
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise HybridMappingError(
            "explicit_pairs mapping requires non-empty pairs_0based"
        )
    pairs = []
    for index, raw in enumerate(raw_pairs, start=1):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise HybridMappingError(
                f"explicit_pairs pairs_0based entry {index} must contain two indices"
            )
        if any(type(value) is not int or value < 0 for value in raw):
            raise HybridMappingError(
                f"explicit_pairs pairs_0based entry {index} must contain "
                "non-negative integers"
            )
        pairs.append((raw[0], raw[1]))
    if len({atom_a for atom_a, _atom_b in pairs}) != len(pairs) or len(
        {atom_b for _atom_a, atom_b in pairs}
    ) != len(pairs):
        raise HybridMappingError("explicit_pairs atom map must be one-to-one")
    return pairs


def _permutation_parity(source, target):
    """Return +1/-1 for the permutation from source ordering to target."""
    if len(source) != len(target) or set(source) != set(target):
        raise HybridMappingError("cannot compare incompatible neighbor orderings")
    positions = {value: index for index, value in enumerate(target)}
    permutation = [positions[value] for value in source]
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _local_neighbor_compatible(atom_a, atom_b, center_a, center_b, molecule_a, molecule_b):
    bond_a = molecule_a.GetBondBetweenAtoms(center_a, atom_a.GetIdx())
    bond_b = molecule_b.GetBondBetweenAtoms(center_b, atom_b.GetIdx())
    return (
        atom_a.GetAtomicNum() == atom_b.GetAtomicNum()
        and atom_a.GetFormalCharge() == atom_b.GetFormalCharge()
        and atom_a.GetIsAromatic() == atom_b.GetIsAromatic()
        and bond_a is not None
        and bond_b is not None
        and bond_a.GetBondType() == bond_b.GetBondType()
        and bond_a.GetIsAromatic() == bond_b.GetIsAromatic()
    )


def _complete_local_neighbor_correspondence(
    molecule_a, molecule_b, center_a, center_b, mapping
):
    """Build an unambiguous local correspondence without extending the atom map."""
    neighbors_a = [atom.GetIdx() for atom in molecule_a.GetAtomWithIdx(center_a).GetNeighbors()]
    neighbors_b = [atom.GetIdx() for atom in molecule_b.GetAtomWithIdx(center_b).GetNeighbors()]
    if len(neighbors_a) != 4 or len(neighbors_b) != 4:
        return None
    neighbor_set_b = set(neighbors_b)
    correspondence = {}
    for atom_a in neighbors_a:
        if atom_a not in mapping:
            continue
        atom_b = mapping[atom_a]
        if atom_b not in neighbor_set_b or atom_b in correspondence.values():
            return None
        correspondence[atom_a] = atom_b

    remaining_a = [atom for atom in neighbors_a if atom not in correspondence]
    remaining_b = [atom for atom in neighbors_b if atom not in correspondence.values()]
    while remaining_a:
        candidates = {
            atom_a: [
                atom_b
                for atom_b in remaining_b
                if _local_neighbor_compatible(
                    molecule_a.GetAtomWithIdx(atom_a),
                    molecule_b.GetAtomWithIdx(atom_b),
                    center_a,
                    center_b,
                    molecule_a,
                    molecule_b,
                )
            ]
            for atom_a in remaining_a
        }
        forced = [
            (atom_a, values[0])
            for atom_a, values in candidates.items()
            if len(values) == 1
            and sum(values[0] in other for other in candidates.values()) == 1
        ]
        if not forced:
            return None
        for atom_a, atom_b in forced:
            correspondence[atom_a] = atom_b
            remaining_a.remove(atom_a)
            remaining_b.remove(atom_b)
    return correspondence


def _inverted_mapped_stereocenters(
    molecule_a,
    molecule_b,
    mapping,
    requested_pairs,
    force_unique_a,
    force_unique_b,
):
    """Identify unambiguous local inversions and their attached hydrogens."""
    molecule_a = Chem.Mol(molecule_a)
    molecule_b = Chem.Mol(molecule_b)
    Chem.AssignStereochemistry(molecule_a, cleanIt=True, force=True)
    Chem.AssignStereochemistry(molecule_b, cleanIt=True, force=True)
    requested = set(requested_pairs)
    detected = []
    automatic_a = set()
    automatic_b = set()
    for center_a, center_b in sorted(mapping.items()):
        atom_a = molecule_a.GetAtomWithIdx(center_a)
        atom_b = molecule_b.GetAtomWithIdx(center_b)
        sign_a = _TETRAHEDRAL_CHIRAL_SIGNS.get(atom_a.GetChiralTag())
        sign_b = _TETRAHEDRAL_CHIRAL_SIGNS.get(atom_b.GetChiralTag())
        if sign_a is None or sign_b is None:
            continue
        hydrogens_a = [
            atom.GetIdx() for atom in atom_a.GetNeighbors() if atom.GetAtomicNum() == 1
        ]
        hydrogens_b = [
            atom.GetIdx() for atom in atom_b.GetNeighbors() if atom.GetAtomicNum() == 1
        ]
        if len(hydrogens_a) != 1 or len(hydrogens_b) != 1:
            continue
        correspondence = _complete_local_neighbor_correspondence(
            molecule_a, molecule_b, center_a, center_b, mapping
        )
        if correspondence is None:
            continue
        order_a_in_b = [
            correspondence[neighbor.GetIdx()] for neighbor in atom_a.GetNeighbors()
        ]
        order_b = [neighbor.GetIdx() for neighbor in atom_b.GetNeighbors()]
        mapped_sign_b = sign_b * _permutation_parity(order_b, order_a_in_b)
        if sign_a == mapped_sign_b:
            continue

        hydrogen_a = hydrogens_a[0]
        hydrogen_b = hydrogens_b[0]
        if (hydrogen_a, hydrogen_b) in requested:
            action = "explicit_mapping_preserved"
        elif hydrogen_a in force_unique_a or hydrogen_b in force_unique_b:
            action = "already_force_unique"
        elif hydrogen_a in mapping or hydrogen_b in mapping.values():
            # Requested non-H transmutations and nonlocal H mappings remain authoritative.
            action = "explicit_mapping_preserved"
        else:
            automatic_a.add(hydrogen_a)
            automatic_b.add(hydrogen_b)
            action = "automatically_forced_unique"
        detected.append(
            {
                "ligand_a_center_0based": center_a,
                "ligand_b_center_0based": center_b,
                "ligand_a_hydrogen_0based": hydrogen_a,
                "ligand_b_hydrogen_0based": hydrogen_b,
                "comparison": "mapped_local_tetrahedral_parity",
                "hydrogen_action": action,
            }
        )
    return detected, automatic_a, automatic_b


def normalize_junction_bonds(raw):
    """Validate the public per-endpoint junction selector structure."""
    if raw is None:
        return {"ligand_a": [], "ligand_b": []}
    if not isinstance(raw, dict) or set(raw) - {"ligand_a", "ligand_b"}:
        raise HybridMappingError(
            "junction_bonds must contain only ligand_a and ligand_b lists"
        )
    normalized = {"ligand_a": [], "ligand_b": []}
    for endpoint in normalized:
        entries = raw.get(endpoint, [])
        if not isinstance(entries, list):
            raise HybridMappingError(f"junction_bonds.{endpoint} must be a list")
        for position, entry in enumerate(entries, start=1):
            field = f"junction_bonds.{endpoint} entry {position}"
            if not isinstance(entry, dict):
                raise HybridMappingError(f"{field} must be a mapping")
            allowed = {
                "atoms_0based",
                "mapping_labels",
                "smarts",
                "bond_labels",
                "inactive_geometry",
            }
            if set(entry) - allowed:
                raise HybridMappingError(f"{field} contains unsupported fields")
            geometry = str(entry.get("inactive_geometry", "bond_only")).lower()
            if geometry not in INACTIVE_GEOMETRIES:
                raise HybridMappingError(
                    f"{field} inactive_geometry must be 'bond_only' or "
                    "'terminal_z_matrix'"
                )
            has_atoms = "atoms_0based" in entry
            has_smarts = "smarts" in entry or "bond_labels" in entry
            has_mapping_labels = "mapping_labels" in entry
            if sum((has_atoms, has_smarts, has_mapping_labels)) != 1:
                raise HybridMappingError(
                    f"{field} must define exactly one of atoms_0based, "
                    "mapping_labels, or smarts with bond_labels"
                )
            if has_atoms:
                atoms = entry["atoms_0based"]
                if (
                    not isinstance(atoms, (list, tuple))
                    or len(atoms) != 2
                    or any(type(value) is not int or value < 0 for value in atoms)
                    or atoms[0] == atoms[1]
                ):
                    raise HybridMappingError(
                        f"{field} atoms_0based must be two distinct non-negative integers"
                    )
                normalized[endpoint].append(
                    {"atoms_0based": list(atoms), "inactive_geometry": geometry}
                )
            elif has_mapping_labels:
                labels = entry["mapping_labels"]
                if (
                    not isinstance(labels, (list, tuple))
                    or len(labels) != 2
                    or any(type(value) is not int or value <= 0 for value in labels)
                    or labels[0] == labels[1]
                ):
                    raise HybridMappingError(
                        f"{field} mapping_labels must contain two distinct positive integers"
                    )
                normalized[endpoint].append(
                    {"mapping_labels": list(labels), "inactive_geometry": geometry}
                )
            else:
                smarts = entry.get("smarts")
                labels = entry.get("bond_labels")
                query = Chem.MolFromSmarts(str(smarts or ""))
                if query is None:
                    raise HybridMappingError(f"{field} smarts is invalid")
                mapped_labels = _mapped_smarts_labels(query, f"{field} smarts")
                if (
                    not isinstance(labels, (list, tuple))
                    or len(labels) != 2
                    or any(type(value) is not int or value <= 0 for value in labels)
                    or labels[0] == labels[1]
                    or any(value not in mapped_labels for value in labels)
                ):
                    raise HybridMappingError(
                        f"{field} bond_labels must identify two distinct SMARTS labels"
                    )
                normalized[endpoint].append(
                    {
                        "smarts": str(smarts),
                        "bond_labels": list(labels),
                        "inactive_geometry": geometry,
                    }
                )
    return normalized


def normalize_alchemical_bonds(raw):
    """Validate at most one endpoint-specific soft bond per endpoint."""
    if raw is None:
        return {"ligand_a": [], "ligand_b": []}
    if not isinstance(raw, dict) or set(raw) - {"ligand_a", "ligand_b"}:
        raise HybridMappingError(
            "alchemical_bonds must contain only ligand_a and ligand_b lists"
        )
    normalized = {"ligand_a": [], "ligand_b": []}
    for endpoint in normalized:
        entries = raw.get(endpoint, [])
        if not isinstance(entries, list):
            raise HybridMappingError(f"alchemical_bonds.{endpoint} must be a list")
        if len(entries) > 1:
            raise HybridMappingError(
                f"initial soft-bond support allows one changing bond in {endpoint}"
            )
        for position, entry in enumerate(entries, start=1):
            field = f"alchemical_bonds.{endpoint} entry {position}"
            if not isinstance(entry, dict) or set(entry) - {"atoms_0based", "mode"}:
                raise HybridMappingError(
                    f"{field} must contain atoms_0based and optional mode"
                )
            atoms = entry.get("atoms_0based")
            if (
                not isinstance(atoms, (list, tuple))
                or len(atoms) != 2
                or any(type(value) is not int or value < 0 for value in atoms)
                or atoms[0] == atoms[1]
            ):
                raise HybridMappingError(
                    f"{field} atoms_0based must be two distinct non-negative integers"
                )
            if str(entry.get("mode", "soft_bond")).lower() != "soft_bond":
                raise HybridMappingError(f"{field} mode must be 'soft_bond'")
            normalized[endpoint].append(
                {"atoms_0based": list(atoms), "mode": "soft_bond"}
            )
    return normalized


def _resolve_junction_selector(molecule, entry, endpoint, matched_labels=None):
    if "atoms_0based" in entry:
        return tuple(entry["atoms_0based"])
    if "mapping_labels" in entry:
        matched_labels = matched_labels or {}
        if any(label not in matched_labels for label in entry["mapping_labels"]):
            raise HybridMappingError(
                f"junction_bonds.{endpoint} mapping_labels are not present in the "
                "resolved paired SMARTS match"
            )
        return tuple(matched_labels[label] for label in entry["mapping_labels"])
    query = Chem.MolFromSmarts(entry["smarts"])
    labels = _mapped_smarts_labels(query, f"junction_bonds.{endpoint} smarts")
    label_a, label_b = entry["bond_labels"]
    matches = molecule.GetSubstructMatches(query, uniquify=False, useChirality=True)
    pairs = {
        (int(match[labels[label_a]]), int(match[labels[label_b]]))
        for match in matches
    }
    if len(pairs) != 1:
        raise HybridMappingError(
            f"junction_bonds.{endpoint} SMARTS must resolve to exactly one ordered "
            f"bond; observed {len(pairs)}"
        )
    return next(iter(pairs))


def _resolve_junction_bonds(
    molecule, mapping_atoms, entries, endpoint, matched_labels=None,
    alchemical_bonds=(),
):
    unique = set(range(molecule.GetNumAtoms())) - set(mapping_atoms)
    open_bonds = {tuple(sorted(pair)) for pair in alchemical_bonds}
    selected = set()
    z_matrix_roots = set()
    resolved = []
    seen = set()
    for entry in entries:
        core, root = _resolve_junction_selector(
            molecule, entry, endpoint, matched_labels
        )
        if core >= molecule.GetNumAtoms() or root >= molecule.GetNumAtoms():
            raise HybridMappingError(
                f"junction_bonds.{endpoint} atom is outside the molecule"
            )
        bond = molecule.GetBondBetweenAtoms(core, root)
        if bond is None:
            raise HybridMappingError(
                f"junction_bonds.{endpoint} atoms {core}:{root} are not bonded"
            )
        if bond.IsInRing() and not open_bonds:
            raise HybridMappingError(
                f"junction_bonds.{endpoint} atoms {core}:{root} form a ring bond"
            )
        if core not in mapping_atoms or root not in unique:
            raise HybridMappingError(
                f"junction_bonds.{endpoint} atoms must be ordered [mapped core, "
                "endpoint-unique branch root]"
            )
        pending = [root]
        component = set()
        while pending:
            atom = pending.pop()
            if atom in component:
                continue
            component.add(atom)
            pending.extend(
                neighbor.GetIdx()
                for neighbor in molecule.GetAtomWithIdx(atom).GetNeighbors()
                if neighbor.GetIdx() != core and neighbor.GetIdx() not in component
                and tuple(sorted((atom, neighbor.GetIdx()))) not in open_bonds
            )
        if not component <= unique:
            raise HybridMappingError(
                f"junction_bonds.{endpoint} {core}:{root} does not isolate an "
                "endpoint-unique branch"
            )
        key = (core, root)
        if key in seen or selected & component:
            raise HybridMappingError(
                f"junction_bonds.{endpoint} contains a duplicate or overlapping branch"
            )
        seen.add(key)
        selected.update(component)
        geometry = entry["inactive_geometry"]
        if geometry == "terminal_z_matrix":
            z_matrix_roots.add(root)
        resolved.append(
            {
                "endpoint": endpoint[-1],
                "boundary_atoms_0based": [core, root],
                "branch_atoms_0based": sorted(component),
                "inactive_geometry": geometry,
                "selection_source": "explicit",
                "fallback_reason": None,
                "selector": dict(entry),
            }
        )
    return selected, z_matrix_roots, resolved


def _has_mapped_heavy_z_matrix_chain(molecule, mapped, core, root):
    for atom_b_obj in molecule.GetAtomWithIdx(core).GetNeighbors():
        atom_b = atom_b_obj.GetIdx()
        if atom_b == root or atom_b not in mapped or atom_b_obj.GetAtomicNum() == 1:
            continue
        for atom_a_obj in atom_b_obj.GetNeighbors():
            atom_a = atom_a_obj.GetIdx()
            if (
                atom_a != core
                and atom_a in mapped
                and atom_a_obj.GetAtomicNum() != 1
            ):
                return True
    return False


def _automatic_junction_bonds(
    molecule, mapping_atoms, endpoint, *, alchemical_bonds=(), excluded_atoms=()
):
    """Resolve every unambiguous endpoint-unique branch junction."""
    mapped = set(mapping_atoms)
    unique = set(range(molecule.GetNumAtoms())) - mapped
    open_bonds = {tuple(sorted(pair)) for pair in alchemical_bonds}
    excluded = set(excluded_atoms)
    pending = set(unique)
    components = []
    while pending:
        start = min(pending)
        stack = [start]
        component = set()
        while stack:
            atom = stack.pop()
            if atom in component:
                continue
            component.add(atom)
            pending.discard(atom)
            stack.extend(
                neighbor.GetIdx()
                for neighbor in molecule.GetAtomWithIdx(atom).GetNeighbors()
                if neighbor.GetIdx() in unique
                and neighbor.GetIdx() not in component
                and tuple(sorted((atom, neighbor.GetIdx()))) not in open_bonds
            )
        components.append(component)

    selected = set()
    z_matrix_roots = set()
    resolved = []
    warnings = []
    for component in components:
        if component & excluded:
            continue
        boundaries = sorted(
            {
                (neighbor.GetIdx(), atom)
                for atom in component
                for neighbor in molecule.GetAtomWithIdx(atom).GetNeighbors()
                if neighbor.GetIdx() in mapped
                and tuple(sorted((atom, neighbor.GetIdx()))) not in open_bonds
            }
        )
        if len(boundaries) != 1:
            if boundaries:
                warnings.append(
                    {
                        "endpoint": endpoint[-1],
                        "branch_atoms_0based": sorted(component),
                        "reason": "multiple_mapped_core_boundaries",
                        "boundary_count": len(boundaries),
                    }
                )
            continue
        core, root = boundaries[0]
        has_frame = _has_mapped_heavy_z_matrix_chain(
            molecule, mapped, core, root
        )
        geometry = "terminal_z_matrix" if has_frame else "bond_only"
        fallback = None if has_frame else "no_mapped_heavy_reference_chain"
        selected.update(component)
        if has_frame:
            z_matrix_roots.add(root)
        else:
            warnings.append(
                {
                    "endpoint": endpoint[-1],
                    "branch_atoms_0based": sorted(component),
                    "reason": fallback,
                    "boundary_atoms_0based": [core, root],
                }
            )
        resolved.append(
            {
                "endpoint": endpoint[-1],
                "boundary_atoms_0based": [core, root],
                "branch_atoms_0based": sorted(component),
                "inactive_geometry": geometry,
                "selection_source": "automatic",
                "fallback_reason": fallback,
                "selector": {"automatic_junction": True},
            }
        )
    return selected, z_matrix_roots, resolved, warnings


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


def _mapped_smarts_labels(query, field):
    labels = {}
    for atom in query.GetAtoms():
        label = int(atom.GetAtomMapNum())
        if label <= 0:
            continue
        if label in labels:
            raise HybridMappingError(f"{field} contains duplicate atom-map label {label}")
        labels[label] = atom.GetIdx()
    if not labels:
        raise HybridMappingError(f"{field} must contain atom-map labels")
    return labels


def _paired_smarts_transmutation_map(molecule_a, molecule_b, settings):
    smarts_a = settings.get("ligand_a_smarts")
    smarts_b = settings.get("ligand_b_smarts")
    query_a = Chem.MolFromSmarts(str(smarts_a or ""))
    query_b = Chem.MolFromSmarts(str(smarts_b or ""))
    if query_a is None or query_b is None:
        raise HybridMappingError("paired transmutation SMARTS is invalid")
    labels_a = _mapped_smarts_labels(query_a, "ligand_a_smarts")
    labels_b = _mapped_smarts_labels(query_b, "ligand_b_smarts")
    shared = set(labels_a) & set(labels_b)
    if not shared:
        raise HybridMappingError("paired transmutation SMARTS have no shared labels")
    matches_a = molecule_a.GetSubstructMatches(query_a, uniquify=False, useChirality=True)
    matches_b = molecule_b.GetSubstructMatches(query_b, uniquify=False, useChirality=True)
    if not matches_a or not matches_b:
        raise HybridMappingError("paired transmutation SMARTS do not match both ligands")

    base = find_covalent_atom_map(molecule_a, molecule_b)
    requested_inactive = settings.get("inactive_bonded_labels") or {}
    selected_labels = {
        side: {int(value) for value in requested_inactive.get(side, [])}
        for side in ("ligand_a", "ligand_b")
    }
    if not selected_labels["ligand_a"] <= (set(labels_a) - set(labels_b)):
        raise HybridMappingError(
            "inactive_bonded_labels.ligand_a must reference ligand-A-only SMARTS labels"
        )
    if not selected_labels["ligand_b"] <= (set(labels_b) - set(labels_a)):
        raise HybridMappingError(
            "inactive_bonded_labels.ligand_b must reference ligand-B-only SMARTS labels"
        )

    candidates = []
    for match_a in matches_a:
        atoms_a = {label: int(match_a[index]) for label, index in labels_a.items()}
        for match_b in matches_b:
            atoms_b = {label: int(match_b[index]) for label, index in labels_b.items()}
            forced = {atoms_a[label]: atoms_b[label] for label in shared}
            mapping = {
                atom_a: atom_b for atom_a, atom_b in base.items()
                if atom_a not in forced and atom_b not in forced.values()
            }
            mapping.update(forced)
            transmuted = {
                (atoms_a[label], atoms_b[label]) for label in shared
                if molecule_a.GetAtomWithIdx(atoms_a[label]).GetAtomicNum()
                != molecule_b.GetAtomWithIdx(atoms_b[label]).GetAtomicNum()
            }
            try:
                mapping = complete_covalent_atom_map(
                    molecule_a,
                    molecule_b,
                    mapping,
                    required_pairs=list(forced.items()),
                    transmuted_pairs=transmuted,
                )
            except HybridMappingError:
                continue
            rmsd = _direct_rmsd(molecule_a, molecule_b, mapping)
            inactive_a = {atoms_a[label] for label in selected_labels["ligand_a"]}
            inactive_b = {atoms_b[label] for label in selected_labels["ligand_b"]}
            candidates.append(
                (-len(mapping), rmsd, tuple(sorted(mapping.items())), mapping,
                 transmuted, inactive_a, inactive_b, atoms_a, atoms_b)
            )
    if not candidates:
        raise HybridMappingError("no valid paired-SMARTS transmutation mapping was found")
    return min(candidates, key=lambda item: item[:3])[3:]


def _explicit_pairs_map(molecule_a, molecule_b, settings):
    pairs = _strict_explicit_pairs(settings.get("pairs_0based"))
    force_unique_a = set(_strict_nonnegative_indices(
        settings.get("force_unique_atoms_a_0based", []),
        "explicit_pairs force_unique_atoms_a_0based",
    ))
    force_unique_b = set(_strict_nonnegative_indices(
        settings.get("force_unique_atoms_b_0based", []),
        "explicit_pairs force_unique_atoms_b_0based",
    ))
    inactive_a = set(
        _strict_nonnegative_indices(
            settings.get("inactive_bonded_atoms_a_0based", []),
            "explicit_pairs inactive_bonded_atoms_a_0based",
        )
    )
    inactive_b = set(
        _strict_nonnegative_indices(
            settings.get("inactive_bonded_atoms_b_0based", []),
            "explicit_pairs inactive_bonded_atoms_b_0based",
        )
    )
    geometry = str(settings.get("inactive_bonded_geometry", "bond_only")).lower()
    if geometry not in {"bond_only", "terminal_z_matrix"}:
        raise HybridMappingError(
            "explicit_pairs inactive_bonded_geometry must be 'bond_only' or "
            "'terminal_z_matrix'"
        )
    if geometry == "terminal_z_matrix" and not (inactive_a or inactive_b):
        raise HybridMappingError(
            "explicit_pairs terminal_z_matrix requires at least one inactive bonded atom"
        )
    mapping = dict(pairs)
    transmuted = {
        (atom_a, atom_b) for atom_a, atom_b in pairs
        if 0 <= atom_a < molecule_a.GetNumAtoms()
        and 0 <= atom_b < molecule_b.GetNumAtoms()
        and molecule_a.GetAtomWithIdx(atom_a).GetAtomicNum()
        != molecule_b.GetAtomWithIdx(atom_b).GetAtomicNum()
    }
    requested_pairs = tuple(pairs)
    requested = set(requested_pairs)
    alchemical = normalize_alchemical_bonds(settings.get("alchemical_bonds"))
    alchemical_a = {
        tuple(sorted(entry["atoms_0based"])) for entry in alchemical["ligand_a"]
    }
    alchemical_b = {
        tuple(sorted(entry["atoms_0based"])) for entry in alchemical["ligand_b"]
    }
    (
        inverted_stereocenters,
        automatic_stereo_hydrogens_a,
        automatic_stereo_hydrogens_b,
    ) = _inverted_mapped_stereocenters(
        molecule_a,
        molecule_b,
        mapping,
        requested_pairs,
        force_unique_a,
        force_unique_b,
    )
    force_unique_a.update(automatic_stereo_hydrogens_a)
    force_unique_b.update(automatic_stereo_hydrogens_b)
    mapping = complete_covalent_atom_map(
        molecule_a,
        molecule_b,
        mapping,
        required_pairs=pairs,
        transmuted_pairs=transmuted,
        alchemical_bonds_a=alchemical_a,
        alchemical_bonds_b=alchemical_b,
        force_unique_atoms_a=force_unique_a,
        force_unique_atoms_b=force_unique_b,
    )
    reverse_mapping = {atom_b: atom_a for atom_a, atom_b in mapping.items()}
    for endpoint, selected in (("ligand_a", alchemical_a), ("ligand_b", alchemical_b)):
        molecule = molecule_a if endpoint == "ligand_a" else molecule_b
        other = molecule_b if endpoint == "ligand_a" else molecule_a
        endpoint_mapping = mapping if endpoint == "ligand_a" else reverse_mapping
        for atom1, atom2 in selected:
            if atom1 >= molecule.GetNumAtoms() or atom2 >= molecule.GetNumAtoms():
                raise HybridMappingError(
                    f"alchemical_bonds.{endpoint} atom is outside the molecule"
                )
            if molecule.GetBondBetweenAtoms(atom1, atom2) is None:
                raise HybridMappingError(
                    f"alchemical_bonds.{endpoint} selected bond does not exist"
                )
            mapped = [atom in endpoint_mapping for atom in (atom1, atom2)]
            if all(mapped) and other.GetBondBetweenAtoms(
                endpoint_mapping[atom1], endpoint_mapping[atom2]
            ) is not None:
                raise HybridMappingError(
                    f"alchemical_bonds.{endpoint} counterpart must be absent in the other ligand"
                )
    if any(atom >= molecule_a.GetNumAtoms() for atom in inactive_a):
        raise HybridMappingError(
            "explicit_pairs inactive ligand-A atom is outside the molecule"
        )
    if any(atom >= molecule_b.GetNumAtoms() for atom in inactive_b):
        raise HybridMappingError(
            "explicit_pairs inactive ligand-B atom is outside the molecule"
        )
    unique_a = set(range(molecule_a.GetNumAtoms())) - set(mapping)
    unique_b = set(range(molecule_b.GetNumAtoms())) - set(mapping.values())
    if not force_unique_a <= unique_a or not force_unique_b <= unique_b:
        raise HybridMappingError("force-unique hydrogens must remain endpoint-unique")
    if not inactive_a <= unique_a:
        raise HybridMappingError(
            "explicit_pairs inactive ligand-A atoms must be endpoint-unique"
        )
    if not inactive_b <= unique_b:
        raise HybridMappingError(
            "explicit_pairs inactive ligand-B atoms must be endpoint-unique"
        )
    completed_hydrogens = set(mapping.items()) - requested
    return (
        mapping,
        transmuted,
        inactive_a,
        inactive_b,
        {},
        {},
        requested_pairs,
        completed_hydrogens,
        force_unique_a,
        force_unique_b,
        inverted_stereocenters,
        automatic_stereo_hydrogens_a,
        automatic_stereo_hydrogens_b,
    )


def build_hybrid_atom_map(parameters_a, parameters_b, settings):
    raw_a = parameters_a.molecule.to_rdkit()
    raw_b = parameters_b.molecule.to_rdkit()
    aromatic_before = {
        "ligand_a": sum(atom.GetIsAromatic() for atom in raw_a.GetAtoms()),
        "ligand_b": sum(atom.GetIsAromatic() for atom in raw_b.GetAtoms()),
    }
    molecule_a = normalize_mapping_aromaticity(raw_a)
    molecule_b = normalize_mapping_aromaticity(raw_b)
    aromatic_after = {
        "ligand_a": sum(atom.GetIsAromatic() for atom in molecule_a.GetAtoms()),
        "ligand_b": sum(atom.GetIsAromatic() for atom in molecule_b.GetAtoms()),
    }
    method = settings.get("method", "mcs")
    if method != "explicit_pairs" and (
        "force_unique_atoms_a_0based" in settings
        or "force_unique_atoms_b_0based" in settings
    ):
        raise HybridMappingError(
            "force_unique_atoms_*_0based requires mapping.method explicit_pairs"
        )
    requested_pairs = ()
    completed_hydrogen_pairs = set()
    force_unique_a = set()
    force_unique_b = set()
    inverted_stereocenters = []
    automatic_stereo_hydrogens_a = set()
    automatic_stereo_hydrogens_b = set()
    if method == "mcs":
        mapping = find_covalent_atom_map(molecule_a, molecule_b)
    elif method == "mcs_core_smarts":
        smarts = settings.get("smarts")
        if not isinstance(smarts, str) or not smarts.strip():
            raise HybridMappingError(
                "mcs_core_smarts mapping requires workflow.alchemy.mapping.smarts"
            )
        mapping = _smarts_constrained_map(molecule_a, molecule_b, smarts.strip())
        transmuted = set()
        inactive_a = set()
        inactive_b = set()
        matched_labels_a = {}
        matched_labels_b = {}
    elif method == "paired_smarts_transmutation":
        (
            mapping,
            transmuted,
            inactive_a,
            inactive_b,
            matched_labels_a,
            matched_labels_b,
        ) = _paired_smarts_transmutation_map(molecule_a, molecule_b, settings)
    elif method == "explicit_pairs":
        (
            mapping,
            transmuted,
            inactive_a,
            inactive_b,
            matched_labels_a,
            matched_labels_b,
            requested_pairs,
            completed_hydrogen_pairs,
            force_unique_a,
            force_unique_b,
            inverted_stereocenters,
            automatic_stereo_hydrogens_a,
            automatic_stereo_hydrogens_b,
        ) = _explicit_pairs_map(molecule_a, molecule_b, settings)
    else:
        raise HybridMappingError(
            "workflow.alchemy.mapping.method must be 'mcs', 'mcs_core_smarts', "
            "'paired_smarts_transmutation', or 'explicit_pairs'"
        )
    if method == "mcs":
        transmuted = set()
        inactive_a = set()
        inactive_b = set()
        matched_labels_a = {}
        matched_labels_b = {}
    alchemical_bonds = normalize_alchemical_bonds(settings.get("alchemical_bonds"))
    alchemical_bond_pairs = {
        endpoint: {
            tuple(sorted(entry["atoms_0based"]))
            for entry in entries
        }
        for endpoint, entries in alchemical_bonds.items()
    }
    if settings.get("alchemical_bonds") is not None and method != "explicit_pairs":
        raise HybridMappingError(
            "alchemical_bonds initially require mapping.method explicit_pairs"
        )
    raw_junctions = settings.get("junction_bonds")
    legacy_requested = bool(
        settings.get("inactive_bonded_labels")
        or settings.get("inactive_bonded_atoms_a_0based")
        or settings.get("inactive_bonded_atoms_b_0based")
        or "inactive_bonded_geometry" in settings
    )
    if raw_junctions is not None and legacy_requested:
        raise HybridMappingError(
            "junction_bonds cannot be combined with legacy inactive_bonded_* settings"
        )
    junction_settings = normalize_junction_bonds(raw_junctions)
    resolved_junctions = []
    junction_warnings = []
    z_matrix_roots_a = set()
    z_matrix_roots_b = set()
    if raw_junctions is not None:
        inactive_a, z_matrix_roots_a, resolved_a = _resolve_junction_bonds(
            molecule_a,
            set(mapping),
            junction_settings["ligand_a"],
            "ligand_a",
            matched_labels_a,
            alchemical_bond_pairs["ligand_a"],
        )
        inactive_b, z_matrix_roots_b, resolved_b = _resolve_junction_bonds(
            molecule_b,
            set(mapping.values()),
            junction_settings["ligand_b"],
            "ligand_b",
            matched_labels_b,
            alchemical_bond_pairs["ligand_b"],
        )
        resolved_junctions = resolved_a + resolved_b
    else:
        geometry = str(settings.get("inactive_bonded_geometry", "bond_only")).lower()
        components_a = _inactive_branch_components(
            molecule_a, inactive_a, set(range(molecule_a.GetNumAtoms())) - set(mapping), "a"
        )
        components_b = _inactive_branch_components(
            molecule_b,
            inactive_b,
            set(range(molecule_b.GetNumAtoms())) - set(mapping.values()),
            "b",
        )
        if geometry == "terminal_z_matrix":
            z_matrix_roots_a = {component["root"] for component in components_a}
            z_matrix_roots_b = {component["root"] for component in components_b}
        resolved_junctions = [
            {
                "endpoint": endpoint,
                "boundary_atoms_0based": [component["center"], component["root"]],
                "branch_atoms_0based": list(component["atoms"]),
                "inactive_geometry": geometry,
                "selection_source": "legacy",
                "fallback_reason": None,
                "selector": {"legacy_inactive_atoms_0based": list(component["atoms"])},
            }
            for endpoint, components in (("a", components_a), ("b", components_b))
            for component in components
        ]
    auto_a, auto_z_a, auto_resolved_a, auto_warnings_a = _automatic_junction_bonds(
        molecule_a,
        set(mapping),
        "ligand_a",
        alchemical_bonds=alchemical_bond_pairs["ligand_a"],
        excluded_atoms=inactive_a,
    )
    auto_b, auto_z_b, auto_resolved_b, auto_warnings_b = _automatic_junction_bonds(
        molecule_b,
        set(mapping.values()),
        "ligand_b",
        alchemical_bonds=alchemical_bond_pairs["ligand_b"],
        excluded_atoms=inactive_b,
    )
    inactive_a.update(auto_a)
    inactive_b.update(auto_b)
    z_matrix_roots_a.update(auto_z_a)
    z_matrix_roots_b.update(auto_z_b)
    resolved_junctions.extend(auto_resolved_a + auto_resolved_b)
    junction_warnings.extend(auto_warnings_a + auto_warnings_b)
    rmsd = _direct_rmsd(molecule_a, molecule_b, mapping)
    maximum = settings.get("max_mapped_rmsd_a")
    if maximum is not None and rmsd > float(maximum):
        raise HybridMappingError(
            f"mapped ligand RMSD {rmsd:.3f} A exceeds max_mapped_rmsd_a {float(maximum):.3f} A"
        )
    return mapping, {
        "schema_version": 3,
        "aromaticity_model": "rdkit",
        "aromatic_atom_count_before": aromatic_before,
        "aromatic_atom_count_after": aromatic_after,
        "method": method,
        "smarts": settings.get("smarts") if method == "mcs_core_smarts" else None,
        "ligand_a_smarts": settings.get("ligand_a_smarts") if method == "paired_smarts_transmutation" else None,
        "ligand_b_smarts": settings.get("ligand_b_smarts") if method == "paired_smarts_transmutation" else None,
        "requested_pairs_0based": [
            list(pair) for pair in requested_pairs
        ],
        "auto_completed_hydrogen_pairs_0based": [
            list(pair) for pair in sorted(completed_hydrogen_pairs)
        ],
        "force_unique_atoms_a_0based": sorted(force_unique_a),
        "force_unique_atoms_b_0based": sorted(force_unique_b),
        "automatically_forced_unique_stereo_hydrogens": {
            "ligand_a": sorted(automatic_stereo_hydrogens_a),
            "ligand_b": sorted(automatic_stereo_hydrogens_b),
        },
        "inverted_stereocenters_0based": inverted_stereocenters,
        "transmuted_pairs_0based": [list(pair) for pair in sorted(transmuted)],
        "transmuted_pairs_1based": [[a + 1, b + 1] for a, b in sorted(transmuted)],
        "inactive_bonded_atoms_a_0based": sorted(inactive_a),
        "inactive_bonded_atoms_b_0based": sorted(inactive_b),
        "inactive_bonded_geometry": str(
            settings.get("inactive_bonded_geometry", "mixed" if len({
                entry["inactive_geometry"] for entry in resolved_junctions
            }) > 1 else (
                resolved_junctions[0]["inactive_geometry"] if resolved_junctions else "bond_only"
            ))
        ).lower(),
        "junction_bonds": junction_settings if raw_junctions is not None else None,
        "resolved_junction_bonds": resolved_junctions,
        "automatic_junction_warnings": junction_warnings,
        "inactive_z_matrix_root_atoms_a_0based": sorted(z_matrix_roots_a),
        "inactive_z_matrix_root_atoms_b_0based": sorted(z_matrix_roots_b),
        "alchemical_bonds": alchemical_bonds if settings.get("alchemical_bonds") is not None else None,
        "matched_smarts_labels_a_0based": matched_labels_a,
        "matched_smarts_labels_b_0based": matched_labels_b,
        "mapped_atom_count": len(mapping),
        "mapped_heavy_atom_count": sum(
            molecule_a.GetAtomWithIdx(atom).GetAtomicNum() != 1 for atom in mapping
        ),
        "mapped_direct_rmsd_angstrom": rmsd,
        "map_a_to_b_0based": {
            int(atom_a): int(atom_b) for atom_a, atom_b in sorted(mapping.items())
        },
    }
