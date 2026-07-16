"""Amber-mask selections extended with role-aware ligand SMARTS leaves."""

from __future__ import annotations

import re
from pathlib import Path


SMARTS_LEAF = re.compile(
    r'#(?P<role>ligand_a|ligand_b|ligand|bound|unbound):"(?P<smarts>(?:\\.|[^"\\])*)"'
)


class SelectionError(ValueError):
    pass


def contains_smarts_selection(expression):
    return isinstance(expression, str) and SMARTS_LEAF.search(expression) is not None


def _endpoint_role(role, endpoint):
    if role in ("ligand", "ligand_a", "ligand_b"):
        return role
    endpoint = None if endpoint is None else str(endpoint).lower()
    if endpoint not in ("a", "b"):
        raise SelectionError(
            f"#{role} requires physical endpoint context 'a' or 'b'; "
            "bound/unbound roles are undefined at the ATM midpoint"
        )
    if endpoint == "a":
        return "ligand_a" if role == "bound" else "ligand_b"
    return "ligand_b" if role == "bound" else "ligand_a"


def _unescape_smarts(value):
    return value.replace(r'\"', '"').replace(r"\\", "\\")


def _metadata(keywords):
    metadata = keywords.get("SELECTION_METADATA") if keywords else None
    if not isinstance(metadata, dict):
        raise SelectionError(
            "SMARTS selection metadata is missing; re-prepare this pair with the current atom-rbfe version"
        )
    return metadata


def _load_molecule(entry, base_dir):
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise SelectionError("Ligand SMARTS selections require RDKit") from exc
    path = Path(entry.get("structure_file", ""))
    if not path.is_absolute():
        path = Path(base_dir or ".") / path
    path = path.resolve()
    if not path.exists():
        raise SelectionError(f"SMARTS selection structure does not exist: {path}")
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    molecule = next((mol for mol in supplier if mol is not None), None)
    if molecule is None:
        raise SelectionError(f"Could not read SMARTS selection structure: {path}")
    mapping = [int(value) for value in entry.get("system_atom_indices", [])]
    if molecule.GetNumAtoms() != len(mapping):
        raise SelectionError(
            f"SMARTS selection mapping mismatch for {path}: molecule has {molecule.GetNumAtoms()} atoms "
            f"but metadata has {len(mapping)} system atom indices"
        )
    return molecule, mapping


def _match_identity(identity, smarts, keywords, base_dir):
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise SelectionError("Ligand SMARTS selections require RDKit") from exc
    entry = _metadata(keywords).get(identity)
    if not isinstance(entry, dict):
        raise SelectionError(f"SMARTS selection metadata has no {identity} entry")
    molecule, mapping = _load_molecule(entry, base_dir)
    if smarts == "*":
        return set(mapping)
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise SelectionError(f"Invalid SMARTS pattern {smarts!r}")
    matches = molecule.GetSubstructMatches(query, uniquify=False)
    if not matches:
        raise SelectionError(f"SMARTS {smarts!r} did not match {identity}")
    return {mapping[index] for match in matches for index in match}


def resolve_smarts_atoms(expression, keywords, *, endpoint=None, base_dir=None):
    """Resolve all SMARTS leaves and return their union as zero-based system indices."""
    if not isinstance(expression, str) or not expression.strip():
        raise SelectionError("selection must be a non-empty string")
    selected = set()
    leaves = list(SMARTS_LEAF.finditer(expression))
    if not leaves:
        raise SelectionError(f"selection contains no role-aware SMARTS leaf: {expression!r}")
    for leaf in leaves:
        role = _endpoint_role(leaf.group("role"), endpoint)
        smarts = _unescape_smarts(leaf.group("smarts"))
        identities = ("ligand_a", "ligand_b") if role == "ligand" else (role,)
        for identity in identities:
            selected.update(_match_identity(identity, smarts, keywords, base_dir))
    return sorted(selected)


def compile_selection_expression(expression, keywords, *, endpoint=None, base_dir=None):
    """Replace SMARTS leaves with explicit 1-based Amber atom-mask expressions."""
    if not isinstance(expression, str) or not expression.strip():
        raise SelectionError("selection must be a non-empty string")
    if "#" in expression and not contains_smarts_selection(expression):
        raise SelectionError(f"Malformed role-aware SMARTS selection in {expression!r}")

    def replace(match):
        role = _endpoint_role(match.group("role"), endpoint)
        smarts = _unescape_smarts(match.group("smarts"))
        identities = ("ligand_a", "ligand_b") if role == "ligand" else (role,)
        atoms = set()
        for identity in identities:
            atoms.update(_match_identity(identity, smarts, keywords, base_dir))
        atom_mask = ",".join(str(index + 1) for index in sorted(atoms))
        return f"(@{atom_mask})"

    return SMARTS_LEAF.sub(replace, expression)
