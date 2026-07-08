from __future__ import annotations

from pathlib import Path

from openmm.app import PDBFile
from openmm.unit import nanometer


def _as_index_list(values):
    if values is None:
        return []
    return [int(value) for value in values]


def atm_swapped_positions(positions, keywords):
    """Return diagnostic positions with RBFE ligand variable regions exchanged."""
    lig1_attach = keywords.get("LIGAND1_ATTACH_ATOM")
    lig2_attach = keywords.get("LIGAND2_ATTACH_ATOM")
    if lig1_attach is None or lig2_attach is None:
        return None

    positions_nm = positions.value_in_unit(nanometer)
    swapped = [positions_nm[i] for i in range(len(positions_nm))]
    lig1_attach = int(lig1_attach)
    lig2_attach = int(lig2_attach)
    displacement = positions_nm[lig2_attach] - positions_nm[lig1_attach]

    lig1_var_atoms = _as_index_list(keywords.get("LIGAND1_VAR_ATOMS") or keywords.get("LIGAND1_ATOMS"))
    lig2_var_atoms = _as_index_list(keywords.get("LIGAND2_VAR_ATOMS") or keywords.get("LIGAND2_ATOMS"))
    for atom in lig1_var_atoms:
        swapped[atom] = positions_nm[atom] + displacement
    for atom in lig2_var_atoms:
        swapped[atom] = positions_nm[atom] - displacement

    return swapped * nanometer


def write_atm_swapped_pdb(topology, positions, keywords, path):
    swapped = atm_swapped_positions(positions, keywords)
    if swapped is None:
        return False
    path = Path(path)
    with open(path, "w") as handle:
        PDBFile.writeFile(topology, swapped, handle, keepIds=True)
    return True
