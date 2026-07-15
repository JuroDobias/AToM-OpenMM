"""Role-aware physical endpoint systems and ATM/native state conversion."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import openmm as mm
from openmm import unit

from atom_openmm.atm_coordinates import atm_swapped_positions
from atom_openmm.ommsystem import OMMSystemRBFENativeEndpoint


@dataclass(frozen=True)
class EndpointLigandRoles:
    endpoint: str
    bound_ligand: str
    unbound_ligand: str
    bound_atoms: tuple[int, ...]
    unbound_atoms: tuple[int, ...]


def endpoint_ligand_roles(keywords, endpoint):
    endpoint = str(endpoint).lower()
    if endpoint not in ("a", "b"):
        raise ValueError("endpoint must be 'a' or 'b'")
    ligand1 = tuple(int(value) for value in keywords["LIGAND1_ATOMS"])
    ligand2 = tuple(int(value) for value in keywords["LIGAND2_ATOMS"])
    if endpoint == "a":
        return EndpointLigandRoles("a", "L1", "L2", ligand1, ligand2)
    return EndpointLigandRoles("b", "L2", "L1", ligand2, ligand1)


def _swap_paired_keyword_values(keywords):
    result = deepcopy(keywords)
    visited = set()
    for key in list(keywords):
        if key in visited:
            continue
        partner = None
        if "LIGAND1" in key:
            partner = key.replace("LIGAND1", "LIGAND2")
        elif "LIGAND2" in key:
            partner = key.replace("LIGAND2", "LIGAND1")
        elif "LIG1" in key:
            partner = key.replace("LIG1", "LIG2")
        elif "LIG2" in key:
            partner = key.replace("LIG2", "LIG1")
        if partner is not None and partner in keywords:
            result[key] = deepcopy(keywords[partner])
            result[partner] = deepcopy(keywords[key])
            visited.update((key, partner))
    return result


def native_endpoint_keywords(keywords, endpoint):
    roles = endpoint_ligand_roles(keywords, endpoint)
    result = deepcopy(keywords) if roles.endpoint == "a" else _swap_paired_keyword_values(keywords)
    # EXCLUSION_POT_MOL1_INDEXES identifies the receptor region in legacy ATM
    # inputs. Its interaction partner must always be the unbound ligand role.
    if result.get("EXCLUSION_POT_MOL1_INDEXES"):
        result["EXCLUSION_POT_MOL2_INDEXES"] = list(roles.unbound_atoms)
    result["NATIVE_ENDPOINT"] = roles.endpoint
    result["NATIVE_BOUND_LIGAND"] = roles.bound_ligand
    result["NATIVE_UNBOUND_LIGAND"] = roles.unbound_ligand
    return result


def map_atm_to_native_positions(positions, endpoint, keywords):
    endpoint = str(endpoint).lower()
    if endpoint == "a":
        return positions
    if endpoint != "b":
        raise ValueError("endpoint must be 'a' or 'b'")
    mapped = atm_swapped_positions(positions, keywords)
    if mapped is None:
        raise ValueError("ATM/native B conversion requires ligand attachment atoms")
    return mapped


def map_native_to_atm_positions(positions, endpoint, keywords):
    # The common/variable-region ATM coordinate exchange is involutive.
    return map_atm_to_native_positions(positions, endpoint, keywords)


def transfer_state_to_context(source_state, context, *, endpoint, keywords, to_native):
    positions = source_state.getPositions()
    mapped = (
        map_atm_to_native_positions(positions, endpoint, keywords)
        if to_native else map_native_to_atm_positions(positions, endpoint, keywords)
    )
    box_vectors = source_state.getPeriodicBoxVectors()
    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(mapped)
    try:
        velocities = source_state.getVelocities()
        if velocities is not None:
            context.setVelocities(velocities)
    except Exception:
        pass
    return mapped


def write_converted_state(
    source_path,
    endpoint_system,
    output_path,
    *,
    endpoint,
    keywords,
    to_native,
    platform=None,
    platform_properties=None,
):
    source = mm.XmlSerializer.deserialize(Path(source_path).read_text())
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    if platform is None:
        context = mm.Context(endpoint_system.system, integrator)
    else:
        context = mm.Context(
            endpoint_system.system, integrator, platform, platform_properties or {}
        )
    try:
        positions = transfer_state_to_context(
            source,
            context,
            endpoint=endpoint,
            keywords=keywords,
            to_native=to_native,
        )
        state = context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=False,
            enforcePeriodicBox=True,
        )
        Path(output_path).write_text(mm.XmlSerializer.serialize(state))
        endpoint_system.positions = positions
        endpoint_system.boxvectors = state.getPeriodicBoxVectors()
    finally:
        del context
        del integrator
    return Path(output_path)


def create_native_endpoint_system(atm_system, endpoint, *, rest2=True, logger=None):
    roles = endpoint_ligand_roles(atm_system.keywords, endpoint)
    keywords = native_endpoint_keywords(atm_system.keywords, endpoint)
    keywords["REST2_ENABLED"] = bool(rest2)
    native = OMMSystemRBFENativeEndpoint(
        atm_system.basename,
        keywords,
        atm_system.pdbtopfile,
        atm_system.systemfile,
        logger or atm_system.logger,
    )
    native.create_system()
    native.endpoint_roles = roles
    return native
