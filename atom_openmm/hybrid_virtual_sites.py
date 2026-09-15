from __future__ import annotations

from dataclasses import replace

import numpy as np
import openmm as mm
from openmm import unit

from atom_openmm.covalent_hybrid import CovalentHybridMolecule
from atom_openmm.covalent_parameters import VirtualSiteParameter


class HybridVirtualSiteError(ValueError):
    pass


def _nonbonded(system: mm.System) -> mm.NonbondedForce:
    forces = [force for force in system.getForces() if isinstance(force, mm.NonbondedForce)]
    if len(forces) != 1:
        raise HybridVirtualSiteError("hybrid endpoint must contain one NonbondedForce")
    return forces[0]


def _exception_map(force: mm.NonbondedForce) -> dict[tuple[int, int], tuple]:
    output = {}
    for index in range(force.getNumExceptions()):
        values = force.getExceptionParameters(index)
        output[tuple(sorted((int(values[0]), int(values[1]))))] = values[2:]
    return output


def _coulomb_14_scale(force: mm.NonbondedForce) -> float:
    scales = []
    for index in range(force.getNumExceptions()):
        atom1, atom2, charge_product, _, epsilon = force.getExceptionParameters(index)
        q1 = force.getParticleParameters(int(atom1))[0]
        q2 = force.getParticleParameters(int(atom2))[0]
        denominator = (q1 * q2).value_in_unit(unit.elementary_charge ** 2)
        product = charge_product.value_in_unit(unit.elementary_charge ** 2)
        if abs(denominator) > 1.0e-10 and abs(product) > 1.0e-10:
            scales.append(product / denominator)
        elif epsilon.value_in_unit(unit.kilojoules_per_mole) > 0.0:
            scales.append(1.0 / 1.2)
    return float(np.median(scales)) if scales else 1.0 / 1.2


def _clone_system(system: mm.System) -> mm.System:
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))


def _add_site_to_endpoint(
    system: mm.System,
    parents: tuple[int, int, int],
    site: VirtualSiteParameter,
) -> int:
    carbon, chlorine, frame = parents
    if len({carbon, chlorine, frame}) != 3:
        raise HybridVirtualSiteError("sigma-hole frame atoms must be distinct")
    particle = system.addParticle(0.0 * unit.dalton)
    system.setVirtualSite(
        particle,
        mm.LocalCoordinatesSite(
            [chlorine, carbon, frame],
            [1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, -1.0, 1.0],
            mm.Vec3(float(site.distance_a) / 10.0, 0.0, 0.0),
        ),
    )
    force = _nonbonded(system)
    existing = _exception_map(force)
    scale14 = _coulomb_14_scale(force)
    force.addParticle(
        float(site.charge_e) * unit.elementary_charge,
        max(float(site.sigma_a), 0.01) * unit.angstrom,
        float(site.epsilon_kj_mol) * unit.kilojoules_per_mole,
    )
    chlorine_charge = force.getParticleParameters(chlorine)[0]
    for other in range(particle):
        if other == chlorine:
            force.addException(
                particle, other, 0.0 * unit.elementary_charge ** 2,
                1.0 * unit.nanometer, 0.0 * unit.kilojoules_per_mole,
            )
            continue
        source = existing.get(tuple(sorted((chlorine, other))))
        if source is None:
            continue
        charge_product, sigma, epsilon = source
        other_charge = force.getParticleParameters(other)[0]
        denominator = (
            chlorine_charge * other_charge
        ).value_in_unit(unit.elementary_charge ** 2)
        source_product = charge_product.value_in_unit(unit.elementary_charge ** 2)
        if abs(source_product) <= 1.0e-12:
            scale = 0.0
        elif abs(denominator) > 1.0e-12:
            scale = source_product / denominator
        elif epsilon.value_in_unit(unit.kilojoules_per_mole) > 0.0:
            scale = scale14
        else:
            raise HybridVirtualSiteError("cannot infer sigma-hole intramolecular scaling")
        force.addException(
            particle,
            other,
            float(site.charge_e) * unit.elementary_charge * other_charge * scale,
            sigma,
            0.0 * unit.kilojoules_per_mole,
        )
    return particle


def _matched_sites(hybrid, parameters_a, parameters_b):
    sites_a = list(parameters_a.virtual_sites)
    sites_b = list(parameters_b.virtual_sites)
    if len(sites_a) != len(sites_b):
        raise HybridVirtualSiteError(
            "hybrid RESP endpoints must contain matching sigma-hole site counts"
        )
    remaining = list(sites_b)
    matched = []
    for site_a in sites_a:
        mapped = tuple(
            hybrid.map_a_to_b.get(int(atom))
            for atom in site_a.parent_atom_indices
        )
        if any(atom is None for atom in mapped):
            raise HybridVirtualSiteError(
                "sigma-hole C-Cl group and frame atom must be mapped in both ligands"
            )
        candidates = [
            site for site in remaining
            if tuple(site.parent_atom_indices) == mapped and site.kind == site_a.kind
        ]
        if len(candidates) != 1:
            raise HybridVirtualSiteError(
                "could not identify one corresponding endpoint sigma-hole site"
            )
        site_b = candidates[0]
        remaining.remove(site_b)
        if not np.isclose(site_a.distance_a, site_b.distance_a, atol=1.0e-8):
            raise HybridVirtualSiteError("endpoint sigma-hole distances differ")
        matched.append((site_a, site_b))
    return matched


def add_common_sigma_holes(
    hybrid: CovalentHybridMolecule, parameters_a, parameters_b
) -> CovalentHybridMolecule:
    if not parameters_a.virtual_sites and not parameters_b.virtual_sites:
        return hybrid
    matched = _matched_sites(hybrid, parameters_a, parameters_b)
    endpoint_a = _clone_system(hybrid.endpoint_a)
    endpoint_b = _clone_system(hybrid.endpoint_b)
    topology = hybrid.topology
    residue = next(topology.residues())
    coordinates = np.asarray(hybrid.positions.value_in_unit(unit.nanometer)).tolist()
    for index, (site_a, site_b) in enumerate(matched, start=1):
        parents_a = tuple(
            hybrid.map_a_to_hybrid[int(atom)] for atom in site_a.parent_atom_indices
        )
        parents_b = tuple(
            hybrid.map_b_to_hybrid[int(atom)] for atom in site_b.parent_atom_indices
        )
        if parents_a != parents_b:
            raise HybridVirtualSiteError("mapped endpoint sigma-hole frames differ")
        particle_a = _add_site_to_endpoint(endpoint_a, parents_a, site_a)
        particle_b = _add_site_to_endpoint(endpoint_b, parents_b, site_b)
        if particle_a != particle_b or particle_a != topology.getNumAtoms():
            raise HybridVirtualSiteError("sigma-hole particle ordering differs")
        topology.addAtom(f"EP{index}", None, residue)
        carbon, chlorine, _ = parents_a
        axis = np.asarray(coordinates[chlorine]) - np.asarray(coordinates[carbon])
        axis /= np.linalg.norm(axis)
        coordinates.append(
            (np.asarray(coordinates[chlorine]) + site_a.distance_a / 10.0 * axis).tolist()
        )
    return replace(
        hybrid,
        topology=topology,
        positions=np.asarray(coordinates) * unit.nanometer,
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
    )
