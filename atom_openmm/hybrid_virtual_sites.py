from __future__ import annotations

from dataclasses import replace

import numpy as np
import openmm as mm
from openmm import unit

from atom_openmm.covalent_hybrid import AlchemicalVirtualSite, CovalentHybridMolecule
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


def _inactive_site(site: VirtualSiteParameter) -> VirtualSiteParameter:
    return replace(site, charge_e=0.0, sigma_a=0.0, epsilon_kj_mol=0.0)


def _add_site_to_endpoint(
    system: mm.System,
    parents: tuple[int, int, int],
    site: VirtualSiteParameter,
) -> int:
    carbon, halogen, frame = parents
    if len({carbon, halogen, frame}) != 3:
        raise HybridVirtualSiteError("sigma-hole frame atoms must be distinct")
    particle = system.addParticle(0.0 * unit.dalton)
    system.setVirtualSite(
        particle,
        mm.LocalCoordinatesSite(
            [halogen, carbon, frame],
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
    halogen_charge = force.getParticleParameters(halogen)[0]
    for other in range(particle):
        if other == halogen:
            force.addException(
                particle, other, 0.0 * unit.elementary_charge ** 2,
                1.0 * unit.nanometer, 0.0 * unit.kilojoules_per_mole,
            )
            continue
        source = existing.get(tuple(sorted((halogen, other))))
        if source is None:
            continue
        charge_product, sigma, epsilon = source
        other_charge = force.getParticleParameters(other)[0]
        denominator = (
            halogen_charge * other_charge
        ).value_in_unit(unit.elementary_charge ** 2)
        source_product = charge_product.value_in_unit(unit.elementary_charge ** 2)
        if abs(source_product) <= 1.0e-12 or abs(float(site.charge_e)) <= 1.0e-12:
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


def _hybrid_parents(hybrid, site, endpoint):
    mapping = hybrid.map_a_to_hybrid if endpoint == "a" else hybrid.map_b_to_hybrid
    try:
        return tuple(mapping[int(atom)] for atom in site.parent_atom_indices)
    except KeyError as exc:
        raise HybridVirtualSiteError(
            f"sigma-hole {endpoint.upper()} parent atom is absent from the hybrid"
        ) from exc


def _site_key_in_b(hybrid, site):
    carbon, halogen, _ = (int(atom) for atom in site.parent_atom_indices)
    mapped_carbon = hybrid.map_a_to_b.get(carbon)
    mapped_halogen = hybrid.map_a_to_b.get(halogen)
    if mapped_carbon is None or mapped_halogen is None:
        return None
    return mapped_carbon, mapped_halogen


def _halogen_element(parameters, site):
    halogen = int(site.parent_atom_indices[1])
    return int(parameters.molecule.atoms[halogen].atomic_number)


def _classify_sites(hybrid, parameters_a, parameters_b):
    sites_a = list(parameters_a.virtual_sites)
    sites_b = list(parameters_b.virtual_sites)
    for site in sites_a + sites_b:
        if site.kind != "sigma_hole" or len(site.parent_atom_indices) != 3:
            raise HybridVirtualSiteError("unsupported cached ligand virtual site")

    by_b_key = {}
    for site in sites_b:
        key = tuple(int(atom) for atom in site.parent_atom_indices[:2])
        if key in by_b_key:
            raise HybridVirtualSiteError("multiple sigma holes use the same C-X bond")
        by_b_key[key] = site

    common = []
    unique_a = []
    consumed_b = set()
    for site_a in sites_a:
        key_b = _site_key_in_b(hybrid, site_a)
        site_b = by_b_key.get(key_b)
        if site_b is None:
            if key_b is not None:
                atom_b = key_b[1]
                if _halogen_element(parameters_a, site_a) == int(
                    parameters_b.molecule.atoms[atom_b].atomic_number
                ):
                    raise HybridVirtualSiteError(
                        "the same mapped halogen has a sigma hole in only endpoint A"
                    )
            unique_a.append(site_a)
            continue
        consumed_b.add(id(site_b))
        same_element = (
            _halogen_element(parameters_a, site_a)
            == _halogen_element(parameters_b, site_b)
        )
        same_distance = np.isclose(
            site_a.distance_a, site_b.distance_a, atol=1.0e-8, rtol=0.0
        )
        if same_element and same_distance:
            common.append((site_a, site_b))
        else:
            unique_a.append(site_a)
            consumed_b.remove(id(site_b))

    unique_b = [site for site in sites_b if id(site) not in consumed_b]
    reverse = {atom_b: atom_a for atom_a, atom_b in hybrid.map_a_to_b.items()}
    for site_b in unique_b:
        carbon_b, halogen_b, _ = (int(atom) for atom in site_b.parent_atom_indices)
        if carbon_b in reverse and halogen_b in reverse:
            atom_a = reverse[halogen_b]
            matching_a = [
                site for site in sites_a
                if int(site.parent_atom_indices[0]) == reverse[carbon_b]
                and int(site.parent_atom_indices[1]) == atom_a
            ]
            if not matching_a and _halogen_element(parameters_b, site_b) == int(
                parameters_a.molecule.atoms[atom_a].atomic_number
            ):
                raise HybridVirtualSiteError(
                    "the same mapped halogen has a sigma hole in only endpoint B"
                )
    return common, unique_a, unique_b


def _site_coordinate(coordinates, parents, distance_a):
    carbon, halogen, _ = parents
    axis = np.asarray(coordinates[halogen]) - np.asarray(coordinates[carbon])
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        raise HybridVirtualSiteError("sigma-hole C-X axis has zero length")
    return (
        np.asarray(coordinates[halogen]) + float(distance_a) / 10.0 * axis / norm
    ).tolist()


def _endpoint_site_role(hybrid, site, endpoint):
    halogen = int(site.parent_atom_indices[1])
    mapped = (
        halogen in hybrid.map_a_to_b
        if endpoint == "a"
        else halogen in set(hybrid.map_a_to_b.values())
    )
    return f"mapped_{endpoint}" if mapped else endpoint


def add_alchemical_sigma_holes(
    hybrid: CovalentHybridMolecule, parameters_a, parameters_b
) -> CovalentHybridMolecule:
    if not parameters_a.virtual_sites and not parameters_b.virtual_sites:
        return hybrid
    common, sites_a, sites_b = _classify_sites(hybrid, parameters_a, parameters_b)
    endpoint_a = _clone_system(hybrid.endpoint_a)
    endpoint_b = _clone_system(hybrid.endpoint_b)
    topology = hybrid.topology
    residue = next(topology.residues())
    coordinates = np.asarray(hybrid.positions.value_in_unit(unit.nanometer)).tolist()
    metadata = []
    unique_particles_a = list(hybrid.unique_particle_indices_a)
    unique_particles_b = list(hybrid.unique_particle_indices_b)

    def append_site(site_a, site_b, parents, role):
        source = site_a if site_a is not None else site_b
        actual_a = site_a if site_a is not None else _inactive_site(source)
        actual_b = site_b if site_b is not None else _inactive_site(source)
        particle_a = _add_site_to_endpoint(endpoint_a, parents, actual_a)
        particle_b = _add_site_to_endpoint(endpoint_b, parents, actual_b)
        if particle_a != particle_b or particle_a != topology.getNumAtoms():
            raise HybridVirtualSiteError("sigma-hole particle ordering differs")
        topology.addAtom(f"EP{particle_a + 1}", None, residue)
        coordinates.append(_site_coordinate(coordinates, parents, source.distance_a))
        if role == "a":
            unique_particles_a.append(particle_a)
        elif role == "b":
            unique_particles_b.append(particle_a)
        metadata.append(AlchemicalVirtualSite(
            particle_index=particle_a,
            role=role,
            parent_particle_indices=parents,
            source_name_a=None if site_a is None else site_a.name,
            source_name_b=None if site_b is None else site_b.name,
            charge_a_e=0.0 if site_a is None else float(site_a.charge_e),
            charge_b_e=0.0 if site_b is None else float(site_b.charge_e),
            distance_a_angstrom=(
                None if site_a is None else float(site_a.distance_a)
            ),
            distance_b_angstrom=(
                None if site_b is None else float(site_b.distance_a)
            ),
        ))

    for site_a, site_b in common:
        parents_a = _hybrid_parents(hybrid, site_a, "a")
        parents_b = _hybrid_parents(hybrid, site_b, "b")
        if parents_a[:2] != parents_b[:2]:
            raise HybridVirtualSiteError("mapped sigma-hole C-X atoms differ")
        append_site(site_a, site_b, parents_a, "common")
    for site in sites_a:
        append_site(
            site,
            None,
            _hybrid_parents(hybrid, site, "a"),
            _endpoint_site_role(hybrid, site, "a"),
        )
    for site in sites_b:
        append_site(
            None,
            site,
            _hybrid_parents(hybrid, site, "b"),
            _endpoint_site_role(hybrid, site, "b"),
        )

    return replace(
        hybrid,
        topology=topology,
        positions=np.asarray(coordinates) * unit.nanometer,
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        unique_particle_indices_a=tuple(sorted(unique_particles_a)),
        unique_particle_indices_b=tuple(sorted(unique_particles_b)),
        alchemical_virtual_sites=tuple(metadata),
    )


def add_common_sigma_holes(
    hybrid: CovalentHybridMolecule, parameters_a, parameters_b
) -> CovalentHybridMolecule:
    """Backward-compatible alias for the alchemical sigma-hole builder."""
    return add_alchemical_sigma_holes(hybrid, parameters_a, parameters_b)


def alchemical_virtual_site_metadata(hybrid: CovalentHybridMolecule):
    return [
        {
            "particle_index": site.particle_index,
            "role": site.role,
            "parent_particle_indices": list(site.parent_particle_indices),
            "source_name_a": site.source_name_a,
            "source_name_b": site.source_name_b,
            "charge_a_e": site.charge_a_e,
            "charge_b_e": site.charge_b_e,
            "distance_a_angstrom": site.distance_a_angstrom,
            "distance_b_angstrom": site.distance_b_angstrom,
        }
        for site in hybrid.alchemical_virtual_sites
    ]
