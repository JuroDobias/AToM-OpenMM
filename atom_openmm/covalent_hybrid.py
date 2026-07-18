from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openmm as mm
from openmm import app, unit
from rdkit import Chem
from rdkit.Chem import rdFMCS

from atom_openmm.covalent_alchemy import CovalentAlchemyError
from atom_openmm.covalent_parameters import CovalentParameterBundle


@dataclass(frozen=True)
class CovalentHybridMolecule:
    topology: app.Topology
    positions: unit.Quantity
    endpoint_a: mm.System
    endpoint_b: mm.System
    map_a_to_b: dict[int, int]
    map_a_to_hybrid: dict[int, int]
    map_b_to_hybrid: dict[int, int]
    unique_a: tuple[int, ...]
    unique_b: tuple[int, ...]


def _rdkit_molecule(bundle: CovalentParameterBundle) -> Chem.Mol:
    molecule = bundle.molecule.to_rdkit()
    if molecule.GetNumConformers() != 1:
        raise CovalentAlchemyError("covalent products require one conformer")
    return molecule


def find_covalent_atom_map(
    molecule_a: Chem.Mol,
    molecule_b: Chem.Mol,
    *,
    required_pairs: list[tuple[int, int]] | None = None,
    timeout_seconds: int = 30,
) -> dict[int, int]:
    if required_pairs:
        return _expand_required_mapping(molecule_a, molecule_b, required_pairs)
    result = rdFMCS.FindMCS(
        [molecule_a, molecule_b],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        matchValences=True,
        timeout=int(timeout_seconds),
    )
    if result.canceled or result.numAtoms == 0:
        raise CovalentAlchemyError("could not determine a complete covalent-product MCS")
    query = Chem.MolFromSmarts(result.smartsString)
    matches_a = molecule_a.GetSubstructMatches(query, uniquify=True, useChirality=True)
    matches_b = molecule_b.GetSubstructMatches(query, uniquify=True, useChirality=True)
    required = set(required_pairs or [])
    candidates = []
    conformer_a = molecule_a.GetConformer()
    conformer_b = molecule_b.GetConformer()
    for match_a in matches_a:
        for match_b in matches_b:
            mapping = dict(zip(match_a, match_b))
            if not required.issubset(mapping.items()):
                continue
            squared = []
            for atom_a, atom_b in mapping.items():
                difference = np.asarray(conformer_a.GetAtomPosition(atom_a)) - np.asarray(
                    conformer_b.GetAtomPosition(atom_b)
                )
                squared.append(float(difference @ difference))
            candidates.append((float(np.sqrt(np.mean(squared))), mapping))
    if not candidates:
        raise CovalentAlchemyError("no MCS mapping contains all required core atom pairs")
    mapping = min(candidates, key=lambda item: item[0])[1]
    for atom_a, atom_b in mapping.items():
        left = molecule_a.GetAtomWithIdx(atom_a)
        right = molecule_b.GetAtomWithIdx(atom_b)
        if (
            left.GetAtomicNum() != right.GetAtomicNum()
            or left.GetFormalCharge() != right.GetFormalCharge()
            or left.GetIsAromatic() != right.GetIsAromatic()
            or left.IsInRing() != right.IsInRing()
        ):
            raise CovalentAlchemyError(f"invalid mapped atom pair {atom_a}:{atom_b}")
    return mapping


def _atom_signature(atom: Chem.Atom):
    return (
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic(),
        atom.IsInRing(),
        atom.GetChiralTag(),
    )


def _bond_signature(bond: Chem.Bond):
    return (bond.GetBondType(), bond.GetIsAromatic(), bond.IsInRing())


def _expand_required_mapping(molecule_a, molecule_b, required_pairs):
    mapping = dict(required_pairs)
    if len(mapping) != len(required_pairs) or len(set(mapping.values())) != len(mapping):
        raise CovalentAlchemyError("required atom pairs must be one-to-one")
    for atom_a, atom_b in mapping.items():
        if _atom_signature(molecule_a.GetAtomWithIdx(atom_a)) != _atom_signature(
            molecule_b.GetAtomWithIdx(atom_b)
        ):
            raise CovalentAlchemyError(f"required atom pair {atom_a}:{atom_b} is chemically incompatible")

    reverse = {atom_b: atom_a for atom_a, atom_b in mapping.items()}
    conformer_a = molecule_a.GetConformer()
    conformer_b = molecule_b.GetConformer()
    changed = True
    while changed:
        changed = False
        proposals = []
        for atom_a, atom_b in list(mapping.items()):
            center_a = molecule_a.GetAtomWithIdx(atom_a)
            center_b = molecule_b.GetAtomWithIdx(atom_b)
            neighbors_a = [neighbor for neighbor in center_a.GetNeighbors() if neighbor.GetIdx() not in mapping]
            neighbors_b = [neighbor for neighbor in center_b.GetNeighbors() if neighbor.GetIdx() not in reverse]
            for neighbor_a in neighbors_a:
                bond_a = molecule_a.GetBondBetweenAtoms(atom_a, neighbor_a.GetIdx())
                candidates = []
                for neighbor_b in neighbors_b:
                    bond_b = molecule_b.GetBondBetweenAtoms(atom_b, neighbor_b.GetIdx())
                    if _atom_signature(neighbor_a) != _atom_signature(neighbor_b):
                        continue
                    if _bond_signature(bond_a) != _bond_signature(bond_b):
                        continue
                    distance = np.linalg.norm(
                        np.asarray(conformer_a.GetAtomPosition(neighbor_a.GetIdx()))
                        - np.asarray(conformer_b.GetAtomPosition(neighbor_b.GetIdx()))
                    )
                    candidates.append((float(distance), neighbor_b.GetIdx()))
                if candidates:
                    proposals.append((min(candidates), neighbor_a.GetIdx()))
        for (distance, atom_b), atom_a in sorted(proposals):
            if atom_a in mapping or atom_b in reverse:
                continue
            mapping[atom_a] = atom_b
            reverse[atom_b] = atom_a
            changed = True
    return mapping


def _source_force(system: mm.System, cls):
    forces = [force for force in system.getForces() if isinstance(force, cls)]
    if len(forces) > 1:
        raise CovalentAlchemyError(f"multiple {cls.__name__} instances are not supported")
    return forces[0] if forces else None


def _mapped_indices(indices, mapping):
    return tuple(mapping[int(index)] for index in indices)


def _add_union_constraints(output, systems_and_maps):
    constraints = {}
    for system, mapping in systems_and_maps:
        for index in range(system.getNumConstraints()):
            particle1, particle2, distance = system.getConstraintParameters(index)
            pair = tuple(sorted((mapping[int(particle1)], mapping[int(particle2)])))
            distance_nm = distance.value_in_unit(unit.nanometer)
            if pair in constraints and not np.isclose(constraints[pair], distance_nm, atol=1.0e-8):
                raise CovalentAlchemyError(
                    f"mapped constraint {pair} has incompatible endpoint distances"
                )
            constraints[pair] = distance_nm
    for pair, distance_nm in sorted(constraints.items()):
        output.addConstraint(*pair, distance_nm * unit.nanometer)


def _add_bonds(target, source, mapping, include):
    if source is None:
        return
    for index in range(source.getNumBonds()):
        particle1, particle2, length, k = source.getBondParameters(index)
        particles = (int(particle1), int(particle2))
        if include(particles):
            target.addBond(*_mapped_indices(particles, mapping), length, k)


def _add_angles(target, source, mapping, include):
    if source is None:
        return
    for index in range(source.getNumAngles()):
        particle1, particle2, particle3, angle, k = source.getAngleParameters(index)
        particles = (int(particle1), int(particle2), int(particle3))
        if include(particles):
            target.addAngle(*_mapped_indices(particles, mapping), angle, k)


def _add_torsions(target, source, mapping, include):
    if source is None:
        return
    for index in range(source.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = source.getTorsionParameters(index)
        particles = (int(p1), int(p2), int(p3), int(p4))
        if include(particles):
            target.addTorsion(*_mapped_indices(particles, mapping), periodicity, phase, k)


def _build_endpoint(
    system_a: mm.System,
    system_b: mm.System,
    map_a_to_hybrid: dict[int, int],
    map_b_to_hybrid: dict[int, int],
    unique_a: set[int],
    unique_b: set[int],
    state: str,
) -> mm.System:
    output = mm.System()
    reverse_a = {hybrid: atom for atom, hybrid in map_a_to_hybrid.items()}
    reverse_b = {hybrid: atom for atom, hybrid in map_b_to_hybrid.items()}
    particle_count = max(max(reverse_a), max(reverse_b)) + 1
    for hybrid in range(particle_count):
        if hybrid in reverse_a:
            mass = system_a.getParticleMass(reverse_a[hybrid])
        else:
            mass = system_b.getParticleMass(reverse_b[hybrid])
        output.addParticle(mass)
    _add_union_constraints(
        output,
        ((system_a, map_a_to_hybrid), (system_b, map_b_to_hybrid)),
    )

    bonds = mm.HarmonicBondForce()
    angles = mm.HarmonicAngleForce()
    torsions = mm.PeriodicTorsionForce()
    force_a_bond = _source_force(system_a, mm.HarmonicBondForce)
    force_b_bond = _source_force(system_b, mm.HarmonicBondForce)
    force_a_angle = _source_force(system_a, mm.HarmonicAngleForce)
    force_b_angle = _source_force(system_b, mm.HarmonicAngleForce)
    force_a_torsion = _source_force(system_a, mm.PeriodicTorsionForce)
    force_b_torsion = _source_force(system_b, mm.PeriodicTorsionForce)
    if state == "a":
        _add_bonds(bonds, force_a_bond, map_a_to_hybrid, lambda _: True)
        _add_angles(angles, force_a_angle, map_a_to_hybrid, lambda _: True)
        _add_torsions(torsions, force_a_torsion, map_a_to_hybrid, lambda _: True)
        _add_bonds(bonds, force_b_bond, map_b_to_hybrid, lambda atoms: any(i in unique_b for i in atoms))
        _add_angles(angles, force_b_angle, map_b_to_hybrid, lambda atoms: any(i in unique_b for i in atoms))
        _add_torsions(torsions, force_b_torsion, map_b_to_hybrid, lambda atoms: any(i in unique_b for i in atoms))
    else:
        _add_bonds(bonds, force_b_bond, map_b_to_hybrid, lambda _: True)
        _add_angles(angles, force_b_angle, map_b_to_hybrid, lambda _: True)
        _add_torsions(torsions, force_b_torsion, map_b_to_hybrid, lambda _: True)
        _add_bonds(bonds, force_a_bond, map_a_to_hybrid, lambda atoms: any(i in unique_a for i in atoms))
        _add_angles(angles, force_a_angle, map_a_to_hybrid, lambda atoms: any(i in unique_a for i in atoms))
        _add_torsions(torsions, force_a_torsion, map_a_to_hybrid, lambda atoms: any(i in unique_a for i in atoms))
    for force in (bonds, angles, torsions):
        if (hasattr(force, "getNumBonds") and force.getNumBonds()) or (
            hasattr(force, "getNumAngles") and force.getNumAngles()
        ) or (hasattr(force, "getNumTorsions") and force.getNumTorsions()):
            output.addForce(force)

    source_nonbonded = _source_force(system_a if state == "a" else system_b, mm.NonbondedForce)
    if source_nonbonded is None:
        raise CovalentAlchemyError("OpenFF endpoint is missing a NonbondedForce")
    source_mapping = map_a_to_hybrid if state == "a" else map_b_to_hybrid
    source_reverse = reverse_a if state == "a" else reverse_b
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(source_nonbonded.getNonbondedMethod())
    nonbonded.setCutoffDistance(source_nonbonded.getCutoffDistance())
    for hybrid in range(particle_count):
        if hybrid in source_reverse:
            nonbonded.addParticle(*source_nonbonded.getParticleParameters(source_reverse[hybrid]))
        else:
            other_source = system_b if state == "a" else system_a
            other_reverse = reverse_b if state == "a" else reverse_a
            other_nb = _source_force(other_source, mm.NonbondedForce)
            _, sigma, _ = other_nb.getParticleParameters(other_reverse[hybrid])
            nonbonded.addParticle(0.0, sigma, 0.0)
    for index in range(source_nonbonded.getNumExceptions()):
        p1, p2, charge, sigma, epsilon = source_nonbonded.getExceptionParameters(index)
        nonbonded.addException(source_mapping[int(p1)], source_mapping[int(p2)], charge, sigma, epsilon)
    output.addForce(nonbonded)
    return output


def _hybrid_topology(molecule_a, molecule_b, map_a_to_hybrid, map_b_to_hybrid):
    topology = app.Topology()
    chain = topology.addChain("A")
    residue = topology.addResidue("HYB", chain, "1")
    atoms = [None] * (max(max(map_a_to_hybrid.values()), max(map_b_to_hybrid.values())) + 1)
    used_names = set()
    for source, mapping, prefix in (
        (molecule_a, map_a_to_hybrid, "A"),
        (molecule_b, map_b_to_hybrid, "B"),
    ):
        for atom in source.GetAtoms():
            hybrid = mapping[atom.GetIdx()]
            if atoms[hybrid] is not None:
                continue
            base = atom.GetProp("_Name") if atom.HasProp("_Name") else f"{prefix}{atom.GetSymbol()}{atom.GetIdx()+1}"
            name = base
            suffix = 1
            while name in used_names:
                suffix += 1
                name = f"{base}{suffix}"
            used_names.add(name)
            atoms[hybrid] = topology.addAtom(name, app.Element.getByAtomicNumber(atom.GetAtomicNum()), residue)
    bonds = set()
    for source, mapping in ((molecule_a, map_a_to_hybrid), (molecule_b, map_b_to_hybrid)):
        for bond in source.GetBonds():
            pair = tuple(sorted((mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()])))
            if pair not in bonds:
                topology.addBond(atoms[pair[0]], atoms[pair[1]])
                bonds.add(pair)
    return topology


def build_covalent_hybrid_molecule(
    parameters_a: CovalentParameterBundle,
    parameters_b: CovalentParameterBundle,
    *,
    required_pairs: list[tuple[int, int]] | None = None,
) -> CovalentHybridMolecule:
    molecule_a = _rdkit_molecule(parameters_a)
    molecule_b = _rdkit_molecule(parameters_b)
    map_a_to_b = find_covalent_atom_map(
        molecule_a, molecule_b, required_pairs=required_pairs
    )
    map_a_to_hybrid = {index: index for index in range(molecule_a.GetNumAtoms())}
    map_b_to_hybrid = {atom_b: atom_a for atom_a, atom_b in map_a_to_b.items()}
    next_index = molecule_a.GetNumAtoms()
    for atom_b in range(molecule_b.GetNumAtoms()):
        if atom_b not in map_b_to_hybrid:
            map_b_to_hybrid[atom_b] = next_index
            next_index += 1
    unique_a = set(range(molecule_a.GetNumAtoms())) - set(map_a_to_b)
    unique_b = set(range(molecule_b.GetNumAtoms())) - set(map_a_to_b.values())
    endpoint_a = _build_endpoint(
        parameters_a.system,
        parameters_b.system,
        map_a_to_hybrid,
        map_b_to_hybrid,
        unique_a,
        unique_b,
        "a",
    )
    endpoint_b = _build_endpoint(
        parameters_a.system,
        parameters_b.system,
        map_a_to_hybrid,
        map_b_to_hybrid,
        unique_a,
        unique_b,
        "b",
    )
    positions = np.zeros((next_index, 3), dtype=float)
    conformer_a = molecule_a.GetConformer()
    conformer_b = molecule_b.GetConformer()
    for atom, hybrid in map_a_to_hybrid.items():
        positions[hybrid] = np.asarray(conformer_a.GetAtomPosition(atom))
    for atom in unique_b:
        positions[map_b_to_hybrid[atom]] = np.asarray(conformer_b.GetAtomPosition(atom))
    topology = _hybrid_topology(molecule_a, molecule_b, map_a_to_hybrid, map_b_to_hybrid)
    return CovalentHybridMolecule(
        topology,
        positions * unit.angstrom,
        endpoint_a,
        endpoint_b,
        map_a_to_b,
        map_a_to_hybrid,
        map_b_to_hybrid,
        tuple(sorted(unique_a)),
        tuple(sorted(unique_b)),
    )
