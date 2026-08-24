from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import permutations

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
    anchor_pairs: tuple[tuple[int, int], ...]
    attachment_pairs: tuple[tuple[int, int], tuple[int, int]] | None
    dummy_bonded_scales: "DummyBondedScales"
    dummy_core_nonbonded: str
    transmuted_pairs: tuple[tuple[int, int], ...] = ()
    inactive_bonded_atoms_a: tuple[int, ...] = ()
    inactive_bonded_atoms_b: tuple[int, ...] = ()
    inactive_bonded_geometry: str = "bond_only"
    inactive_z_matrix_terms: tuple["InactiveZMatrixTerm", ...] = ()


@dataclass(frozen=True)
class InactiveZMatrixTerm:
    endpoint: str
    dummy_atom: int
    angle_atoms: tuple[int, int, int]
    torsion_atoms: tuple[int, int, int, int]
    hybrid_dummy_atom: int
    hybrid_angle_atoms: tuple[int, int, int]
    hybrid_torsion_atoms: tuple[int, int, int, int]
    angle_degrees: float
    angle_k_kj_mol_rad2: float
    torsion_barrier_kj_mol: float
    torsion_terms: tuple[tuple[int, float, float], ...]


@dataclass(frozen=True)
class DummyBondedScales:
    bond: float = 1.0
    angle: float = 1.0
    proper_torsion: float = 1.0
    junction_angle: float = 1.0
    junction_proper_torsion: float = 1.0
    internal_rotatable_torsion: float | None = None
    junction_rotatable_torsion: float | None = None

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if value is not None and float(value) < 0.0:
                raise CovalentAlchemyError(f"dummy bonded scale {name} cannot be negative")


def _rdkit_molecule(bundle: CovalentParameterBundle) -> Chem.Mol:
    molecule = normalize_mapping_aromaticity(bundle.molecule.to_rdkit())
    if molecule.GetNumConformers() != 1:
        raise CovalentAlchemyError("covalent products require one conformer")
    return molecule


def normalize_mapping_aromaticity(molecule: Chem.Mol) -> Chem.Mol:
    """Return an atom-order-preserving copy with RDKit aromaticity perceived."""
    copied = Chem.Mol(molecule)
    Chem.SetAromaticity(copied, Chem.AromaticityModel.AROMATICITY_RDKIT)
    return copied


def complete_covalent_atom_map(
    molecule_a: Chem.Mol,
    molecule_b: Chem.Mol,
    mapping: dict[int, int],
    *,
    required_pairs: list[tuple[int, int]] | None = None,
    transmuted_pairs: set[tuple[int, int]] | None = None,
) -> dict[int, int]:
    """Complete mapped heavy atoms with compatible hydrogens and validate the map."""
    molecule_a = normalize_mapping_aromaticity(molecule_a)
    molecule_b = normalize_mapping_aromaticity(molecule_b)
    mapping = {int(atom_a): int(atom_b) for atom_a, atom_b in mapping.items()}
    required = set(required_pairs or [])
    transmuted = set(transmuted_pairs or ())
    if len(required) != len(required_pairs or ()):
        raise CovalentAlchemyError("required covalent atom pairs contain duplicates")
    if len(set(mapping.values())) != len(mapping):
        raise CovalentAlchemyError("covalent atom map must be one-to-one")
    if not transmuted.issubset(mapping.items()):
        raise CovalentAlchemyError("transmuted atom pairs must be present in the atom map")
    if any(
        atom_a < 0 or atom_a >= molecule_a.GetNumAtoms()
        or atom_b < 0 or atom_b >= molecule_b.GetNumAtoms()
        for atom_a, atom_b in set(mapping.items()) | required
    ):
        raise CovalentAlchemyError("covalent atom pair is outside the molecule")
    required_heavy = {
        (atom_a, atom_b)
        for atom_a, atom_b in required
        if molecule_a.GetAtomWithIdx(atom_a).GetAtomicNum() != 1
        and molecule_b.GetAtomWithIdx(atom_b).GetAtomicNum() != 1
    }
    if not required_heavy.issubset(mapping.items()):
        raise CovalentAlchemyError("covalent atom map does not contain all required pairs")

    required_by_a = dict(required)
    conformer_a = molecule_a.GetConformer()
    conformer_b = molecule_b.GetConformer()
    for atom_a, atom_b in list(mapping.items()):
        if molecule_a.GetAtomWithIdx(atom_a).GetAtomicNum() == 1:
            continue
        hydrogens_a = sorted(
            atom.GetIdx() for atom in molecule_a.GetAtomWithIdx(atom_a).GetNeighbors()
            if atom.GetAtomicNum() == 1
        )
        hydrogens_b = sorted(
            atom.GetIdx() for atom in molecule_b.GetAtomWithIdx(atom_b).GetNeighbors()
            if atom.GetAtomicNum() == 1
        )
        if len(hydrogens_a) != len(hydrogens_b):
            continue
        fixed = {
            hydrogen_a: required_by_a[hydrogen_a]
            for hydrogen_a in hydrogens_a if hydrogen_a in required_by_a
        }
        if any(hydrogen_b not in hydrogens_b for hydrogen_b in fixed.values()):
            continue
        remaining_a = [
            atom for atom in hydrogens_a
            if atom not in fixed and atom not in mapping
        ]
        mapped_b = set(mapping.values()) | set(fixed.values())
        remaining_b = [atom for atom in hydrogens_b if atom not in mapped_b]
        if len(remaining_a) != len(remaining_b):
            continue
        best = None
        for ordered_b in permutations(remaining_b):
            squared = 0.0
            for hydrogen_a, hydrogen_b in zip(remaining_a, ordered_b):
                difference = np.asarray(conformer_a.GetAtomPosition(hydrogen_a)) - np.asarray(
                    conformer_b.GetAtomPosition(hydrogen_b)
                )
                squared += float(difference @ difference)
            candidate = (squared, ordered_b)
            if best is None or candidate < best:
                best = candidate
        mapping.update(fixed)
        if best is not None:
            mapping.update(zip(remaining_a, best[1]))

    if not required.issubset(mapping.items()):
        raise CovalentAlchemyError(
            "covalent atom map could not preserve all required explicit-hydrogen pairs"
        )
    for atom_a, atom_b in mapping.items():
        left = molecule_a.GetAtomWithIdx(atom_a)
        right = molecule_b.GetAtomWithIdx(atom_b)
        if (atom_a, atom_b) not in transmuted and (
            left.GetAtomicNum() != right.GetAtomicNum()
            or left.GetFormalCharge() != right.GetFormalCharge()
            or left.GetIsAromatic() != right.GetIsAromatic()
            or left.IsInRing() != right.IsInRing()
        ):
            raise CovalentAlchemyError(f"invalid mapped atom pair {atom_a}:{atom_b}")

    mapped_heavy = {
        atom_a for atom_a in mapping
        if molecule_a.GetAtomWithIdx(atom_a).GetAtomicNum() != 1
    }
    for atom_a in mapped_heavy:
        atom_b = mapping[atom_a]
        for neighbor_a in molecule_a.GetAtomWithIdx(atom_a).GetNeighbors():
            other_a = neighbor_a.GetIdx()
            if other_a not in mapped_heavy or atom_a > other_a:
                continue
            other_b = mapping[other_a]
            bond_a = molecule_a.GetBondBetweenAtoms(atom_a, other_a)
            bond_b = molecule_b.GetBondBetweenAtoms(atom_b, other_b)
            changed_atom = (atom_a, atom_b) in transmuted or (
                other_a, other_b
            ) in transmuted
            if bond_b is None or (
                not changed_atom and _bond_signature(bond_a) != _bond_signature(bond_b)
            ):
                raise CovalentAlchemyError(
                    f"mapped bond {atom_a}:{other_a} is incompatible with {atom_b}:{other_b}"
                )
    reverse_mapping = {atom_b: atom_a for atom_a, atom_b in mapping.items()}
    mapped_heavy_b = set(reverse_mapping)
    for atom_b in mapped_heavy_b:
        atom_a = reverse_mapping[atom_b]
        for neighbor_b in molecule_b.GetAtomWithIdx(atom_b).GetNeighbors():
            other_b = neighbor_b.GetIdx()
            if other_b not in mapped_heavy_b or atom_b > other_b:
                continue
            other_a = reverse_mapping[other_b]
            bond_b = molecule_b.GetBondBetweenAtoms(atom_b, other_b)
            bond_a = molecule_a.GetBondBetweenAtoms(atom_a, other_a)
            changed_atom = (atom_a, atom_b) in transmuted or (
                other_a, other_b
            ) in transmuted
            if bond_a is None or (
                not changed_atom and _bond_signature(bond_a) != _bond_signature(bond_b)
            ):
                raise CovalentAlchemyError(
                    f"mapped bond {atom_b}:{other_b} is incompatible with {atom_a}:{other_a}"
                )
    if mapped_heavy:
        pending = [next(iter(mapped_heavy))]
        visited = set()
        while pending:
            atom = pending.pop()
            if atom in visited:
                continue
            visited.add(atom)
            pending.extend(
                neighbor.GetIdx()
                for neighbor in molecule_a.GetAtomWithIdx(atom).GetNeighbors()
                if neighbor.GetIdx() in mapped_heavy
            )
        if visited != mapped_heavy:
            raise CovalentAlchemyError(
                "covalent common atom map must be a connected protein-linked subgraph"
            )
    return mapping


def find_covalent_atom_map(
    molecule_a: Chem.Mol,
    molecule_b: Chem.Mol,
    *,
    required_pairs: list[tuple[int, int]] | None = None,
    timeout_seconds: int = 30,
) -> dict[int, int]:
    molecule_a = normalize_mapping_aromaticity(molecule_a)
    molecule_b = normalize_mapping_aromaticity(molecule_b)
    heavy_molecules = []
    for molecule in (molecule_a, molecule_b):
        copied = Chem.Mol(molecule)
        for atom in copied.GetAtoms():
            atom.SetIntProp("_covalent_original_index", atom.GetIdx())
        heavy_molecules.append(Chem.RemoveHs(copied))
    result = rdFMCS.FindMCS(
        heavy_molecules,
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
    matches_a = heavy_molecules[0].GetSubstructMatches(query, uniquify=True, useChirality=True)
    matches_b = heavy_molecules[1].GetSubstructMatches(query, uniquify=True, useChirality=True)
    required = set(required_pairs or [])
    if len(required) != len(required_pairs or ()):
        raise CovalentAlchemyError("required covalent atom pairs contain duplicates")
    if len({atom_a for atom_a, _ in required}) != len(required) or len(
        {atom_b for _, atom_b in required}
    ) != len(required):
        raise CovalentAlchemyError("required covalent atom pairs must be one-to-one")
    if any(
        atom_a < 0 or atom_a >= molecule_a.GetNumAtoms()
        or atom_b < 0 or atom_b >= molecule_b.GetNumAtoms()
        for atom_a, atom_b in required
    ):
        raise CovalentAlchemyError("required covalent atom pair is outside the molecule")
    required_heavy = {
        pair for pair in required
        if molecule_a.GetAtomWithIdx(pair[0]).GetAtomicNum() != 1
        and molecule_b.GetAtomWithIdx(pair[1]).GetAtomicNum() != 1
    }
    candidates = []
    conformer_a = molecule_a.GetConformer()
    conformer_b = molecule_b.GetConformer()
    for match_a in matches_a:
        for match_b in matches_b:
            mapping = {
                heavy_molecules[0].GetAtomWithIdx(atom_a).GetIntProp("_covalent_original_index"):
                heavy_molecules[1].GetAtomWithIdx(atom_b).GetIntProp("_covalent_original_index")
                for atom_a, atom_b in zip(match_a, match_b)
            }
            if not required_heavy.issubset(mapping.items()):
                continue
            squared = []
            for atom_a, atom_b in mapping.items():
                difference = np.asarray(conformer_a.GetAtomPosition(atom_a)) - np.asarray(
                    conformer_b.GetAtomPosition(atom_b)
                )
                squared.append(float(difference @ difference))
            candidates.append((float(np.sqrt(np.mean(squared))), mapping))
    if not candidates:
        raise CovalentAlchemyError("no connected MCS mapping contains all covalent anchor pairs")
    mapping = min(candidates, key=lambda item: item[0])[1]
    return complete_covalent_atom_map(
        molecule_a,
        molecule_b,
        mapping,
        required_pairs=list(required),
    )


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


def _add_bonds(target, source, mapping, include, scale=lambda _: 1.0):
    if source is None:
        return
    for index in range(source.getNumBonds()):
        particle1, particle2, length, k = source.getBondParameters(index)
        particles = (int(particle1), int(particle2))
        if include(particles):
            target.addBond(*_mapped_indices(particles, mapping), length, k * scale(particles))


def _add_angles(target, source, mapping, include, scale=lambda _: 1.0):
    if source is None:
        return
    for index in range(source.getNumAngles()):
        particle1, particle2, particle3, angle, k = source.getAngleParameters(index)
        particles = (int(particle1), int(particle2), int(particle3))
        if include(particles):
            target.addAngle(*_mapped_indices(particles, mapping), angle, k * scale(particles))


def _add_torsions(target, source, mapping, include, scale=lambda _: 1.0):
    if source is None:
        return
    for index in range(source.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = source.getTorsionParameters(index)
        particles = (int(p1), int(p2), int(p3), int(p4))
        if include(particles):
            target.addTorsion(
                *_mapped_indices(particles, mapping), periodicity, phase, k * scale(particles)
            )


def _is_proper_torsion(molecule: Chem.Mol, atoms: tuple[int, ...]) -> bool:
    return all(
        molecule.GetBondBetweenAtoms(atoms[index], atoms[index + 1]) is not None
        for index in range(3)
    )


def _is_rotatable_central_bond(
    molecule: Chem.Mol, atoms: tuple[int, ...]
) -> bool:
    if not _is_proper_torsion(molecule, atoms):
        return False
    atom1 = molecule.GetAtomWithIdx(atoms[1])
    atom2 = molecule.GetAtomWithIdx(atoms[2])
    bond = molecule.GetBondBetweenAtoms(atoms[1], atoms[2])
    if (
        bond is None
        or bond.GetBondType() != Chem.BondType.SINGLE
        or bond.IsInRing()
        or atom1.GetAtomicNum() == 1
        or atom2.GetAtomicNum() == 1
    ):
        return False

    def heavy_degree(atom):
        return sum(neighbor.GetAtomicNum() > 1 for neighbor in atom.GetNeighbors())

    return heavy_degree(atom1) > 1 and heavy_degree(atom2) > 1


def _inactive_scales(molecule, unique, scales):
    def angle(atoms):
        terminal_unique = sum(atom in unique for atom in (atoms[0], atoms[2]))
        if terminal_unique == 1 and atoms[1] not in unique:
            return scales.junction_angle
        return scales.angle

    def torsion(atoms):
        if not _is_proper_torsion(molecule, atoms):
            return scales.proper_torsion
        if _is_rotatable_central_bond(molecule, atoms):
            central_unique = sum(atom in unique for atom in atoms[1:3])
            if central_unique == 1:
                return (
                    scales.junction_proper_torsion
                    if scales.junction_rotatable_torsion is None
                    else scales.junction_rotatable_torsion
                )
            if central_unique == 2:
                return (
                    scales.proper_torsion
                    if scales.internal_rotatable_torsion is None
                    else scales.internal_rotatable_torsion
                )
        terminal_unique = sum(atom in unique for atom in (atoms[0], atoms[3]))
        if terminal_unique == 1:
            return scales.junction_proper_torsion
        return scales.proper_torsion

    return angle, torsion


def _canonical_torsion(atoms):
    atoms = tuple(int(atom) for atom in atoms)
    reverse = atoms[::-1]
    return min(atoms, reverse)


def _torsion_barrier_kj_mol(terms):
    phi = np.linspace(-np.pi, np.pi, 4097)
    energy = np.zeros_like(phi)
    for periodicity, phase, k in terms:
        energy += k * (1.0 + np.cos(periodicity * phi - phase))
    return float(np.max(energy) - np.min(energy))


def _select_inactive_z_matrix_terms(
    molecule,
    system,
    selected,
    unique,
    mapping,
    endpoint,
):
    """Select one nonredundant bond-angle-torsion frame per terminal dummy."""
    if not selected:
        return {}
    angle_force = _source_force(system, mm.HarmonicAngleForce)
    torsion_force = _source_force(system, mm.PeriodicTorsionForce)
    if angle_force is None or torsion_force is None:
        raise CovalentAlchemyError(
            "terminal_z_matrix requires harmonic angles and periodic torsions"
        )
    angles = {}
    for index in range(angle_force.getNumAngles()):
        atom1, center, atom3, theta, k = angle_force.getAngleParameters(index)
        key = (min(int(atom1), int(atom3)), int(center), max(int(atom1), int(atom3)))
        angles[key] = (theta, k)
    torsions = {}
    for index in range(torsion_force.getNumTorsions()):
        atom1, atom2, atom3, atom4, periodicity, phase, k = (
            torsion_force.getTorsionParameters(index)
        )
        key = _canonical_torsion((atom1, atom2, atom3, atom4))
        torsions.setdefault(key, []).append((periodicity, phase, k))

    canonical_ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    result = {}
    for dummy in sorted(selected):
        atom_d = molecule.GetAtomWithIdx(dummy)
        neighbors = [atom.GetIdx() for atom in atom_d.GetNeighbors()]
        if len(neighbors) != 1:
            raise CovalentAlchemyError(
                f"terminal_z_matrix atom {dummy} in endpoint {endpoint} must have "
                "exactly one bonded neighbor"
            )
        center = neighbors[0]
        if center in unique:
            raise CovalentAlchemyError(
                f"terminal_z_matrix atom {dummy} in endpoint {endpoint} must attach "
                "directly to a mapped atom"
            )
        candidates = []
        for atom_b_obj in molecule.GetAtomWithIdx(center).GetNeighbors():
            atom_b = atom_b_obj.GetIdx()
            if atom_b == dummy or atom_b in unique or atom_b_obj.GetAtomicNum() == 1:
                continue
            angle_key = (min(atom_b, dummy), center, max(atom_b, dummy))
            if angle_key not in angles:
                continue
            theta, angle_k = angles[angle_key]
            for atom_a_obj in atom_b_obj.GetNeighbors():
                atom_a = atom_a_obj.GetIdx()
                if atom_a == center or atom_a in unique or atom_a_obj.GetAtomicNum() == 1:
                    continue
                quartet = (atom_a, atom_b, center, dummy)
                key = _canonical_torsion(quartet)
                grouped = torsions.get(key)
                if not grouped:
                    continue
                numeric_terms = tuple(
                    (
                        int(periodicity),
                        float(phase.value_in_unit(unit.radian)),
                        float(k.value_in_unit(unit.kilojoule_per_mole)),
                    )
                    for periodicity, phase, k in grouped
                )
                barrier = _torsion_barrier_kj_mol(numeric_terms)
                bond_ab = molecule.GetBondBetweenAtoms(atom_a, atom_b)
                rigid_ab = int(
                    bond_ab.IsInRing()
                    or bond_ab.GetIsAromatic()
                    or bond_ab.GetBondType() != Chem.BondType.SINGLE
                )
                score = (
                    rigid_ab,
                    int(bond_ab.IsInRing()),
                    barrier,
                    float(
                        angle_k.value_in_unit(
                            unit.kilojoule_per_mole / unit.radian**2
                        )
                    ),
                    -canonical_ranks[atom_a],
                    -canonical_ranks[atom_b],
                    -atom_a,
                    -atom_b,
                )
                candidates.append((score, quartet, theta, angle_k, numeric_terms, barrier))
        if not candidates:
            raise CovalentAlchemyError(
                f"terminal_z_matrix atom {dummy} in endpoint {endpoint} has no "
                "mapped heavy-atom proper-torsion reference chain"
            )
        _, quartet, theta, angle_k, numeric_terms, barrier = max(
            candidates, key=lambda item: item[0]
        )
        atom_a, atom_b, center, dummy = quartet
        result[dummy] = InactiveZMatrixTerm(
            endpoint=endpoint,
            dummy_atom=dummy,
            angle_atoms=(atom_b, center, dummy),
            torsion_atoms=quartet,
            hybrid_dummy_atom=mapping[dummy],
            hybrid_angle_atoms=_mapped_indices((atom_b, center, dummy), mapping),
            hybrid_torsion_atoms=_mapped_indices(quartet, mapping),
            angle_degrees=float(theta.value_in_unit(unit.degree)),
            angle_k_kj_mol_rad2=float(
                angle_k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)
            ),
            torsion_barrier_kj_mol=barrier,
            torsion_terms=numeric_terms,
        )
    return result


def _exception_parameters(force: mm.NonbondedForce):
    return {
        tuple(sorted((int(force.getExceptionParameters(index)[0]), int(force.getExceptionParameters(index)[1])))):
        force.getExceptionParameters(index)[2:]
        for index in range(force.getNumExceptions())
    }


def _add_unique_vacuum_nonbonded(
    output,
    sources_and_maps,
    endpoint_nonbonded,
    inactive_core_source=None,
):
    """Keep unique vacuum interactions, optionally including the inactive core."""
    vacuum = mm.CustomBondForce(
        "ONE_4PI_EPS0*chargeprod/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6)"
    )
    vacuum.setName("CovalentUniqueVacuumNonbondedForce")
    vacuum.addGlobalParameter("ONE_4PI_EPS0", 138.935456)
    for parameter in ("chargeprod", "sigma", "epsilon"):
        vacuum.addPerBondParameter(parameter)

    existing = {
        tuple(sorted((int(endpoint_nonbonded.getExceptionParameters(index)[0]),
                      int(endpoint_nonbonded.getExceptionParameters(index)[1])))): index
        for index in range(endpoint_nonbonded.getNumExceptions())
    }
    def pair_parameters(source_nb, exceptions, atom1, atom2):
        pair = tuple(sorted((atom1, atom2)))
        if pair in exceptions:
            chargeprod, sigma, epsilon = exceptions[pair]
            if (
                abs(chargeprod.value_in_unit(unit.elementary_charge**2)) < 1.0e-14
                and abs(epsilon.value_in_unit(unit.kilojoule_per_mole)) < 1.0e-14
            ):
                return None
            return chargeprod, sigma, epsilon
        q1, sigma1, epsilon1 = source_nb.getParticleParameters(atom1)
        q2, sigma2, epsilon2 = source_nb.getParticleParameters(atom2)
        return (
            q1 * q2,
            0.5 * (sigma1 + sigma2),
            (epsilon1 * epsilon2) ** 0.5,
        )

    for source, mapping, unique in sources_and_maps:
        source_nb = _source_force(source, mm.NonbondedForce)
        exceptions = _exception_parameters(source_nb)
        for offset, atom1 in enumerate(sorted(unique)):
            for atom2 in sorted(unique)[offset + 1:]:
                parameters = pair_parameters(source_nb, exceptions, atom1, atom2)
                hybrid_pair = tuple(sorted((mapping[atom1], mapping[atom2])))
                if parameters is not None:
                    vacuum.addBond(*hybrid_pair, parameters)
                if hybrid_pair in existing:
                    index = existing[hybrid_pair]
                    p1, p2, _, sigma, _ = endpoint_nonbonded.getExceptionParameters(index)
                    endpoint_nonbonded.setExceptionParameters(index, p1, p2, 0.0, sigma, 0.0)
                else:
                    existing[hybrid_pair] = endpoint_nonbonded.addException(
                        *hybrid_pair, 0.0, 1.0, 0.0
                    )

    if inactive_core_source is not None:
        source, mapping, unique, common = inactive_core_source
        source_nb = _source_force(source, mm.NonbondedForce)
        exceptions = _exception_parameters(source_nb)
        for atom1 in sorted(unique):
            for atom2 in sorted(common):
                parameters = pair_parameters(source_nb, exceptions, atom1, atom2)
                if parameters is not None:
                    vacuum.addBond(mapping[atom1], mapping[atom2], parameters)
    if vacuum.getNumBonds():
        output.addForce(vacuum)


def _build_endpoint(
    system_a: mm.System,
    system_b: mm.System,
    map_a_to_hybrid: dict[int, int],
    map_b_to_hybrid: dict[int, int],
    unique_a: set[int],
    unique_b: set[int],
    state: str,
    molecule_a: Chem.Mol,
    molecule_b: Chem.Mol,
    dummy_bonded_scales: DummyBondedScales,
    dummy_core_nonbonded: str,
    inactive_bonded_atoms_a: set[int],
    inactive_bonded_atoms_b: set[int],
    inactive_z_matrix_a: dict[int, InactiveZMatrixTerm],
    inactive_z_matrix_b: dict[int, InactiveZMatrixTerm],
) -> mm.System:
    output = mm.System()
    reverse_a = {hybrid: atom for atom, hybrid in map_a_to_hybrid.items()}
    reverse_b = {hybrid: atom for atom, hybrid in map_b_to_hybrid.items()}
    particle_count = max(max(reverse_a), max(reverse_b)) + 1
    for hybrid in range(particle_count):
        if state == "a" and hybrid in reverse_a:
            mass = system_a.getParticleMass(reverse_a[hybrid])
        elif state == "b" and hybrid in reverse_b:
            mass = system_b.getParticleMass(reverse_b[hybrid])
        elif hybrid in reverse_a:
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
    angle_scale_a, torsion_scale_a = _inactive_scales(
        molecule_a, unique_a, dummy_bonded_scales
    )
    angle_scale_b, torsion_scale_b = _inactive_scales(
        molecule_b, unique_b, dummy_bonded_scales
    )

    def local_angle_scale(base, atoms, selected, z_matrix):
        selected_atoms = set(atoms) & selected
        if not selected_atoms:
            return base(atoms)
        if len(selected_atoms) == 1:
            term = z_matrix.get(next(iter(selected_atoms)))
            if term is not None and (
                atoms == term.angle_atoms or atoms[::-1] == term.angle_atoms
            ):
                return 1.0
        return 0.0

    def local_torsion_scale(base, atoms, selected, z_matrix):
        selected_atoms = set(atoms) & selected
        if not selected_atoms:
            return base(atoms)
        if len(selected_atoms) == 1:
            term = z_matrix.get(next(iter(selected_atoms)))
            if term is not None and _canonical_torsion(atoms) == _canonical_torsion(
                term.torsion_atoms
            ):
                return 1.0
        return 0.0

    def local_bond_scale(atoms, selected):
        return 1.0 if any(atom in selected for atom in atoms) else dummy_bonded_scales.bond
    if state == "a":
        _add_bonds(bonds, force_a_bond, map_a_to_hybrid, lambda _: True)
        _add_angles(angles, force_a_angle, map_a_to_hybrid, lambda _: True)
        _add_torsions(torsions, force_a_torsion, map_a_to_hybrid, lambda _: True)
        include = lambda atoms: any(i in unique_b for i in atoms)
        _add_bonds(bonds, force_b_bond, map_b_to_hybrid, include, lambda atoms: local_bond_scale(atoms, inactive_bonded_atoms_b))
        _add_angles(angles, force_b_angle, map_b_to_hybrid, include, lambda atoms: local_angle_scale(angle_scale_b, atoms, inactive_bonded_atoms_b, inactive_z_matrix_b))
        _add_torsions(torsions, force_b_torsion, map_b_to_hybrid, include, lambda atoms: local_torsion_scale(torsion_scale_b, atoms, inactive_bonded_atoms_b, inactive_z_matrix_b))
    else:
        _add_bonds(bonds, force_b_bond, map_b_to_hybrid, lambda _: True)
        _add_angles(angles, force_b_angle, map_b_to_hybrid, lambda _: True)
        _add_torsions(torsions, force_b_torsion, map_b_to_hybrid, lambda _: True)
        include = lambda atoms: any(i in unique_a for i in atoms)
        _add_bonds(bonds, force_a_bond, map_a_to_hybrid, include, lambda atoms: local_bond_scale(atoms, inactive_bonded_atoms_a))
        _add_angles(angles, force_a_angle, map_a_to_hybrid, include, lambda atoms: local_angle_scale(angle_scale_a, atoms, inactive_bonded_atoms_a, inactive_z_matrix_a))
        _add_torsions(torsions, force_a_torsion, map_a_to_hybrid, include, lambda atoms: local_torsion_scale(torsion_scale_a, atoms, inactive_bonded_atoms_a, inactive_z_matrix_a))
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
    for atom_a in unique_a:
        hybrid_a = map_a_to_hybrid[atom_a]
        for atom_b in unique_b:
            hybrid_b = map_b_to_hybrid[atom_b]
            nonbonded.addException(
                hybrid_a,
                hybrid_b,
                0.0,
                1.0 * unit.nanometer,
                0.0,
            )
    output.addForce(nonbonded)
    _add_unique_vacuum_nonbonded(
        output,
        (
            (system_a, map_a_to_hybrid, unique_a),
            (system_b, map_b_to_hybrid, unique_b),
        ),
        nonbonded,
        inactive_core_source=(
            (
                system_b,
                map_b_to_hybrid,
                unique_b,
                set(range(molecule_b.GetNumAtoms())) - unique_b,
            )
            if state == "a" and dummy_core_nonbonded == "retain"
            else (
                system_a,
                map_a_to_hybrid,
                unique_a,
                set(range(molecule_a.GetNumAtoms())) - unique_a,
            )
            if state == "b" and dummy_core_nonbonded == "retain"
            else None
        ),
    )
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
            raw_name = atom.GetProp("_Name") if atom.HasProp("_Name") else ""
            base = re.sub(r"[^A-Za-z0-9_]", "", raw_name)
            if not base:
                base = f"{prefix}{atom.GetSymbol()}{atom.GetIdx()+1}"
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


def vacuum_nonbonded_pair_counts(hybrid: CovalentHybridMolecule):
    common = {
        hybrid.map_a_to_hybrid[index] for index in hybrid.map_a_to_b
    }
    unique_a = {
        hybrid.map_a_to_hybrid[index] for index in hybrid.unique_a
    }
    unique_b = {
        hybrid.map_b_to_hybrid[index] for index in hybrid.unique_b
    }

    def counts(system):
        force = next(
            (
                force
                for force in system.getForces()
                if force.getName() == "CovalentUniqueVacuumNonbondedForce"
            ),
            None,
        )
        result = {"unique_unique": 0, "unique_common": 0}
        if force is None:
            return result
        for index in range(force.getNumBonds()):
            atom1, atom2, _ = force.getBondParameters(index)
            pair = {int(atom1), int(atom2)}
            if pair <= unique_a or pair <= unique_b:
                result["unique_unique"] += 1
            elif pair & common and pair & (unique_a | unique_b):
                result["unique_common"] += 1
        return result

    return {
        "endpoint_a": counts(hybrid.endpoint_a),
        "endpoint_b": counts(hybrid.endpoint_b),
    }


def build_covalent_hybrid_molecule(
    parameters_a: CovalentParameterBundle,
    parameters_b: CovalentParameterBundle,
    *,
    required_pairs: list[tuple[int, int]] | None = None,
    atom_map: dict[int, int] | None = None,
    attachment_pairs: tuple[tuple[int, int], tuple[int, int]] | None = None,
    dummy_bonded_scales: DummyBondedScales | None = None,
    dummy_core_nonbonded: str = "off",
    transmuted_pairs: set[tuple[int, int]] | None = None,
    inactive_bonded_atoms_a: set[int] | None = None,
    inactive_bonded_atoms_b: set[int] | None = None,
    inactive_bonded_geometry: str = "bond_only",
) -> CovalentHybridMolecule:
    molecule_a = _rdkit_molecule(parameters_a)
    molecule_b = _rdkit_molecule(parameters_b)
    scales = dummy_bonded_scales or DummyBondedScales()
    dummy_core_nonbonded = str(dummy_core_nonbonded).lower()
    if dummy_core_nonbonded not in {"off", "retain"}:
        raise CovalentAlchemyError(
            "dummy_core_nonbonded must be 'off' or 'retain'"
        )
    if atom_map is None:
        map_a_to_b = find_covalent_atom_map(
            molecule_a, molecule_b, required_pairs=required_pairs
        )
    else:
        map_a_to_b = complete_covalent_atom_map(
            molecule_a,
            molecule_b,
            atom_map,
            required_pairs=required_pairs,
            transmuted_pairs=transmuted_pairs,
        )
    inactive_bonded_atoms_a = set(inactive_bonded_atoms_a or ())
    inactive_bonded_atoms_b = set(inactive_bonded_atoms_b or ())
    inactive_bonded_geometry = str(inactive_bonded_geometry).lower()
    if inactive_bonded_geometry not in {"bond_only", "terminal_z_matrix"}:
        raise CovalentAlchemyError(
            "inactive_bonded_geometry must be 'bond_only' or 'terminal_z_matrix'"
        )
    if attachment_pairs is not None:
        sulfur_pair, ligand_pair = attachment_pairs
        if map_a_to_b.get(sulfur_pair[0]) != sulfur_pair[1] or map_a_to_b.get(
            ligand_pair[0]
        ) != ligand_pair[1]:
            raise CovalentAlchemyError("covalent MCS does not contain the required attachment atoms")
        bond_a = molecule_a.GetBondBetweenAtoms(sulfur_pair[0], ligand_pair[0])
        bond_b = molecule_b.GetBondBetweenAtoms(sulfur_pair[1], ligand_pair[1])
        if bond_a is None or bond_b is None or _bond_signature(bond_a) != _bond_signature(bond_b):
            raise CovalentAlchemyError("covalent attachment bond differs between endpoints")
    map_a_to_hybrid = {index: index for index in range(molecule_a.GetNumAtoms())}
    map_b_to_hybrid = {atom_b: atom_a for atom_a, atom_b in map_a_to_b.items()}
    next_index = molecule_a.GetNumAtoms()
    for atom_b in range(molecule_b.GetNumAtoms()):
        if atom_b not in map_b_to_hybrid:
            map_b_to_hybrid[atom_b] = next_index
            next_index += 1
    unique_a = set(range(molecule_a.GetNumAtoms())) - set(map_a_to_b)
    unique_b = set(range(molecule_b.GetNumAtoms())) - set(map_a_to_b.values())
    if not inactive_bonded_atoms_a <= unique_a:
        raise CovalentAlchemyError("inactive ligand-A bonded atoms must be endpoint-unique")
    if not inactive_bonded_atoms_b <= unique_b:
        raise CovalentAlchemyError("inactive ligand-B bonded atoms must be endpoint-unique")
    inactive_z_matrix_a = (
        _select_inactive_z_matrix_terms(
            molecule_a,
            parameters_a.system,
            inactive_bonded_atoms_a,
            unique_a,
            map_a_to_hybrid,
            "a",
        )
        if inactive_bonded_geometry == "terminal_z_matrix"
        else {}
    )
    inactive_z_matrix_b = (
        _select_inactive_z_matrix_terms(
            molecule_b,
            parameters_b.system,
            inactive_bonded_atoms_b,
            unique_b,
            map_b_to_hybrid,
            "b",
        )
        if inactive_bonded_geometry == "terminal_z_matrix"
        else {}
    )
    endpoint_a = _build_endpoint(
        parameters_a.system,
        parameters_b.system,
        map_a_to_hybrid,
        map_b_to_hybrid,
        unique_a,
        unique_b,
        "a",
        molecule_a,
        molecule_b,
        scales,
        dummy_core_nonbonded,
        inactive_bonded_atoms_a,
        inactive_bonded_atoms_b,
        inactive_z_matrix_a,
        inactive_z_matrix_b,
    )
    endpoint_b = _build_endpoint(
        parameters_a.system,
        parameters_b.system,
        map_a_to_hybrid,
        map_b_to_hybrid,
        unique_a,
        unique_b,
        "b",
        molecule_a,
        molecule_b,
        scales,
        dummy_core_nonbonded,
        inactive_bonded_atoms_a,
        inactive_bonded_atoms_b,
        inactive_z_matrix_a,
        inactive_z_matrix_b,
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
        topology=topology,
        positions=positions * unit.angstrom,
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        map_a_to_b=map_a_to_b,
        map_a_to_hybrid=map_a_to_hybrid,
        map_b_to_hybrid=map_b_to_hybrid,
        unique_a=tuple(sorted(unique_a)),
        unique_b=tuple(sorted(unique_b)),
        anchor_pairs=tuple(required_pairs or ()),
        attachment_pairs=attachment_pairs,
        dummy_bonded_scales=scales,
        dummy_core_nonbonded=dummy_core_nonbonded,
        transmuted_pairs=tuple(sorted(transmuted_pairs or ())),
        inactive_bonded_atoms_a=tuple(sorted(inactive_bonded_atoms_a)),
        inactive_bonded_atoms_b=tuple(sorted(inactive_bonded_atoms_b)),
        inactive_bonded_geometry=inactive_bonded_geometry,
        inactive_z_matrix_terms=tuple(
            inactive_z_matrix_a[index] for index in sorted(inactive_z_matrix_a)
        ) + tuple(
            inactive_z_matrix_b[index] for index in sorted(inactive_z_matrix_b)
        ),
    )


# Generic names for noncovalent and covalent dual-topology preparation.  The
# covalent names remain available for existing internal callers.
HybridMolecule = CovalentHybridMolecule
HybridBondedScales = DummyBondedScales
build_hybrid_molecule = build_covalent_hybrid_molecule
