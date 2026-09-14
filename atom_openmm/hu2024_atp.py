from __future__ import annotations

import ast
from pathlib import Path
import re

import openmm as mm
from openmm import unit


class HuATPError(ValueError):
    pass


def _normalized_name(name):
    return str(name).replace("'", "*") if name.startswith(("O", "C")) else str(name)


def read_prepi(path):
    atoms = {}
    in_atoms = False
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) >= 11 and fields[0].isdigit():
            in_atoms = True
            name, atom_type, charge = fields[1], fields[2], fields[-1]
            if name != "DUMM":
                atoms[_normalized_name(name)] = {
                    "type": atom_type, "charge_e": float(charge)
                }
        elif in_atoms and line.strip() == "LOOP":
            break
    if not atoms:
        raise HuATPError(f"no ATP atoms found in {path}")
    return atoms


def read_frcmod(path):
    sections = {name: [] for name in ("BOND", "ANGLE", "DIHE")}
    current = None
    for line in Path(path).read_text().splitlines():
        label = line.strip()
        if label in {"MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON"}:
            current = label
            continue
        if current in sections and label:
            count = {"BOND": 2, "ANGLE": 3, "DIHE": 4}[current]
            pattern = r"^\s*((?:[A-Za-z0-9*]+\s*-\s*){" + str(count - 1) + r"}[A-Za-z0-9*]+)\s+(.+)$"
            match = re.match(pattern, line)
            if not match:
                raise HuATPError(f"invalid Hu {current} parameter: {line!r}")
            atom_pattern = "-".join(part.strip() for part in match.group(1).split("-"))
            sections[current].append([atom_pattern, *match.group(2).split()])
    return sections


def read_b3_cmap(path):
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "cmap_matrix_B3"
            for target in node.targets
        ):
            values = ast.literal_eval(node.value)
            if len(values) != 24 * 24:
                raise HuATPError("Hu B3 CMAP must contain 576 values")
            return [float(value) for value in values]
    raise HuATPError(f"cmap_matrix_B3 not found in {path}")


def _atp_atoms(topology, residue_name="ATP"):
    residues = [residue for residue in topology.residues() if residue.name == residue_name]
    if len(residues) != 1:
        raise HuATPError(f"expected one {residue_name} residue, found {len(residues)}")
    return {atom.name: atom for atom in residues[0].atoms()}


def _force(system, force_type):
    found = [force for force in system.getForces() if isinstance(force, force_type)]
    if len(found) != 1:
        raise HuATPError(f"expected one {force_type.__name__}, found {len(found)}")
    return found[0]


def _types_match(pattern, observed):
    return all(expected == "X" or expected == actual for expected, actual in zip(pattern, observed))


def apply_hu2024_atp(system, topology, *, prepi, frcmod, mod_py,
                     atom_classes=None, residue_name="ATP"):
    """Apply the Hu et al. B3 ATP charges, bonded terms, CMAP, and Mg cross LJ."""
    atoms = _atp_atoms(topology, residue_name)
    parameters = read_prepi(prepi)
    missing = sorted(set(parameters) - set(atoms))
    if missing:
        raise HuATPError(f"ATP topology is missing Hu atoms: {missing}")
    atom_types = {atoms[name].index: value["type"] for name, value in parameters.items()}
    nonbonded = _force(system, mm.NonbondedForce)
    previous_charges = {
        index: nonbonded.getParticleParameters(index)[0].value_in_unit(unit.elementary_charge)
        for index in atom_types
    }
    for name, value in parameters.items():
        index = atoms[name].index
        _, sigma, epsilon = nonbonded.getParticleParameters(index)
        nonbonded.setParticleParameters(index, value["charge_e"], sigma, epsilon)
    updated_exceptions = 0
    for exception_index in range(nonbonded.getNumExceptions()):
        i, j, product, sigma, epsilon = nonbonded.getExceptionParameters(exception_index)
        if i not in atom_types or j not in atom_types:
            continue
        old_product = previous_charges[i] * previous_charges[j]
        if abs(old_product) < 1e-10:
            if abs(product.value_in_unit(unit.elementary_charge**2)) > 1e-10:
                raise HuATPError("cannot reconstruct ATP 1-4 charge scaling from zero original charges")
            continue
        scale = product.value_in_unit(unit.elementary_charge**2) / old_product
        if abs(scale) > 1e-10:
            new_i = parameters[next(name for name, atom in atoms.items() if atom.index == i)]["charge_e"]
            new_j = parameters[next(name for name, atom in atoms.items() if atom.index == j)]["charge_e"]
            nonbonded.setExceptionParameters(
                exception_index, i, j, scale * new_i * new_j * unit.elementary_charge**2,
                sigma, epsilon,
            )
            updated_exceptions += 1

    bonded = read_frcmod(frcmod)
    bond_force = _force(system, mm.HarmonicBondForce)
    bond_values = {}
    for fields in bonded["BOND"]:
        types = tuple(fields[0].split("-"))
        bond_values[frozenset(types)] = (float(fields[1]), float(fields[2]))
    changed_bonds = 0
    for term in range(bond_force.getNumBonds()):
        i, j, length, k = bond_force.getBondParameters(term)
        if i in atom_types and j in atom_types:
            value = bond_values.get(frozenset((atom_types[i], atom_types[j])))
            if value:
                force_constant, length_a = value
                bond_force.setBondParameters(
                    term, i, j, length_a * unit.angstrom,
                    2.0 * force_constant * unit.kilocalorie_per_mole / unit.angstrom**2,
                )
                changed_bonds += 1

    angle_force = _force(system, mm.HarmonicAngleForce)
    angle_values = {}
    for fields in bonded["ANGLE"]:
        types = tuple(part.strip() for part in fields[0].split("-"))
        angle_values[types] = (float(fields[1]), float(fields[2]))
    changed_angles = 0
    for term in range(angle_force.getNumAngles()):
        i, j, kidx, theta, force_constant = angle_force.getAngleParameters(term)
        if all(index in atom_types for index in (i, j, kidx)):
            observed = (atom_types[i], atom_types[j], atom_types[kidx])
            value = angle_values.get(observed) or angle_values.get(observed[::-1])
            if value:
                amber_k, degrees = value
                angle_force.setAngleParameters(
                    term, i, j, kidx, degrees * unit.degree,
                    2.0 * amber_k * unit.kilocalorie_per_mole / unit.radian**2,
                )
                changed_angles += 1

    torsion_force = _force(system, mm.PeriodicTorsionForce)
    definitions = []
    for fields in bonded["DIHE"]:
        if len(fields) != 5:
            raise HuATPError(f"invalid Hu torsion definition: {fields}")
        divisor = float(fields[1])
        if divisor == 0:
            raise HuATPError("Hu torsion divisor must be nonzero")
        definitions.append((
            tuple(part.strip() for part in fields[0].split("-")),
            float(fields[3]), int(abs(float(fields[4]))), float(fields[2]) / divisor,
        ))
    changed_torsions = 0
    updated_quartets = set()
    for term in range(torsion_force.getNumTorsions()):
        i, j, kidx, l, periodicity, phase, force_constant = torsion_force.getTorsionParameters(term)
        if all(index in atom_types for index in (i, j, kidx, l)):
            observed = (atom_types[i], atom_types[j], atom_types[kidx], atom_types[l])
            matches = [value for value in definitions if _types_match(value[0], observed) or _types_match(value[0], observed[::-1])]
            if matches:
                torsion_force.setTorsionParameters(term, i, j, kidx, l, periodicity, phase, 0.0)
                quartet = (i, j, kidx, l)
                if quartet not in updated_quartets and quartet[::-1] not in updated_quartets:
                    for _, phase_deg, new_periodicity, amber_k in matches:
                        torsion_force.addTorsion(
                            i, j, kidx, l, new_periodicity, phase_deg * unit.degree,
                            amber_k * unit.kilocalorie_per_mole,
                        )
                    updated_quartets.add(quartet)
                changed_torsions += 1

    cmap = mm.CMAPTorsionForce()
    cmap.setName("Hu2024 ATP polyphosphate CMAP (B3)")
    cmap_index = cmap.addMap(24, [value * unit.kilocalorie_per_mole for value in read_b3_cmap(mod_py)])
    sequence = [atoms[name].index for name in ("O3B", "PB", "O3A", "PA", "O5*")]
    cmap.addTorsion(cmap_index, *sequence[:4], *sequence[1:])
    system.addForce(cmap)

    hu_cross = {
        "O3": (56958.78, 69.26), "P": (133097.78, 154.94),
        "OS": (26407.56, 66.18), "O2": (56958.78, 69.26),
        "OY": (26407.56, 66.18),
    }
    topology_atoms = list(topology.atoms())
    if atom_classes is None:
        atom_classes = [None] * len(topology_atoms)
    if len(atom_classes) != len(topology_atoms):
        raise HuATPError("Amber atom classes must match the system particle count")
    global_types = {
        atom.index: atom_types.get(atom.index, atom_classes[atom.index])
        for atom in topology_atoms
    }
    correction = mm.CustomNonbondedForce(
        "A/r^12-B/r^6-4*sqrt(eps1*eps2)*((0.5*(sig1+sig2)/r)^12-(0.5*(sig1+sig2)/r)^6);"
        "A=A1+A2; B=B1+B2"
    )
    correction.setName("Hu2024 Mg-ATP pair-specific LJ correction")
    for name in ("A", "B", "sig", "eps"):
        correction.addPerParticleParameter(name)
    for atom in topology_atoms:
        _, sigma, epsilon = nonbonded.getParticleParameters(atom.index)
        atom_type = global_types[atom.index]
        a, b = hu_cross.get(atom_type, (0.0, 0.0))
        correction.addParticle([a * 4.184e-12, b * 4.184e-6, sigma, epsilon])
    mg_indices = [atom.index for atom in topology_atoms if atom.element and atom.element.symbol == "Mg"]
    corrected_atoms = {index for index, atom_type in global_types.items() if atom_type in hu_cross}
    correction.addInteractionGroup(set(mg_indices), corrected_atoms)
    from atom_openmm.metal_ions import copy_nonbonded_exclusions

    copy_nonbonded_exclusions(nonbonded, correction)
    correction.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    correction.setCutoffDistance(nonbonded.getCutoffDistance())
    corrected_pairs = len(mg_indices) * len(corrected_atoms)
    system.addForce(correction)
    return {
        "model": "Hu2024-B3", "changed_bonds": changed_bonds,
        "changed_angles": changed_angles, "changed_torsions": changed_torsions,
        "cmap_terms": 1, "mg_atp_lj_pairs": corrected_pairs,
        "updated_14_exceptions": updated_exceptions,
    }
