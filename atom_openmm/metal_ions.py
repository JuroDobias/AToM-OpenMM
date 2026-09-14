from __future__ import annotations

from pathlib import Path

import openmm as mm
from openmm import unit


class MetalIonParameterError(ValueError):
    pass


def c4_kcal_a4_to_kj_nm4(value):
    return float(value) * 4.184e-4


def read_polarizabilities(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2:
            try:
                values[fields[0]] = float(fields[1])
            except ValueError:
                continue
    if values.get("OW") != 1.444 or values.get("O2") != 0.569:
        raise MetalIonParameterError("unexpected Amber 12-6-4 polarizability table")
    return values


def copy_nonbonded_exclusions(nonbonded, custom):
    """Match the NonbondedForce exception list used by OpenMM's CPU neighbor list."""
    for index in range(nonbonded.getNumExceptions()):
        atom_a, atom_b, *_ = nonbonded.getExceptionParameters(index)
        custom.addExclusion(atom_a, atom_b)


def apply_panteva_m1264(system, topology, *, atom_classes,
                       polarizability_table, atp_residue_name="ATP",
                       water_model="tip4pew"):
    """Apply TIP4P-Ew Li-Merz Mg 12-6-4 with Panteva ATP overrides."""
    if str(water_model).lower() not in {"tip4pew", "tip4p-ew"}:
        raise MetalIonParameterError(
            "Panteva m12-6-4 parameters require the TIP4P-Ew water model"
        )
    nonbonded = next(
        (force for force in system.getForces() if isinstance(force, mm.NonbondedForce)), None
    )
    if nonbonded is None:
        raise MetalIonParameterError("system has no NonbondedForce")
    atoms = list(topology.atoms())
    if atom_classes is None or len(atom_classes) != len(atoms):
        raise MetalIonParameterError("one Amber/GAFF2 atom class is required for every particle")
    polarizabilities = read_polarizabilities(polarizability_table)
    mg_indices = [atom.index for atom in atoms if atom.element and atom.element.symbol == "Mg"]
    if not mg_indices:
        raise MetalIonParameterError("topology contains no magnesium ions")
    for index in mg_indices:
        charge, _, _ = nonbonded.getParticleParameters(index)
        # Amber Rmin/2 -> OpenMM sigma.
        sigma = 2.0 * 1.437 / (2.0 ** (1.0 / 6.0)) * unit.angstrom
        nonbonded.setParticleParameters(index, charge, sigma, 0.02257962 * unit.kilocalorie_per_mole)

    c4_values = []
    for atom, atom_class in zip(atoms, atom_classes):
        if atom_class == "EP" and system.isVirtualSite(atom.index):
            c4_values.append(0.0)
            continue
        if atom_class not in polarizabilities and atom_class is not None and atom_class.upper() in polarizabilities:
            atom_class = atom_class.upper()
        if atom_class not in polarizabilities:
            raise MetalIonParameterError(
                f"missing Amber 12-6-4 polarizability for atom {atom.index} "
                f"({atom.residue.name} {atom.name}, class {atom_class!r})"
            )
        alpha = polarizabilities[atom_class]
        c4 = 180.5 * alpha / 1.444
        if atom.residue.name == atp_residue_name and atom.name in {"O1A", "O2A", "O1B", "O2B", "O1G", "O2G", "O3G"}:
            c4 = 21.25
        elif atom.residue.name == atp_residue_name and atom.name == "N7":
            c4 = 238.75
        c4_values.append(c4_kcal_a4_to_kj_nm4(c4))

    force = mm.CustomNonbondedForce("-(isMg1*c42+isMg2*c41)/r^4")
    force.setName("Panteva modified Mg 12-6-4")
    force.addPerParticleParameter("isMg")
    force.addPerParticleParameter("c4")
    mg_set = set(mg_indices)
    for atom in atoms:
        force.addParticle([1.0 if atom.index in mg_set else 0.0, c4_values[atom.index]])
    force.addInteractionGroup(mg_set, set(range(len(atoms))) - mg_set)
    copy_nonbonded_exclusions(nonbonded, force)
    if nonbonded.getNonbondedMethod() == mm.NonbondedForce.NoCutoff:
        force.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
    else:
        force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force.setCutoffDistance(nonbonded.getCutoffDistance())
    force.setUseLongRangeCorrection(False)
    system.addForce(force)
    return {
        "model": "Panteva-m12-6-4", "magnesium_count": len(mg_indices),
        "atp_nonbridging_oxygen_c4_kcal_a4": 21.25,
        "atp_n7_c4_kcal_a4": 238.75, "water_oxygen_c4_kcal_a4": 180.5,
    }
