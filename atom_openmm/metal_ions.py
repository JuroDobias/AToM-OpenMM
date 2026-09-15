from __future__ import annotations

import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import openmm as mm
from openmm import unit


class MetalIonParameterError(ValueError):
    pass


PANTEVA_FORCE_NAME = "Panteva modified Mg 12-6-4"


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


def gaff2_atom_classes(sdf, positions):
    """Return GAFF2 atom classes without changing the input atom order."""
    from parmed import load_file

    with tempfile.TemporaryDirectory(prefix="atom-gaff2-types-") as directory:
        mol2 = Path(directory) / "typing.mol2"
        subprocess.run(
            [
                "antechamber", "-i", str(sdf), "-fi", "sdf",
                "-o", str(mol2), "-fo", "mol2", "-at", "gaff2",
                "-seq", "n", "-s", "0", "-pf", "y",
            ],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )
        typed = load_file(str(mol2))
        expected = np.asarray(positions.value_in_unit(unit.angstrom))
        observed = np.asarray(typed.coordinates)
        if observed.shape != expected.shape or np.max(np.abs(observed - expected)) > 0.02:
            raise MetalIonParameterError(
                "GAFF2 typing changed ligand atom order or coordinates"
            )
        return [atom.type for atom in typed.atoms]


def copy_nonbonded_exclusions(nonbonded, custom):
    """Match the exclusions required across OpenMM nonbonded forces."""
    for index in range(nonbonded.getNumExceptions()):
        atom_a, atom_b, *_ = nonbonded.getExceptionParameters(index)
        custom.addExclusion(atom_a, atom_b)


def apply_panteva_m1264(system, topology, *, atom_classes,
                       polarizability_table, atp_residue_name="ATP",
                       water_model="tip4pew", active_atom_indices=None):
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

    active = (
        set(range(len(atoms)))
        if active_atom_indices is None
        else {int(index) for index in active_atom_indices}
    )
    c4_values = []
    for atom, atom_class in zip(atoms, atom_classes):
        if atom.index not in active:
            c4_values.append(0.0)
            continue
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
    force.setName(PANTEVA_FORCE_NAME)
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


def merge_panteva_endpoint_forces(output, endpoint_a, endpoint_b, unique_a, unique_b):
    """Merge physical endpoint C4 forces into the hybrid switching Hamiltonian."""
    forces_a = [force for force in endpoint_a.getForces() if force.getName() == PANTEVA_FORCE_NAME]
    forces_b = [force for force in endpoint_b.getForces() if force.getName() == PANTEVA_FORCE_NAME]
    if not forces_a and not forces_b:
        return False
    if len(forces_a) != 1 or len(forces_b) != 1:
        raise MetalIonParameterError(
            "hybrid endpoints must each contain exactly one Panteva C4 force"
        )
    force_a, force_b = forces_a[0], forces_b[0]
    if force_a.getNumParticles() != force_b.getNumParticles():
        raise MetalIonParameterError("Panteva endpoint particle counts differ")
    unique_a = {int(index) for index in unique_a}
    unique_b = {int(index) for index in unique_b}
    expression = (
        "-(isMg1*c4eff2+isMg2*c4eff1)/r^4;"
        "c4eff1=c4a1*(env1+ua1*COVALENT_STERICS_A+mapped1*(1-COVALENT_STERICS))"
        "+c4b1*(ub1*COVALENT_STERICS_B+mapped1*COVALENT_STERICS);"
        "c4eff2=c4a2*(env2+ua2*COVALENT_STERICS_A+mapped2*(1-COVALENT_STERICS))"
        "+c4b2*(ub2*COVALENT_STERICS_B+mapped2*COVALENT_STERICS)"
    )
    merged = mm.CustomNonbondedForce(expression)
    merged.setName(PANTEVA_FORCE_NAME)
    for name in ("COVALENT_STERICS_A", "COVALENT_STERICS_B", "COVALENT_STERICS"):
        merged.addGlobalParameter(name, 0.0)
    for name in ("isMg", "c4a", "c4b", "env", "ua", "ub", "mapped"):
        merged.addPerParticleParameter(name)

    mg_indices = set()
    for index in range(force_a.getNumParticles()):
        params_a = list(force_a.getParticleParameters(index))
        params_b = list(force_b.getParticleParameters(index))
        is_mg_a, c4_a = float(params_a[0]), float(params_a[1])
        is_mg_b, c4_b = float(params_b[0]), float(params_b[1])
        if is_mg_a != is_mg_b:
            raise MetalIonParameterError("Panteva endpoint magnesium identities differ")
        if is_mg_a:
            mg_indices.add(index)
        ua = float(index in unique_a)
        ub = float(index in unique_b)
        mapped = float(
            not ua and not ub and not is_mg_a
            and not math.isclose(c4_a, c4_b, rel_tol=0.0, abs_tol=1.0e-15)
        )
        # Unchanged mapped atoms are equivalent to environment atoms for this force.
        env = float(not ua and not ub and not mapped)
        merged.addParticle([is_mg_a, c4_a, c4_b, env, ua, ub, mapped])
    exclusions_a = {
        tuple(sorted(int(value) for value in force_a.getExclusionParticles(index)))
        for index in range(force_a.getNumExclusions())
        if any(
            int(value) in mg_indices
            for value in force_a.getExclusionParticles(index)
        )
    }
    exclusions_b = {
        tuple(sorted(int(value) for value in force_b.getExclusionParticles(index)))
        for index in range(force_b.getNumExclusions())
        if any(
            int(value) in mg_indices
            for value in force_b.getExclusionParticles(index)
        )
    }
    if exclusions_a != exclusions_b:
        raise MetalIonParameterError(
            "Panteva endpoint Mg exclusion lists differ"
        )
    non_mg = set(range(force_a.getNumParticles())) - mg_indices
    merged.addInteractionGroup(mg_indices, non_mg)
    existing_custom = [
        force
        for force in output.getForces()
        if isinstance(force, mm.CustomNonbondedForce)
    ]
    if existing_custom:
        canonical_exclusions = {
            tuple(
                sorted(
                    int(value)
                    for value in existing_custom[0].getExclusionParticles(index)
                )
            )
            for index in range(existing_custom[0].getNumExclusions())
        }
    else:
        canonical_exclusions = {
            tuple(sorted(int(value) for value in force.getExclusionParticles(index)))
            for force in (force_a, force_b)
            for index in range(force.getNumExclusions())
        }
    for atom_a, atom_b in sorted(canonical_exclusions):
        merged.addExclusion(atom_a, atom_b)
    merged.setNonbondedMethod(force_a.getNonbondedMethod())
    if force_a.getNonbondedMethod() != mm.CustomNonbondedForce.NoCutoff:
        merged.setCutoffDistance(force_a.getCutoffDistance())
    merged.setUseLongRangeCorrection(False)
    merged.setForceGroup(force_a.getForceGroup())
    output.addForce(merged)
    return True
