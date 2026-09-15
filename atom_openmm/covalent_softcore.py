from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
import statistics
import openmm as mm
from openmm import unit

from atom_openmm.covalent_alchemy import (
    CovalentAlchemyError,
    _assert_compatible_endpoints,
    _clone_force,
    _clone_virtual_site,
)


CHARGE_A_PARAMETER = "COVALENT_CHARGE_A"
CHARGE_B_PARAMETER = "COVALENT_CHARGE_B"
MAPPED_CHARGE_PARAMETER = "COVALENT_MAPPED_CHARGE"
STERICS_PARAMETER = "COVALENT_STERICS"
STERICS_A_PARAMETER = "COVALENT_STERICS_A"
STERICS_B_PARAMETER = "COVALENT_STERICS_B"
BONDED_A_PARAMETER = "COVALENT_BONDED_A"
BONDED_B_PARAMETER = "COVALENT_BONDED_B"
SEPARATE_BONDED_PARAMETER = "COVALENT_SEPARATE_BONDED"
SOFT_BOND_A_PARAMETER = "COVALENT_SOFT_BOND_A"
SOFT_BOND_B_PARAMETER = "COVALENT_SOFT_BOND_B"
SOFT_ANGLE_A_PARAMETER = "COVALENT_SOFT_ANGLE_A"
SOFT_ANGLE_B_PARAMETER = "COVALENT_SOFT_ANGLE_B"
SOFT_TORSION_A_PARAMETER = "COVALENT_SOFT_TORSION_A"
SOFT_TORSION_B_PARAMETER = "COVALENT_SOFT_TORSION_B"
BOND_NONBONDED_CHARGE_A_PARAMETER = "COVALENT_BOND_NONBONDED_CHARGE_A"
BOND_NONBONDED_CHARGE_B_PARAMETER = "COVALENT_BOND_NONBONDED_CHARGE_B"
BOND_NONBONDED_VDW_A_PARAMETER = "COVALENT_BOND_NONBONDED_VDW_A"
BOND_NONBONDED_VDW_B_PARAMETER = "COVALENT_BOND_NONBONDED_VDW_B"
BOND_ONE_FOUR_CHARGE_A_PARAMETER = "COVALENT_BOND_ONE_FOUR_CHARGE_A"
BOND_ONE_FOUR_CHARGE_B_PARAMETER = "COVALENT_BOND_ONE_FOUR_CHARGE_B"
BOND_ONE_FOUR_VDW_A_PARAMETER = "COVALENT_BOND_ONE_FOUR_VDW_A"
BOND_ONE_FOUR_VDW_B_PARAMETER = "COVALENT_BOND_ONE_FOUR_VDW_B"
RECIPROCAL_A_CHARGE_PARAMETER = "COVALENT_RECIPROCAL_A_CHARGE"
RECIPROCAL_B_CHARGE_PARAMETER = "COVALENT_RECIPROCAL_B_CHARGE"
RECIPROCAL_A_EXCEPTION_PARAMETER = "COVALENT_RECIPROCAL_A_EXCEPTION"
RECIPROCAL_B_EXCEPTION_PARAMETER = "COVALENT_RECIPROCAL_B_EXCEPTION"
GAPSYS_RECIPROCAL_A_CHARGE_PARAMETER = "COVALENT_GAPSYS_RECIPROCAL_A_CHARGE"
GAPSYS_RECIPROCAL_B_CHARGE_PARAMETER = "COVALENT_GAPSYS_RECIPROCAL_B_CHARGE"
GAPSYS_RECIPROCAL_A_EXCEPTION_PARAMETER = "COVALENT_GAPSYS_RECIPROCAL_A_EXCEPTION"
GAPSYS_RECIPROCAL_B_EXCEPTION_PARAMETER = "COVALENT_GAPSYS_RECIPROCAL_B_EXCEPTION"
SOFTCORE_NONBONDED_FORCE_GROUP = 31
ONE_4PI_EPS0 = 138.935456
AMBER_SSC2_IMPLEMENTATION = "amber_gti_ssc2_v1"
LEGACY_SSC2_IMPLEMENTATION = "effective_distance_ssc2_v1"
GROMACS_GAPSYS_IMPLEMENTATION = "gromacs_gapsys_2026_v1"


EXPLICIT_PATH_CONTROLS = (
    "charge_a", "charge_b", "sterics_a", "sterics_b",
    "mapped_charge", "mapped_vdw", "bonded_a", "bonded_b",
    "soft_bond_a", "soft_bond_b", "soft_angles_a", "soft_angles_b",
    "soft_torsions_a", "soft_torsions_b",
    "bond_nonbonded_charge_a", "bond_nonbonded_charge_b",
    "bond_nonbonded_vdw_a", "bond_nonbonded_vdw_b",
    "bond_one_four_charge_a", "bond_one_four_charge_b",
    "bond_one_four_vdw_a", "bond_one_four_vdw_b",
)

EXPLICIT_ENDPOINT_A_CONTROLS = {
    "charge_a": 1.0, "charge_b": 0.0,
    "sterics_a": 1.0, "sterics_b": 0.0,
    "mapped_charge": 0.0, "mapped_vdw": 0.0,
    "bonded_a": 1.0, "bonded_b": 0.0,
    "soft_bond_a": 1.0, "soft_bond_b": 0.0,
    "soft_angles_a": 1.0, "soft_angles_b": 0.0,
    "soft_torsions_a": 1.0, "soft_torsions_b": 0.0,
    "bond_nonbonded_charge_a": 0.0, "bond_nonbonded_charge_b": 1.0,
    "bond_nonbonded_vdw_a": 0.0, "bond_nonbonded_vdw_b": 1.0,
    "bond_one_four_charge_a": 1.0, "bond_one_four_charge_b": 0.0,
    "bond_one_four_vdw_a": 1.0, "bond_one_four_vdw_b": 0.0,
}
EXPLICIT_ENDPOINT_B_CONTROLS = {
    name: EXPLICIT_ENDPOINT_A_CONTROLS[
        name[:-1] + ("b" if name.endswith("a") else "a")
    ]
    if name.endswith(("a", "b"))
    else 1.0
    for name in EXPLICIT_PATH_CONTROLS
}


@dataclass(frozen=True)
class CovalentSoftcoreHamiltonian:
    system: mm.System
    parameter_values: dict[str, list[float]]
    segment_steps: list[int]
    total_steps: int
    resolved_path: dict
    stage_interpolation: str = "linear"


def _force(system: mm.System, cls):
    forces = [force for force in system.getForces() if isinstance(force, cls)]
    if len(forces) > 1:
        raise CovalentAlchemyError(f"multiple {cls.__name__} instances are not supported")
    return forces[0] if forces else None


def _float(value, target_unit):
    return float(value.value_in_unit(target_unit))


def _smoothstep2(value):
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def _copy_force_metadata(source, target):
    target.setName(source.getName())
    target.setForceGroup(source.getForceGroup())
    if hasattr(source, "usesPeriodicBoundaryConditions") and hasattr(
        target, "setUsesPeriodicBoundaryConditions"
    ):
        target.setUsesPeriodicBoundaryConditions(source.usesPeriodicBoundaryConditions())


def _system_shell(endpoint_a: mm.System) -> mm.System:
    output = mm.System()
    for index in range(endpoint_a.getNumParticles()):
        output.addParticle(endpoint_a.getParticleMass(index))
        if endpoint_a.isVirtualSite(index):
            output.setVirtualSite(index, _clone_virtual_site(endpoint_a.getVirtualSite(index)))
    for index in range(endpoint_a.getNumConstraints()):
        output.addConstraint(*endpoint_a.getConstraintParameters(index))
    output.setDefaultPeriodicBoxVectors(*endpoint_a.getDefaultPeriodicBoxVectors())
    return output


def _common_mass_endpoint_copies(endpoint_a: mm.System, endpoint_b: mm.System):
    if endpoint_a.getNumParticles() != endpoint_b.getNumParticles():
        raise CovalentAlchemyError("endpoint systems must contain identical particle counts")
    copied_a = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(endpoint_a))
    copied_b = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(endpoint_b))
    for index in range(endpoint_a.getNumParticles()):
        mass_a = endpoint_a.getParticleMass(index)
        mass_b = endpoint_b.getParticleMass(index)
        common = max(
            mass_a.value_in_unit(unit.dalton),
            mass_b.value_in_unit(unit.dalton),
        ) * unit.dalton
        copied_a.setParticleMass(index, common)
        copied_b.setParticleMass(index, common)
    return copied_a, copied_b


def _canonical_angle(particles):
    particles = tuple(int(value) for value in particles)
    reverse = tuple(reversed(particles))
    return min(particles, reverse)


def _term_contains_bond(atoms, selected_bonds):
    atoms = {int(atom) for atom in atoms}
    return any(set(pair) <= atoms for pair in selected_bonds)


def _bond_terms(force):
    if force is None:
        return {}
    terms = {}
    for index in range(force.getNumBonds()):
        p1, p2, length, k = force.getBondParameters(index)
        key = tuple(sorted((int(p1), int(p2))))
        terms[key] = (
            _float(length, unit.nanometer),
            _float(k, unit.kilojoule_per_mole / unit.nanometer**2),
        )
    return terms


def _angle_terms(force):
    if force is None:
        return {}
    terms = {}
    for index in range(force.getNumAngles()):
        p1, p2, p3, theta, k = force.getAngleParameters(index)
        key = _canonical_angle((p1, p2, p3))
        terms[key] = (
            _float(theta, unit.radian),
            _float(k, unit.kilojoule_per_mole / unit.radian**2),
        )
    return terms


def _torsion_terms(force):
    terms = []
    if force is None:
        return terms
    for index in range(force.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(index)
        # Preserve torsion orientation. Reversing atom order changes the sign of
        # theta and is only equivalent for special phase values.
        particles = tuple(int(value) for value in (p1, p2, p3, p4))
        terms.append(
            (
                particles,
                int(periodicity),
                _float(phase, unit.radian),
                _float(k, unit.kilojoule_per_mole),
            )
        )
    return terms


def _split_identical(left, right):
    left_counts = Counter(left)
    right_counts = Counter(right)
    common = left_counts & right_counts
    only_left = left_counts - common
    only_right = right_counts - common
    return list(common.elements()), list(only_left.elements()), list(only_right.elements())


def _add_bonded_forces(
    output,
    endpoint_a,
    endpoint_b,
    *,
    soft_bond_alpha_nm2=100.0,
    soft_bond_pairs=(),
):
    bond_a = _force(endpoint_a, mm.HarmonicBondForce)
    bond_b = _force(endpoint_b, mm.HarmonicBondForce)
    bonds_a = _bond_terms(bond_a)
    bonds_b = _bond_terms(bond_b)
    common_bonds = mm.HarmonicBondForce()
    changed_bonds = mm.CustomBondForce(
        "0.5*(wA*kA*(r-rA)^2+wB*kB*(r-rB)^2);"
        "wA=(1-separate)*(1-COVALENT_STERICS)+separate*COVALENT_BONDED_A;"
        "wB=(1-separate)*COVALENT_STERICS+separate*COVALENT_BONDED_B;"
        "separate=COVALENT_SEPARATE_BONDED"
    )
    changed_bonds.addGlobalParameter(BONDED_A_PARAMETER, 0.0)
    changed_bonds.addGlobalParameter(BONDED_B_PARAMETER, 0.0)
    changed_bonds.addGlobalParameter(STERICS_PARAMETER, 0.0)
    changed_bonds.addGlobalParameter(SEPARATE_BONDED_PARAMETER, 0.0)
    for name in ("rA", "kA", "rB", "kB"):
        changed_bonds.addPerBondParameter(name)
    soft_bonds = mm.CustomBondForce(
        "0.5*(wA*kA*drA^2/(1+SOFT_BOND_ALPHA*(1-wA)*drA^2)"
        "+wB*kB*drB^2/(1+SOFT_BOND_ALPHA*(1-wB)*drB^2));"
        "wA=COVALENT_SOFT_BOND_A;wB=COVALENT_SOFT_BOND_B;"
        "drA=r-rA;drB=r-rB"
    )
    soft_bonds.addGlobalParameter(SOFT_BOND_A_PARAMETER, 0.0)
    soft_bonds.addGlobalParameter(SOFT_BOND_B_PARAMETER, 0.0)
    soft_bonds.addGlobalParameter("SOFT_BOND_ALPHA", float(soft_bond_alpha_nm2))
    for name in ("rA", "kA", "rB", "kB"):
        soft_bonds.addPerBondParameter(name)
    soft_bond_pairs = {tuple(sorted(pair)) for pair in soft_bond_pairs}
    for particles in sorted(set(bonds_a) | set(bonds_b)):
        value_a = bonds_a.get(particles)
        value_b = bonds_b.get(particles)
        if value_a == value_b:
            common_bonds.addBond(*particles, value_a[0], value_a[1])
        elif (
            value_a is not None and value_b is not None
        ) or particles not in soft_bond_pairs:
            fallback = value_a or value_b
            r_a, k_a = value_a or (fallback[0], 0.0)
            r_b, k_b = value_b or (fallback[0], 0.0)
            changed_bonds.addBond(*particles, [r_a, k_a, r_b, k_b])
        else:
            fallback = value_a or value_b
            r_a, k_a = value_a or (fallback[0], 0.0)
            r_b, k_b = value_b or (fallback[0], 0.0)
            soft_bonds.addBond(*particles, [r_a, k_a, r_b, k_b])
    if common_bonds.getNumBonds():
        if bond_a is not None:
            _copy_force_metadata(bond_a, common_bonds)
        common_bonds.setName("CovalentCommonBonds")
        output.addForce(common_bonds)
    if changed_bonds.getNumBonds():
        changed_bonds.setName("CovalentInterpolatedBonds")
        output.addForce(changed_bonds)
    if soft_bonds.getNumBonds():
        soft_bonds.setName("CovalentSoftBonds")
        output.addForce(soft_bonds)

    angle_a = _force(endpoint_a, mm.HarmonicAngleForce)
    angle_b = _force(endpoint_b, mm.HarmonicAngleForce)
    angles_a = _angle_terms(angle_a)
    angles_b = _angle_terms(angle_b)
    common_angles = mm.HarmonicAngleForce()
    changed_angles = mm.CustomAngleForce(
        "0.5*(wA*kA*(theta-thetaA)^2+wB*kB*(theta-thetaB)^2);"
        "wA=(1-separate)*(1-COVALENT_STERICS)+separate*COVALENT_BONDED_A;"
        "wB=(1-separate)*COVALENT_STERICS+separate*COVALENT_BONDED_B;"
        "separate=COVALENT_SEPARATE_BONDED"
    )
    changed_angles.addGlobalParameter(BONDED_A_PARAMETER, 0.0)
    changed_angles.addGlobalParameter(BONDED_B_PARAMETER, 0.0)
    changed_angles.addGlobalParameter(STERICS_PARAMETER, 0.0)
    changed_angles.addGlobalParameter(SEPARATE_BONDED_PARAMETER, 0.0)
    for name in ("thetaA", "kA", "thetaB", "kB"):
        changed_angles.addPerAngleParameter(name)
    soft_angles = mm.CustomAngleForce(
        "0.5*(COVALENT_SOFT_ANGLE_A*kA*(theta-thetaA)^2"
        "+COVALENT_SOFT_ANGLE_B*kB*(theta-thetaB)^2)"
    )
    soft_angles.addGlobalParameter(SOFT_ANGLE_A_PARAMETER, 0.0)
    soft_angles.addGlobalParameter(SOFT_ANGLE_B_PARAMETER, 0.0)
    for name in ("thetaA", "kA", "thetaB", "kB"):
        soft_angles.addPerAngleParameter(name)
    for particles in sorted(set(angles_a) | set(angles_b)):
        value_a = angles_a.get(particles)
        value_b = angles_b.get(particles)
        if value_a == value_b:
            common_angles.addAngle(*particles, value_a[0], value_a[1])
        else:
            fallback = value_a or value_b
            theta_a, k_a = value_a or (fallback[0], 0.0)
            theta_b, k_b = value_b or (fallback[0], 0.0)
            target = soft_angles if _term_contains_bond(particles, soft_bond_pairs) else changed_angles
            target.addAngle(*particles, [theta_a, k_a, theta_b, k_b])
    if common_angles.getNumAngles():
        if angle_a is not None:
            _copy_force_metadata(angle_a, common_angles)
        common_angles.setName("CovalentCommonAngles")
        output.addForce(common_angles)
    if changed_angles.getNumAngles():
        changed_angles.setName("CovalentInterpolatedAngles")
        output.addForce(changed_angles)
    if soft_angles.getNumAngles():
        soft_angles.setName("CovalentSoftBondAngles")
        output.addForce(soft_angles)

    torsion_a = _force(endpoint_a, mm.PeriodicTorsionForce)
    torsion_b = _force(endpoint_b, mm.PeriodicTorsionForce)
    common, only_a, only_b = _split_identical(
        _torsion_terms(torsion_a), _torsion_terms(torsion_b)
    )
    common_torsions = mm.PeriodicTorsionForce()
    for particles, periodicity, phase, k in common:
        common_torsions.addTorsion(*particles, periodicity, phase, k)
    if common_torsions.getNumTorsions():
        if torsion_a is not None:
            _copy_force_metadata(torsion_a, common_torsions)
        common_torsions.setName("CovalentCommonTorsions")
        output.addForce(common_torsions)
    for label, terms, expression in (
        (
            "A",
            only_a,
            "((1-COVALENT_SEPARATE_BONDED)*COVALENT_STERICS_A"
            "+COVALENT_SEPARATE_BONDED*COVALENT_BONDED_A)"
            "*k*(1+cos(periodicity*theta-phase))",
        ),
        (
            "B",
            only_b,
            "((1-COVALENT_SEPARATE_BONDED)*COVALENT_STERICS_B"
            "+COVALENT_SEPARATE_BONDED*COVALENT_BONDED_B)"
            "*k*(1+cos(periodicity*theta-phase))",
        ),
    ):
        regular_terms = []
        closure_terms = []
        for term in terms:
            target = (
                closure_terms
                if _term_contains_bond(term[0], soft_bond_pairs)
                else regular_terms
            )
            target.append(term)
        force = mm.CustomTorsionForce(expression)
        force.addGlobalParameter(
            BONDED_A_PARAMETER if label == "A" else BONDED_B_PARAMETER,
            0.0,
        )
        force.addGlobalParameter(
            STERICS_A_PARAMETER if label == "A" else STERICS_B_PARAMETER,
            0.0,
        )
        force.addGlobalParameter(SEPARATE_BONDED_PARAMETER, 0.0)
        for name in ("periodicity", "phase", "k"):
            force.addPerTorsionParameter(name)
        for particles, periodicity, phase, k in regular_terms:
            force.addTorsion(*particles, [periodicity, phase, k])
        if force.getNumTorsions():
            force.setName(f"CovalentInterpolatedTorsions{label}")
            output.addForce(force)
        soft_scale = SOFT_TORSION_A_PARAMETER if label == "A" else SOFT_TORSION_B_PARAMETER
        soft_force = mm.CustomTorsionForce(
            f"{soft_scale}*k*(1+cos(periodicity*theta-phase))"
        )
        soft_force.addGlobalParameter(soft_scale, 0.0)
        for name in ("periodicity", "phase", "k"):
            soft_force.addPerTorsionParameter(name)
        for particles, periodicity, phase, k in closure_terms:
            soft_force.addTorsion(*particles, [periodicity, phase, k])
        if soft_force.getNumTorsions():
            soft_force.setName(f"CovalentSoftBondTorsions{label}")
            output.addForce(soft_force)

    parameter_anchor = mm.CustomExternalForce(
        "0*(COVALENT_BONDED_A+COVALENT_BONDED_B+COVALENT_SEPARATE_BONDED"
        "+COVALENT_SOFT_BOND_A+COVALENT_SOFT_BOND_B"
        "+COVALENT_SOFT_ANGLE_A+COVALENT_SOFT_ANGLE_B"
        "+COVALENT_SOFT_TORSION_A+COVALENT_SOFT_TORSION_B"
        "+COVALENT_BOND_NONBONDED_CHARGE_A+COVALENT_BOND_NONBONDED_CHARGE_B"
        "+COVALENT_BOND_NONBONDED_VDW_A+COVALENT_BOND_NONBONDED_VDW_B"
        "+COVALENT_BOND_ONE_FOUR_CHARGE_A+COVALENT_BOND_ONE_FOUR_CHARGE_B"
        "+COVALENT_BOND_ONE_FOUR_VDW_A+COVALENT_BOND_ONE_FOUR_VDW_B)"
    )
    parameter_anchor.addGlobalParameter(BONDED_A_PARAMETER, 0.0)
    parameter_anchor.addGlobalParameter(BONDED_B_PARAMETER, 0.0)
    parameter_anchor.addGlobalParameter(SEPARATE_BONDED_PARAMETER, 0.0)
    for parameter in (
        SOFT_BOND_A_PARAMETER, SOFT_BOND_B_PARAMETER,
        SOFT_ANGLE_A_PARAMETER, SOFT_ANGLE_B_PARAMETER,
        SOFT_TORSION_A_PARAMETER, SOFT_TORSION_B_PARAMETER,
        BOND_NONBONDED_CHARGE_A_PARAMETER, BOND_NONBONDED_CHARGE_B_PARAMETER,
        BOND_NONBONDED_VDW_A_PARAMETER, BOND_NONBONDED_VDW_B_PARAMETER,
        BOND_ONE_FOUR_CHARGE_A_PARAMETER, BOND_ONE_FOUR_CHARGE_B_PARAMETER,
        BOND_ONE_FOUR_VDW_A_PARAMETER, BOND_ONE_FOUR_VDW_B_PARAMETER,
    ):
        parameter_anchor.addGlobalParameter(parameter, 0.0)
    if output.getNumParticles():
        parameter_anchor.addParticle(0, [])
    parameter_anchor.setName("CovalentBondedParameterAnchor")
    output.addForce(parameter_anchor)


def _configure_nonbonded_like(source, target):
    target.setNonbondedMethod(source.getNonbondedMethod())
    target.setCutoffDistance(source.getCutoffDistance())
    target.setReactionFieldDielectric(source.getReactionFieldDielectric())
    target.setEwaldErrorTolerance(source.getEwaldErrorTolerance())
    target.setUseDispersionCorrection(source.getUseDispersionCorrection())
    if hasattr(source, "getUseSwitchingFunction") and source.getUseSwitchingFunction():
        target.setUseSwitchingFunction(True)
        target.setSwitchingDistance(source.getSwitchingDistance())


def _configure_custom_nonbonded_like(
    source, target, *, use_long_range_correction=True
):
    method = source.getNonbondedMethod()
    target.setNonbondedMethod(
        mm.CustomNonbondedForce.NoCutoff
        if method == mm.NonbondedForce.NoCutoff
        else mm.CustomNonbondedForce.CutoffPeriodic
    )
    if method != mm.NonbondedForce.NoCutoff:
        target.setCutoffDistance(source.getCutoffDistance())
    if source.getUseSwitchingFunction():
        target.setUseSwitchingFunction(True)
        target.setSwitchingDistance(source.getSwitchingDistance())
    target.setUseLongRangeCorrection(
        bool(use_long_range_correction and source.getUseDispersionCorrection())
    )


def _ewald_alpha(source):
    alpha, _, _, _ = source.getPMEParameters()
    alpha_nm = _float(alpha, unit.nanometer**-1)
    if alpha_nm == 0.0:
        cutoff_nm = _float(source.getCutoffDistance(), unit.nanometer)
        tolerance = float(source.getEwaldErrorTolerance())
        alpha_nm = math.sqrt(-math.log(2.0 * tolerance)) / cutoff_nm
    return alpha_nm


def _exception_dict(force):
    values = {}
    if force is None:
        return values
    for index in range(force.getNumExceptions()):
        p1, p2, charge, sigma, epsilon = force.getExceptionParameters(index)
        values[tuple(sorted((int(p1), int(p2))))] = (
            _float(charge, unit.elementary_charge**2),
            _float(sigma, unit.nanometer),
            _float(epsilon, unit.kilojoule_per_mole),
        )
    return values


def _add_offset(force, parameter, particle, charge=0.0, sigma=0.0, epsilon=0.0):
    if charge == 0.0 and sigma == 0.0 and epsilon == 0.0:
        return
    force.addParticleParameterOffset(parameter, particle, charge, sigma, epsilon)


def _add_exception_offset(force, parameter, exception, charge=0.0, sigma=0.0, epsilon=0.0):
    if charge == 0.0 and sigma == 0.0 and epsilon == 0.0:
        return
    force.addExceptionParameterOffset(parameter, exception, charge, sigma, epsilon)


def _infer_one_four_scales(force):
    charge_scales = []
    epsilon_scales = []
    for pair, (chargeprod, _sigma, epsilon) in _exception_dict(force).items():
        atom1, atom2 = pair
        q1, sigma1, epsilon1 = force.getParticleParameters(atom1)
        q2, sigma2, epsilon2 = force.getParticleParameters(atom2)
        ordinary_charge = _float(q1 * q2, unit.elementary_charge**2)
        ordinary_epsilon = math.sqrt(
            max(0.0, _float(epsilon1, unit.kilojoule_per_mole))
            * max(0.0, _float(epsilon2, unit.kilojoule_per_mole))
        )
        if abs(ordinary_charge) > 1.0e-10 and abs(chargeprod) > 1.0e-10:
            charge_scales.append(abs(chargeprod / ordinary_charge))
        if ordinary_epsilon > 1.0e-10 and epsilon > 1.0e-10:
            epsilon_scales.append(epsilon / ordinary_epsilon)
    return (
        float(statistics.median(charge_scales)) if charge_scales else 1.0,
        float(statistics.median(epsilon_scales)) if epsilon_scales else 1.0,
    )


def _pair_parameters_for_class(
    force, pair, distance_class, one_four_scales, *, use_existing_exception
):
    atom1, atom2 = pair
    q1, sigma1, epsilon1 = force.getParticleParameters(atom1)
    q2, sigma2, epsilon2 = force.getParticleParameters(atom2)
    sigma = 0.5 * (
        _float(sigma1, unit.nanometer) + _float(sigma2, unit.nanometer)
    )
    if distance_class <= 2:
        return 0.0, sigma, 0.0
    if distance_class == 3:
        exception = (
            _exception_dict(force).get(tuple(sorted(pair)))
            if use_existing_exception
            else None
        )
        if exception is not None:
            return exception
        charge_scale, epsilon_scale = one_four_scales
    else:
        charge_scale = epsilon_scale = 1.0
    return (
        _float(q1 * q2, unit.elementary_charge**2) * charge_scale,
        sigma,
        math.sqrt(
            max(0.0, _float(epsilon1, unit.kilojoule_per_mole))
            * max(0.0, _float(epsilon2, unit.kilojoule_per_mole))
        ) * epsilon_scale,
    )


def _new_soft_bond_topology_pair_force(label):
    lower = label.lower()
    force = mm.CustomBondForce(
        "ONE_4PI_EPS0*((wcq-baseline*qbranch)*qClosed+woq*qOpen)/r"
        "+4*(wcv-baseline*vbranch)*eClosed*((sClosed/r)^12-(sClosed/r)^6)"
        "+4*wov*eOpen*((sOpen/r)^12-(sOpen/r)^6);"
        f"wcq=qbranch*COVALENT_BOND_ONE_FOUR_CHARGE_{label};"
        f"wcv=vbranch*COVALENT_BOND_ONE_FOUR_VDW_{label};"
        f"woq=qbranch*COVALENT_BOND_NONBONDED_CHARGE_{label};"
        f"wov=vbranch*COVALENT_BOND_NONBONDED_VDW_{label};"
        f"qbranch=COVALENT_CHARGE_{label};vbranch=COVALENT_STERICS_{label}"
    )
    force.setName(f"CovalentSoftBondTopologyPairs{label}")
    for parameter in (
        CHARGE_A_PARAMETER if lower == "a" else CHARGE_B_PARAMETER,
        STERICS_A_PARAMETER if lower == "a" else STERICS_B_PARAMETER,
        BOND_NONBONDED_CHARGE_A_PARAMETER if lower == "a" else BOND_NONBONDED_CHARGE_B_PARAMETER,
        BOND_NONBONDED_VDW_A_PARAMETER if lower == "a" else BOND_NONBONDED_VDW_B_PARAMETER,
        BOND_ONE_FOUR_CHARGE_A_PARAMETER if lower == "a" else BOND_ONE_FOUR_CHARGE_B_PARAMETER,
        BOND_ONE_FOUR_VDW_A_PARAMETER if lower == "a" else BOND_ONE_FOUR_VDW_B_PARAMETER,
    ):
        force.addGlobalParameter(parameter, 0.0)
    force.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    for parameter in (
        "qClosed", "sClosed", "eClosed", "qOpen", "sOpen", "eOpen",
        "baseline",
    ):
        force.addPerBondParameter(parameter)
    force.setUsesPeriodicBoundaryConditions(True)
    return force


def _beutler_expression(scale):
    return (
        f"4*({scale})*epsilon*(x*x-x);"
        "x=sigma^6/(r^6+SOFTCORE_ALPHA*(1-(" + scale + "))^SOFTCORE_POWER*SOFTCORE_SIGMA^6);"
        "sigma=0.5*(sigma1+sigma2);epsilon=sqrt(epsilon1*epsilon2)"
    )


def _beutler_bond_expression(scale):
    return (
        f"4*({scale})*epsilon*(x*x-x);"
        "x=sigma^6/(r^6+SOFTCORE_ALPHA*(1-(" + scale + "))^SOFTCORE_POWER*SOFTCORE_SIGMA^6)"
    )


def _gapsys_energy_expression(scale, *, mixing):
    definitions = (
        "C6=4*epsilon*sigma^6;"
        "C12=4*epsilon*sigma^12;"
        "rsc=max(1e-6,GAPSYS_SCALE_LINPOINT_LJ"
        "*((26.0/7.0)*sigmaEff^6*(1-(" + scale + ")))^(1.0/6.0));"
        "sigmaEff=select(delta(epsilon),GAPSYS_SIGMA,sigma)"
    )
    if mixing:
        definitions += ";sigma=0.5*(sigma1+sigma2);epsilon=sqrt(epsilon1*epsilon2)"
    return (
        f"({scale})*select(step(r-rsc),Vhard,Vlinear);"
        "Vhard=C12/max(r,1e-6)^12-C6/max(r,1e-6)^6;"
        "Vlinear=(78*C12/rsc^14-21*C6/rsc^8)*r^2"
        "-(168*C12/rsc^13-48*C6/rsc^7)*r"
        "+91*C12/rsc^12-28*C6/rsc^6;"
        + definitions
    )


def _gapsys_coulomb_correction_expression(scale, *, mixing):
    definitions = (
        "Vquadratic=rSafe^2/rQSafe^3-3*rSafe/rQSafe^2+3/rQSafe;"
        "rSafe=max(r,1e-6);"
        "rQSafe=max(rQ,1e-6);"
        "rQ=min(CUTOFF,GAPSYS_SCALE_LINPOINT_Q"
        f"*(max(0,1-({scale})))^(1.0/6.0)*(1+abs(chargeprod)))"
    )
    if mixing:
        definitions += ";chargeprod=charge1*charge2"
    return (
        f"({scale})*ONE_4PI_EPS0*chargeprod*step(rQ-r)"
        "*(Vquadratic-1/rSafe);" + definitions
    )


def _amber_ssc2_energy_expression(scale, *, mixing, cutoff):
    definitions = (
        "x=(sigma^2/reff2)^3;"
        "reff2=r^2+SSC2_ALPHA_LJ*fsw*(1-w)*sigma^2;"
    )
    if cutoff:
        definitions += (
            "fsw=1-rsw^3*(10+rsw*(-15+6*rsw));"
            "rsw=min(1,max(0,(r-SSC2_SWITCH_START)"
            "/(SSC2_SWITCH_END-SSC2_SWITCH_START)));"
        )
    else:
        definitions += "fsw=1;"
    definitions += (
        "w=s^3*(10+s*(-15+6*s));"
        f"s=min(1,max(0,{scale}))"
    )
    if mixing:
        definitions += ";sigma=0.5*(sigma1+sigma2);epsilon=sqrt(epsilon1*epsilon2)"
    return "4*w*epsilon*(x*x-x);" + definitions


def _effective_distance_ssc2_coulomb_exception_expression(scale):
    return (
        "ONE_4PI_EPS0*chargeprod*w/reff;"
        "reff=sqrt(r^2+SSC2_ALPHA_COUL*(1-w)*sigma^2);"
        "w=s^3*(10+s*(-15+6*s));"
        f"s=min(1,max(0,{scale}))"
    )


def _amber_ssc2_coulomb_exception_expression(scale):
    return (
        "ONE_4PI_EPS0*chargeprod*w*(1/rsc-1/r);"
        "rsc=sqrt(r^2+SSC2_BETA_COUL_14_NM2*(1-w));"
        "w=s^3*(10+s*(-15+6*s));"
        f"s=min(1,max(0,{scale}))"
    )


def _effective_distance_ssc2_combined_direct_expression():
    return (
        "Eenv+Ea+Eb;"
        "Eenv=envpair*(ONE_4PI_EPS0*qenv1*qenv2*erfc(EWALD_ALPHA*renv)/renv"
        "+4*eenv*((senv/renv)^12-(senv/renv)^6));"
        "Ea=apair*(ONE_4PI_EPS0*qprodA*wca*erfc(EWALD_ALPHA*rca)/rca"
        "+4*wsa*epsilonA*(xA*xA-xA));"
        "Eb=bpair*(ONE_4PI_EPS0*qprodB*wcb*erfc(EWALD_ALPHA*rcb)/rcb"
        "+4*wsb*epsilonB*(xB*xB-xB));"
        "xA=(sigmaA/rsa)^6;xB=(sigmaB/rsb)^6;"
        "rca=sqrt(r^2+SSC2_ALPHA_COUL*fsw*(1-wca)*sigmaA^2);"
        "rcb=sqrt(r^2+SSC2_ALPHA_COUL*fsw*(1-wcb)*sigmaB^2);"
        "rsa=sqrt(r^2+SSC2_ALPHA_LJ*fsw*(1-wsa)*sigmaA^2);"
        "rsb=sqrt(r^2+SSC2_ALPHA_LJ*fsw*(1-wsb)*sigmaB^2);"
        "wca=sca^3*(10+sca*(-15+6*sca));"
        "wcb=scb^3*(10+scb*(-15+6*scb));"
        "wsa=ssa^3*(10+ssa*(-15+6*ssa));"
        "wsb=ssb^3*(10+ssb*(-15+6*ssb));"
        "sca=min(1,max(0,COVALENT_CHARGE_A));"
        "scb=min(1,max(0,COVALENT_CHARGE_B));"
        "ssa=min(1,max(0,COVALENT_STERICS_A));"
        "ssb=min(1,max(0,COVALENT_STERICS_B));"
        "fsw=1-rsw^3*(10+rsw*(-15+6*rsw));"
        "rsw=min(1,max(0,(r-SSC2_SWITCH_START)"
        "/(SSC2_SWITCH_END-SSC2_SWITCH_START)));"
        "renv=r+(1-envpair)*SSC2_SWITCH_END;"
        "qprodA=a1*env2*qA1*qenvA2+env1*a2*qenvA1*qA2;"
        "qprodB=b1*env2*qB1*qenvB2+env1*b2*qenvB1*qB2;"
        "qenv1=qA1+COVALENT_MAPPED_CHARGE*(qB1-qA1);"
        "qenv2=qA2+COVALENT_MAPPED_CHARGE*(qB2-qA2);"
        "qenvA1=qA1+COVALENT_MAPPED_CHARGE*(qB1-qA1);"
        "qenvA2=qA2+COVALENT_MAPPED_CHARGE*(qB2-qA2);"
        "qenvB1=qA1+COVALENT_MAPPED_CHARGE*(qB1-qA1);"
        "qenvB2=qA2+COVALENT_MAPPED_CHARGE*(qB2-qA2);"
        "senv=0.5*(senv1+senv2);eenv=sqrt(eenv1*eenv2);"
        "senv1=sA1+COVALENT_STERICS*(sB1-sA1);"
        "senv2=sA2+COVALENT_STERICS*(sB2-sA2);"
        "eenv1=eA1+COVALENT_STERICS*(eB1-eA1);"
        "eenv2=eA2+COVALENT_STERICS*(eB2-eA2);"
        "sigmaA=0.5*(sA1+sA2);epsilonA=sqrt(eA1*eA2);"
        "sigmaB=0.5*(sB1+sB2);epsilonB=sqrt(eB1*eB2);"
        "envpair=env1*env2;apair=a1*env2+env1*a2;"
        "bpair=b1*env2+env1*b2;"
        "env1=delta(role1);env2=delta(role2);"
        "a1=delta(role1-1);a2=delta(role2-1);"
        "b1=delta(role1-2);b2=delta(role2-2)"
    )


def _amber_ssc2_combined_direct_expression():
    return (
        "Eenv+Ea+Eb;"
        "Eenv=envpair*4*(wsa*epsilonEnvA*((sigmaEnvA/renv)^12"
        "-(sigmaEnvA/renv)^6)+wsb*epsilonEnvB*((sigmaEnvB/renv)^12"
        "-(sigmaEnvB/renv)^6));"
        "Ea=apair*(ONE_4PI_EPS0*qprodA*wca*erfc(EWALD_ALPHA*r)*(1/rca-1/r)"
        "+4*wsa*epsilonA*(xA*xA-xA));"
        "Eb=bpair*(ONE_4PI_EPS0*qprodB*wcb*erfc(EWALD_ALPHA*r)*(1/rcb-1/r)"
        "+4*wsb*epsilonB*(xB*xB-xB));"
        "xA=(sigmaA/rsa)^6;xB=(sigmaB/rsb)^6;"
        "rca=sqrt(r^2+SSC2_BETA_COUL*fsw*(1-wca)*betaScaleA);"
        "rcb=sqrt(r^2+SSC2_BETA_COUL*fsw*(1-wcb)*betaScaleB);"
        "betaScaleA=select(delta(epsilonA),SSC2_MIN_COUL_R2,"
        "max(SSC2_MIN_COUL_R2,sigmaA^2));"
        "betaScaleB=select(delta(epsilonB),SSC2_MIN_COUL_R2,"
        "max(SSC2_MIN_COUL_R2,sigmaB^2));"
        "rsa=sqrt(r^2+SSC2_ALPHA_LJ*fsw*(1-wsa)*sigmaA^2);"
        "rsb=sqrt(r^2+SSC2_ALPHA_LJ*fsw*(1-wsb)*sigmaB^2);"
        "wca=sca^3*(10+sca*(-15+6*sca));"
        "wcb=scb^3*(10+scb*(-15+6*scb));"
        "wsa=ssa^3*(10+ssa*(-15+6*ssa));"
        "wsb=ssb^3*(10+ssb*(-15+6*ssb));"
        "sca=min(1,max(0,COVALENT_CHARGE_A));"
        "scb=min(1,max(0,COVALENT_CHARGE_B));"
        "ssa=min(1,max(0,COVALENT_STERICS_A));"
        "ssb=min(1,max(0,COVALENT_STERICS_B));"
        "fsw=1-rsw^3*(10+rsw*(-15+6*rsw));"
        "rsw=min(1,max(0,(r-SSC2_SWITCH_START)"
        "/(SSC2_SWITCH_END-SSC2_SWITCH_START)));"
        "renv=r+(1-envpair)*SSC2_SWITCH_END;"
        "qprodA=a1*env2*qA1*qenvA2+env1*a2*qenvA1*qA2;"
        "qprodB=b1*env2*qB1*qenvB2+env1*b2*qenvB1*qB2;"
        "qenv1=qA1+COVALENT_MAPPED_CHARGE*(qB1-qA1);"
        "qenv2=qA2+COVALENT_MAPPED_CHARGE*(qB2-qA2);"
        "qenvA1=qA1;qenvA2=qA2;qenvB1=qB1;qenvB2=qB2;"
        "sigmaEnvA=0.5*(sA1+sA2);epsilonEnvA=sqrt(eA1*eA2);"
        "sigmaEnvB=0.5*(sB1+sB2);epsilonEnvB=sqrt(eB1*eB2);"
        "sigmaA=0.5*(sA1+sA2);epsilonA=sqrt(eA1*eA2);"
        "sigmaB=0.5*(sB1+sB2);epsilonB=sqrt(eB1*eB2);"
        "envpair=env1*env2;apair=a1*env2+env1*a2;"
        "bpair=b1*env2+env1*b2;"
        "env1=delta(role1);env2=delta(role2);"
        "a1=delta(role1-1);a2=delta(role2-1);"
        "b1=delta(role1-2);b2=delta(role2-2)"
    )


def _amber_ssc2_combined_lrc_expression():
    return (
        "tail*(EenvLJ+EaLJ+EbLJ);"
        "EenvLJ=envpair*4*(wsa*epsilonEnvA*((sigmaEnvA/rtail)^12"
        "-(sigmaEnvA/rtail)^6)+wsb*epsilonEnvB*((sigmaEnvB/rtail)^12"
        "-(sigmaEnvB/rtail)^6));"
        "EaLJ=apair*4*wsa*epsilonA*((sigmaA/rtail)^12-(sigmaA/rtail)^6);"
        "EbLJ=bpair*4*wsb*epsilonB*((sigmaB/rtail)^12-(sigmaB/rtail)^6);"
        "rtail=r+(1-tail)*CUTOFF;tail=step(r-CUTOFF);"
        "wsa=ssa^3*(10+ssa*(-15+6*ssa));"
        "wsb=ssb^3*(10+ssb*(-15+6*ssb));"
        "ssa=min(1,max(0,COVALENT_STERICS_A));"
        "ssb=min(1,max(0,COVALENT_STERICS_B));"
        "sigmaEnvA=0.5*(sA1+sA2);epsilonEnvA=sqrt(eA1*eA2);"
        "sigmaEnvB=0.5*(sB1+sB2);epsilonEnvB=sqrt(eB1*eB2);"
        "sigmaA=0.5*(sA1+sA2);epsilonA=sqrt(eA1*eA2);"
        "sigmaB=0.5*(sB1+sB2);epsilonB=sqrt(eB1*eB2);"
        "envpair=env1*env2;apair=a1*env2+env1*a2;"
        "bpair=b1*env2+env1*b2;"
        "env1=delta(role1);env2=delta(role2);"
        "a1=delta(role1-1);a2=delta(role2-1);"
        "b1=delta(role1-2);b2=delta(role2-2)"
    )


def _exception_reciprocal_correction_expression():
    return (
        "-ONE_4PI_EPS0*q1*q2*erf(EWALD_ALPHA*r)/r;"
        "q1=qbase1+COVALENT_CHARGE_A*qA1+COVALENT_CHARGE_B*qB1"
        "+COVALENT_MAPPED_CHARGE*qmapped1;"
        "q2=qbase2+COVALENT_CHARGE_A*qA2+COVALENT_CHARGE_B*qB2"
        "+COVALENT_MAPPED_CHARGE*qmapped2"
    )


def _environment_exception_expression(*, include_coulomb):
    if not include_coulomb:
        return (
            "4*(wA*eA*((sA/r)^12-(sA/r)^6)"
            "+wB*eB*((sB/r)^12-(sB/r)^6));"
            "wA=xA^3*(10+xA*(-15+6*xA));"
            "wB=xB^3*(10+xB*(-15+6*xB));"
            "xA=min(1,max(0,COVALENT_STERICS_A));"
            "xB=min(1,max(0,COVALENT_STERICS_B))"
        )
    coulomb = "ONE_4PI_EPS0*q/r+" if include_coulomb else ""
    return (
        coulomb + "4*epsilon*((sigma/r)^12-(sigma/r)^6);"
        "q=qA+COVALENT_MAPPED_CHARGE*(qB-qA);"
        "sigma=sA+COVALENT_STERICS*(sB-sA);"
        "epsilon=eA+COVALENT_STERICS*(eB-eA)"
    )


def _softcore_expression(scale, function, *, cutoff=True):
    if function == "gapsys":
        return _gapsys_energy_expression(scale, mixing=True)
    if function in {"amber_ssc2", "effective_distance_ssc2"}:
        return _amber_ssc2_energy_expression(scale, mixing=True, cutoff=cutoff)
    return _beutler_expression(scale)


def _softcore_bond_expression(scale, function):
    if function == "gapsys":
        return _gapsys_energy_expression(scale, mixing=False)
    if function in {"amber_ssc2", "effective_distance_ssc2"}:
        return _amber_ssc2_energy_expression(scale, mixing=False, cutoff=False)
    return _beutler_bond_expression(scale)


def _new_softcore_force(
    source,
    label,
    scale,
    function,
    alpha,
    sigma_nm,
    power,
    gapsys_scale_linpoint_lj,
    gapsys_sigma_nm,
    ssc2_alpha_lj,
    ssc2_switch_width_nm,
    *,
    use_long_range_correction,
):
    has_cutoff = source.getNonbondedMethod() != mm.NonbondedForce.NoCutoff
    force = mm.CustomNonbondedForce(
        _softcore_expression(scale, function, cutoff=has_cutoff)
    )
    force.setName(f"CovalentSoftcoreNonbonded{label}")
    force.addGlobalParameter(scale, 0.0)
    if function == "gapsys":
        force.addGlobalParameter(
            "GAPSYS_SCALE_LINPOINT_LJ", float(gapsys_scale_linpoint_lj)
        )
        force.addGlobalParameter("GAPSYS_SIGMA", float(gapsys_sigma_nm))
    elif function in {"amber_ssc2", "effective_distance_ssc2"}:
        force.addGlobalParameter("SSC2_ALPHA_LJ", float(ssc2_alpha_lj))
        if has_cutoff:
            cutoff_nm = _float(source.getCutoffDistance(), unit.nanometer)
            if float(ssc2_switch_width_nm) >= cutoff_nm:
                raise CovalentAlchemyError(
                    "ssc2_switch_width_nm must be smaller than the nonbonded cutoff"
                )
            force.addGlobalParameter(
                "SSC2_SWITCH_START", cutoff_nm - float(ssc2_switch_width_nm)
            )
            force.addGlobalParameter("SSC2_SWITCH_END", cutoff_nm)
    else:
        force.addGlobalParameter("SOFTCORE_ALPHA", float(alpha))
        force.addGlobalParameter("SOFTCORE_SIGMA", float(sigma_nm))
        force.addGlobalParameter("SOFTCORE_POWER", float(power))
    force.addPerParticleParameter("sigma")
    force.addPerParticleParameter("epsilon")
    force.setForceGroup(SOFTCORE_NONBONDED_FORCE_GROUP)
    _configure_custom_nonbonded_like(
        source,
        force,
        use_long_range_correction=use_long_range_correction,
    )
    return force


def _new_softcore_exception_force(
    label,
    scale,
    function,
    alpha,
    sigma_nm,
    power,
    gapsys_scale_linpoint_lj,
    gapsys_sigma_nm,
    ssc2_alpha_lj,
    ssc2_switch_width_nm,
):
    force = mm.CustomBondForce(_softcore_bond_expression(scale, function))
    force.setName(f"CovalentSoftcoreExceptions{label}")
    force.addGlobalParameter(scale, 0.0)
    if function == "gapsys":
        force.addGlobalParameter(
            "GAPSYS_SCALE_LINPOINT_LJ", float(gapsys_scale_linpoint_lj)
        )
        force.addGlobalParameter("GAPSYS_SIGMA", float(gapsys_sigma_nm))
    elif function in {"amber_ssc2", "effective_distance_ssc2"}:
        force.addGlobalParameter("SSC2_ALPHA_LJ", float(ssc2_alpha_lj))
    else:
        force.addGlobalParameter("SOFTCORE_ALPHA", float(alpha))
        force.addGlobalParameter("SOFTCORE_SIGMA", float(sigma_nm))
        force.addGlobalParameter("SOFTCORE_POWER", float(power))
    force.addPerBondParameter("sigma")
    force.addPerBondParameter("epsilon")
    force.setUsesPeriodicBoundaryConditions(True)
    return force


def _new_ssc2_combined_direct_force(
    source,
    coulomb_function,
    alpha_lj,
    alpha_coul,
    beta_coul,
    switch_width_nm,
):
    if coulomb_function == "amber_ssc2":
        expression = _amber_ssc2_combined_direct_expression()
        name = "CovalentAmberGTISSC2CombinedDirect"
    else:
        expression = _effective_distance_ssc2_combined_direct_expression()
        name = "CovalentEffectiveDistanceSSC2CombinedDirect"
    force = mm.CustomNonbondedForce(expression)
    force.setName(name)
    for parameter in (
        CHARGE_A_PARAMETER,
        CHARGE_B_PARAMETER,
        MAPPED_CHARGE_PARAMETER,
        STERICS_PARAMETER,
        STERICS_A_PARAMETER,
        STERICS_B_PARAMETER,
    ):
        force.addGlobalParameter(parameter, 0.0)
    force.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    force.addGlobalParameter("EWALD_ALPHA", _ewald_alpha(source))
    force.addGlobalParameter("SSC2_ALPHA_LJ", float(alpha_lj))
    if coulomb_function == "amber_ssc2":
        force.addGlobalParameter("SSC2_BETA_COUL", float(beta_coul))
        force.addGlobalParameter("SSC2_MIN_COUL_R2", 0.04)
    else:
        force.addGlobalParameter("SSC2_ALPHA_COUL", float(alpha_coul))
    cutoff_nm = _float(source.getCutoffDistance(), unit.nanometer)
    if float(switch_width_nm) >= cutoff_nm:
        raise CovalentAlchemyError(
            "ssc2_switch_width_nm must be smaller than the nonbonded cutoff"
        )
    force.addGlobalParameter("SSC2_SWITCH_START", cutoff_nm - float(switch_width_nm))
    force.addGlobalParameter("SSC2_SWITCH_END", cutoff_nm)
    for name in ("role", "qA", "qB", "sA", "sB", "eA", "eB"):
        force.addPerParticleParameter(name)
    force.setForceGroup(SOFTCORE_NONBONDED_FORCE_GROUP)
    _configure_custom_nonbonded_like(source, force, use_long_range_correction=False)
    return force


def _new_ssc2_combined_lrc_force(source):
    force = mm.CustomNonbondedForce(_amber_ssc2_combined_lrc_expression())
    force.setName("CovalentSSC2CombinedLRC")
    for parameter in (STERICS_PARAMETER, STERICS_A_PARAMETER, STERICS_B_PARAMETER):
        force.addGlobalParameter(parameter, 0.0)
    force.addGlobalParameter(
        "CUTOFF", _float(source.getCutoffDistance(), unit.nanometer)
    )
    for name in ("role", "sA", "sB", "eA", "eB"):
        force.addPerParticleParameter(name)
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(source.getCutoffDistance())
    force.setUseLongRangeCorrection(True)
    force.setForceGroup(SOFTCORE_NONBONDED_FORCE_GROUP)
    return force


def _new_ssc2_coulomb_exception_force(
    label, scale, coulomb_function, alpha_coul, beta_coul
):
    if coulomb_function == "amber_ssc2":
        expression = _amber_ssc2_coulomb_exception_expression(scale)
    else:
        expression = _effective_distance_ssc2_coulomb_exception_expression(scale)
    force = mm.CustomBondForce(expression)
    force.setName(f"CovalentSoftcoreCoulombExceptions{label}")
    force.addGlobalParameter(scale, 0.0)
    force.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    if coulomb_function == "amber_ssc2":
        # Amber applies unscaled scbeta to 1-4 electrostatics.  Its input is in
        # A^2, while OpenMM CustomBondForce distances are expressed in nm.
        force.addGlobalParameter("SSC2_BETA_COUL_14_NM2", float(beta_coul) * 0.01)
    else:
        force.addGlobalParameter("SSC2_ALPHA_COUL", float(alpha_coul))
    for name in ("chargeprod", "sigma"):
        force.addPerBondParameter(name)
    force.setUsesPeriodicBoundaryConditions(True)
    force.setForceGroup(SOFTCORE_NONBONDED_FORCE_GROUP)
    return force


def _new_exception_reciprocal_correction_force(source):
    force = mm.CustomBondForce(_exception_reciprocal_correction_expression())
    force.setName("CovalentPMEExceptionReciprocalCorrection")
    for name in (
        CHARGE_A_PARAMETER,
        CHARGE_B_PARAMETER,
        MAPPED_CHARGE_PARAMETER,
    ):
        force.addGlobalParameter(name, 0.0)
    force.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    force.addGlobalParameter("EWALD_ALPHA", _ewald_alpha(source))
    for suffix in ("1", "2"):
        for prefix in ("qbase", "qA", "qB", "qmapped"):
            force.addPerBondParameter(prefix + suffix)
    force.setUsesPeriodicBoundaryConditions(True)
    force.setForceGroup(SOFTCORE_NONBONDED_FORCE_GROUP)
    return force


def _new_environment_exception_force(*, include_coulomb=True):
    force = mm.CustomBondForce(
        _environment_exception_expression(include_coulomb=include_coulomb)
    )
    force.setName("CovalentEnvironmentExceptions")
    if include_coulomb:
        force.addGlobalParameter(MAPPED_CHARGE_PARAMETER, 0.0)
        force.addGlobalParameter(STERICS_PARAMETER, 0.0)
    else:
        force.addGlobalParameter(STERICS_A_PARAMETER, 0.0)
        force.addGlobalParameter(STERICS_B_PARAMETER, 0.0)
    force.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    for name in ("qA", "qB", "sA", "sB", "eA", "eB"):
        force.addPerBondParameter(name)
    force.setUsesPeriodicBoundaryConditions(True)
    force.setForceGroup(SOFTCORE_NONBONDED_FORCE_GROUP)
    return force


def _new_endpoint_reciprocal_force(source, label, exception_pairs, *, model="amber"):
    """Build one weighted endpoint electrostatic Hamiltonian.

    PME energies are quadratic in particle charges, so particle charges use
    sqrt(weight), while exception charge products use weight directly. Lennard-
    Jones terms are omitted and evaluated by separate custom forces.
    """
    if model == "gapsys":
        charge_parameter = (
            GAPSYS_RECIPROCAL_A_CHARGE_PARAMETER
            if label == "A"
            else GAPSYS_RECIPROCAL_B_CHARGE_PARAMETER
        )
        exception_parameter = (
            GAPSYS_RECIPROCAL_A_EXCEPTION_PARAMETER
            if label == "A"
            else GAPSYS_RECIPROCAL_B_EXCEPTION_PARAMETER
        )
        force_name = f"CovalentGromacsGapsysEndpointElectrostatics{label}"
    else:
        charge_parameter = (
            RECIPROCAL_A_CHARGE_PARAMETER
            if label == "A"
            else RECIPROCAL_B_CHARGE_PARAMETER
        )
        exception_parameter = (
            RECIPROCAL_A_EXCEPTION_PARAMETER
            if label == "A"
            else RECIPROCAL_B_EXCEPTION_PARAMETER
        )
        force_name = f"CovalentAmberGTIEndpointElectrostatics{label}"
    force = mm.NonbondedForce()
    force.setName(force_name)
    _configure_nonbonded_like(source, force)
    force.addGlobalParameter(charge_parameter, 0.0)
    force.addGlobalParameter(exception_parameter, 0.0)
    for index in range(source.getNumParticles()):
        charge, sigma, _ = source.getParticleParameters(index)
        charge_value = _float(charge, unit.elementary_charge)
        force.addParticle(charge, sigma, 0.0)
        _add_offset(force, charge_parameter, index, charge=charge_value)
    source_exceptions = _exception_dict(source)
    for atom1, atom2 in sorted(exception_pairs):
        chargeprod, sigma, _ = source_exceptions.get(
            (atom1, atom2), (0.0, 1.0, 0.0)
        )
        chargeprod_value = float(chargeprod)
        exception = force.addException(atom1, atom2, chargeprod, sigma, 0.0)
        _add_exception_offset(
            force,
            exception_parameter,
            exception,
            charge=chargeprod_value,
        )
    return force


def _new_gapsys_coulomb_force(source, label, scale, scale_linpoint_q):
    force = mm.CustomNonbondedForce(
        _gapsys_coulomb_correction_expression(scale, mixing=True)
    )
    force.setName(f"CovalentGromacsGapsysCoulombCorrection{label}")
    force.addGlobalParameter(scale, 0.0)
    force.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    force.addGlobalParameter("GAPSYS_SCALE_LINPOINT_Q", float(scale_linpoint_q))
    force.addGlobalParameter(
        "CUTOFF", _float(source.getCutoffDistance(), unit.nanometer)
    )
    force.addPerParticleParameter("charge")
    force.setForceGroup(SOFTCORE_NONBONDED_FORCE_GROUP)
    _configure_custom_nonbonded_like(
        source, force, use_long_range_correction=False
    )
    return force


def _new_gapsys_coulomb_exception_force(label, scale, scale_linpoint_q, cutoff_nm):
    force = mm.CustomBondForce(
        _gapsys_coulomb_correction_expression(scale, mixing=False)
    )
    force.setName(f"CovalentGromacsGapsysCoulombExceptions{label}")
    force.addGlobalParameter(scale, 0.0)
    force.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    force.addGlobalParameter("GAPSYS_SCALE_LINPOINT_Q", float(scale_linpoint_q))
    force.addGlobalParameter("CUTOFF", float(cutoff_nm))
    force.addPerBondParameter("chargeprod")
    force.setUsesPeriodicBoundaryConditions(True)
    force.setForceGroup(SOFTCORE_NONBONDED_FORCE_GROUP)
    return force


def _add_nonbonded_forces(
    output,
    endpoint_a,
    endpoint_b,
    unique_a,
    unique_b,
    *,
    function,
    coulomb_function,
    alpha,
    sigma_nm,
    power,
    gapsys_scale_linpoint_lj,
    gapsys_scale_linpoint_q,
    gapsys_sigma_nm,
    ssc2_alpha_lj,
    ssc2_alpha_coul,
    ssc2_beta_coul,
    ssc2_switch_width_nm,
    use_long_range_correction,
    soft_bond_pair_changes=(),
    enable_soft_bond_topology=False,
):
    source_a = _force(endpoint_a, mm.NonbondedForce)
    source_b = _force(endpoint_b, mm.NonbondedForce)
    if source_a is None or source_b is None:
        raise CovalentAlchemyError("softcore endpoints require one NonbondedForce")
    if source_a.getNonbondedMethod() != source_b.getNonbondedMethod():
        raise CovalentAlchemyError("softcore endpoint nonbonded methods differ")
    use_ssc2_coulomb = coulomb_function in {
        "amber_ssc2",
        "effective_distance_ssc2",
    }
    use_gapsys_coulomb = coulomb_function == "gapsys"
    use_amber_reciprocal = coulomb_function == "amber_ssc2"
    use_endpoint_electrostatics = use_amber_reciprocal or use_gapsys_coulomb
    if (use_ssc2_coulomb or use_gapsys_coulomb) and source_a.getNonbondedMethod() not in {
        mm.NonbondedForce.PME,
        mm.NonbondedForce.Ewald,
    }:
        raise CovalentAlchemyError("softcore Coulomb requires PME or Ewald")
    unique_a = set(int(index) for index in unique_a)
    unique_b = set(int(index) for index in unique_b)
    if unique_a & unique_b:
        raise CovalentAlchemyError("softcore unique branches overlap")
    all_particles = set(range(endpoint_a.getNumParticles()))
    environment = all_particles - unique_a - unique_b

    topology_changes = {}
    for raw in (soft_bond_pair_changes or ()) if enable_soft_bond_topology else ():
        endpoint = str(raw["endpoint"]).lower()
        pair = tuple(sorted(int(value) for value in raw["system_atoms_0based"]))
        if endpoint not in {"a", "b"} or len(pair) != 2:
            raise CovalentAlchemyError("invalid soft-bond topology-pair metadata")
        endpoint_unique = unique_a if endpoint == "a" else unique_b
        if not endpoint_unique.intersection(pair):
            raise CovalentAlchemyError(
                "soft-bond topology changes between mapped common atoms are not supported"
            )
        topology_changes[pair] = {
            "endpoint": endpoint,
            "closed_class": int(raw["closed_class"]),
            "open_class": int(raw["open_class"]),
        }

    exceptions_a = _exception_dict(source_a)
    exceptions_b = _exception_dict(source_b)
    exception_pairs = set(exceptions_a) | set(exceptions_b)
    exception_pairs.update(topology_changes)
    exception_pairs.update(
        tuple(sorted((particle_a, particle_b)))
        for particle_a in unique_a for particle_b in unique_b
    )
    if use_ssc2_coulomb or use_gapsys_coulomb:
        exception_pairs.update(combinations(sorted(unique_a), 2))
        exception_pairs.update(combinations(sorted(unique_b), 2))

    if use_endpoint_electrostatics:
        model = "gapsys" if use_gapsys_coulomb else "amber"
        reciprocal_a = _new_endpoint_reciprocal_force(
            source_a, "A", exception_pairs, model=model
        )
        reciprocal_b = _new_endpoint_reciprocal_force(
            source_b, "B", exception_pairs, model=model
        )
    else:
        reciprocal_a = reciprocal_b = None

    if use_amber_reciprocal:
        force = None
    else:
        force = mm.NonbondedForce()
        force.setName("CovalentInterpolatedPMENonbonded")
        _configure_nonbonded_like(source_a, force)
        if use_ssc2_coulomb:
            force.setIncludeDirectSpace(False)
        for parameter in (
            CHARGE_A_PARAMETER,
            CHARGE_B_PARAMETER,
            MAPPED_CHARGE_PARAMETER,
            STERICS_PARAMETER,
            STERICS_A_PARAMETER,
            STERICS_B_PARAMETER,
        ):
            force.addGlobalParameter(parameter, 0.0)
    charge_coefficients = {}
    endpoint_charges = {}
    for particle in range(endpoint_a.getNumParticles()):
        q_a, sigma_a, epsilon_a = source_a.getParticleParameters(particle)
        q_b, sigma_b, epsilon_b = source_b.getParticleParameters(particle)
        qa = _float(q_a, unit.elementary_charge)
        qb = _float(q_b, unit.elementary_charge)
        sa = _float(sigma_a, unit.nanometer)
        sb = _float(sigma_b, unit.nanometer)
        ea = _float(epsilon_a, unit.kilojoule_per_mole)
        eb = _float(epsilon_b, unit.kilojoule_per_mole)
        endpoint_charges[particle] = (qa, qb)
        if force is not None:
            if particle in unique_a:
                force.addParticle(0.0, sa, 0.0)
                if not use_gapsys_coulomb:
                    _add_offset(force, CHARGE_A_PARAMETER, particle, charge=qa)
            elif particle in unique_b:
                force.addParticle(0.0, sb, 0.0)
                if not use_gapsys_coulomb:
                    _add_offset(force, CHARGE_B_PARAMETER, particle, charge=qb)
            else:
                force.addParticle(0.0 if use_gapsys_coulomb else qa, sa, ea)
                if not use_gapsys_coulomb:
                    _add_offset(force, MAPPED_CHARGE_PARAMETER, particle, charge=qb - qa)
                _add_offset(
                    force,
                    STERICS_PARAMETER,
                    particle,
                    sigma=sb - sa,
                    epsilon=eb - ea,
                )

    combined_direct = combined_lrc = None
    gapsys_coulomb_a = gapsys_coulomb_b = None
    exception_coulomb_a = exception_coulomb_b = None
    reciprocal_exception_correction = environment_exceptions = None
    if use_ssc2_coulomb:
        combined_direct = _new_ssc2_combined_direct_force(
            source_a,
            coulomb_function,
            ssc2_alpha_lj,
            ssc2_alpha_coul,
            ssc2_beta_coul,
            ssc2_switch_width_nm,
        )
        if use_long_range_correction:
            combined_lrc = _new_ssc2_combined_lrc_force(source_a)
        exception_coulomb_a = _new_ssc2_coulomb_exception_force(
            "A",
            CHARGE_A_PARAMETER,
            coulomb_function,
            ssc2_alpha_coul,
            ssc2_beta_coul,
        )
        exception_coulomb_b = _new_ssc2_coulomb_exception_force(
            "B",
            CHARGE_B_PARAMETER,
            coulomb_function,
            ssc2_alpha_coul,
            ssc2_beta_coul,
        )
        reciprocal_exception_correction = (
            None
            if use_amber_reciprocal
            else _new_exception_reciprocal_correction_force(source_a)
        )
        environment_exceptions = _new_environment_exception_force(
            include_coulomb=not use_amber_reciprocal
        )
        softcore_a = softcore_b = None
    else:
        softcore_a = _new_softcore_force(
            source_a,
            "A",
            STERICS_A_PARAMETER,
            function,
            alpha,
            sigma_nm,
            power,
            gapsys_scale_linpoint_lj,
            gapsys_sigma_nm,
            ssc2_alpha_lj,
            ssc2_switch_width_nm,
            use_long_range_correction=use_long_range_correction,
        )
        if use_gapsys_coulomb:
            cutoff_nm = _float(source_a.getCutoffDistance(), unit.nanometer)
            gapsys_coulomb_a = _new_gapsys_coulomb_force(
                source_a, "A", CHARGE_A_PARAMETER, gapsys_scale_linpoint_q
            )
            gapsys_coulomb_b = _new_gapsys_coulomb_force(
                source_b, "B", CHARGE_B_PARAMETER, gapsys_scale_linpoint_q
            )
            exception_coulomb_a = _new_gapsys_coulomb_exception_force(
                "A", CHARGE_A_PARAMETER, gapsys_scale_linpoint_q, cutoff_nm
            )
            exception_coulomb_b = _new_gapsys_coulomb_exception_force(
                "B", CHARGE_B_PARAMETER, gapsys_scale_linpoint_q, cutoff_nm
            )
        softcore_b = _new_softcore_force(
            source_b,
            "B",
            STERICS_B_PARAMETER,
            function,
            alpha,
            sigma_nm,
            power,
            gapsys_scale_linpoint_lj,
            gapsys_sigma_nm,
            ssc2_alpha_lj,
            ssc2_switch_width_nm,
            use_long_range_correction=use_long_range_correction,
        )
    for particle in range(endpoint_a.getNumParticles()):
        charge_a, sigma_a, epsilon_a = source_a.getParticleParameters(particle)
        charge_b, sigma_b, epsilon_b = source_b.getParticleParameters(particle)
        if use_ssc2_coulomb:
            qa_value = _float(charge_a, unit.elementary_charge)
            qb_value = _float(charge_b, unit.elementary_charge)
            role = 1.0 if particle in unique_a else 2.0 if particle in unique_b else 0.0
            combined_direct.addParticle(
                [role, charge_a, charge_b, sigma_a, sigma_b, epsilon_a, epsilon_b]
            )
            if combined_lrc is not None:
                combined_lrc.addParticle(
                    [role, sigma_a, sigma_b, epsilon_a, epsilon_b]
                )
            if particle in unique_a:
                charge_coefficients[particle] = [0.0, qa_value, 0.0, 0.0]
            elif particle in unique_b:
                charge_coefficients[particle] = [0.0, 0.0, qb_value, 0.0]
            else:
                charge_coefficients[particle] = [
                    qa_value,
                    0.0,
                    0.0,
                    qb_value - qa_value,
                ]
        else:
            softcore_a.addParticle([sigma_a, epsilon_a])
            softcore_b.addParticle([sigma_b, epsilon_b])
            if use_gapsys_coulomb:
                gapsys_coulomb_a.addParticle([charge_a])
                gapsys_coulomb_b.addParticle([charge_b])
    if not use_ssc2_coulomb:
        if unique_a and environment:
            softcore_a.addInteractionGroup(unique_a, environment)
            if use_gapsys_coulomb:
                gapsys_coulomb_a.addInteractionGroup(unique_a, environment)
        if unique_b and environment:
            softcore_b.addInteractionGroup(unique_b, environment)
            if use_gapsys_coulomb:
                gapsys_coulomb_b.addInteractionGroup(unique_b, environment)
    exception_lj_a = _new_softcore_exception_force(
        "A",
        STERICS_A_PARAMETER,
        function,
        alpha,
        sigma_nm,
        power,
        gapsys_scale_linpoint_lj,
        gapsys_sigma_nm,
        ssc2_alpha_lj,
        ssc2_switch_width_nm,
    )
    exception_lj_b = _new_softcore_exception_force(
        "B",
        STERICS_B_PARAMETER,
        function,
        alpha,
        sigma_nm,
        power,
        gapsys_scale_linpoint_lj,
        gapsys_sigma_nm,
        ssc2_alpha_lj,
        ssc2_switch_width_nm,
    )
    topology_pair_a = _new_soft_bond_topology_pair_force("A")
    topology_pair_b = _new_soft_bond_topology_pair_force("B")
    one_four_a = _infer_one_four_scales(source_a)
    one_four_b = _infer_one_four_scales(source_b)

    for pair in sorted(exception_pairs):
        value_a = exceptions_a.get(pair, (0.0, 1.0, 0.0))
        value_b = exceptions_b.get(pair, (0.0, 1.0, 0.0))
        q_a, sigma_a, epsilon_a = value_a
        q_b, sigma_b, epsilon_b = value_b
        p1, p2 = pair
        in_a = int(p1 in unique_a) + int(p2 in unique_a)
        in_b = int(p1 in unique_b) + int(p2 in unique_b)
        topology_change = topology_changes.get(pair)
        if use_ssc2_coulomb and not use_amber_reciprocal:
            index = force.addException(p1, p2, 0.0, 1.0, 0.0)
        elif use_amber_reciprocal:
            index = None
        elif in_a == 2 or in_b == 2 or (in_a and in_b):
            index = force.addException(p1, p2, 0.0, 1.0, 0.0)
        elif in_a:
            index = force.addException(p1, p2, 0.0, sigma_a, 0.0)
            if use_ssc2_coulomb:
                if q_a != 0.0:
                    exception_coulomb_a.addBond(p1, p2, [q_a, sigma_a])
            elif use_gapsys_coulomb:
                if q_a != 0.0:
                    exception_coulomb_a.addBond(p1, p2, [q_a])
            else:
                _add_exception_offset(force, CHARGE_A_PARAMETER, index, charge=q_a)
            if epsilon_a != 0.0:
                exception_lj_a.addBond(p1, p2, [sigma_a, epsilon_a])
        elif in_b:
            index = force.addException(p1, p2, 0.0, sigma_b, 0.0)
            if use_ssc2_coulomb:
                if q_b != 0.0:
                    exception_coulomb_b.addBond(p1, p2, [q_b, sigma_b])
            elif use_gapsys_coulomb:
                if q_b != 0.0:
                    exception_coulomb_b.addBond(p1, p2, [q_b])
            else:
                _add_exception_offset(force, CHARGE_B_PARAMETER, index, charge=q_b)
            if epsilon_b != 0.0:
                exception_lj_b.addBond(p1, p2, [sigma_b, epsilon_b])
        else:
            index = force.addException(
                p1, p2, 0.0 if use_gapsys_coulomb else q_a, sigma_a, epsilon_a
            )
            if not use_gapsys_coulomb:
                _add_exception_offset(
                    force, MAPPED_CHARGE_PARAMETER, index, charge=q_b - q_a
                )
            _add_exception_offset(
                force,
                STERICS_PARAMETER,
                index,
                sigma=sigma_b - sigma_a,
                epsilon=epsilon_b - epsilon_a,
            )
        if use_ssc2_coulomb:
            if reciprocal_exception_correction is not None:
                if use_amber_reciprocal:
                    qa1, qb1 = endpoint_charges[p1]
                    qa2, qb2 = endpoint_charges[p2]
                    reciprocal_exception_correction.addBond(
                        p1, p2, [qa1, qa2, qb1, qb2]
                    )
                else:
                    reciprocal_exception_correction.addBond(
                        p1,
                        p2,
                        [*charge_coefficients[p1], *charge_coefficients[p2]],
                    )
            if not in_a and not in_b:
                environment_exceptions.addBond(
                    p1,
                    p2,
                    [q_a, q_b, sigma_a, sigma_b, epsilon_a, epsilon_b],
                )
            elif in_a == 1:
                if q_a != 0.0:
                    exception_coulomb_a.addBond(p1, p2, [q_a, sigma_a])
                if epsilon_a != 0.0:
                    exception_lj_a.addBond(p1, p2, [sigma_a, epsilon_a])
            elif in_b == 1:
                if q_b != 0.0:
                    exception_coulomb_b.addBond(p1, p2, [q_b, sigma_b])
                if epsilon_b != 0.0:
                    exception_lj_b.addBond(p1, p2, [sigma_b, epsilon_b])
        if topology_change is not None:
            endpoint = topology_change["endpoint"]
            source = source_a if endpoint == "a" else source_b
            scales = one_four_a if endpoint == "a" else one_four_b
            closed = _pair_parameters_for_class(
                source,
                pair,
                topology_change["closed_class"],
                scales,
                use_existing_exception=True,
            )
            opened = _pair_parameters_for_class(
                source,
                pair,
                topology_change["open_class"],
                scales,
                use_existing_exception=False,
            )
            target = topology_pair_a if endpoint == "a" else topology_pair_b
            target.addBond(p1, p2, [*closed, *opened, 1.0])
        if use_ssc2_coulomb:
            combined_direct.addExclusion(p1, p2)
            if combined_lrc is not None:
                combined_lrc.addExclusion(p1, p2)
        else:
            softcore_a.addExclusion(p1, p2)
            softcore_b.addExclusion(p1, p2)
            if use_gapsys_coulomb:
                gapsys_coulomb_a.addExclusion(p1, p2)
                gapsys_coulomb_b.addExclusion(p1, p2)

    if force is not None:
        output.addForce(force)
    if use_endpoint_electrostatics:
        output.addForce(reciprocal_a)
        output.addForce(reciprocal_b)
    if use_ssc2_coulomb:
        output.addForce(combined_direct)
        if combined_lrc is not None:
            output.addForce(combined_lrc)
        if environment_exceptions.getNumBonds():
            output.addForce(environment_exceptions)
        if (
            reciprocal_exception_correction is not None
            and reciprocal_exception_correction.getNumBonds()
        ):
            output.addForce(reciprocal_exception_correction)
    if unique_a:
        if not use_ssc2_coulomb:
            output.addForce(softcore_a)
            if use_gapsys_coulomb:
                output.addForce(gapsys_coulomb_a)
    if unique_b:
        if not use_ssc2_coulomb:
            output.addForce(softcore_b)
            if use_gapsys_coulomb:
                output.addForce(gapsys_coulomb_b)
    if exception_lj_a.getNumBonds():
        output.addForce(exception_lj_a)
    if exception_lj_b.getNumBonds():
        output.addForce(exception_lj_b)
    if (use_ssc2_coulomb or use_gapsys_coulomb) and exception_coulomb_a.getNumBonds():
        output.addForce(exception_coulomb_a)
    if (use_ssc2_coulomb or use_gapsys_coulomb) and exception_coulomb_b.getNumBonds():
        output.addForce(exception_coulomb_b)
    if topology_pair_a.getNumBonds():
        output.addForce(topology_pair_a)
    if topology_pair_b.getNumBonds():
        output.addForce(topology_pair_b)


def _vacuum_bond_terms(force):
    if force is None:
        return []
    terms = []
    for index in range(force.getNumBonds()):
        atom1, atom2, parameters = force.getBondParameters(index)
        terms.append(
            (
                tuple(sorted((int(atom1), int(atom2)))),
                tuple(float(value) for value in parameters),
            )
        )
    return terms


def _vacuum_subset(source, terms, *, scale=None, name):
    expression = source.getEnergyFunction()
    if scale is not None:
        expression = f"({scale})*({expression})"
    force = mm.CustomBondForce(expression)
    force.setName(name)
    existing_globals = set()
    for index in range(source.getNumGlobalParameters()):
        parameter = source.getGlobalParameterName(index)
        existing_globals.add(parameter)
        force.addGlobalParameter(
            parameter, source.getGlobalParameterDefaultValue(index)
        )
    if scale is not None:
        parameter = (
            STERICS_B_PARAMETER if "STERICS_B" in scale else STERICS_A_PARAMETER
        )
        if parameter not in existing_globals:
            force.addGlobalParameter(parameter, 0.0)
    for index in range(source.getNumPerBondParameters()):
        force.addPerBondParameter(source.getPerBondParameterName(index))
    for particles, parameters in terms:
        force.addBond(*particles, parameters)
    _copy_force_metadata(source, force)
    force.setName(name)
    return force


def _merge_unique_vacuum_force(output, endpoint_a, endpoint_b):
    name = "CovalentUniqueVacuumNonbondedForce"
    force_a = next(
        (force for force in endpoint_a.getForces() if force.getName() == name),
        None,
    )
    force_b = next(
        (force for force in endpoint_b.getForces() if force.getName() == name),
        None,
    )
    if force_a is None and force_b is None:
        return
    if force_a is None or force_b is None:
        raise CovalentAlchemyError(
            "hybrid endpoints must both contain the unique vacuum force"
        )
    if (
        force_a.getEnergyFunction() != force_b.getEnergyFunction()
        or force_a.getNumGlobalParameters() != force_b.getNumGlobalParameters()
        or force_a.getNumPerBondParameters() != force_b.getNumPerBondParameters()
    ):
        raise CovalentAlchemyError("hybrid endpoint vacuum force definitions differ")
    common, only_a, only_b = _split_identical(
        _vacuum_bond_terms(force_a), _vacuum_bond_terms(force_b)
    )
    if common:
        output.addForce(
            _vacuum_subset(force_a, common, name=name)
        )
    if only_a:
        output.addForce(
            _vacuum_subset(
                force_a,
                only_a,
                scale=f"1-{STERICS_B_PARAMETER}",
                name="CovalentInactiveBToCoreVacuumNonbondedForce",
            )
        )
    if only_b:
        output.addForce(
            _vacuum_subset(
                force_b,
                only_b,
                scale=f"1-{STERICS_A_PARAMETER}",
                name="CovalentInactiveAToCoreVacuumNonbondedForce",
            )
        )


def _copy_other_forces(output, endpoint_a, endpoint_b, unique_a, unique_b):
    from atom_openmm.metal_ions import (
        PANTEVA_FORCE_NAME,
        merge_panteva_endpoint_forces,
    )

    supported = (
        mm.HarmonicBondForce,
        mm.HarmonicAngleForce,
        mm.PeriodicTorsionForce,
        mm.NonbondedForce,
    )
    forces_b_by_name = defaultdict(list)
    for force in endpoint_b.getForces():
        if (
            not isinstance(force, supported)
            and force.getName() not in {
                "CovalentUniqueVacuumNonbondedForce",
                PANTEVA_FORCE_NAME,
            }
        ):
            forces_b_by_name[(type(force), force.getName())].append(force)
    for force in endpoint_a.getForces():
        if (
            isinstance(force, supported)
            or force.getName() in {
                "CovalentUniqueVacuumNonbondedForce",
                PANTEVA_FORCE_NAME,
            }
        ):
            continue
        matches = forces_b_by_name[(type(force), force.getName())]
        if not matches:
            raise CovalentAlchemyError(
                f"endpoint B is missing force {type(force).__name__} {force.getName()!r}"
            )
        other = matches.pop(0)
        if isinstance(force, mm.MonteCarloBarostat):
            output.addForce(_clone_force(force))
        elif isinstance(force, mm.CMMotionRemover):
            output.addForce(_clone_force(force))
        elif mm.XmlSerializer.serialize(force) == mm.XmlSerializer.serialize(other):
            output.addForce(_clone_force(force))
        else:
            raise CovalentAlchemyError(
                f"unsupported changing force {type(force).__name__} {force.getName()!r}"
            )
    leftovers = [key for key, values in forces_b_by_name.items() if values]
    if leftovers:
        raise CovalentAlchemyError(f"endpoint A is missing forces present in endpoint B: {leftovers}")
    _merge_unique_vacuum_force(output, endpoint_a, endpoint_b)
    merge_panteva_endpoint_forces(output, endpoint_a, endpoint_b, unique_a, unique_b)


def _allocate_interval_steps(total_steps, nodes, segment_counts):
    boundaries = [0.0, *nodes, 1.0]
    widths = [
        boundaries[index + 1] - boundaries[index]
        for index in range(len(boundaries) - 1)
    ]
    raw = [float(total_steps) * width for width in widths]
    steps = [int(value) for value in raw]
    remainder = int(total_steps) - sum(steps)
    order = sorted(
        range(len(raw)),
        key=lambda index: (raw[index] - steps[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        steps[index] += 1
    if any(total < count for total, count in zip(steps, segment_counts)):
        raise CovalentAlchemyError(
            "softcore total_steps must provide at least one step per segment"
        )
    return steps


def resolve_softcore_path(
    *,
    charge_steps_per_stage=10000,
    sterics_steps=30000,
    subdivisions_per_stage=1,
    total_steps=None,
    path_nodes=None,
    vdw_a=None,
    charge_a=None,
    segments_per_interval=None,
    path_mode=None,
    control_nodes=None,
):
    if control_nodes is not None:
        if path_mode is not None or any(
            value is not None for value in (path_nodes, vdw_a, charge_a)
        ):
            raise CovalentAlchemyError(
                "explicit softcore control nodes cannot be combined with a named "
                "path mode or legacy path arrays"
            )
        if total_steps is None or segments_per_interval is None:
            raise CovalentAlchemyError(
                "explicit softcore control nodes require total_steps and "
                "segments_per_interval"
            )
        if not isinstance(control_nodes, (list, tuple)) or len(control_nodes) < 2:
            raise CovalentAlchemyError(
                "softcore control nodes must contain at least two mappings"
            )
        normalized = []
        previous = None
        for index, raw in enumerate(control_nodes):
            if not isinstance(raw, dict) or set(raw) - {"at", "label", "controls"}:
                raise CovalentAlchemyError(
                    f"softcore control node {index + 1} must contain at, optional "
                    "label, and controls"
                )
            controls = raw.get("controls")
            if not isinstance(controls, dict) or set(controls) - set(EXPLICIT_PATH_CONTROLS):
                raise CovalentAlchemyError(
                    f"softcore control node {index + 1} contains invalid controls"
                )
            if previous is None:
                missing = set(EXPLICIT_PATH_CONTROLS) - set(controls)
                if missing:
                    raise CovalentAlchemyError(
                        "the first softcore control node must define every control; "
                        f"missing {', '.join(sorted(missing))}"
                    )
                resolved_controls = {}
            else:
                resolved_controls = dict(previous["controls"])
            for name, value in controls.items():
                value = float(value)
                if not 0.0 <= value <= 1.0:
                    raise CovalentAlchemyError(
                        f"softcore control {name} at node {index + 1} must be in [0, 1]"
                    )
                resolved_controls[name] = value
            node = {
                "at": float(raw.get("at", -1.0)),
                "label": str(raw.get("label", f"node_{index}")),
                "controls": resolved_controls,
            }
            normalized.append(node)
            previous = node
        coordinates = [node["at"] for node in normalized]
        if coordinates[0] != 0.0 or coordinates[-1] != 1.0 or any(
            right <= left for left, right in zip(coordinates, coordinates[1:])
        ):
            raise CovalentAlchemyError(
                "softcore control-node coordinates must increase strictly from 0 to 1"
            )
        for label, controls, expected in (
            ("first", normalized[0]["controls"], EXPLICIT_ENDPOINT_A_CONTROLS),
            ("last", normalized[-1]["controls"], EXPLICIT_ENDPOINT_B_CONTROLS),
        ):
            mismatched = [
                name for name, value in expected.items()
                if not math.isclose(controls[name], value, abs_tol=1.0e-12)
            ]
            if mismatched:
                raise CovalentAlchemyError(
                    f"the {label} softcore control node is not a physical endpoint; "
                    f"invalid controls: {', '.join(mismatched)}"
                )
        segments = [int(value) for value in segments_per_interval]
        if len(segments) != len(normalized) - 1 or any(value < 1 for value in segments):
            raise CovalentAlchemyError(
                "segments_per_interval must contain one positive value per control-node interval"
            )
        total = int(total_steps)
        if total < 1:
            raise CovalentAlchemyError("softcore total_steps must be positive")
        internal_nodes = coordinates[1:-1]
        interval_steps = _allocate_interval_steps(total, internal_nodes, segments)
        values = {
            name: [node["controls"][name] for node in normalized]
            for name in EXPLICIT_PATH_CONTROLS
        }
        return {
            "source": "explicit_nodes",
            "nodes": internal_nodes,
            "control_nodes": normalized,
            "vdw_a": values["sterics_a"],
            "vdw_b": values["sterics_b"],
            "charge_a": values["charge_a"],
            "charge_b": values["charge_b"],
            "bonded_a": values["bonded_a"],
            "bonded_b": values["bonded_b"],
            "separate_bonded": [1.0] * len(normalized),
            "mapped_vdw": values["mapped_vdw"],
            "mapped_charge": values["mapped_charge"],
            "soft_bond_a": values["soft_bond_a"],
            "soft_bond_b": values["soft_bond_b"],
            "soft_angles_a": values["soft_angles_a"],
            "soft_angles_b": values["soft_angles_b"],
            "soft_torsions_a": values["soft_torsions_a"],
            "soft_torsions_b": values["soft_torsions_b"],
            "bond_nonbonded_charge_a": values["bond_nonbonded_charge_a"],
            "bond_nonbonded_charge_b": values["bond_nonbonded_charge_b"],
            "bond_nonbonded_vdw_a": values["bond_nonbonded_vdw_a"],
            "bond_nonbonded_vdw_b": values["bond_nonbonded_vdw_b"],
            "bond_one_four_charge_a": values["bond_one_four_charge_a"],
            "bond_one_four_charge_b": values["bond_one_four_charge_b"],
            "bond_one_four_vdw_a": values["bond_one_four_vdw_a"],
            "bond_one_four_vdw_b": values["bond_one_four_vdw_b"],
            "stage_labels": [node["label"] for node in normalized[1:]],
            "reverse_stage_labels": [node["label"] for node in reversed(normalized[:-1])],
            "segments_per_interval": segments,
            "interval_steps": interval_steps,
            "total_steps": total,
        }
    if path_mode is not None:
        path_mode = str(path_mode).lower()
        if path_mode not in {"concerted", "staged_bonded", "scheme1_soft_bond"}:
            raise CovalentAlchemyError(
                "softcore path mode must be 'concerted', 'staged_bonded', or "
                "'scheme1_soft_bond'"
            )
        if any(value is not None for value in (path_nodes, vdw_a, charge_a)):
            raise CovalentAlchemyError(
                "softcore named path modes cannot be combined with explicit path arrays"
            )
        if path_mode in {"concerted", "scheme1_soft_bond"}:
            path_nodes = []
            if path_mode == "scheme1_soft_bond":
                path_nodes = [0.5]
                vdw_a = [1.0, 0.5, 0.0]
                charge_a = [1.0, 0.5, 0.0]
            else:
                vdw_a = [1.0, 0.0]
                charge_a = [1.0, 0.0]
        else:
            path_nodes = [0.1, 0.3, 0.7, 0.9]
            vdw_a = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
            charge_a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if segments_per_interval is None:
            segments_per_interval = [1] * (len(path_nodes) + 1)
    general_values = (total_steps, path_nodes, vdw_a, charge_a, segments_per_interval)
    general = any(value is not None for value in general_values)
    if general:
        if any(value is None for value in general_values):
            raise CovalentAlchemyError(
                "general softcore paths require total_steps, path_nodes, vdw_a, "
                "charge_a, and segments_per_interval"
            )
        nodes = [float(value) for value in path_nodes]
        vdw = [float(value) for value in vdw_a]
        charge = [float(value) for value in charge_a]
        segments = [int(value) for value in segments_per_interval]
        total = int(total_steps)
        if charge_steps_per_stage != 10000 or sterics_steps != 30000:
            raise CovalentAlchemyError(
                "general softcore paths cannot be combined with legacy stage steps"
            )
        if subdivisions_per_stage != 1:
            raise CovalentAlchemyError(
                "segments_per_interval replaces subdivisions_per_stage for "
                "general softcore paths"
            )
        if any(
            not 0.0 < value < 1.0 for value in nodes
        ) or any(right <= left for left, right in zip(nodes, nodes[1:])):
            raise CovalentAlchemyError(
                "softcore path nodes must be strictly increasing values in (0, 1)"
            )
        expected_values = len(nodes) + 2
        if len(vdw) != expected_values or len(charge) != expected_values:
            raise CovalentAlchemyError(
                "softcore vdw_a and charge_a must contain len(nodes)+2 values"
            )
        if len(segments) != len(nodes) + 1 or any(value < 1 for value in segments):
            raise CovalentAlchemyError(
                "segments_per_interval must contain one positive value per path interval"
            )
        if any(value < 0.0 or value > 1.0 for value in (*vdw, *charge)):
            raise CovalentAlchemyError("softcore path scales must be in [0, 1]")
        if vdw[0] != 1.0 or vdw[-1] != 0.0:
            raise CovalentAlchemyError("softcore vdw_a must begin at 1 and end at 0")
        if charge[0] != 1.0 or charge[-1] != 0.0:
            raise CovalentAlchemyError(
                "softcore charge_a must begin at 1 and end at 0"
            )
        if total < 1:
            raise CovalentAlchemyError("softcore total_steps must be positive")
        interval_steps = _allocate_interval_steps(total, nodes, segments)
        source = path_mode if path_mode is not None else "general"
    else:
        if charge_steps_per_stage < 1 or sterics_steps < 1:
            raise CovalentAlchemyError("softcore stage steps must be positive")
        if subdivisions_per_stage < 1:
            raise CovalentAlchemyError(
                "softcore subdivisions_per_stage must be positive"
            )
        if min(charge_steps_per_stage, sterics_steps) < subdivisions_per_stage:
            raise CovalentAlchemyError(
                "softcore stage steps must be at least subdivisions_per_stage"
            )
        total = 2 * int(charge_steps_per_stage) + int(sterics_steps)
        nodes = [
            float(charge_steps_per_stage) / total,
            float(charge_steps_per_stage + sterics_steps) / total,
        ]
        vdw = [1.0, 1.0, 0.0, 0.0]
        charge = [1.0, 0.0, 0.0, 0.0]
        segments = [int(subdivisions_per_stage)] * 3
        interval_steps = [
            int(charge_steps_per_stage),
            int(sterics_steps),
            int(charge_steps_per_stage),
        ]
        source = "legacy_staged"

    vdw_b = list(reversed(vdw))
    charge_b = list(reversed(charge))
    if source == "staged_bonded":
        bonded_a = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        mapped_vdw = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        mapped_charge = list(mapped_vdw)
        stage_labels = [
            "discharge_a",
            "promote_b_bonded",
            "exchange_sterics_and_mapped",
            "demote_a_bonded",
            "charge_b",
        ]
        reverse_stage_labels = [
            "discharge_b",
            "promote_a_bonded",
            "exchange_sterics_and_mapped",
            "demote_b_bonded",
            "charge_a",
        ]
    else:
        bonded_a = list(vdw)
        mapped_vdw = [
            0.5 * (1.0 - left + right) for left, right in zip(vdw, vdw_b)
        ]
        mapped_charge = [
            0.5 * (1.0 - left + right) for left, right in zip(charge, charge_b)
        ]
        stage_labels = [f"interval_{index + 1}" for index in range(len(nodes) + 1)]
        reverse_stage_labels = list(reversed(stage_labels))
    bonded_b = list(reversed(bonded_a))
    separate_bonded = [1.0 if source == "staged_bonded" else 0.0] * len(vdw)
    result = {
        "source": source,
        "nodes": nodes,
        "vdw_a": vdw,
        "vdw_b": vdw_b,
        "charge_a": charge,
        "charge_b": charge_b,
        "bonded_a": bonded_a,
        "bonded_b": bonded_b,
        "separate_bonded": separate_bonded,
        "mapped_vdw": mapped_vdw,
        "mapped_charge": mapped_charge,
        "stage_labels": stage_labels,
        "reverse_stage_labels": reverse_stage_labels,
        "segments_per_interval": segments,
        "interval_steps": interval_steps,
        "total_steps": total,
    }
    if source == "scheme1_soft_bond":
        result.update({
            "soft_bond_a": [1.0, 0.5, 0.0],
            "soft_bond_b": [0.0, 0.5, 1.0],
            "soft_angles_a": [1.0, 0.0, 0.0],
            "soft_angles_b": [0.0, 0.0, 1.0],
            "soft_torsions_a": [1.0, 0.0, 0.0],
            "soft_torsions_b": [0.0, 0.0, 1.0],
            "bond_nonbonded_charge_a": [0.0, 0.0, 1.0],
            "bond_nonbonded_charge_b": [1.0, 0.0, 0.0],
            "bond_nonbonded_vdw_a": [0.0, 1.0, 1.0],
            "bond_nonbonded_vdw_b": [1.0, 1.0, 0.0],
            "bond_one_four_charge_a": [1.0, 0.0, 0.0],
            "bond_one_four_charge_b": [0.0, 0.0, 1.0],
            "bond_one_four_vdw_a": [1.0, 1.0, 0.0],
            "bond_one_four_vdw_b": [0.0, 1.0, 1.0],
            "stage_labels": ["soften_topology_a", "form_topology_b"],
            "reverse_stage_labels": ["soften_topology_b", "form_topology_a"],
        })
    else:
        if source == "staged_bonded":
            legacy_soft_bond_a = list(result["bonded_a"])
            legacy_soft_bond_b = list(result["bonded_b"])
        else:
            legacy_soft_bond_a = [1.0 - value for value in result["mapped_vdw"]]
            legacy_soft_bond_b = list(result["mapped_vdw"])
        if source == "legacy_staged":
            topology_b = list(result["mapped_vdw"])
            topology_a = [1.0 - value for value in topology_b]
        else:
            topology_a = topology_b = None
        result.update({
            "soft_bond_a": legacy_soft_bond_a,
            "soft_bond_b": legacy_soft_bond_b,
            "soft_angles_a": list(result["bonded_a"]),
            "soft_angles_b": list(result["bonded_b"]),
            "soft_torsions_a": list(result["bonded_a"]),
            "soft_torsions_b": list(result["bonded_b"]),
            "bond_nonbonded_charge_a": (
                topology_b if topology_b is not None
                else [0.0] * len(result["bonded_a"])
            ),
            "bond_nonbonded_charge_b": (
                topology_a if topology_a is not None
                else [0.0] * len(result["bonded_b"])
            ),
            "bond_nonbonded_vdw_a": (
                topology_b if topology_b is not None
                else [0.0] * len(result["bonded_a"])
            ),
            "bond_nonbonded_vdw_b": (
                topology_a if topology_a is not None
                else [0.0] * len(result["bonded_b"])
            ),
            "bond_one_four_charge_a": (
                topology_a if topology_a is not None
                else [1.0] * len(result["bonded_a"])
            ),
            "bond_one_four_charge_b": (
                topology_b if topology_b is not None
                else [1.0] * len(result["bonded_b"])
            ),
            "bond_one_four_vdw_a": (
                topology_a if topology_a is not None
                else [1.0] * len(result["bonded_a"])
            ),
            "bond_one_four_vdw_b": (
                topology_b if topology_b is not None
                else [1.0] * len(result["bonded_b"])
            ),
        })
    return result


def _expand_node_values(nodes, resolved):
    values = [nodes[0]]
    for interval, count in enumerate(resolved["segments_per_interval"]):
        for subdivision in range(int(count)):
            fraction = (subdivision + 1) / float(count)
            values.append(
                nodes[interval]
                + fraction * (nodes[interval + 1] - nodes[interval])
            )
    return values


def _expand_path(resolved):
    node_values = {
        CHARGE_A_PARAMETER: resolved["charge_a"],
        CHARGE_B_PARAMETER: resolved["charge_b"],
        MAPPED_CHARGE_PARAMETER: resolved["mapped_charge"],
        STERICS_A_PARAMETER: resolved["vdw_a"],
        STERICS_B_PARAMETER: resolved["vdw_b"],
        STERICS_PARAMETER: resolved["mapped_vdw"],
        BONDED_A_PARAMETER: resolved["bonded_a"],
        BONDED_B_PARAMETER: resolved["bonded_b"],
        SEPARATE_BONDED_PARAMETER: resolved["separate_bonded"],
        SOFT_BOND_A_PARAMETER: resolved["soft_bond_a"],
        SOFT_BOND_B_PARAMETER: resolved["soft_bond_b"],
        SOFT_ANGLE_A_PARAMETER: resolved["soft_angles_a"],
        SOFT_ANGLE_B_PARAMETER: resolved["soft_angles_b"],
        SOFT_TORSION_A_PARAMETER: resolved["soft_torsions_a"],
        SOFT_TORSION_B_PARAMETER: resolved["soft_torsions_b"],
        BOND_NONBONDED_CHARGE_A_PARAMETER: resolved["bond_nonbonded_charge_a"],
        BOND_NONBONDED_CHARGE_B_PARAMETER: resolved["bond_nonbonded_charge_b"],
        BOND_NONBONDED_VDW_A_PARAMETER: resolved["bond_nonbonded_vdw_a"],
        BOND_NONBONDED_VDW_B_PARAMETER: resolved["bond_nonbonded_vdw_b"],
        BOND_ONE_FOUR_CHARGE_A_PARAMETER: resolved["bond_one_four_charge_a"],
        BOND_ONE_FOUR_CHARGE_B_PARAMETER: resolved["bond_one_four_charge_b"],
        BOND_ONE_FOUR_VDW_A_PARAMETER: resolved["bond_one_four_vdw_a"],
        BOND_ONE_FOUR_VDW_B_PARAMETER: resolved["bond_one_four_vdw_b"],
    }
    values = {
        name: _expand_node_values(nodes, resolved)
        for name, nodes in node_values.items()
    }
    steps = []
    for interval, (count, total) in enumerate(
        zip(resolved["segments_per_interval"], resolved["interval_steps"])
    ):
        quotient, remainder = divmod(int(total), int(count))
        for subdivision in range(int(count)):
            steps.append(quotient + (1 if subdivision < remainder else 0))
    return values, steps


def _add_amber_reciprocal_path(values, resolved):
    weight_a = [_smoothstep2(value) for value in resolved["charge_a"]]
    weight_b = [_smoothstep2(value) for value in resolved["charge_b"]]
    reciprocal_a = [
        math.sqrt(value) - 1.0 for value in weight_a
    ]
    reciprocal_b = [
        math.sqrt(value) - 1.0 for value in weight_b
    ]
    values[RECIPROCAL_A_CHARGE_PARAMETER] = _expand_node_values(
        reciprocal_a, resolved
    )
    values[RECIPROCAL_B_CHARGE_PARAMETER] = _expand_node_values(
        reciprocal_b, resolved
    )
    values[RECIPROCAL_A_EXCEPTION_PARAMETER] = _expand_node_values(
        [value - 1.0 for value in weight_a], resolved
    )
    values[RECIPROCAL_B_EXCEPTION_PARAMETER] = _expand_node_values(
        [value - 1.0 for value in weight_b], resolved
    )


def _add_gapsys_reciprocal_path(values, resolved):
    weight_a = resolved["charge_a"]
    weight_b = resolved["charge_b"]
    values[GAPSYS_RECIPROCAL_A_CHARGE_PARAMETER] = _expand_node_values(
        [math.sqrt(value) - 1.0 for value in weight_a], resolved
    )
    values[GAPSYS_RECIPROCAL_B_CHARGE_PARAMETER] = _expand_node_values(
        [math.sqrt(value) - 1.0 for value in weight_b], resolved
    )
    values[GAPSYS_RECIPROCAL_A_EXCEPTION_PARAMETER] = _expand_node_values(
        [value - 1.0 for value in weight_a], resolved
    )
    values[GAPSYS_RECIPROCAL_B_EXCEPTION_PARAMETER] = _expand_node_values(
        [value - 1.0 for value in weight_b], resolved
    )


def create_softcore_hamiltonian(
    endpoint_a: mm.System,
    endpoint_b: mm.System,
    unique_a,
    unique_b,
    *,
    function: str = "beutler",
    coulomb_function: str = "linear_pme",
    alpha: float = 0.3,
    sigma_nm: float = 0.25,
    power: int = 1,
    gapsys_scale_linpoint_lj: float = 0.85,
    gapsys_scale_linpoint_q: float = 0.30,
    gapsys_sigma_nm: float = 0.30,
    ssc2_alpha_lj: float = 0.5,
    ssc2_alpha_coul: float = 1.0,
    ssc2_beta_coul: float = 1.0,
    ssc2_switch_width_nm: float = 0.2,
    soft_bond_alpha_nm2: float = 100.0,
    soft_bond_pairs=(),
    soft_bond_pair_changes=(),
    charge_steps_per_stage: int = 10000,
    sterics_steps: int = 30000,
    subdivisions_per_stage: int = 1,
    total_steps: int | None = None,
    path_nodes=None,
    vdw_a=None,
    charge_a=None,
    segments_per_interval=None,
    path_mode: str | None = None,
    control_nodes=None,
    stage_interpolation: str = "linear",
    use_long_range_correction: bool = True,
) -> CovalentSoftcoreHamiltonian:
    endpoint_a, endpoint_b = _common_mass_endpoint_copies(endpoint_a, endpoint_b)
    _assert_compatible_endpoints(endpoint_a, endpoint_b)
    function = str(function).lower()
    coulomb_function = str(coulomb_function).lower()
    stage_interpolation = str(stage_interpolation).lower()
    if function not in {
        "beutler",
        "gapsys",
        "amber_ssc2",
        "effective_distance_ssc2",
    }:
        raise CovalentAlchemyError(
            "softcore function must be 'beutler', 'gapsys', 'amber_ssc2', "
            "or 'effective_distance_ssc2'"
        )
    if coulomb_function not in {
        "linear_pme",
        "gapsys",
        "amber_ssc2",
        "effective_distance_ssc2",
    }:
        raise CovalentAlchemyError(
            "softcore coulomb_function must be 'linear_pme', 'gapsys', "
            "'amber_ssc2', or 'effective_distance_ssc2'"
        )
    if stage_interpolation not in {"linear", "smoothstep2"}:
        raise CovalentAlchemyError(
            "stage_interpolation must be 'linear' or 'smoothstep2'"
        )
    if alpha <= 0.0 or sigma_nm <= 0.0 or power < 1:
        raise CovalentAlchemyError("softcore alpha, sigma_nm, and power must be positive")
    if soft_bond_alpha_nm2 <= 0.0:
        raise CovalentAlchemyError("soft_bond_alpha_nm2 must be positive")
    if (
        gapsys_scale_linpoint_lj <= 0.0
        or gapsys_scale_linpoint_q <= 0.0
        or gapsys_sigma_nm <= 0.0
    ):
        raise CovalentAlchemyError(
            "Gapsys scale linearization point and sigma must be positive"
        )
    if (
        ssc2_alpha_lj <= 0.0
        or ssc2_alpha_coul <= 0.0
        or ssc2_beta_coul <= 0.0
        or ssc2_switch_width_nm <= 0.0
    ):
        raise CovalentAlchemyError("Amber SSC(2) LJ parameters must be positive")
    if coulomb_function in {"amber_ssc2", "effective_distance_ssc2"}:
        if function != coulomb_function:
            raise CovalentAlchemyError(
                f"{coulomb_function} Coulomb requires function: {coulomb_function}"
            )
        if stage_interpolation != "linear":
            raise CovalentAlchemyError(
                "Amber SSC(2) Coulomb requires stage_interpolation: linear"
            )
        if str(path_mode).lower() != "concerted":
            raise CovalentAlchemyError(
                "Amber SSC(2) Coulomb requires softcore path mode: concerted"
            )
    resolved_path = resolve_softcore_path(
        charge_steps_per_stage=int(charge_steps_per_stage),
        sterics_steps=int(sterics_steps),
        subdivisions_per_stage=int(subdivisions_per_stage),
        total_steps=total_steps,
        path_nodes=path_nodes,
        vdw_a=vdw_a,
        charge_a=charge_a,
        segments_per_interval=segments_per_interval,
        path_mode=path_mode,
        control_nodes=control_nodes,
    )
    topology_path_sources = {
        "legacy_staged",
        "scheme1_soft_bond",
        "explicit_nodes",
    }
    if soft_bond_pair_changes and resolved_path["source"] not in topology_path_sources:
        raise CovalentAlchemyError(
            "the selected alchemical bond requires the legacy staged, "
            "scheme1_soft_bond, or explicit control-node path"
        )
    if coulomb_function == "gapsys":
        if function != "gapsys":
            raise CovalentAlchemyError(
                "Gapsys Coulomb requires function: gapsys"
            )
        if any(
            not math.isclose(left + right, 1.0, abs_tol=1.0e-12)
            for left, right in zip(
                resolved_path["charge_a"], resolved_path["charge_b"]
            )
        ):
            raise CovalentAlchemyError(
                "Gapsys Coulomb requires complementary A/B charge schedules"
            )
    output = _system_shell(endpoint_a)
    _add_bonded_forces(
        output,
        endpoint_a,
        endpoint_b,
        soft_bond_alpha_nm2=soft_bond_alpha_nm2,
        soft_bond_pairs=soft_bond_pairs,
    )
    _add_nonbonded_forces(
        output,
        endpoint_a,
        endpoint_b,
        unique_a,
        unique_b,
        function=function,
        coulomb_function=coulomb_function,
        alpha=float(alpha),
        sigma_nm=float(sigma_nm),
        power=int(power),
        gapsys_scale_linpoint_lj=float(gapsys_scale_linpoint_lj),
        gapsys_scale_linpoint_q=float(gapsys_scale_linpoint_q),
        gapsys_sigma_nm=float(gapsys_sigma_nm),
        ssc2_alpha_lj=float(ssc2_alpha_lj),
        ssc2_alpha_coul=float(ssc2_alpha_coul),
        ssc2_beta_coul=float(ssc2_beta_coul),
        ssc2_switch_width_nm=float(ssc2_switch_width_nm),
        use_long_range_correction=bool(use_long_range_correction),
        soft_bond_pair_changes=soft_bond_pair_changes,
        enable_soft_bond_topology=resolved_path["source"] in topology_path_sources,
    )
    _copy_other_forces(output, endpoint_a, endpoint_b, unique_a, unique_b)
    values, steps = _expand_path(resolved_path)
    if coulomb_function == "amber_ssc2":
        _add_amber_reciprocal_path(values, resolved_path)
    elif coulomb_function == "gapsys":
        _add_gapsys_reciprocal_path(values, resolved_path)
    return CovalentSoftcoreHamiltonian(
        output,
        values,
        steps,
        sum(steps),
        resolved_path,
        stage_interpolation,
    )
