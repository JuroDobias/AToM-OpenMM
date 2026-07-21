from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
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
SOFTCORE_NONBONDED_FORCE_GROUP = 31


@dataclass(frozen=True)
class CovalentSoftcoreHamiltonian:
    system: mm.System
    parameter_values: dict[str, list[float]]
    segment_steps: list[int]
    total_steps: int


def _force(system: mm.System, cls):
    forces = [force for force in system.getForces() if isinstance(force, cls)]
    if len(forces) > 1:
        raise CovalentAlchemyError(f"multiple {cls.__name__} instances are not supported")
    return forces[0] if forces else None


def _float(value, target_unit):
    return float(value.value_in_unit(target_unit))


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


def _canonical_angle(particles):
    particles = tuple(int(value) for value in particles)
    reverse = tuple(reversed(particles))
    return min(particles, reverse)


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


def _add_bonded_forces(output, endpoint_a, endpoint_b):
    bond_a = _force(endpoint_a, mm.HarmonicBondForce)
    bond_b = _force(endpoint_b, mm.HarmonicBondForce)
    bonds_a = _bond_terms(bond_a)
    bonds_b = _bond_terms(bond_b)
    common_bonds = mm.HarmonicBondForce()
    changed_bonds = mm.CustomBondForce(
        "0.5*((1-COVALENT_STERICS)*kA*(r-rA)^2+COVALENT_STERICS*kB*(r-rB)^2)"
    )
    changed_bonds.addGlobalParameter(STERICS_PARAMETER, 0.0)
    for name in ("rA", "kA", "rB", "kB"):
        changed_bonds.addPerBondParameter(name)
    for particles in sorted(set(bonds_a) | set(bonds_b)):
        value_a = bonds_a.get(particles)
        value_b = bonds_b.get(particles)
        if value_a == value_b:
            common_bonds.addBond(*particles, value_a[0], value_a[1])
        else:
            fallback = value_a or value_b
            r_a, k_a = value_a or (fallback[0], 0.0)
            r_b, k_b = value_b or (fallback[0], 0.0)
            changed_bonds.addBond(*particles, [r_a, k_a, r_b, k_b])
    if common_bonds.getNumBonds():
        if bond_a is not None:
            _copy_force_metadata(bond_a, common_bonds)
        common_bonds.setName("CovalentCommonBonds")
        output.addForce(common_bonds)
    if changed_bonds.getNumBonds():
        changed_bonds.setName("CovalentInterpolatedBonds")
        output.addForce(changed_bonds)

    angle_a = _force(endpoint_a, mm.HarmonicAngleForce)
    angle_b = _force(endpoint_b, mm.HarmonicAngleForce)
    angles_a = _angle_terms(angle_a)
    angles_b = _angle_terms(angle_b)
    common_angles = mm.HarmonicAngleForce()
    changed_angles = mm.CustomAngleForce(
        "0.5*((1-COVALENT_STERICS)*kA*(theta-thetaA)^2"
        "+COVALENT_STERICS*kB*(theta-thetaB)^2)"
    )
    changed_angles.addGlobalParameter(STERICS_PARAMETER, 0.0)
    for name in ("thetaA", "kA", "thetaB", "kB"):
        changed_angles.addPerAngleParameter(name)
    for particles in sorted(set(angles_a) | set(angles_b)):
        value_a = angles_a.get(particles)
        value_b = angles_b.get(particles)
        if value_a == value_b:
            common_angles.addAngle(*particles, value_a[0], value_a[1])
        else:
            fallback = value_a or value_b
            theta_a, k_a = value_a or (fallback[0], 0.0)
            theta_b, k_b = value_b or (fallback[0], 0.0)
            changed_angles.addAngle(*particles, [theta_a, k_a, theta_b, k_b])
    if common_angles.getNumAngles():
        if angle_a is not None:
            _copy_force_metadata(angle_a, common_angles)
        common_angles.setName("CovalentCommonAngles")
        output.addForce(common_angles)
    if changed_angles.getNumAngles():
        changed_angles.setName("CovalentInterpolatedAngles")
        output.addForce(changed_angles)

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
        ("A", only_a, "(1-COVALENT_STERICS)*k*(1+cos(periodicity*theta-phase))"),
        ("B", only_b, "COVALENT_STERICS*k*(1+cos(periodicity*theta-phase))"),
    ):
        force = mm.CustomTorsionForce(expression)
        force.addGlobalParameter(STERICS_PARAMETER, 0.0)
        for name in ("periodicity", "phase", "k"):
            force.addPerTorsionParameter(name)
        for particles, periodicity, phase, k in terms:
            force.addTorsion(*particles, [periodicity, phase, k])
        if force.getNumTorsions():
            force.setName(f"CovalentInterpolatedTorsions{label}")
            output.addForce(force)


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


def _softcore_expression(scale):
    return (
        f"4*({scale})*epsilon*(x*x-x);"
        "x=sigma^6/(r^6+SOFTCORE_ALPHA*(1-(" + scale + "))^SOFTCORE_POWER*SOFTCORE_SIGMA^6);"
        "sigma=0.5*(sigma1+sigma2);epsilon=sqrt(epsilon1*epsilon2)"
    )


def _softcore_bond_expression(scale):
    return (
        f"4*({scale})*epsilon*(x*x-x);"
        "x=sigma^6/(r^6+SOFTCORE_ALPHA*(1-(" + scale + "))^SOFTCORE_POWER*SOFTCORE_SIGMA^6)"
    )


def _new_softcore_force(
    source,
    label,
    scale,
    alpha,
    sigma_nm,
    power,
    *,
    use_long_range_correction,
):
    force = mm.CustomNonbondedForce(_softcore_expression(scale))
    force.setName(f"CovalentSoftcoreNonbonded{label}")
    force.addGlobalParameter(STERICS_PARAMETER, 0.0)
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


def _new_softcore_exception_force(label, scale, alpha, sigma_nm, power):
    force = mm.CustomBondForce(_softcore_bond_expression(scale))
    force.setName(f"CovalentSoftcoreExceptions{label}")
    force.addGlobalParameter(STERICS_PARAMETER, 0.0)
    force.addGlobalParameter("SOFTCORE_ALPHA", float(alpha))
    force.addGlobalParameter("SOFTCORE_SIGMA", float(sigma_nm))
    force.addGlobalParameter("SOFTCORE_POWER", float(power))
    force.addPerBondParameter("sigma")
    force.addPerBondParameter("epsilon")
    force.setUsesPeriodicBoundaryConditions(True)
    return force


def _add_nonbonded_forces(
    output,
    endpoint_a,
    endpoint_b,
    unique_a,
    unique_b,
    *,
    alpha,
    sigma_nm,
    power,
    use_long_range_correction,
):
    source_a = _force(endpoint_a, mm.NonbondedForce)
    source_b = _force(endpoint_b, mm.NonbondedForce)
    if source_a is None or source_b is None:
        raise CovalentAlchemyError("softcore endpoints require one NonbondedForce")
    if source_a.getNonbondedMethod() != source_b.getNonbondedMethod():
        raise CovalentAlchemyError("softcore endpoint nonbonded methods differ")
    unique_a = set(int(index) for index in unique_a)
    unique_b = set(int(index) for index in unique_b)
    if unique_a & unique_b:
        raise CovalentAlchemyError("softcore unique branches overlap")
    all_particles = set(range(endpoint_a.getNumParticles()))
    environment = all_particles - unique_a - unique_b

    force = mm.NonbondedForce()
    force.setName("CovalentInterpolatedPMENonbonded")
    _configure_nonbonded_like(source_a, force)
    for parameter in (
        CHARGE_A_PARAMETER,
        CHARGE_B_PARAMETER,
        MAPPED_CHARGE_PARAMETER,
        STERICS_PARAMETER,
    ):
        force.addGlobalParameter(parameter, 0.0)
    for particle in range(endpoint_a.getNumParticles()):
        q_a, sigma_a, epsilon_a = source_a.getParticleParameters(particle)
        q_b, sigma_b, epsilon_b = source_b.getParticleParameters(particle)
        qa = _float(q_a, unit.elementary_charge)
        qb = _float(q_b, unit.elementary_charge)
        sa = _float(sigma_a, unit.nanometer)
        sb = _float(sigma_b, unit.nanometer)
        ea = _float(epsilon_a, unit.kilojoule_per_mole)
        eb = _float(epsilon_b, unit.kilojoule_per_mole)
        if particle in unique_a:
            force.addParticle(0.0, sa, 0.0)
            _add_offset(force, CHARGE_A_PARAMETER, particle, charge=qa)
        elif particle in unique_b:
            force.addParticle(0.0, sb, 0.0)
            _add_offset(force, CHARGE_B_PARAMETER, particle, charge=qb)
        else:
            force.addParticle(qa, sa, ea)
            _add_offset(force, MAPPED_CHARGE_PARAMETER, particle, charge=qb - qa)
            _add_offset(force, STERICS_PARAMETER, particle, sigma=sb - sa, epsilon=eb - ea)

    exceptions_a = _exception_dict(source_a)
    exceptions_b = _exception_dict(source_b)
    exception_pairs = set(exceptions_a) | set(exceptions_b)
    exception_pairs.update(
        tuple(sorted((particle_a, particle_b)))
        for particle_a in unique_a for particle_b in unique_b
    )
    softcore_a = _new_softcore_force(
        source_a,
        "A",
        f"1-{STERICS_PARAMETER}",
        alpha,
        sigma_nm,
        power,
        use_long_range_correction=use_long_range_correction,
    )
    softcore_b = _new_softcore_force(
        source_b,
        "B",
        STERICS_PARAMETER,
        alpha,
        sigma_nm,
        power,
        use_long_range_correction=use_long_range_correction,
    )
    for particle in range(endpoint_a.getNumParticles()):
        _, sigma_a, epsilon_a = source_a.getParticleParameters(particle)
        _, sigma_b, epsilon_b = source_b.getParticleParameters(particle)
        softcore_a.addParticle([sigma_a, epsilon_a])
        softcore_b.addParticle([sigma_b, epsilon_b])
    if unique_a and environment:
        softcore_a.addInteractionGroup(unique_a, environment)
    if unique_b and environment:
        softcore_b.addInteractionGroup(unique_b, environment)
    exception_lj_a = _new_softcore_exception_force(
        "A", f"1-{STERICS_PARAMETER}", alpha, sigma_nm, power
    )
    exception_lj_b = _new_softcore_exception_force(
        "B", STERICS_PARAMETER, alpha, sigma_nm, power
    )

    for pair in sorted(exception_pairs):
        value_a = exceptions_a.get(pair, (0.0, 1.0, 0.0))
        value_b = exceptions_b.get(pair, (0.0, 1.0, 0.0))
        q_a, sigma_a, epsilon_a = value_a
        q_b, sigma_b, epsilon_b = value_b
        p1, p2 = pair
        in_a = int(p1 in unique_a) + int(p2 in unique_a)
        in_b = int(p1 in unique_b) + int(p2 in unique_b)
        if in_a == 2 or in_b == 2 or (in_a and in_b):
            index = force.addException(p1, p2, 0.0, 1.0, 0.0)
        elif in_a:
            index = force.addException(p1, p2, 0.0, sigma_a, 0.0)
            _add_exception_offset(force, CHARGE_A_PARAMETER, index, charge=q_a)
            if epsilon_a != 0.0:
                exception_lj_a.addBond(p1, p2, [sigma_a, epsilon_a])
        elif in_b:
            index = force.addException(p1, p2, 0.0, sigma_b, 0.0)
            _add_exception_offset(force, CHARGE_B_PARAMETER, index, charge=q_b)
            if epsilon_b != 0.0:
                exception_lj_b.addBond(p1, p2, [sigma_b, epsilon_b])
        else:
            index = force.addException(p1, p2, q_a, sigma_a, epsilon_a)
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
        softcore_a.addExclusion(p1, p2)
        softcore_b.addExclusion(p1, p2)

    output.addForce(force)
    if unique_a:
        output.addForce(softcore_a)
    if unique_b:
        output.addForce(softcore_b)
    if exception_lj_a.getNumBonds():
        output.addForce(exception_lj_a)
    if exception_lj_b.getNumBonds():
        output.addForce(exception_lj_b)


def _copy_other_forces(output, endpoint_a, endpoint_b):
    supported = (
        mm.HarmonicBondForce,
        mm.HarmonicAngleForce,
        mm.PeriodicTorsionForce,
        mm.NonbondedForce,
    )
    forces_b_by_name = defaultdict(list)
    for force in endpoint_b.getForces():
        if not isinstance(force, supported):
            forces_b_by_name[(type(force), force.getName())].append(force)
    for force in endpoint_a.getForces():
        if isinstance(force, supported):
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


def create_softcore_hamiltonian(
    endpoint_a: mm.System,
    endpoint_b: mm.System,
    unique_a,
    unique_b,
    *,
    alpha: float = 0.3,
    sigma_nm: float = 0.25,
    power: int = 1,
    charge_steps_per_stage: int = 10000,
    sterics_steps: int = 30000,
    use_long_range_correction: bool = True,
) -> CovalentSoftcoreHamiltonian:
    _assert_compatible_endpoints(endpoint_a, endpoint_b)
    if alpha <= 0.0 or sigma_nm <= 0.0 or power < 1:
        raise CovalentAlchemyError("softcore alpha, sigma_nm, and power must be positive")
    if charge_steps_per_stage < 1 or sterics_steps < 1:
        raise CovalentAlchemyError("softcore stage steps must be positive")
    output = _system_shell(endpoint_a)
    _add_bonded_forces(output, endpoint_a, endpoint_b)
    _add_nonbonded_forces(
        output,
        endpoint_a,
        endpoint_b,
        unique_a,
        unique_b,
        alpha=float(alpha),
        sigma_nm=float(sigma_nm),
        power=int(power),
        use_long_range_correction=bool(use_long_range_correction),
    )
    _copy_other_forces(output, endpoint_a, endpoint_b)
    steps = [int(charge_steps_per_stage), int(sterics_steps), int(charge_steps_per_stage)]
    values = {
        CHARGE_A_PARAMETER: [1.0, 0.0, 0.0, 0.0],
        CHARGE_B_PARAMETER: [0.0, 0.0, 0.0, 1.0],
        MAPPED_CHARGE_PARAMETER: [0.0, 0.5, 0.5, 1.0],
        STERICS_PARAMETER: [0.0, 0.0, 1.0, 1.0],
    }
    return CovalentSoftcoreHamiltonian(output, values, steps, sum(steps))
