"""Reusable REST2 force scaling for standard fixed-charge OpenMM systems."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional

import openmm as mm
from openmm import unit


class REST2Error(ValueError):
    """Raised when a System cannot be transformed safely for REST2."""


@dataclass(frozen=True)
class REST2System:
    system: mm.System
    solute_atoms: tuple[int, ...]
    scale_parameter: str
    sqrt_scale_parameter: str


@dataclass(frozen=True)
class MultiREST2System:
    system: mm.System
    regions: dict[str, tuple[int, ...]]
    parameters: dict[str, tuple[str, str]]


def _copy_force_metadata(source, target):
    target.setForceGroup(source.getForceGroup())
    target.setName(source.getName())
    if hasattr(source, "usesPeriodicBoundaryConditions") and hasattr(target, "setUsesPeriodicBoundaryConditions"):
        target.setUsesPeriodicBoundaryConditions(source.usesPeriodicBoundaryConditions())


def _term_weights(atom_indices, solute):
    count = sum(index in solute for index in atom_indices)
    if count == len(atom_indices):
        return (0.0, 1.0, 0.0)
    if count:
        return (0.0, 0.0, 1.0)
    return (1.0, 0.0, 0.0)


def _scale_expression(energy, scale_parameter, sqrt_scale_parameter):
    return (
        f"(rest_unscaled + rest_scaled*{scale_parameter} + "
        f"rest_mixed*{sqrt_scale_parameter})*({energy})"
    )


def _transform_bonds(force, solute, scale_parameter, sqrt_scale_parameter):
    transformed = mm.CustomBondForce(
        _scale_expression("0.5*k*(r-r0)^2", scale_parameter, sqrt_scale_parameter)
    )
    transformed.addGlobalParameter(scale_parameter, 1.0)
    transformed.addGlobalParameter(sqrt_scale_parameter, 1.0)
    for name in ("r0", "k", "rest_unscaled", "rest_scaled", "rest_mixed"):
        transformed.addPerBondParameter(name)
    for index in range(force.getNumBonds()):
        atom1, atom2, r0, k = force.getBondParameters(index)
        weights = _term_weights((atom1, atom2), solute)
        transformed.addBond(atom1, atom2, [r0, k, *weights])
    _copy_force_metadata(force, transformed)
    return transformed


def _transform_custom_bonds(force, solute, scale_parameter, sqrt_scale_parameter):
    if force.getName() != "CovalentUniqueVacuumNonbondedForce":
        raise REST2Error(f"unsupported CustomBondForce for REST2: {force.getName()!r}")
    transformed = mm.CustomBondForce(
        _scale_expression(f"({force.getEnergyFunction()})", scale_parameter, sqrt_scale_parameter)
    )
    existing_globals = set()
    for index in range(force.getNumGlobalParameters()):
        name = force.getGlobalParameterName(index)
        existing_globals.add(name)
        transformed.addGlobalParameter(name, force.getGlobalParameterDefaultValue(index))
    if scale_parameter in existing_globals or sqrt_scale_parameter in existing_globals:
        raise REST2Error("covalent vacuum force conflicts with REST2 global parameter names")
    transformed.addGlobalParameter(scale_parameter, 1.0)
    transformed.addGlobalParameter(sqrt_scale_parameter, 1.0)
    for index in range(force.getNumPerBondParameters()):
        transformed.addPerBondParameter(force.getPerBondParameterName(index))
    for name in ("rest_unscaled", "rest_scaled", "rest_mixed"):
        transformed.addPerBondParameter(name)
    for index in range(force.getNumBonds()):
        atom1, atom2, parameters = force.getBondParameters(index)
        transformed.addBond(atom1, atom2, [*parameters, *_term_weights((atom1, atom2), solute)])
    _copy_force_metadata(force, transformed)
    return transformed


def _transform_angles(force, solute, scale_parameter, sqrt_scale_parameter):
    transformed = mm.CustomAngleForce(
        _scale_expression("0.5*k*(theta-theta0)^2", scale_parameter, sqrt_scale_parameter)
    )
    transformed.addGlobalParameter(scale_parameter, 1.0)
    transformed.addGlobalParameter(sqrt_scale_parameter, 1.0)
    for name in ("theta0", "k", "rest_unscaled", "rest_scaled", "rest_mixed"):
        transformed.addPerAngleParameter(name)
    for index in range(force.getNumAngles()):
        atom1, atom2, atom3, theta0, k = force.getAngleParameters(index)
        weights = _term_weights((atom1, atom2, atom3), solute)
        transformed.addAngle(atom1, atom2, atom3, [theta0, k, *weights])
    _copy_force_metadata(force, transformed)
    return transformed


def _transform_torsions(force, solute, scale_parameter, sqrt_scale_parameter):
    transformed = mm.CustomTorsionForce(
        _scale_expression(
            "k*(1+cos(periodicity*theta-phase))", scale_parameter, sqrt_scale_parameter
        )
    )
    transformed.addGlobalParameter(scale_parameter, 1.0)
    transformed.addGlobalParameter(sqrt_scale_parameter, 1.0)
    for name in (
        "periodicity", "phase", "k", "rest_unscaled", "rest_scaled", "rest_mixed"
    ):
        transformed.addPerTorsionParameter(name)
    for index in range(force.getNumTorsions()):
        atom1, atom2, atom3, atom4, periodicity, phase, k = force.getTorsionParameters(index)
        weights = _term_weights((atom1, atom2, atom3, atom4), solute)
        transformed.addTorsion(atom1, atom2, atom3, atom4, [periodicity, phase, k, *weights])
    _copy_force_metadata(force, transformed)
    return transformed


def _transform_nonbonded(force, solute, scale_parameter, sqrt_scale_parameter):
    force.addGlobalParameter(scale_parameter, 1.0)
    force.addGlobalParameter(sqrt_scale_parameter, 1.0)
    for atom in sorted(solute):
        charge, sigma, epsilon = force.getParticleParameters(atom)
        force.setParticleParameters(atom, 0.0 * charge, sigma, 0.0 * epsilon)
        force.addParticleParameterOffset(sqrt_scale_parameter, atom, charge, 0.0 * sigma, 0.0 * epsilon)
        force.addParticleParameterOffset(scale_parameter, atom, 0.0 * charge, 0.0 * sigma, epsilon)
    for index in range(force.getNumExceptions()):
        atom1, atom2, charge_product, sigma, epsilon = force.getExceptionParameters(index)
        count = int(atom1 in solute) + int(atom2 in solute)
        if not count:
            continue
        parameter = scale_parameter if count == 2 else sqrt_scale_parameter
        force.setExceptionParameters(index, atom1, atom2, 0.0 * charge_product, sigma, 0.0 * epsilon)
        force.addExceptionParameterOffset(
            parameter, index, charge_product, 0.0 * sigma, epsilon
        )
    return force


def create_rest2_system(
    system: mm.System,
    solute_atoms: Iterable[int],
    *,
    scale_parameter: str = "REST2_SCALE",
    sqrt_scale_parameter: str = "REST2_SQRT_SCALE",
) -> REST2System:
    """Clone ``system`` and make its standard forces REST2 scale-aware."""
    atoms = tuple(sorted(set(int(index) for index in solute_atoms)))
    if not atoms:
        raise REST2Error("REST2 solute_atoms must not be empty")
    if atoms[0] < 0 or atoms[-1] >= system.getNumParticles():
        raise REST2Error("REST2 solute atom index is outside the System particle range")
    if scale_parameter == sqrt_scale_parameter:
        raise REST2Error("REST2 scale parameter names must be different")

    transformed_system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    solute = set(atoms)
    replacements = []
    nonbonded_count = 0
    for index in range(transformed_system.getNumForces()):
        force = transformed_system.getForce(index)
        if isinstance(force, mm.HarmonicBondForce):
            replacements.append((index, _transform_bonds(force, solute, scale_parameter, sqrt_scale_parameter)))
        elif isinstance(force, mm.CustomBondForce):
            replacements.append(
                (index, _transform_custom_bonds(force, solute, scale_parameter, sqrt_scale_parameter))
            )
        elif isinstance(force, mm.HarmonicAngleForce):
            replacements.append((index, _transform_angles(force, solute, scale_parameter, sqrt_scale_parameter)))
        elif isinstance(force, mm.PeriodicTorsionForce):
            replacements.append((index, _transform_torsions(force, solute, scale_parameter, sqrt_scale_parameter)))
        elif isinstance(force, mm.NonbondedForce):
            _transform_nonbonded(force, solute, scale_parameter, sqrt_scale_parameter)
            nonbonded_count += 1
        elif isinstance(force, mm.CMAPTorsionForce):
            for torsion_index in range(force.getNumTorsions()):
                parameters = force.getTorsionParameters(torsion_index)
                atoms = [int(index) for index in parameters[1:]]
                if any(index in solute for index in atoms):
                    raise REST2Error(
                        "CMAP terms touching REST2 solute atoms are not supported"
                    )
            # Protein-backbone CMAP terms outside the hot region remain unscaled.
            continue
        elif isinstance(force, (mm.CMMotionRemover, mm.MonteCarloBarostat, mm.CustomExternalForce)):
            continue
        else:
            raise REST2Error(
                f"unsupported force for REST2 transformation: {force.__class__.__name__}"
            )
    if nonbonded_count != 1:
        raise REST2Error(f"REST2 requires exactly one NonbondedForce; found {nonbonded_count}")

    for index, replacement in reversed(replacements):
        transformed_system.removeForce(index)
        transformed_system.addForce(replacement)
    return REST2System(
        transformed_system, atoms, scale_parameter, sqrt_scale_parameter
    )


def set_rest2_scale(
    context: mm.Context, scale: float, rest2_system: Optional[REST2System] = None
):
    """Set a REST2 scale and its square root on an existing Context."""
    scale = float(scale)
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise REST2Error("REST2 scale must be finite and in the interval (0, 1]")
    scale_parameter = rest2_system.scale_parameter if rest2_system else "REST2_SCALE"
    sqrt_parameter = rest2_system.sqrt_scale_parameter if rest2_system else "REST2_SQRT_SCALE"
    context.setParameter(scale_parameter, scale)
    context.setParameter(sqrt_parameter, math.sqrt(scale))


def create_multi_rest2_system(
    system: mm.System,
    regions: dict[str, Iterable[int]],
) -> MultiREST2System:
    """Create independently scalable REST2 regions.

    Regions must be disjoint. Terms touching two regions are rejected because
    no such terms are expected between the two ligand copies in an ATM system.
    """
    normalized = {
        str(name): tuple(sorted(set(int(index) for index in atoms)))
        for name, atoms in regions.items()
    }
    if not normalized or any(not atoms for atoms in normalized.values()):
        raise REST2Error("multi-region REST2 requires non-empty regions")
    owners = {}
    for name, atoms in normalized.items():
        for atom in atoms:
            if atom < 0 or atom >= system.getNumParticles():
                raise REST2Error("REST2 solute atom index is outside the System particle range")
            if atom in owners:
                raise REST2Error(
                    f"REST2 regions {owners[atom]!r} and {name!r} overlap at atom {atom}"
                )
            owners[atom] = name
    parameters = {
        name: (f"AWH_REST2_{name.upper()}_SCALE", f"AWH_REST2_{name.upper()}_SQRT_SCALE")
        for name in normalized
    }
    transformed = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))

    def term_region(atom_indices):
        touched = {owners[index] for index in atom_indices if index in owners}
        if len(touched) > 1:
            raise REST2Error(
                f"bonded term {tuple(atom_indices)} touches multiple REST2 regions"
            )
        if not touched:
            return None, False
        name = next(iter(touched))
        return name, all(index in normalized[name] for index in atom_indices)

    replacements = []
    nonbonded_count = 0
    for force_index in range(transformed.getNumForces()):
        force = transformed.getForce(force_index)
        if isinstance(force, mm.HarmonicBondForce):
            expression = "unscaled"
            for name, (scale, sqrt_scale) in parameters.items():
                expression += f"+full_{name}*{scale}+mixed_{name}*{sqrt_scale}"
            custom = mm.CustomBondForce(f"({expression})*0.5*k*(r-r0)^2")
            for scale, sqrt_scale in parameters.values():
                custom.addGlobalParameter(scale, 1.0)
                custom.addGlobalParameter(sqrt_scale, 1.0)
            for parameter in ("r0", "k", "unscaled"):
                custom.addPerBondParameter(parameter)
            for name in normalized:
                custom.addPerBondParameter(f"full_{name}")
                custom.addPerBondParameter(f"mixed_{name}")
            for index in range(force.getNumBonds()):
                a, b, r0, k = force.getBondParameters(index)
                name, complete = term_region((a, b))
                weights = [1.0 if name is None else 0.0]
                for region in normalized:
                    weights.extend([
                        1.0 if name == region and complete else 0.0,
                        1.0 if name == region and not complete else 0.0,
                    ])
                custom.addBond(a, b, [r0, k, *weights])
            _copy_force_metadata(force, custom)
            replacements.append((force_index, custom))
        elif isinstance(force, mm.HarmonicAngleForce):
            expression = "unscaled"
            for name, (scale, sqrt_scale) in parameters.items():
                expression += f"+full_{name}*{scale}+mixed_{name}*{sqrt_scale}"
            custom = mm.CustomAngleForce(f"({expression})*0.5*k*(theta-theta0)^2")
            for scale, sqrt_scale in parameters.values():
                custom.addGlobalParameter(scale, 1.0)
                custom.addGlobalParameter(sqrt_scale, 1.0)
            for parameter in ("theta0", "k", "unscaled"):
                custom.addPerAngleParameter(parameter)
            for name in normalized:
                custom.addPerAngleParameter(f"full_{name}")
                custom.addPerAngleParameter(f"mixed_{name}")
            for index in range(force.getNumAngles()):
                a, b, c, theta0, k = force.getAngleParameters(index)
                name, complete = term_region((a, b, c))
                weights = [1.0 if name is None else 0.0]
                for region in normalized:
                    weights.extend([
                        1.0 if name == region and complete else 0.0,
                        1.0 if name == region and not complete else 0.0,
                    ])
                custom.addAngle(a, b, c, [theta0, k, *weights])
            _copy_force_metadata(force, custom)
            replacements.append((force_index, custom))
        elif isinstance(force, mm.PeriodicTorsionForce):
            expression = "unscaled"
            for name, (scale, sqrt_scale) in parameters.items():
                expression += f"+full_{name}*{scale}+mixed_{name}*{sqrt_scale}"
            custom = mm.CustomTorsionForce(
                f"({expression})*k*(1+cos(periodicity*theta-phase))"
            )
            for scale, sqrt_scale in parameters.values():
                custom.addGlobalParameter(scale, 1.0)
                custom.addGlobalParameter(sqrt_scale, 1.0)
            for parameter in ("periodicity", "phase", "k", "unscaled"):
                custom.addPerTorsionParameter(parameter)
            for name in normalized:
                custom.addPerTorsionParameter(f"full_{name}")
                custom.addPerTorsionParameter(f"mixed_{name}")
            for index in range(force.getNumTorsions()):
                a, b, c, d, periodicity, phase, k = force.getTorsionParameters(index)
                name, complete = term_region((a, b, c, d))
                weights = [1.0 if name is None else 0.0]
                for region in normalized:
                    weights.extend([
                        1.0 if name == region and complete else 0.0,
                        1.0 if name == region and not complete else 0.0,
                    ])
                custom.addTorsion(a, b, c, d, [periodicity, phase, k, *weights])
            _copy_force_metadata(force, custom)
            replacements.append((force_index, custom))
        elif isinstance(force, mm.NonbondedForce):
            for scale, sqrt_scale in parameters.values():
                force.addGlobalParameter(scale, 1.0)
                force.addGlobalParameter(sqrt_scale, 1.0)
            for atom, name in sorted(owners.items()):
                charge, sigma, epsilon = force.getParticleParameters(atom)
                force.setParticleParameters(atom, 0.0 * charge, sigma, 0.0 * epsilon)
                scale, sqrt_scale = parameters[name]
                force.addParticleParameterOffset(
                    sqrt_scale, atom, charge, 0.0 * sigma, 0.0 * epsilon
                )
                force.addParticleParameterOffset(
                    scale, atom, 0.0 * charge, 0.0 * sigma, epsilon
                )
            for index in range(force.getNumExceptions()):
                a, b, charge_product, sigma, epsilon = force.getExceptionParameters(index)
                region_a = owners.get(int(a))
                region_b = owners.get(int(b))
                if region_a and region_b and region_a != region_b:
                    charge_value = charge_product.value_in_unit(
                        unit.elementary_charge**2
                    )
                    epsilon_value = epsilon.value_in_unit(
                        unit.kilojoule_per_mole
                    )
                    if abs(charge_value) > 0 or abs(epsilon_value) > 0:
                        raise REST2Error("nonzero exception connects two REST2 regions")
                    continue
                name = region_a or region_b
                if name is None:
                    continue
                parameter = parameters[name][0 if region_a == region_b else 1]
                force.setExceptionParameters(
                    index, a, b, 0.0 * charge_product, sigma, 0.0 * epsilon
                )
                force.addExceptionParameterOffset(
                    parameter, index, charge_product, 0.0 * sigma, epsilon
                )
            nonbonded_count += 1
        elif isinstance(force, mm.CMAPTorsionForce):
            for index in range(force.getNumTorsions()):
                atoms = [int(value) for value in force.getTorsionParameters(index)[1:]]
                if any(atom in owners for atom in atoms):
                    raise REST2Error("CMAP terms touching REST2 atoms are unsupported")
        elif isinstance(force, (mm.CMMotionRemover, mm.MonteCarloBarostat, mm.CustomExternalForce)):
            continue
        else:
            raise REST2Error(
                f"unsupported force for multi-region REST2: {force.__class__.__name__}"
            )
    if nonbonded_count != 1:
        raise REST2Error(
            f"multi-region REST2 requires exactly one NonbondedForce; found {nonbonded_count}"
        )
    for index, replacement in reversed(replacements):
        transformed.removeForce(index)
        transformed.addForce(replacement)
    return MultiREST2System(transformed, normalized, parameters)


def set_multi_rest2_scales(
    context: mm.Context,
    scales: dict[str, float],
    rest2_system: MultiREST2System,
):
    for name, (scale_parameter, sqrt_parameter) in rest2_system.parameters.items():
        scale = float(scales.get(name, 1.0))
        if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise REST2Error("REST2 scale must be finite and in the interval (0, 1]")
        context.setParameter(scale_parameter, scale)
        context.setParameter(sqrt_parameter, math.sqrt(scale))
