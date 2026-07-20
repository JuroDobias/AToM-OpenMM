import math
import copy

import numpy as np
import openmm as mm
from openmm import unit
import pytest


def _energy_and_forces(system, positions):
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    return energy, np.asarray(forces)


def _standard_system():
    system = mm.System()
    for _ in range(4):
        system.addParticle(12.0)
    bonds = mm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, 500.0)
    bonds.addBond(1, 2, 0.15, 400.0)
    system.addForce(bonds)
    angles = mm.HarmonicAngleForce()
    angles.addAngle(0, 1, 2, 2.0, 50.0)
    system.addForce(angles)
    torsions = mm.PeriodicTorsionForce()
    torsions.addTorsion(0, 1, 2, 3, 2, 0.3, 4.0)
    system.addForce(torsions)
    nonbonded = mm.NonbondedForce()
    for charge, sigma, epsilon in [(0.2, 0.25, 0.3), (-0.2, 0.27, 0.2), (0.1, 0.3, 0.4), (-0.1, 0.28, 0.25)]:
        nonbonded.addParticle(charge, sigma, epsilon)
    nonbonded.addException(0, 1, -0.04, 0.25, 0.1)
    system.addForce(nonbonded)
    positions = np.array([[0, 0, 0], [0.16, 0, 0], [0.27, 0.12, 0], [0.36, 0.15, 0.13]])
    return system, positions


def _test_rest2_scale_one_matches_original_energy_and_forces():
    from atom_openmm.rest2 import create_rest2_system

    system, positions = _standard_system()
    original_xml = mm.XmlSerializer.serialize(system)
    transformed = create_rest2_system(system, [0, 1]).system
    original_energy, original_forces = _energy_and_forces(system, positions)
    transformed_energy, transformed_forces = _energy_and_forces(transformed, positions)

    assert transformed_energy == pytest.approx(original_energy, abs=1e-5)
    assert np.max(np.abs(transformed_forces - original_forces)) < 1e-4
    assert mm.XmlSerializer.serialize(system) == original_xml


def _pair_energy(solute_atoms, scale):
    from atom_openmm.rest2 import create_rest2_system, set_rest2_scale

    system = mm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)
    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    force.addParticle(0.5, 0.3, 0.4)
    force.addParticle(-0.4, 0.32, 0.2)
    system.addForce(force)
    rest2 = create_rest2_system(system, solute_atoms)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(rest2.system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0, 0, 0], [0.5, 0, 0]])
    set_rest2_scale(context, scale, rest2)
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _test_rest2_nonbonded_scaling_matches_solute_classification():
    scale = 0.49
    all_solute = _pair_energy([0, 1], scale)
    mixed = _pair_energy([0], scale)
    physical = _pair_energy([0, 1], 1.0)

    assert all_solute == pytest.approx(scale * physical, rel=1e-6)
    assert mixed == pytest.approx(math.sqrt(scale) * physical, rel=1e-6)


def _test_rest2_bond_scaling_and_context_validation():
    from atom_openmm.rest2 import REST2Error, create_rest2_system, set_rest2_scale

    system = mm.System()
    for _ in range(3):
        system.addParticle(12.0)
    bonds = mm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.1, 100.0)
    bonds.addBond(1, 2, 0.1, 100.0)
    system.addForce(bonds)
    nonbonded = mm.NonbondedForce()
    for _ in range(3):
        nonbonded.addParticle(0.0, 0.3, 0.0)
    system.addForce(nonbonded)
    rest2 = create_rest2_system(system, [0, 1])
    context = mm.Context(rest2.system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0, 0, 0], [0.2, 0, 0], [0.4, 0, 0]])
    physical = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    set_rest2_scale(context, 0.25, rest2)
    scaled = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert scaled == pytest.approx((0.25 + 0.5) * physical / 2.0)
    with pytest.raises(REST2Error, match="interval"):
        set_rest2_scale(context, 0.0, rest2)


def _test_rest2_rejects_unsupported_force():
    from atom_openmm.rest2 import REST2Error, create_rest2_system

    system = mm.System()
    system.addParticle(12.0)
    system.addForce(mm.CustomBondForce("r"))
    nonbonded = mm.NonbondedForce()
    nonbonded.addParticle(0.0, 0.3, 0.0)
    system.addForce(nonbonded)
    with pytest.raises(REST2Error, match="CustomBondForce"):
        create_rest2_system(system, [0])


def _test_rest2_scales_covalent_unique_vacuum_force():
    from atom_openmm.rest2 import create_rest2_system, set_rest2_scale

    system = mm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)
    vacuum = mm.CustomBondForce("k/r")
    vacuum.setName("CovalentUniqueVacuumNonbondedForce")
    vacuum.addPerBondParameter("k")
    vacuum.addBond(0, 1, [2.0])
    system.addForce(vacuum)
    nonbonded = mm.NonbondedForce()
    nonbonded.addParticle(0.0, 0.3, 0.0)
    nonbonded.addParticle(0.0, 0.3, 0.0)
    system.addForce(nonbonded)
    rest2 = create_rest2_system(system, [0, 1])
    context = mm.Context(
        rest2.system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions([[0, 0, 0], [0.5, 0, 0]])
    physical = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    set_rest2_scale(context, 0.25, rest2)
    scaled = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    assert scaled == pytest.approx(0.25 * physical)


def _test_rest2_parameters_remain_active_inside_atmforce():
    from atom_openmm.rest2 import create_rest2_system, set_rest2_scale

    system = mm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)
    bonds = mm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.1, 100.0)
    system.addForce(bonds)
    nonbonded = mm.NonbondedForce()
    nonbonded.addParticle(0.0, 0.3, 0.0)
    nonbonded.addParticle(0.0, 0.3, 0.0)
    system.addForce(nonbonded)
    rest2 = create_rest2_system(system, [0, 1])

    atmforce = mm.ATMForce("u0")
    for force in rest2.system.getForces():
        atmforce.addForce(copy.copy(force))
    atmforce.addParticle(mm.Vec3(0, 0, 0))
    atmforce.addParticle(mm.Vec3(0, 0, 0))
    nested = mm.System()
    nested.addParticle(12.0)
    nested.addParticle(12.0)
    nested.addForce(atmforce)
    context = mm.Context(
        nested, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions([[0, 0, 0], [0.2, 0, 0]])
    physical = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    set_rest2_scale(context, 0.5, rest2)
    scaled = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )

    assert scaled == pytest.approx(0.5 * physical)
