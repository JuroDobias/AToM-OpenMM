import numpy as np
import openmm as mm
from openmm import unit

from atom_openmm.covalent_alchemy import (
    CovalentAlchemyError,
    create_endpoint_hamiltonian,
)


def _endpoint(k, equilibrium):
    system = mm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)
    force = mm.HarmonicBondForce()
    force.addBond(0, 1, equilibrium * unit.nanometer, k * unit.kilojoules_per_mole / unit.nanometer**2)
    system.addForce(force)
    return system


def _energy(system, positions, parameter=None):
    context = mm.Context(system, mm.VerletIntegrator(0.001))
    context.setPositions(positions)
    if parameter:
        context.setParameter(*parameter)
    value = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    del context
    return value


def _test_endpoint_hamiltonian_has_exact_endpoint_energies():
    system_a = _endpoint(100.0, 0.1)
    system_b = _endpoint(250.0, 0.2)
    positions = np.asarray([[0, 0, 0], [0.15, 0, 0]]) * unit.nanometer
    for interpolation in ("linear", "envelope"):
        hybrid = create_endpoint_hamiltonian(system_a, system_b, interpolation=interpolation)
        assert np.isclose(
            _energy(hybrid.system, positions, (hybrid.lambda_parameter, 0.0)),
            _energy(system_a, positions),
        )
        assert np.isclose(
            _energy(hybrid.system, positions, (hybrid.lambda_parameter, 1.0)),
            _energy(system_b, positions),
        )


def _test_endpoint_hamiltonian_rejects_particle_mismatch():
    system_a = _endpoint(100.0, 0.1)
    system_b = _endpoint(100.0, 0.1)
    system_b.addParticle(1.0)
    try:
        create_endpoint_hamiltonian(system_a, system_b)
    except CovalentAlchemyError as exc:
        assert "particle counts" in str(exc)
    else:
        raise AssertionError("particle mismatch was accepted")


def _test_endpoint_hamiltonian_preserves_virtual_sites():
    system_a = _endpoint(100.0, 0.1)
    system_b = _endpoint(100.0, 0.1)
    for system in (system_a, system_b):
        system.addParticle(0.0)
        system.setVirtualSite(2, mm.TwoParticleAverageSite(0, 1, 0.5, 0.5))
    hybrid = create_endpoint_hamiltonian(system_a, system_b)
    assert hybrid.system.isVirtualSite(2)
    positions = np.asarray([[0, 0, 0], [0.2, 0, 0], [0, 0, 0]]) * unit.nanometer
    context = mm.Context(hybrid.system, mm.VerletIntegrator(0.001))
    context.setPositions(positions)
    context.computeVirtualSites()
    observed = context.getState(getPositions=True).getPositions(asNumpy=True)
    assert np.isclose(observed[2, 0].value_in_unit(unit.nanometer), 0.1)
    del context
