from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openmm as mm
from openmm import unit


class CovalentAlchemyError(ValueError):
    pass


@dataclass(frozen=True)
class EndpointHamiltonian:
    system: mm.System
    lambda_parameter: str
    interpolation: str


def _clone_force(force: mm.Force) -> mm.Force:
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(force))


def _clone_virtual_site(site):
    particles = [int(site.getParticle(index)) for index in range(site.getNumParticles())]
    if isinstance(site, mm.TwoParticleAverageSite):
        return mm.TwoParticleAverageSite(
            particles[0], particles[1], site.getWeight(0), site.getWeight(1)
        )
    if isinstance(site, mm.ThreeParticleAverageSite):
        return mm.ThreeParticleAverageSite(
            particles[0], particles[1], particles[2],
            site.getWeight(0), site.getWeight(1), site.getWeight(2),
        )
    if isinstance(site, mm.OutOfPlaneSite):
        return mm.OutOfPlaneSite(
            particles[0], particles[1], particles[2],
            site.getWeight12(), site.getWeight13(), site.getWeightCross(),
        )
    if isinstance(site, mm.LocalCoordinatesSite):
        return mm.LocalCoordinatesSite(
            particles,
            site.getOriginWeights(),
            site.getXWeights(),
            site.getYWeights(),
            site.getLocalPosition(),
        )
    raise CovalentAlchemyError(f"unsupported virtual site {type(site).__name__}")


def _virtual_site_signature(system, index):
    if not system.isVirtualSite(index):
        return None
    site = system.getVirtualSite(index)
    particles = tuple(int(site.getParticle(i)) for i in range(site.getNumParticles()))
    if isinstance(site, (mm.TwoParticleAverageSite, mm.ThreeParticleAverageSite)):
        return type(site).__name__, particles, tuple(
            float(site.getWeight(i)) for i in range(site.getNumParticles())
        )
    if isinstance(site, mm.OutOfPlaneSite):
        return (
            type(site).__name__, particles,
            float(site.getWeight12()), float(site.getWeight13()), float(site.getWeightCross()),
        )
    if isinstance(site, mm.LocalCoordinatesSite):
        local = site.getLocalPosition()
        return (
            type(site).__name__, particles,
            tuple(float(value) for value in site.getOriginWeights()),
            tuple(float(value) for value in site.getXWeights()),
            tuple(float(value) for value in site.getYWeights()),
            (float(local.x), float(local.y), float(local.z)),
        )
    raise CovalentAlchemyError(f"unsupported virtual site {type(site).__name__}")


def _endpoint_energy_force(system: mm.System, label: str) -> mm.CustomCVForce:
    expressions = []
    wrapper = mm.CustomCVForce("0")
    for index, force in enumerate(system.getForces()):
        if isinstance(force, (mm.CMMotionRemover, mm.MonteCarloBarostat)):
            continue
        name = f"{label}_{index}"
        wrapper.addCollectiveVariable(name, _clone_force(force))
        expressions.append(name)
    if not expressions:
        raise CovalentAlchemyError(f"endpoint {label} contains no energy forces")
    wrapper.setEnergyFunction("+".join(expressions))
    return wrapper


def _assert_compatible_endpoints(system_a: mm.System, system_b: mm.System):
    if system_a.getNumParticles() != system_b.getNumParticles():
        raise CovalentAlchemyError("endpoint systems must contain identical particle counts")
    for index in range(system_a.getNumParticles()):
        mass_a = system_a.getParticleMass(index).value_in_unit(unit.dalton)
        mass_b = system_b.getParticleMass(index).value_in_unit(unit.dalton)
        if not np.isclose(mass_a, mass_b, atol=1.0e-8):
            raise CovalentAlchemyError(f"endpoint particle {index} has different masses")
        if _virtual_site_signature(system_a, index) != _virtual_site_signature(system_b, index):
            raise CovalentAlchemyError(f"endpoint virtual site {index} differs")
    if system_a.getNumConstraints() != system_b.getNumConstraints():
        raise CovalentAlchemyError("endpoint systems must contain identical constraints")
    for index in range(system_a.getNumConstraints()):
        a1, a2, distance_a = system_a.getConstraintParameters(index)
        b1, b2, distance_b = system_b.getConstraintParameters(index)
        if (int(a1), int(a2)) != (int(b1), int(b2)) or not np.isclose(
            distance_a.value_in_unit(unit.nanometer),
            distance_b.value_in_unit(unit.nanometer),
            atol=1.0e-8,
        ):
            raise CovalentAlchemyError(f"endpoint constraint {index} differs")


def create_endpoint_hamiltonian(
    system_a: mm.System,
    system_b: mm.System,
    *,
    interpolation: str = "envelope",
    temperature_k: float = 300.0,
    lambda_parameter: str = "COVALENT_LAMBDA",
) -> EndpointHamiltonian:
    _assert_compatible_endpoints(system_a, system_b)
    output = mm.System()
    for index in range(system_a.getNumParticles()):
        output.addParticle(system_a.getParticleMass(index))
        if system_a.isVirtualSite(index):
            output.setVirtualSite(index, _clone_virtual_site(system_a.getVirtualSite(index)))
    for index in range(system_a.getNumConstraints()):
        particle1, particle2, distance = system_a.getConstraintParameters(index)
        output.addConstraint(particle1, particle2, distance)
    output.setDefaultPeriodicBoxVectors(*system_a.getDefaultPeriodicBoxVectors())

    if interpolation == "linear":
        expression = "(1-COVALENT_LAMBDA)*uA+COVALENT_LAMBDA*uB"
    elif interpolation == "envelope":
        beta = 1.0 / (
            unit.MOLAR_GAS_CONSTANT_R
            * float(temperature_k)
            * unit.kelvin
        ).value_in_unit(unit.kilojoules_per_mole)
        expression = (
            "umin-log((1-COVALENT_LAMBDA)*exp(-BETA*(uA-umin))"
            "+COVALENT_LAMBDA*exp(-BETA*(uB-umin)))/BETA;"
            "umin=min(uA,uB)"
        )
    else:
        raise CovalentAlchemyError("interpolation must be 'linear' or 'envelope'")
    expression = expression.replace("COVALENT_LAMBDA", lambda_parameter)
    force = mm.CustomCVForce(expression)
    force.addGlobalParameter(lambda_parameter, 0.0)
    if interpolation == "envelope":
        force.addGlobalParameter("BETA", beta)
    force.addCollectiveVariable("uA", _endpoint_energy_force(system_a, "a"))
    force.addCollectiveVariable("uB", _endpoint_energy_force(system_b, "b"))
    force.addEnergyParameterDerivative(lambda_parameter)
    force.setName("CovalentEndpointHamiltonian")
    output.addForce(force)

    for force_a in system_a.getForces():
        if isinstance(force_a, mm.MonteCarloBarostat):
            output.addForce(_clone_force(force_a))
            break
    if not any(isinstance(force, mm.CMMotionRemover) for force in output.getForces()):
        output.addForce(mm.CMMotionRemover())
    return EndpointHamiltonian(output, lambda_parameter, interpolation)
