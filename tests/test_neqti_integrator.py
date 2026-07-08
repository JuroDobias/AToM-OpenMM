import pytest
import openmm as mm
from openmm.unit import femtosecond, kelvin, kilojoules_per_mole, picosecond


def _constant_parameter_system(initial_value):
    system = mm.System()
    system.addParticle(1.0)
    force = mm.CustomExternalForce("switch_parameter")
    force.addGlobalParameter("switch_parameter", initial_value)
    force.addParticle(0, [])
    system.addForce(force)
    return system


def _test_custom_integrator_accumulates_forward_protocol_work():
    from atom_openmm.neqti_integrator import ATMNonequilibriumLangevinIntegrator

    system = _constant_parameter_system(0.0)
    integrator = ATMNonequilibriumLangevinIntegrator(
        temperature=300.0 * kelvin,
        collision_rate=1.0 / picosecond,
        timestep=1.0 * femtosecond,
        parameter_values={"switch_parameter": [0.0, 1.0]},
        steps_per_segment=2,
        random_seed=1,
    )
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.0, 0.0, 0.0]])

    integrator.step(2)

    assert context.getParameter("switch_parameter") == pytest.approx(1.0)
    assert integrator.get_protocol_work() / kilojoules_per_mole == pytest.approx(1.0)


def _test_custom_integrator_accumulates_reverse_protocol_work():
    from atom_openmm.neqti_integrator import ATMNonequilibriumLangevinIntegrator

    system = _constant_parameter_system(1.0)
    integrator = ATMNonequilibriumLangevinIntegrator(
        temperature=300.0 * kelvin,
        collision_rate=1.0 / picosecond,
        timestep=1.0 * femtosecond,
        parameter_values={"switch_parameter": [1.0, 0.0]},
        steps_per_segment=4,
        random_seed=2,
    )
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.0, 0.0, 0.0]])

    integrator.step(4)

    assert context.getParameter("switch_parameter") == pytest.approx(0.0)
    assert integrator.get_protocol_work() / kilojoules_per_mole == pytest.approx(-1.0)


def _test_direction_expression_changes_only_at_segment_endpoint():
    from atom_openmm.neqti_integrator import _piecewise_expression

    expression = _piecewise_expression([1.0, -1.0], 4, discrete=True)
    system = _constant_parameter_system(1.0)
    integrator = mm.CustomIntegrator(0.0)
    integrator.addGlobalVariable("neq_step", 0.0)
    integrator.addComputeGlobal("neq_step", "neq_step+1")
    integrator.addComputeGlobal("switch_parameter", expression)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.0, 0.0, 0.0]])

    observed = []
    for _ in range(4):
        integrator.step(1)
        observed.append(context.getParameter("switch_parameter"))

    assert observed == pytest.approx([1.0, 1.0, 1.0, -1.0])


def _test_custom_integrators_share_seed_in_compound_context():
    from atom_openmm.neqti_integrator import ATMNonequilibriumLangevinIntegrator, _set_shared_random_seed

    system = _constant_parameter_system(0.0)
    compound = mm.CompoundIntegrator()
    equilibrium = mm.LangevinMiddleIntegrator(300.0 * kelvin, 1.0 / picosecond, 1.0 * femtosecond)
    equilibrium.setRandomNumberSeed(17)
    integrators = [equilibrium]
    compound.addIntegrator(equilibrium)
    for _ in range(2):
        integrator = ATMNonequilibriumLangevinIntegrator(
            temperature=300.0 * kelvin,
            collision_rate=1.0 / picosecond,
            timestep=1.0 * femtosecond,
            parameter_values={"switch_parameter": [0.0, 1.0]},
            steps_per_segment=2,
            random_seed=2026,
        )
        integrators.append(integrator)
        compound.addIntegrator(integrator)
    _set_shared_random_seed(integrators, 2026)

    context = mm.Context(system, compound, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.0, 0.0, 0.0]])
    compound.setCurrentIntegrator(1)
    compound.step(1)
    compound.setCurrentIntegrator(2)
    compound.step(1)
