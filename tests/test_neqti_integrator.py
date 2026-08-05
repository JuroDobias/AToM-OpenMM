import pytest
import openmm as mm
from openmm.unit import femtosecond, kelvin, kilojoules_per_mole, nanometer, picosecond


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


def _test_sampled_work_interval_one_matches_exact_and_other_intervals_share_trajectory():
    from atom_openmm.neqti_integrator import ATMNonequilibriumLangevinIntegrator

    system = _constant_parameter_system(0.0)
    integrator = ATMNonequilibriumLangevinIntegrator(
        temperature=300.0 * kelvin,
        collision_rate=1.0 / picosecond,
        timestep=1.0 * femtosecond,
        parameter_values={"switch_parameter": [0.0, 1.0]},
        steps_per_segment=50,
        random_seed=3,
        work_sample_intervals=[1, 5, 10, 25, 50],
    )
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.0, 0.0, 0.0]])

    integrator.step(50)

    exact = integrator.get_protocol_work() / kilojoules_per_mole
    sampled = {
        interval: work / kilojoules_per_mole
        for interval, work in integrator.get_sampled_protocol_work().items()
    }
    assert exact == pytest.approx(1.0)
    assert sampled[1] == pytest.approx(exact)
    assert sampled == pytest.approx({1: 1.0, 5: 1.0, 10: 1.0, 25: 1.0, 50: 1.0})


def _test_custom_integrator_updates_segment_boundaries_and_duration_without_rebuild():
    from atom_openmm.neqti_integrator import ATMNonequilibriumLangevinIntegrator

    system = _constant_parameter_system(0.0)
    integrator = ATMNonequilibriumLangevinIntegrator(
        temperature=300.0 * kelvin,
        collision_rate=1.0 / picosecond,
        timestep=1.0 * femtosecond,
        parameter_values={"switch_parameter": [0.0, 0.25, 1.0]},
        steps_per_segment=[2, 2],
        random_seed=4,
    )
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.0, 0.0, 0.0]])

    integrator.set_segment_steps([3, 5])
    integrator.step(3)
    assert context.getParameter("switch_parameter") == pytest.approx(0.25)
    integrator.step(5)
    assert context.getParameter("switch_parameter") == pytest.approx(1.0)
    assert integrator.get_protocol_work() / kilojoules_per_mole == pytest.approx(1.0)
    assert integrator.get_segment_steps() == [3, 5]


def _test_smoothstep2_spans_stage_subdivisions_without_resetting_at_boundary():
    from atom_openmm.neqti_integrator import (
        ATMNonequilibriumLangevinIntegrator,
        parameter_values_at_step,
        stage_interpolation_fraction,
    )

    system = _constant_parameter_system(0.0)
    parameter_values = {"switch_parameter": [0.0, 0.5, 1.0]}
    segment_steps = [2, 6]
    integrator = ATMNonequilibriumLangevinIntegrator(
        temperature=300.0 * kelvin,
        collision_rate=1.0 / picosecond,
        timestep=1.0 * femtosecond,
        parameter_values=parameter_values,
        steps_per_segment=segment_steps,
        segments_per_stage=[2],
        stage_interpolation="smoothstep2",
        random_seed=5,
    )
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.0, 0.0, 0.0]])

    integrator.step(1)
    expected = stage_interpolation_fraction(0.25, "smoothstep2")
    assert context.getParameter("switch_parameter") == pytest.approx(expected)
    observed = parameter_values_at_step(
        parameter_values,
        segment_steps,
        0,
        1,
        segments_per_stage=[2],
        stage_interpolation="smoothstep2",
    )
    assert observed["switch_parameter"] == pytest.approx(expected)

    integrator.step(1)
    assert context.getParameter("switch_parameter") == pytest.approx(0.5)
    integrator.step(6)
    assert context.getParameter("switch_parameter") == pytest.approx(1.0)
    assert integrator.get_protocol_work() / kilojoules_per_mole == pytest.approx(1.0)


def _test_smoothstep2_math_has_symmetric_zero_slope_endpoints():
    from atom_openmm.neqti_integrator import stage_interpolation_fraction

    curve = lambda value: stage_interpolation_fraction(value, "smoothstep2")
    grid = [index / 100.0 for index in range(101)]
    values = [curve(value) for value in grid]
    assert values[0] == pytest.approx(0.0)
    assert values[50] == pytest.approx(0.5)
    assert values[-1] == pytest.approx(1.0)
    assert values == sorted(values)
    epsilon = 1.0e-5
    assert (curve(epsilon) - curve(0.0)) / epsilon == pytest.approx(0.0, abs=1.0e-8)
    assert (curve(1.0) - curve(1.0 - epsilon)) / epsilon == pytest.approx(0.0, abs=1.0e-8)
    for value in (0.1, 0.25, 0.7, 0.9):
        assert curve(1.0 - value) == pytest.approx(1.0 - curve(value))


def _test_smoothstep2_integrator_matches_python_updates_at_every_step():
    from atom_openmm.neqti_integrator import (
        ATMNonequilibriumLangevinIntegrator,
        parameter_values_at_step,
    )

    system = _constant_parameter_system(0.0)
    parameter_values = {"switch_parameter": [0.0, 0.5, 1.0, 2.0]}
    segment_steps = [2, 3, 4]
    integrator = ATMNonequilibriumLangevinIntegrator(
        temperature=300.0 * kelvin,
        collision_rate=1.0 / picosecond,
        timestep=1.0 * femtosecond,
        parameter_values=parameter_values,
        steps_per_segment=segment_steps,
        segments_per_stage=[2, 1],
        stage_interpolation="smoothstep2",
        random_seed=6,
    )
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.0, 0.0, 0.0]])

    for segment, steps in enumerate(segment_steps):
        for local_step in range(1, steps + 1):
            integrator.step(1)
            expected = parameter_values_at_step(
                parameter_values,
                segment_steps,
                segment,
                local_step,
                segments_per_stage=[2, 1],
                stage_interpolation="smoothstep2",
            )
            assert context.getParameter("switch_parameter") == pytest.approx(
                expected["switch_parameter"]
            )


def _test_custom_integrator_drift_does_not_double_velocity():
    from atom_openmm.neqti_integrator import ATMNonequilibriumLangevinIntegrator

    system = _constant_parameter_system(0.0)
    integrator = ATMNonequilibriumLangevinIntegrator(
        temperature=0.0 * kelvin,
        collision_rate=0.0 / picosecond,
        timestep=1.0 * femtosecond,
        parameter_values={"switch_parameter": [0.0, 0.0]},
        steps_per_segment=1,
    )
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.0, 0.0, 0.0]])
    context.setVelocities([[1.0, 0.0, 0.0]] * nanometer / picosecond)

    integrator.step(1)

    state = context.getState(getPositions=True, getVelocities=True)
    assert state.getPositions(asNumpy=True)[0, 0] / nanometer == pytest.approx(0.001)
    assert state.getVelocities(asNumpy=True)[0, 0] / (nanometer / picosecond) == pytest.approx(1.0)


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


def _test_context_parameter_values_uses_softplus_parameters():
    from atom_openmm.neqti_integrator import _context_parameter_values

    class ATMForce:
        Lambda1 = staticmethod(lambda: "Lambda1")
        Lambda2 = staticmethod(lambda: "Lambda2")
        Alpha = staticmethod(lambda: "Alpha")
        Uh = staticmethod(lambda: "Uh")
        W0 = staticmethod(lambda: "W0")
        Umax = staticmethod(lambda: "Umax")
        Ubcore = staticmethod(lambda: "Ubcore")
        Acore = staticmethod(lambda: "Acore")

    state = {"lambda1": 0.5, "lambda2": 0.5, "alpha": 0.1 / kilojoules_per_mole,
        "uh": 100 * kilojoules_per_mole, "w0": 0 * kilojoules_per_mole,
        "Umax": 200 * kilojoules_per_mole, "Ubcore": 100 * kilojoules_per_mole,
        "Acore": 0.0625, "uoffset": 0 * kilojoules_per_mole}
    system = type("System", (), {"atmforce": ATMForce(), "multisoftplus": False})()

    values = _context_parameter_values(system, [state])

    assert values["Lambda1"] == [0.5]
    assert values["Lambda2"] == [0.5]
