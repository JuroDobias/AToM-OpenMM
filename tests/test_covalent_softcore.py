import numpy as np
import openmm as mm
from openmm import unit

from atom_openmm.covalent_softcore import (
    SOFTCORE_NONBONDED_FORCE_GROUP,
    create_softcore_hamiltonian,
)
from atom_openmm.covalent_workflow import (
    _EndpointLRCCorrectionEvaluator,
    _run_segmented_protocol,
)
from atom_openmm.neqti_integrator import ATMNonequilibriumLangevinIntegrator


def _endpoint(state):
    system = mm.System()
    for _ in range(5):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(2.5, 0.0, 0.0),
        mm.Vec3(0.0, 2.5, 0.0),
        mm.Vec3(0.0, 0.0, 2.5),
    )
    bonds = mm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, 1000.0)
    bonds.addBond(1, 2 if state == "a" else 3, 0.14 if state == "a" else 0.16, 800.0)
    system.addForce(bonds)
    angles = mm.HarmonicAngleForce()
    angles.addAngle(0, 1, 2 if state == "a" else 3, 2.0, 100.0)
    system.addForce(angles)
    torsions = mm.PeriodicTorsionForce()
    torsions.addTorsion(0, 1, 2 if state == "a" else 3, 4, 3, 0.0, 2.0)
    system.addForce(torsions)
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(1.0)
    nonbonded.setEwaldErrorTolerance(1.0e-5)
    nonbonded.setUseDispersionCorrection(True)
    parameters = [
        (-0.20, 0.30, 0.40),
        (0.10, 0.31, 0.30),
        (0.15, 0.32, 0.20) if state == "a" else (0.0, 0.32, 0.0),
        (0.0, 0.33, 0.0) if state == "a" else (-0.05, 0.33, 0.25),
        (0.0, 0.34, 0.20),
    ]
    for values in parameters:
        nonbonded.addParticle(*values)
    nonbonded.addException(0, 1, 0.0, 0.3, 0.0)
    nonbonded.addException(
        1, 2, -0.015 if state == "a" else 0.0, 0.315, 0.08 if state == "a" else 0.0
    )
    nonbonded.addException(
        1, 3, 0.0 if state == "a" else -0.005, 0.32, 0.0 if state == "a" else 0.07
    )
    nonbonded.addException(2, 3, 0.0, 1.0, 0.0)
    system.addForce(nonbonded)
    return system


def _energy_forces(system, positions, parameters=None):
    context = mm.Context(system, mm.VerletIntegrator(0.001))
    context.setPositions(positions)
    for name, value in (parameters or {}).items():
        context.setParameter(name, value)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    del context
    return energy, forces


def _test_softcore_nodes_reproduce_endpoint_energies_and_forces():
    endpoint_a = _endpoint("a")
    endpoint_b = _endpoint("b")
    hamiltonian = create_softcore_hamiltonian(
        endpoint_a,
        endpoint_b,
        [2],
        [3],
        charge_steps_per_stage=2,
        sterics_steps=3,
    )
    positions = np.asarray(
        [[0, 0, 0], [0.15, 0, 0], [0.28, 0.08, 0], [0.29, -0.09, 0.03], [0.7, 0.4, 0.3]]
    ) * unit.nanometer
    for endpoint, node in ((endpoint_a, 0), (endpoint_b, 3)):
        expected_energy, expected_forces = _energy_forces(endpoint, positions)
        parameters = {
            name: values[node] for name, values in hamiltonian.parameter_values.items()
        }
        observed_energy, observed_forces = _energy_forces(
            hamiltonian.system, positions, parameters
        )
        assert np.isclose(observed_energy, expected_energy, atol=1.0e-5)
        assert np.allclose(observed_forces, expected_forces, atol=1.0e-3)


def _test_softcore_schedule_is_symmetric_and_finite_at_overlap():
    hamiltonian = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        charge_steps_per_stage=11,
        sterics_steps=17,
    )
    assert hamiltonian.segment_steps == [11, 17, 11]
    assert hamiltonian.total_steps == 39
    positions = np.asarray(
        [[0, 0, 0], [0.15, 0, 0], [0.28, 0.08, 0], [0.28, 0.08, 0], [0.7, 0.4, 0.3]]
    ) * unit.nanometer
    midpoint = {
        name: 0.5 * (values[1] + values[2])
        for name, values in hamiltonian.parameter_values.items()
    }
    energy, forces = _energy_forces(hamiltonian.system, positions, midpoint)
    assert np.isfinite(energy)
    assert np.all(np.isfinite(forces))


def _test_softcore_subdivision_preserves_path_and_total_steps():
    hamiltonian = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        charge_steps_per_stage=11,
        sterics_steps=17,
        subdivisions_per_stage=4,
    )

    assert len(hamiltonian.segment_steps) == 12
    assert sum(hamiltonian.segment_steps) == 39
    assert hamiltonian.parameter_values["COVALENT_CHARGE_A"][0] == 1.0
    assert hamiltonian.parameter_values["COVALENT_CHARGE_A"][4] == 0.0
    assert hamiltonian.parameter_values["COVALENT_STERICS"][4] == 0.0
    assert hamiltonian.parameter_values["COVALENT_STERICS"][8] == 1.0
    assert hamiltonian.parameter_values["COVALENT_CHARGE_B"][-1] == 1.0
    assert hamiltonian.segment_steps == [3, 3, 3, 2, 5, 4, 4, 4, 3, 3, 3, 2]


def _test_segment_work_increments_sum_to_total_protocol_work():
    hamiltonian = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        charge_steps_per_stage=4,
        sterics_steps=6,
        subdivisions_per_stage=2,
    )
    integrator = ATMNonequilibriumLangevinIntegrator(
        temperature=300.0 * unit.kelvin,
        collision_rate=1.0 / unit.picosecond,
        timestep=1.0 * unit.femtosecond,
        parameter_values=hamiltonian.parameter_values,
        steps_per_segment=hamiltonian.segment_steps,
        random_seed=9,
    )
    context = mm.Context(hamiltonian.system, integrator)
    positions = np.asarray(
        [[0, 0, 0], [0.15, 0, 0], [0.28, 0.08, 0], [0.29, -0.09, 0.03], [0.7, 0.4, 0.3]]
    ) * unit.nanometer
    context.setPositions(positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 9)
    for name, values in hamiltonian.parameter_values.items():
        context.setParameter(name, values[0])

    total, increments = _run_segmented_protocol(
        integrator, hamiltonian.segment_steps
    )

    assert len(increments) == 6
    assert np.isclose(sum(increments), total)
    assert np.isclose(
        total,
        integrator.get_protocol_work().value_in_unit(unit.kilojoule_per_mole),
    )


def _test_softcore_schedule_runs_forward_and_reverse_on_device():
    hamiltonian = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        charge_steps_per_stage=2,
        sterics_steps=3,
    )
    positions = np.asarray(
        [[0, 0, 0], [0.15, 0, 0], [0.28, 0.08, 0], [0.29, -0.09, 0.03], [0.7, 0.4, 0.3]]
    ) * unit.nanometer
    for reverse in (False, True):
        schedules = {
            name: list(reversed(values)) if reverse else values
            for name, values in hamiltonian.parameter_values.items()
        }
        steps = list(reversed(hamiltonian.segment_steps)) if reverse else hamiltonian.segment_steps
        integrator = ATMNonequilibriumLangevinIntegrator(
            temperature=300.0 * unit.kelvin,
            collision_rate=1.0 / unit.picosecond,
            timestep=1.0 * unit.femtosecond,
            parameter_values=schedules,
            steps_per_segment=steps,
            random_seed=9,
        )
        context = mm.Context(hamiltonian.system, integrator)
        context.setPositions(positions)
        context.setVelocitiesToTemperature(300.0 * unit.kelvin, 9)
        for name, values in schedules.items():
            context.setParameter(name, values[0])
        integrator.step(hamiltonian.total_steps)
        work = integrator.get_protocol_work().value_in_unit(unit.kilojoule_per_mole)
        assert np.isfinite(work)
        del context


def _test_disabling_lrc_changes_energy_but_not_forces():
    enabled = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        use_long_range_correction=True,
    )
    disabled = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        use_long_range_correction=False,
    )
    positions = np.asarray(
        [[0, 0, 0], [0.15, 0, 0], [0.28, 0.08, 0], [0.29, -0.09, 0.03], [0.7, 0.4, 0.3]]
    ) * unit.nanometer
    parameters = {
        name: values[0] for name, values in enabled.parameter_values.items()
    }
    energy_on, forces_on = _energy_forces(enabled.system, positions, parameters)
    energy_off, forces_off = _energy_forces(disabled.system, positions, parameters)
    assert not np.isclose(energy_on, energy_off, atol=1.0e-8, rtol=0.0)
    assert np.allclose(forces_on, forces_off, atol=1.0e-5, rtol=0.0)
    for hamiltonian, expected in ((enabled, True), (disabled, False)):
        forces = [
            force
            for force in hamiltonian.system.getForces()
            if isinstance(force, mm.CustomNonbondedForce)
        ]
        assert len(forces) == 2
        assert all(force.getForceGroup() == SOFTCORE_NONBONDED_FORCE_GROUP for force in forces)
        assert all(force.getUseLongRangeCorrection() is expected for force in forces)


def _test_endpoint_lrc_difference_repairs_fixed_volume_work():
    enabled = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        charge_steps_per_stage=2,
        sterics_steps=3,
        use_long_range_correction=True,
    )
    disabled = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        charge_steps_per_stage=2,
        sterics_steps=3,
        use_long_range_correction=False,
    )
    positions = np.asarray(
        [[0, 0, 0], [0.15, 0, 0], [0.28, 0.08, 0], [0.29, -0.09, 0.03], [0.7, 0.4, 0.3]]
    ) * unit.nanometer
    contexts = []
    for hamiltonian in (enabled, disabled):
        integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
        context = mm.Context(hamiltonian.system, integrator)
        context.setPositions(positions)
        contexts.append(context)

    def group_energy(context, hamiltonian, node):
        for name, values in hamiltonian.parameter_values.items():
            context.setParameter(name, values[node])
        return context.getState(
            getEnergy=True, groups={SOFTCORE_NONBONDED_FORCE_GROUP}
        ).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    energies_on = [group_energy(contexts[0], enabled, node) for node in range(4)]
    energies_off = [group_energy(contexts[1], disabled, node) for node in range(4)]
    work_on = sum(right - left for left, right in zip(energies_on, energies_on[1:]))
    work_off = sum(right - left for left, right in zip(energies_off, energies_off[1:]))
    initial_delta = energies_on[0] - energies_off[0]
    final_delta = energies_on[-1] - energies_off[-1]
    assert np.isclose(work_on, work_off + final_delta - initial_delta, atol=1.0e-5)


def test_endpoint_lrc_correction_is_coordinate_invariant():
    enabled = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        use_long_range_correction=True,
    )
    disabled = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        use_long_range_correction=False,
    )
    positions = np.asarray(
        [[0, 0, 0], [0.15, 0, 0], [0.28, 0.08, 0], [0.29, -0.09, 0.03], [0.7, 0.4, 0.3]]
    ) * unit.nanometer

    def state_at(coordinates):
        context = mm.Context(_endpoint("a"), mm.VerletIntegrator(0.001))
        context.setPositions(coordinates)
        state = context.getState(getPositions=True)
        del context
        return state

    evaluator = _EndpointLRCCorrectionEvaluator(disabled, enabled)
    first = evaluator.correction(state_at(positions), enabled.parameter_values, 0)
    displaced = positions + np.asarray(
        [[0.02, 0.01, 0], [0, 0.01, 0.02], [-0.01, 0, 0.01], [0.01, -0.02, 0], [0, 0.02, -0.01]]
    ) * unit.nanometer
    second = evaluator.correction(state_at(displaced), enabled.parameter_values, 0)

    assert np.isfinite(first)
    assert np.isclose(
        first,
        second,
        atol=1.0e-4,
        rtol=0.0,
    )
    evaluator.close()
