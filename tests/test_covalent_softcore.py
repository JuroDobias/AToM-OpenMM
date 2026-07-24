import numpy as np
import openmm as mm
from openmm import unit

from atom_openmm.covalent_softcore import (
    SOFTCORE_NONBONDED_FORCE_GROUP,
    STERICS_A_PARAMETER,
    STERICS_B_PARAMETER,
    _gapsys_energy_expression,
    create_softcore_hamiltonian,
)
from atom_openmm.covalent_workflow import (
    _EndpointLRCCorrectionEvaluator,
    _merge_work_profile_rows,
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


def _gapsys_reference_energy(r, sigma, epsilon, scale, alpha):
    c6 = 4.0 * epsilon * sigma**6
    c12 = 4.0 * epsilon * sigma**12
    rsc = alpha * ((26.0 / 7.0) * sigma**6 * (1.0 - scale)) ** (1.0 / 6.0)
    if r >= rsc:
        potential = c12 / r**12 - c6 / r**6
    else:
        potential = (
            (78.0 * c12 / rsc**14 - 21.0 * c6 / rsc**8) * r**2
            - (168.0 * c12 / rsc**13 - 48.0 * c6 / rsc**7) * r
            + 91.0 * c12 / rsc**12
            - 28.0 * c6 / rsc**6
        )
    return scale * potential


def _test_gapsys_pair_energy_matches_reference_equation():
    sigma = 0.32
    epsilon = 0.40
    scale = 0.35
    alpha = 0.85
    force = mm.CustomBondForce(
        _gapsys_energy_expression("COVALENT_STERICS_A", mixing=False)
    )
    force.addGlobalParameter("COVALENT_STERICS_A", scale)
    force.addGlobalParameter("GAPSYS_SCALE_LINPOINT_LJ", alpha)
    force.addGlobalParameter("GAPSYS_SIGMA", 0.30)
    force.addPerBondParameter("sigma")
    force.addPerBondParameter("epsilon")
    force.addBond(0, 1, [sigma, epsilon])
    system = mm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)
    system.addForce(force)
    rsc = alpha * ((26.0 / 7.0) * sigma**6 * (1.0 - scale)) ** (1.0 / 6.0)

    boundary_forces = []
    for distance in (0.05, 0.8 * rsc, 0.999999 * rsc, rsc, 1.000001 * rsc, 1.2 * rsc):
        positions = np.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        observed, forces = _energy_forces(
            system, positions * unit.nanometer
        )
        expected = _gapsys_reference_energy(
            distance, sigma, epsilon, scale, alpha
        )
        assert np.isclose(observed, expected, atol=1.0e-8, rtol=1.0e-7)
        assert np.all(np.isfinite(forces))
        if 0.999998 < distance / rsc < 1.000002:
            boundary_forces.append(float(forces[1, 0]))
    assert np.allclose(
        boundary_forces,
        boundary_forces[0],
        atol=1.0e-3,
        rtol=1.0e-4,
    )


def _test_gapsys_reproduces_endpoint_energies_and_forces():
    endpoint_a = _endpoint("a")
    endpoint_b = _endpoint("b")
    hamiltonian = create_softcore_hamiltonian(
        endpoint_a,
        endpoint_b,
        [2],
        [3],
        function="gapsys",
        charge_steps_per_stage=2,
        sterics_steps=3,
    )
    positions = np.asarray(
        [[0, 0, 0], [0.15, 0, 0], [0.28, 0.08, 0], [0.29, -0.09, 0.03], [0.7, 0.4, 0.3]]
    ) * unit.nanometer
    for endpoint, node in ((endpoint_a, 0), (endpoint_b, -1)):
        expected_energy, expected_forces = _energy_forces(endpoint, positions)
        parameters = {
            name: values[node]
            for name, values in hamiltonian.parameter_values.items()
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


def _test_general_linear_path_preserves_endpoints_and_budget():
    endpoint_a = _endpoint("a")
    endpoint_b = _endpoint("b")
    hamiltonian = create_softcore_hamiltonian(
        endpoint_a,
        endpoint_b,
        [2],
        [3],
        total_steps=150,
        path_nodes=[],
        vdw_a=[1.0, 0.0],
        charge_a=[1.0, 0.0],
        segments_per_interval=[30],
    )
    assert hamiltonian.segment_steps == [5] * 30
    assert hamiltonian.resolved_path["vdw_b"] == [0.0, 1.0]
    assert hamiltonian.resolved_path["mapped_charge"] == [0.0, 1.0]
    positions = np.asarray(
        [[0, 0, 0], [0.15, 0, 0], [0.28, 0.08, 0], [0.29, -0.09, 0.03], [0.7, 0.4, 0.3]]
    ) * unit.nanometer
    for endpoint, node in ((endpoint_a, 0), (endpoint_b, -1)):
        expected_energy, expected_forces = _energy_forces(endpoint, positions)
        parameters = {
            name: values[node]
            for name, values in hamiltonian.parameter_values.items()
        }
        observed_energy, observed_forces = _energy_forces(
            hamiltonian.system, positions, parameters
        )
        assert np.isclose(observed_energy, expected_energy, atol=1.0e-5)
        assert np.allclose(observed_forces, expected_forces, atol=1.0e-3)


def _test_general_midpoint_path_keeps_both_vdw_branches_fully_coupled():
    hamiltonian = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        total_steps=150,
        path_nodes=[0.5],
        vdw_a=[1.0, 1.0, 0.0],
        charge_a=[1.0, 0.5, 0.0],
        segments_per_interval=[15, 15],
    )
    midpoint = 15
    assert hamiltonian.parameter_values[STERICS_A_PARAMETER][midpoint] == 1.0
    assert hamiltonian.parameter_values[STERICS_B_PARAMETER][midpoint] == 1.0
    assert hamiltonian.parameter_values["COVALENT_CHARGE_A"][midpoint] == 0.5
    assert hamiltonian.parameter_values["COVALENT_CHARGE_B"][midpoint] == 0.5
    assert hamiltonian.parameter_values["COVALENT_STERICS"][midpoint] == 0.5
    positions = np.asarray(
        [[0, 0, 0], [0.15, 0, 0], [0.28, 0.08, 0], [0.28, 0.08, 0], [0.7, 0.4, 0.3]]
    ) * unit.nanometer
    parameters = {
        name: values[midpoint]
        for name, values in hamiltonian.parameter_values.items()
    }
    energy, forces = _energy_forces(hamiltonian.system, positions, parameters)
    assert np.isfinite(energy)
    assert np.all(np.isfinite(forces))


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


def _test_windowed_work_profile_sums_to_exact_protocol_work():
    hamiltonian = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        function="gapsys",
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
    rows = []

    total, increments = _run_segmented_protocol(
        integrator,
        hamiltonian.segment_steps,
        profile_rows=rows,
        profile_interval_steps=3,
        parameter_values=hamiltonian.parameter_values,
        profile_metadata={
            "phase": "optimizer",
            "environment": "protein",
            "direction": "forward",
            "sample": 1,
        },
    )

    assert rows
    assert rows[-1]["step_end"] == hamiltonian.total_steps
    assert np.isclose(
        sum(row["window_work_kj_per_mol"] for row in rows), total
    )
    assert np.isclose(sum(increments), total)
    assert all(row["delta_lambda"] > 0.0 for row in rows)
    assert np.isclose(rows[-1]["lambda_end"], 1.0)
    del context


def _test_work_profile_merge_replaces_resume_duplicate(tmp_path):
    row = {
        "phase": "optimizer",
        "environment": "protein",
        "direction": "forward",
        "sample": 1,
        "status": "running",
        "failure_message": "",
        "segment": 1,
        "step_start": 0,
        "step_end": 100,
        "lambda_start": 0.0,
        "lambda_end": 0.1,
        "delta_lambda": 0.1,
        "window_work_kj_per_mol": 2.0,
        "cumulative_work_kj_per_mol": 2.0,
        "delta_work_over_delta_lambda_kj_per_mol": 20.0,
    }
    for name in (
        "covalent_charge_a",
        "covalent_charge_b",
        "covalent_mapped_charge",
        "covalent_sterics_a",
        "covalent_sterics_b",
        "covalent_sterics",
    ):
        row[f"{name}_end"] = 0.5
    path = tmp_path / "switch_work_profile.csv"
    _merge_work_profile_rows(path, [row], status="failed", failure_message="first")
    replacement = dict(row)
    replacement["window_work_kj_per_mol"] = 1.5
    _merge_work_profile_rows(path, [replacement], status="completed")

    import csv

    with path.open(newline="") as handle:
        observed = list(csv.DictReader(handle))
    assert len(observed) == 1
    assert observed[0]["status"] == "completed"
    assert float(observed[0]["window_work_kj_per_mol"]) == 1.5


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


def _test_gapsys_endpoint_lrc_corrections_are_finite():
    enabled = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        function="gapsys",
        use_long_range_correction=True,
    )
    disabled = create_softcore_hamiltonian(
        _endpoint("a"),
        _endpoint("b"),
        [2],
        [3],
        function="gapsys",
        use_long_range_correction=False,
    )
    positions = np.asarray(
        [[0, 0, 0], [0.15, 0, 0], [0.28, 0.08, 0], [0.29, -0.09, 0.03], [0.7, 0.4, 0.3]]
    ) * unit.nanometer
    context = mm.Context(_endpoint("a"), mm.VerletIntegrator(0.001))
    context.setPositions(positions)
    state = context.getState(getPositions=True)
    del context
    evaluator = _EndpointLRCCorrectionEvaluator(disabled, enabled)

    corrections = [
        evaluator.correction(state, enabled.parameter_values, node)
        for node in (0, -1)
    ]

    assert np.all(np.isfinite(corrections))
    evaluator.close()


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
