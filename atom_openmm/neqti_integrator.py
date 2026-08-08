from __future__ import annotations

import math

import openmm as mm
from openmm.unit import MOLAR_GAS_CONSTANT_R, kilojoules_per_mole

from atom_openmm.ommworker import OMMWorkerATMSync

# The Eold/parameter-update/Enew protocol-work pattern follows the MIT-licensed
# openmmtools AlchemicalNonequilibriumLangevinIntegrator design.
# https://github.com/choderalab/openmmtools/blob/main/openmmtools/integrators.py


_AMBER_RECIPROCAL_TRANSFORMS = {
    "COVALENT_RECIPROCAL_A_CHARGE": (
        "sqrt(max(0,COVALENT_CHARGE_A^3*(10+COVALENT_CHARGE_A*"
        "(-15+6*COVALENT_CHARGE_A))))-1"
    ),
    "COVALENT_RECIPROCAL_B_CHARGE": (
        "sqrt(max(0,COVALENT_CHARGE_B^3*(10+COVALENT_CHARGE_B*"
        "(-15+6*COVALENT_CHARGE_B))))-1"
    ),
    "COVALENT_RECIPROCAL_A_EXCEPTION": (
        "COVALENT_CHARGE_A^3*(10+COVALENT_CHARGE_A*"
        "(-15+6*COVALENT_CHARGE_A))-1"
    ),
    "COVALENT_RECIPROCAL_B_EXCEPTION": (
        "COVALENT_CHARGE_B^3*(10+COVALENT_CHARGE_B*"
        "(-15+6*COVALENT_CHARGE_B))-1"
    ),
}


def _apply_amber_reciprocal_transforms(values):
    for label in ("A", "B"):
        charge = values.get(f"COVALENT_CHARGE_{label}")
        if charge is None:
            continue
        weight = charge**3 * (10.0 + charge * (-15.0 + 6.0 * charge))
        charge_name = f"COVALENT_RECIPROCAL_{label}_CHARGE"
        exception_name = f"COVALENT_RECIPROCAL_{label}_EXCEPTION"
        if charge_name in values:
            values[charge_name] = math.sqrt(max(0.0, weight)) - 1.0
        if exception_name in values:
            values[exception_name] = weight - 1.0
    return values


def normalize_segment_steps(steps_per_segment, nsegments, *, label="steps_per_segment"):
    if isinstance(steps_per_segment, int):
        steps = [int(steps_per_segment)] * nsegments
    else:
        steps = [int(step) for step in steps_per_segment]
    if len(steps) != nsegments:
        raise ValueError(f"{label} must contain {nsegments} values")
    if any(step < 1 for step in steps):
        raise ValueError(f"{label} values must be positive")
    return steps


def normalize_segments_per_stage(segments_per_stage, nsegments):
    if segments_per_stage is None:
        return [1] * nsegments
    groups = [int(value) for value in segments_per_stage]
    if not groups or any(value < 1 for value in groups):
        raise ValueError("segments_per_stage must contain positive integers")
    if sum(groups) != nsegments:
        raise ValueError(f"segments_per_stage must account for {nsegments} segments")
    return groups


def _normalize_stage_interpolation(value):
    value = str(value).lower()
    if value not in {"linear", "smoothstep2"}:
        raise ValueError("stage_interpolation must be 'linear' or 'smoothstep2'")
    return value


def stage_interpolation_fraction(progress, interpolation):
    progress = float(progress)
    interpolation = _normalize_stage_interpolation(interpolation)
    if interpolation == "linear":
        return progress
    return progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))


def _stage_progress_expression(boundary_names, first_segment, segment_count):
    terms = []
    for segment in range(first_segment, first_segment + segment_count):
        start = "0" if segment == 0 else boundary_names[segment - 1]
        end = boundary_names[segment]
        terms.append(f"min(1,max(0,(neq_step-{start})/({end}-{start})))")
    return f"(({'+'.join(terms)})/{segment_count})"


def _stage_curve_expression(progress, interpolation):
    if interpolation == "linear":
        return progress
    return f"(({progress})^3*(10+({progress})*(-15+6*({progress}))))"


def _piecewise_expression(values, boundary_names, segments_per_stage=None, stage_interpolation="linear"):
    values = [float(value) for value in values]
    interpolation = _normalize_stage_interpolation(stage_interpolation)
    groups = normalize_segments_per_stage(segments_per_stage, len(boundary_names))
    expression = f"({values[0]:.17g})"
    first_segment = 0
    for segment_count in groups:
        last_segment = first_segment + segment_count
        delta = values[last_segment] - values[first_segment]
        if delta != 0.0:
            progress = _stage_progress_expression(
                boundary_names, first_segment, segment_count
            )
            curve = _stage_curve_expression(progress, interpolation)
            expression += f"+({delta:.17g})*({curve})"
        first_segment = last_segment
    return expression


def parameter_values_at_step(
    parameter_values,
    segment_steps,
    segment,
    local_step,
    *,
    segments_per_stage=None,
    stage_interpolation="linear",
):
    groups = normalize_segments_per_stage(segments_per_stage, len(segment_steps))
    interpolation = _normalize_stage_interpolation(stage_interpolation)
    first_segment = 0
    for segment_count in groups:
        last_segment = first_segment + segment_count
        if segment < last_segment:
            progress = (
                segment - first_segment
                + float(local_step) / float(segment_steps[segment])
            ) / float(segment_count)
            fraction = stage_interpolation_fraction(progress, interpolation)
            values = {
                name: float(values[first_segment])
                + fraction
                * (float(values[last_segment]) - float(values[first_segment]))
                for name, values in parameter_values.items()
            }
            return _apply_amber_reciprocal_transforms(values)
        first_segment = last_segment
    raise IndexError("segment index is outside the switching schedule")


def _set_shared_random_seed(integrators, random_seed):
    for integrator in integrators:
        if hasattr(integrator, "setRandomNumberSeed"):
            integrator.setRandomNumberSeed(int(random_seed))


class ATMNonequilibriumLangevinIntegrator(mm.CustomIntegrator):
    """BAOAB-style ATM switching with on-device protocol-work accumulation."""

    def __init__(
        self,
        *,
        temperature,
        collision_rate,
        timestep,
        parameter_values,
        steps_per_segment,
        random_seed=None,
        work_sample_intervals=None,
        segments_per_stage=None,
        stage_interpolation="linear",
    ):
        super().__init__(timestep)
        intervals = tuple(sorted({int(value) for value in (work_sample_intervals or ())}))
        if any(value <= 0 for value in intervals):
            raise ValueError("work_sample_intervals must contain positive integers")
        self._work_sample_intervals = intervals
        nsegments = len(next(iter(parameter_values.values()))) - 1
        self._segment_steps = normalize_segment_steps(steps_per_segment, nsegments)
        self._segments_per_stage = normalize_segments_per_stage(
            segments_per_stage, nsegments
        )
        self._stage_interpolation = _normalize_stage_interpolation(stage_interpolation)
        self._segment_boundary_names = [f"neq_segment_end_{index}" for index in range(nsegments)]
        self._temperature = temperature
        self.addGlobalVariable("kT", MOLAR_GAS_CONSTANT_R * temperature)
        self.addGlobalVariable("protocol_work", 0.0)
        self.addGlobalVariable("Eold", 0.0)
        self.addGlobalVariable("Enew", 0.0)
        self.addGlobalVariable("neq_step", 0.0)
        cumulative = 0
        for name, steps in zip(self._segment_boundary_names, self._segment_steps):
            cumulative += steps
            self.addGlobalVariable(name, float(cumulative))
        for interval in intervals:
            self.addGlobalVariable(f"protocol_work_interval_{interval}", 0.0)
        self.addPerDofVariable("x1", 0.0)

        gamma_dt = collision_rate * timestep
        a = math.exp(-float(gamma_dt))
        b = math.sqrt(1.0 - a * a)

        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self._add_constrained_drift("0.5*dt")

        self.addComputeGlobal("Eold", "energy")
        self.addComputeGlobal("neq_step", "neq_step+1")
        for name, values in parameter_values.items():
            if len(values) != nsegments + 1:
                raise ValueError("all switching parameter schedules must have the same length")
            expression = _AMBER_RECIPROCAL_TRANSFORMS.get(name)
            if expression is None:
                expression = _piecewise_expression(
                    values,
                    self._segment_boundary_names,
                    self._segments_per_stage,
                    self._stage_interpolation,
                )
            self.addComputeGlobal(name, expression)
        self.addComputeGlobal("Enew", "energy")
        self.addComputeGlobal("protocol_work", "protocol_work+Enew-Eold")
        for interval in intervals:
            sample = f"delta(neq_step-{interval}*floor(neq_step/{interval}))"
            self.addComputeGlobal(
                f"protocol_work_interval_{interval}",
                f"protocol_work_interval_{interval}+{interval}*({sample})*(Enew-Eold)",
            )

        self.addComputePerDof("v", f"{a:.17g}*v+{b:.17g}*sqrt(kT/m)*gaussian")
        self._add_constrained_drift("0.5*dt")
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addConstrainVelocities()
        self.setConstraintTolerance(0.00001)
        if random_seed is not None:
            self.setRandomNumberSeed(int(random_seed))

    def _add_constrained_drift(self, step):
        self.addComputePerDof("x1", "x")
        self.addComputePerDof("x", f"x+({step})*v")
        self.addConstrainPositions()
        self.addComputePerDof("v", f"(x-x1)/({step})")

    def setTemperature(self, temperature):
        self._temperature = temperature
        self.setGlobalVariableByName("kT", MOLAR_GAS_CONSTANT_R * temperature)

    def getTemperature(self):
        return self._temperature

    def reset_protocol(self):
        self.setGlobalVariableByName("protocol_work", 0.0)
        self.setGlobalVariableByName("Eold", 0.0)
        self.setGlobalVariableByName("Enew", 0.0)
        self.setGlobalVariableByName("neq_step", 0.0)
        for interval in self._work_sample_intervals:
            self.setGlobalVariableByName(f"protocol_work_interval_{interval}", 0.0)

    def get_protocol_work(self):
        return self.getGlobalVariableByName("protocol_work") * kilojoules_per_mole

    def get_sampled_protocol_work(self):
        return {
            interval: self.getGlobalVariableByName(f"protocol_work_interval_{interval}")
            * kilojoules_per_mole
            for interval in self._work_sample_intervals
        }

    def set_segment_steps(self, steps_per_segment):
        steps = normalize_segment_steps(
            steps_per_segment, len(self._segment_steps), label="segment_steps"
        )
        cumulative = 0
        for name, value in zip(self._segment_boundary_names, steps):
            cumulative += value
            self.setGlobalVariableByName(name, float(cumulative))
        self._segment_steps = steps

    def get_segment_steps(self):
        return list(self._segment_steps)

    def get_segments_per_stage(self):
        return list(self._segments_per_stage)

    def get_stage_interpolation(self):
        return self._stage_interpolation


def _context_parameter_values(ommsystem, schedule):
    atmforce = ommsystem.atmforce
    values = {
        atmforce.Lambda1(): [state["lambda1"] for state in schedule],
        atmforce.Lambda2(): [state["lambda2"] for state in schedule],
        atmforce.Alpha(): [state["alpha"] * kilojoules_per_mole for state in schedule],
        atmforce.Uh(): [state["uh"] / kilojoules_per_mole for state in schedule],
        atmforce.W0(): [state["w0"] / kilojoules_per_mole for state in schedule],
        atmforce.Umax(): [state[atmforce.Umax()] / kilojoules_per_mole for state in schedule],
        atmforce.Ubcore(): [state[atmforce.Ubcore()] / kilojoules_per_mole for state in schedule],
        atmforce.Acore(): [state[atmforce.Acore()] for state in schedule],
        "UOffset": [state["uoffset"] / kilojoules_per_mole for state in schedule],
    }
    if ommsystem.multisoftplus:
        values["Lambda3"] = [state["lambda3"] for state in schedule]
        values["Uh1"] = [state["uh1"] / kilojoules_per_mole for state in schedule]
    return values


class OMMWorkerATMNEQTI(OMMWorkerATMSync):
    def __init__(self, *args, switch_schedules, steps_per_segment, random_seed, work_sample_intervals=None, **kwargs):
        self._switch_schedules = switch_schedules
        self._steps_per_segment = steps_per_segment
        self._random_seed = random_seed
        self._work_sample_intervals = work_sample_intervals
        super().__init__(*args, **kwargs)

    def _openmm_worker_body(self):
        self.ommsystem.create_system()
        self.system = self.ommsystem.system
        self.topology = self.ommsystem.topology
        self.positions = self.ommsystem.positions
        self.boxvectors = self.ommsystem.boxvectors

        self.equilibrium_integrator = self.ommsystem.integrator
        if isinstance(self.equilibrium_integrator, mm.DrudeLangevinIntegrator):
            raise ValueError(
                "The custom NEQTI switching integrator does not support Drude systems; "
                "use workflow.neqti.switch_integrator: python"
            )
        temperature = self.equilibrium_integrator.getTemperature()
        collision_rate = self.ommsystem.frictionCoeff
        timestep = self.ommsystem.MDstepsize
        self.switch_integrators = {}
        for name, schedule in self._switch_schedules.items():
            steps_per_segment = (
                self._steps_per_segment[name]
                if isinstance(self._steps_per_segment, dict)
                else self._steps_per_segment
            )
            self.switch_integrators[name] = ATMNonequilibriumLangevinIntegrator(
                temperature=temperature,
                collision_rate=collision_rate,
                timestep=timestep,
                parameter_values=_context_parameter_values(self.ommsystem, schedule),
                steps_per_segment=steps_per_segment,
                random_seed=self._random_seed,
                work_sample_intervals=self._work_sample_intervals,
            )
        _set_shared_random_seed([self.equilibrium_integrator, *self.switch_integrators.values()], self._random_seed)
        self.compound_integrator = mm.CompoundIntegrator()
        self.compound_integrator.addIntegrator(self.equilibrium_integrator)
        self._integrator_indices = {}
        for name, integrator in self.switch_integrators.items():
            self._integrator_indices[name] = self.compound_integrator.getNumIntegrators()
            self.compound_integrator.addIntegrator(integrator)
        self.compound_integrator.setCurrentIntegrator(0)
        self.integrator = self.compound_integrator
        self.ommsystem.integrator = self.compound_integrator

    def set_state(self, par):
        integrator = self.integrator
        self.integrator = self.equilibrium_integrator
        try:
            self._worker_setstate(par)
        finally:
            self.integrator = integrator
        for switch_integrator in self.switch_integrators.values():
            switch_integrator.setTemperature(par["temperature"])

    def select_equilibrium(self):
        self.compound_integrator.setCurrentIntegrator(0)

    def begin_switch(self, switch_name, start_state):
        self.select_equilibrium()
        self.set_state(start_state)
        integrator = self.switch_integrators[switch_name]
        integrator.reset_protocol()
        self.compound_integrator.setCurrentIntegrator(self._integrator_indices[switch_name])
        return integrator

    def end_switch(self):
        self.select_equilibrium()

    def set_switch_segment_steps(self, switch_name, steps):
        self.switch_integrators[switch_name].set_segment_steps(steps)

    def get_switch_segment_steps(self, switch_name):
        return self.switch_integrators[switch_name].get_segment_steps()

    def get_chkpt(self):
        self.select_equilibrium()
        return super().get_chkpt()

    def set_chkpt(self, chkpt):
        self.select_equilibrium()
        return super().set_chkpt(chkpt)
