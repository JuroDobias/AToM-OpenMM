from __future__ import annotations

import math

import openmm as mm
from openmm.unit import MOLAR_GAS_CONSTANT_R, kilojoules_per_mole

from atom_openmm.ommworker import OMMWorkerATMSync

# The Eold/parameter-update/Enew protocol-work pattern follows the MIT-licensed
# openmmtools AlchemicalNonequilibriumLangevinIntegrator design.
# https://github.com/choderalab/openmmtools/blob/main/openmmtools/integrators.py


def _piecewise_expression(values, steps_per_segment):
    values = [float(value) for value in values]
    expression = f"({values[0]:.17g})"
    for segment, (start, end) in enumerate(zip(values[:-1], values[1:])):
        delta = end - start
        if delta == 0.0:
            continue
        progress = (
            f"min(1,max(0,(neq_step-{segment * steps_per_segment})/"
            f"{float(steps_per_segment):.17g}))"
        )
        expression += f"+({delta:.17g})*({progress})"
    return expression


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
    ):
        super().__init__(timestep)
        intervals = tuple(sorted({int(value) for value in (work_sample_intervals or ())}))
        if any(value <= 0 for value in intervals):
            raise ValueError("work_sample_intervals must contain positive integers")
        self._work_sample_intervals = intervals
        self._temperature = temperature
        self.addGlobalVariable("kT", MOLAR_GAS_CONSTANT_R * temperature)
        self.addGlobalVariable("protocol_work", 0.0)
        self.addGlobalVariable("Eold", 0.0)
        self.addGlobalVariable("Enew", 0.0)
        self.addGlobalVariable("neq_step", 0.0)
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
            self.addComputeGlobal(name, _piecewise_expression(values, steps_per_segment))
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
            self.switch_integrators[name] = ATMNonequilibriumLangevinIntegrator(
                temperature=temperature,
                collision_rate=collision_rate,
                timestep=timestep,
                parameter_values=_context_parameter_values(self.ommsystem, schedule),
                steps_per_segment=self._steps_per_segment,
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

    def get_chkpt(self):
        self.select_equilibrium()
        return super().get_chkpt()

    def set_chkpt(self, chkpt):
        self.select_equilibrium()
        return super().set_chkpt(chkpt)
