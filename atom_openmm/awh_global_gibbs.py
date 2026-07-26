"""Device-side REST2 scans and analytical ATM energies for global AWH moves."""

from __future__ import annotations

import math

import numpy as np
import openmm as mm
from openmm import unit

from atom_openmm.atm_energy import reconstruct_atm_energies
from atom_openmm.ommworker import OMMWorkerATMSync
from atom_openmm.rest2_exchange import _atm_parameter_values


def _energy_value(value):
    if hasattr(value, "value_in_unit"):
        return value.value_in_unit(unit.kilojoule_per_mole)
    return float(value)


def _node_parameter_values(ommsystem, atm_states, node):
    values = _atm_parameter_values(ommsystem, atm_states[node["atm_state"]])
    if ommsystem.rest2_system is not None:
        for region, (scale_name, sqrt_name) in ommsystem.rest2_system.parameters.items():
            scale = float(node["rest2"].get(region, 1.0))
            values[scale_name] = scale
            values[sqrt_name] = math.sqrt(scale)
    return values


class AWHREST2EnergyScanIntegrator(mm.CustomIntegrator):
    """Evaluate REST2 hot nodes and one physical reference without dynamics."""

    def __init__(self, ommsystem, atm_states, graph, reference_atm_state=0):
        super().__init__(0.0)
        self._energy_variables = {}
        reference_node = {
            "atm_state": int(reference_atm_state),
            "rest2": {"a": 1.0, "b": 1.0},
        }
        scan_nodes = [
            (index, node)
            for index, node in enumerate(graph)
            if node["kind"].startswith("rest2_")
        ]
        for index, node in scan_nodes:
            for name, value in _node_parameter_values(ommsystem, atm_states, node).items():
                self.addComputeGlobal(name, f"{float(value):.17g}")
            variable = f"awh_scan_energy_{index}"
            self.addGlobalVariable(variable, 0.0)
            self.addComputeGlobal(variable, "energy")
            self._energy_variables[index] = variable

        for name, value in _node_parameter_values(
            ommsystem, atm_states, reference_node
        ).items():
            self.addComputeGlobal(name, f"{float(value):.17g}")
        self.addGlobalVariable("awh_reference_energy", 0.0)
        self.addComputeGlobal("awh_reference_energy", "energy")

    def scanned_energies(self):
        return {
            index: self.getGlobalVariableByName(variable)
            for index, variable in self._energy_variables.items()
        }

    def reference_energy(self):
        return self.getGlobalVariableByName("awh_reference_energy")


class OMMWorkerAWHGlobalGibbs(OMMWorkerATMSync):
    """Synchronous ATM worker with a resident device-side AWH energy scanner."""

    def __init__(self, *args, atm_states, graph, **kwargs):
        self._awh_atm_states = atm_states
        self._awh_graph = graph
        super().__init__(*args, **kwargs)

    def _openmm_worker_body(self):
        self.ommsystem.create_system()
        self.system = self.ommsystem.system
        self.topology = self.ommsystem.topology
        self.positions = self.ommsystem.positions
        self.boxvectors = self.ommsystem.boxvectors
        self.equilibrium_integrator = self.ommsystem.integrator
        self.scan_integrator = AWHREST2EnergyScanIntegrator(
            self.ommsystem,
            self._awh_atm_states,
            self._awh_graph,
        )
        self.compound_integrator = mm.CompoundIntegrator()
        self.compound_integrator.addIntegrator(self.equilibrium_integrator)
        self.compound_integrator.addIntegrator(self.scan_integrator)
        self.compound_integrator.setCurrentIntegrator(0)
        self.integrator = self.compound_integrator
        self.ommsystem.integrator = self.compound_integrator

    def select_equilibrium(self):
        self.compound_integrator.setCurrentIntegrator(0)

    def run_energy_scan(self):
        self.compound_integrator.setCurrentIntegrator(1)
        try:
            self.compound_integrator.step(1)
        finally:
            self.select_equilibrium()

    def all_graph_energies(self):
        self.run_energy_scan()
        u1, u0, bias = self.ommsystem.atmforce.getPerturbationEnergy(self.context)
        physical = reconstruct_atm_energies(
            self._awh_atm_states,
            reference_total_energy=self.scan_integrator.reference_energy(),
            reference_bias_energy=_energy_value(bias),
            reference_direction=self._awh_atm_states[0]["atmdirection"],
            u0=_energy_value(u0),
            u1=_energy_value(u1),
            multisoftplus=self.ommsystem.multisoftplus,
        )
        scanned = self.scan_integrator.scanned_energies()
        result = np.empty(len(self._awh_graph), dtype=float)
        for index, node in enumerate(self._awh_graph):
            if index in scanned:
                result[index] = scanned[index]
            else:
                result[index] = physical[node["atm_state"]]
        return result

    def get_chkpt(self):
        self.select_equilibrium()
        return super().get_chkpt()

    def set_chkpt(self, chkpt):
        self.select_equilibrium()
        return super().set_chkpt(chkpt)
