"""Reusable synchronous REST2 replica exchange for fixed ATM states."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit

from atom_openmm.rest2 import set_rest2_scale


R_KJ_MOL_K = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
    unit.kilojoule_per_mole / unit.kelvin
)


def exchange_log_acceptance(beta, u_ii, u_ij, u_jj, u_ji):
    return -float(beta) * ((u_ij + u_ji) - (u_ii + u_jj))


def set_atm_state(context, ommsystem, state):
    """Apply one fixed ATM state directly to an OpenMM Context."""
    atmforce = ommsystem.atmforce
    context.setParameter(atmforce.Lambda1(), state["lambda1"])
    context.setParameter(atmforce.Lambda2(), state["lambda2"])
    if ommsystem.multisoftplus:
        context.setParameter("Lambda3", state["lambda3"])
        context.setParameter("Uh1", state["uh1"] / unit.kilojoule_per_mole)
    context.setParameter(atmforce.Alpha(), state["alpha"] * unit.kilojoule_per_mole)
    context.setParameter(atmforce.Uh(), state["uh"] / unit.kilojoule_per_mole)
    context.setParameter(atmforce.W0(), state["w0"] / unit.kilojoule_per_mole)
    context.setParameter(atmforce.Direction(), state["atmdirection"])
    context.setParameter(atmforce.Umax(), state[atmforce.Umax()] / unit.kilojoule_per_mole)
    context.setParameter(atmforce.Ubcore(), state[atmforce.Ubcore()] / unit.kilojoule_per_mole)
    context.setParameter(atmforce.Acore(), state[atmforce.Acore()])
    context.setParameter("UOffset", state["uoffset"] / unit.kilojoule_per_mole)


def _energy_kj(context):
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )


def _append_csv(path, fields, values):
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(fields)
        writer.writerow(values)


class REST2ExchangeSampler:
    """One resident REST2 ladder with independent checkpoint banks per ATM state."""

    def __init__(
        self,
        *,
        system,
        topology,
        base_integrator,
        ommsystem=None,
        rest2_system=None,
        state_files,
        atm_states=None,
        config,
        platform,
        platform_properties,
        output_dir="neqti_rest2",
        resume=True,
        random_seed=2026,
        logger=None,
    ):
        self.system = system
        self.topology = topology
        self.ommsystem = ommsystem
        self.rest2_system = rest2_system or (
            None if ommsystem is None else ommsystem.rest2_system
        )
        self.state_files = {key: str(value) for key, value in state_files.items()}
        self.atm_states = atm_states
        self.config = config
        self.output_dir = Path(output_dir)
        self.resume = bool(resume)
        self.logger = logger
        self.exchange_interval = int(config["exchange_interval_steps"])
        self.checkpoint_interval = int(config.get("checkpoint_interval_cycles", 10))
        self.temperatures = [float(value) for value in config["effective_temperatures_k"]]
        self.physical_temperature = float(self.temperatures[0])
        self.scales = [self.physical_temperature / value for value in self.temperatures]
        self.beta = 1.0 / (R_KJ_MOL_K * self.physical_temperature)
        self.rng = np.random.default_rng(int(random_seed) + 71000)
        self.contexts = []
        self.integrators = []
        self.assignments = list(range(len(self.scales)))
        self.attempts = np.zeros(len(self.scales) - 1, dtype=int)
        self.accepts = np.zeros(len(self.scales) - 1, dtype=int)
        self.round_trips = np.zeros(len(self.scales), dtype=int)
        self.trip_phase = np.asarray([1, *([0] * (len(self.scales) - 1))], dtype=int)
        self.active_ensemble = None
        if not self.resume and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for index, scale in enumerate(self.scales):
            integrator = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(base_integrator))
            if hasattr(integrator, "setRandomNumberSeed"):
                integrator.setRandomNumberSeed(int(random_seed) + 72000 + index)
            context = mm.Context(system, integrator, platform, platform_properties)
            set_rest2_scale(context, scale, self.rest2_system)
            self.integrators.append(integrator)
            self.contexts.append(context)

    def _directory(self, ensemble):
        return self.output_dir / str(ensemble)

    def _metadata_path(self, ensemble):
        return self._directory(ensemble) / "state.json"

    def has_bank(self, ensemble):
        return self._metadata_path(ensemble).exists()

    def _initialize_bank(self, ensemble):
        source = mm.XmlSerializer.deserialize(Path(self.state_files[ensemble]).read_text())
        temperature = (
            self.atm_states[ensemble]["temperature"]
            if self.atm_states is not None else self.physical_temperature * unit.kelvin
        )
        for index, context in enumerate(self.contexts):
            if self.atm_states is not None:
                context.setState(source)
                set_atm_state(context, self.ommsystem, self.atm_states[ensemble])
            else:
                box_vectors = source.getPeriodicBoxVectors()
                if box_vectors is not None:
                    context.setPeriodicBoxVectors(*box_vectors)
                context.setPositions(source.getPositions())
                try:
                    velocities = source.getVelocities()
                    if velocities is not None:
                        context.setVelocities(velocities)
                except Exception:
                    pass
            set_rest2_scale(context, self.scales[index], self.rest2_system)
            ensemble_offset = {"a": 1000, "m": 2000, "b": 3000}.get(ensemble, 4000)
            context.setVelocitiesToTemperature(temperature, 73000 + ensemble_offset + index)
        self.assignments = list(range(len(self.scales)))
        self.attempts = np.zeros(len(self.scales) - 1, dtype=int)
        self.accepts = np.zeros(len(self.scales) - 1, dtype=int)
        self.round_trips = np.zeros(len(self.scales), dtype=int)
        self.trip_phase = np.asarray([1, *([0] * (len(self.scales) - 1))], dtype=int)
        self.cycle = 0

    def _load_bank(self, ensemble):
        directory = self._directory(ensemble)
        metadata = json.loads(self._metadata_path(ensemble).read_text())
        for index, context in enumerate(self.contexts):
            context.loadCheckpoint((directory / f"walker_{index}.chk").read_bytes())
        self.assignments = [int(value) for value in metadata["assignments"]]
        self.attempts = np.asarray(metadata["attempts"], dtype=int)
        self.accepts = np.asarray(metadata["accepts"], dtype=int)
        self.round_trips = np.asarray(metadata.get("round_trips", [0] * len(self.scales)), dtype=int)
        self.trip_phase = np.asarray(metadata.get("trip_phase", [0] * len(self.scales)), dtype=int)
        self.cycle = int(metadata["cycle"])
        self.rng.bit_generator.state = metadata["rng_state"]
        for walker, context in enumerate(self.contexts):
            if self.atm_states is not None:
                set_atm_state(context, self.ommsystem, self.atm_states[ensemble])
            set_rest2_scale(
                context, self.scales[self.assignments[walker]], self.rest2_system
            )

    def activate(self, ensemble):
        if self.active_ensemble == ensemble:
            return
        if self.active_ensemble is not None:
            self.save_bank()
        if self._metadata_path(ensemble).exists():
            self._load_bank(ensemble)
            if self.logger:
                self.logger.info("REST2 resumed %s at exchange cycle %d", ensemble, self.cycle)
        else:
            self._initialize_bank(ensemble)
            self.save_bank(ensemble)
        self.active_ensemble = ensemble

    def save_bank(self, ensemble=None):
        ensemble = ensemble or self.active_ensemble
        if ensemble is None:
            return
        directory = self._directory(ensemble)
        directory.mkdir(parents=True, exist_ok=True)
        for index, context in enumerate(self.contexts):
            temporary = directory / f"walker_{index}.chk.tmp"
            temporary.write_bytes(context.createCheckpoint())
            os.replace(temporary, directory / f"walker_{index}.chk")
        payload = {
            "schema_version": 1,
            "ensemble": ensemble,
            "cycle": self.cycle,
            "assignments": self.assignments,
            "attempts": self.attempts.tolist(),
            "accepts": self.accepts.tolist(),
            "round_trips": self.round_trips.tolist(),
            "trip_phase": self.trip_phase.tolist(),
            "rng_state": self.rng.bit_generator.state,
            "effective_temperatures_k": self.temperatures,
            "exchange_interval_steps": self.exchange_interval,
        }
        temporary = directory / "state.json.tmp"
        temporary.write_text(json.dumps(payload) + "\n")
        os.replace(temporary, self._metadata_path(ensemble))

    def _attempt_exchange(self, lower_state):
        upper_state = lower_state + 1
        walker_i = self.assignments.index(lower_state)
        walker_j = self.assignments.index(upper_state)
        context_i = self.contexts[walker_i]
        context_j = self.contexts[walker_j]
        u_ii = _energy_kj(context_i)
        u_jj = _energy_kj(context_j)
        set_rest2_scale(context_i, self.scales[upper_state], self.rest2_system)
        set_rest2_scale(context_j, self.scales[lower_state], self.rest2_system)
        u_ij = _energy_kj(context_i)
        u_ji = _energy_kj(context_j)
        log_acceptance = exchange_log_acceptance(self.beta, u_ii, u_ij, u_jj, u_ji)
        accepted = math.log(self.rng.random()) < min(0.0, log_acceptance)
        self.attempts[lower_state] += 1
        if accepted:
            self.accepts[lower_state] += 1
            self.assignments[walker_i], self.assignments[walker_j] = upper_state, lower_state
        else:
            set_rest2_scale(context_i, self.scales[lower_state], self.rest2_system)
            set_rest2_scale(context_j, self.scales[upper_state], self.rest2_system)
        _append_csv(
            self._directory(self.active_ensemble) / "exchanges.csv",
            ["cycle", "lower_state", "upper_state", "walker_i", "walker_j", "accepted", "log_acceptance"],
            [self.cycle, lower_state, upper_state, walker_i, walker_j, int(accepted), log_acceptance],
        )

    def run_steps(self, ensemble, steps, label=None):
        self.activate(ensemble)
        steps = int(steps)
        if steps % self.exchange_interval:
            raise ValueError("REST2 sampling steps must be divisible by exchange_interval_steps")
        cycles = steps // self.exchange_interval
        for _ in range(cycles):
            for integrator in self.integrators:
                integrator.step(self.exchange_interval)
            parity = self.cycle % 2
            self.cycle += 1
            for lower_state in range(parity, len(self.scales) - 1, 2):
                self._attempt_exchange(lower_state)
            for walker, state_index in enumerate(self.assignments):
                if state_index == 0:
                    if self.trip_phase[walker] == 2:
                        self.round_trips[walker] += 1
                    self.trip_phase[walker] = 1
                elif state_index == len(self.scales) - 1 and self.trip_phase[walker] == 1:
                    self.trip_phase[walker] = 2
            _append_csv(
                self._directory(ensemble) / "state_trace.csv",
                ["cycle", *[f"walker_{i}" for i in range(len(self.scales))]],
                [self.cycle, *self.assignments],
            )
            if self.cycle % self.checkpoint_interval == 0:
                self.save_bank()
        self.save_bank()
        if self.logger:
            rates = np.divide(
                self.accepts, self.attempts,
                out=np.zeros_like(self.accepts, dtype=float), where=self.attempts > 0,
            )
            self.logger.info(
                "%s REST2 complete: %d steps per replica, cycle %d, neighbor acceptance %s",
                label or ensemble, steps, self.cycle, np.round(rates, 3).tolist(),
            )

    def physical_state(self, ensemble):
        self.activate(ensemble)
        walker = self.assignments.index(0)
        return self.contexts[walker].getState(
            getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True
        )

    def summary(self):
        rates = np.divide(
            self.accepts, self.attempts,
            out=np.zeros_like(self.accepts, dtype=float), where=self.attempts > 0,
        )
        return {
            "effective_temperatures_k": self.temperatures,
            "acceptance_rates": rates.tolist(),
            "attempts": self.attempts.tolist(),
            "accepts": self.accepts.tolist(),
            "round_trips": self.round_trips.tolist(),
        }

    def close(self):
        self.save_bank()
        self.contexts.clear()
        self.integrators.clear()
