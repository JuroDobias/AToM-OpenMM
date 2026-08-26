"""Reusable synchronous REST2 replica exchange for fixed ATM states."""

from __future__ import annotations

import csv
import base64
import json
import math
import os
import pickle
import random
import shutil
import time
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit
import openmm.app as app
from openmm.app import DCDFile

from atom_openmm.rest2 import set_rest2_scale
from atom_openmm.rest2_process import REST2ReplicaProcess


R_KJ_MOL_K = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
    unit.kilojoule_per_mole / unit.kelvin
)


def normalize_rest2_sampler_backend(config):
    """Return the canonical REST2 sampler backend name."""
    backend = str(config.get("sampler_backend", "custom")).strip().lower()
    backend = {"native": "openmm_native", "openmm": "openmm_native"}.get(
        backend, backend
    )
    if backend not in {"custom", "openmm_native"}:
        raise ValueError(
            "REST2 sampler_backend must be 'custom' or 'openmm_native'"
        )
    return backend


def create_rest2_exchange_sampler(**kwargs):
    """Create the configured production REST2 sampler."""
    backend = normalize_rest2_sampler_backend(kwargs["config"])
    if backend == "openmm_native":
        return OpenMMREST2ExchangeSampler(**kwargs)
    return REST2ExchangeSampler(**kwargs)


def _bank_backend(metadata):
    # Banks created before backend provenance was added are custom-sampler banks.
    return str(metadata.get("sampler_backend", "custom"))


def _validate_bank_backend(path, expected):
    metadata = json.loads(Path(path).read_text())
    actual = _bank_backend(metadata)
    if actual != expected:
        raise ValueError(
            f"REST2 bank {path} was created by sampler_backend={actual}; "
            f"it cannot be resumed with sampler_backend={expected}"
        )
    return metadata


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


def _atm_parameter_values(ommsystem, state):
    atmforce = ommsystem.atmforce
    values = {
        atmforce.Lambda1(): state["lambda1"],
        atmforce.Lambda2(): state["lambda2"],
        atmforce.Alpha(): state["alpha"] * unit.kilojoule_per_mole,
        atmforce.Uh(): state["uh"] / unit.kilojoule_per_mole,
        atmforce.W0(): state["w0"] / unit.kilojoule_per_mole,
        atmforce.Direction(): state["atmdirection"],
        atmforce.Umax(): state[atmforce.Umax()] / unit.kilojoule_per_mole,
        atmforce.Ubcore(): state[atmforce.Ubcore()] / unit.kilojoule_per_mole,
        atmforce.Acore(): state[atmforce.Acore()],
        "UOffset": state["uoffset"] / unit.kilojoule_per_mole,
    }
    if ommsystem.multisoftplus:
        values["Lambda3"] = state["lambda3"]
        values["Uh1"] = state["uh1"] / unit.kilojoule_per_mole
    return {name: float(value) for name, value in values.items()}


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


def observed_transition_diagnostics(state_trace, replicas):
    """Summarize observed native-sampler state changes from a walker trace."""
    path = Path(state_trace)
    counts = np.zeros(max(0, replicas - 1), dtype=int)
    total = 0
    rows = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    columns = [f"walker_{index}" for index in range(replicas)]
    trace = [[int(row[name]) for name in columns] for row in rows]
    for previous, current in zip(trace[:-1], trace[1:]):
        for old_state, new_state in zip(previous, current):
            if old_state == new_state:
                continue
            total += 1
            lower, upper = sorted((old_state, new_state))
            if upper - lower == 1:
                counts[lower] += 1
    denominator = max(1, (len(trace) - 1) * replicas)
    return {
        "metric": "observed_state_transitions",
        "total_state_changes": int(total),
        "neighbor_transition_counts": counts.tolist(),
        "neighbor_transitions_per_replica_iteration": (counts / denominator).tolist(),
        "note": (
            "OpenMM's global exchange sampler does not expose proposal "
            "acceptance counters."
        ),
    }


def _truncate_csv_after_cycle(path, cycle):
    path = Path(path)
    if not path.exists():
        return
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return
    kept = [rows[0], *(
        row for row in rows[1:] if row and int(row[0]) <= int(cycle)
    )]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        csv.writer(handle).writerows(kept)
    os.replace(temporary, path)


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
        self.sampler_backend = "custom"
        self.system = system
        self.topology = topology
        self.ommsystem = ommsystem
        self.rest2_system = rest2_system or (
            None if ommsystem is None else ommsystem.rest2_system
        )
        self.state_files = {key: str(value) for key, value in state_files.items()}
        self.atm_states = atm_states
        self.config = config
        self.timestep_fs = float(
            base_integrator.getStepSize().value_in_unit(unit.femtosecond)
        )
        self.execution = str(config.get("execution", "serial"))
        if self.execution not in {"serial", "process"}:
            raise ValueError("REST2 execution must be 'serial' or 'process'")
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
        self.workers = []
        self.assignments = list(range(len(self.scales)))
        self.attempts = np.zeros(len(self.scales) - 1, dtype=int)
        self.accepts = np.zeros(len(self.scales) - 1, dtype=int)
        self.round_trips = np.zeros(len(self.scales), dtype=int)
        self.trip_phase = np.asarray([1, *([0] * (len(self.scales) - 1))], dtype=int)
        self.active_ensemble = None
        self.last_run_performance = None
        reporter = config.get("coordinate_reporter") or {}
        self.coordinate_reporter_enabled = bool(reporter.get("enabled", False))
        self.coordinate_reporter_interval = int(reporter.get("interval_cycles", 10))
        state_indices = reporter.get("state_indices", "all")
        if state_indices == "all":
            state_indices = list(range(len(self.scales)))
        self.coordinate_reporter_states = tuple(int(value) for value in state_indices)
        if self.coordinate_reporter_enabled:
            if topology is None:
                raise ValueError("REST2 coordinate reporting requires a topology")
            if self.coordinate_reporter_interval < 1:
                raise ValueError("REST2 coordinate reporter interval_cycles must be positive")
            if len(set(self.coordinate_reporter_states)) != len(
                self.coordinate_reporter_states
            ) or any(
                index < 0 or index >= len(self.scales)
                for index in self.coordinate_reporter_states
            ):
                raise ValueError("REST2 coordinate reporter state_indices are invalid")
        if not self.resume and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        serialized_system = mm.XmlSerializer.serialize(system)
        device_indices = config.get("device_indices")
        if device_indices is not None:
            device_indices = [str(value) for value in device_indices]
            if len(device_indices) == 1:
                device_indices *= len(self.scales)
            if len(device_indices) != len(self.scales):
                raise ValueError(
                    "REST2 device_indices must contain one value or one value per replica"
                )
        for index, scale in enumerate(self.scales):
            integrator = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(base_integrator))
            if hasattr(integrator, "setRandomNumberSeed"):
                integrator.setRandomNumberSeed(int(random_seed) + 72000 + index)
            properties = dict(platform_properties)
            if device_indices is not None:
                properties["DeviceIndex"] = device_indices[index]
            if self.execution == "serial":
                context = mm.Context(system, integrator, platform, properties)
                set_rest2_scale(context, scale, self.rest2_system)
                self.integrators.append(integrator)
                self.contexts.append(context)
            else:
                worker = REST2ReplicaProcess(
                    system_xml=serialized_system,
                    integrator_xml=mm.XmlSerializer.serialize(integrator),
                    platform_name=platform.getName(),
                    platform_properties=properties,
                    scale_parameter=self.rest2_system.scale_parameter,
                    sqrt_scale_parameter=self.rest2_system.sqrt_scale_parameter,
                )
                self.workers.append(worker)
        if self.execution == "process":
            try:
                for worker in self.workers:
                    worker.wait_ready()
                for worker, scale in zip(self.workers, self.scales):
                    worker.request("set_scale", scale)
            except Exception:
                for worker in self.workers:
                    worker.close()
                raise

    def _set_scale(self, walker, scale):
        if self.execution == "serial":
            set_rest2_scale(self.contexts[walker], scale, self.rest2_system)
        else:
            self.workers[walker].request("set_scale", scale)

    def _set_atm_state(self, walker, ensemble):
        if self.atm_states is None:
            return
        if self.execution == "serial":
            set_atm_state(
                self.contexts[walker], self.ommsystem, self.atm_states[ensemble]
            )
        else:
            self.workers[walker].request(
                "set_parameters",
                _atm_parameter_values(self.ommsystem, self.atm_states[ensemble]),
            )

    def _energy(self, walker):
        if self.execution == "serial":
            return _energy_kj(self.contexts[walker])
        return float(self.workers[walker].request("energy"))

    def _coordinate_state(self, walker):
        if self.execution == "serial":
            return self.contexts[walker].getState(
                getPositions=True,
                enforcePeriodicBox=True,
            )
        return mm.XmlSerializer.deserialize(
            self.workers[walker].request("coordinate_state")
        )

    def _step_all(self):
        if self.execution == "serial":
            for integrator in self.integrators:
                integrator.step(self.exchange_interval)
            return
        for worker in self.workers:
            worker.send("step", self.exchange_interval)
        for worker in self.workers:
            worker.receive()

    def _directory(self, ensemble):
        return self.output_dir / str(ensemble)

    def _report_coordinates(self, ensemble):
        if (
            not self.coordinate_reporter_enabled
            or self.cycle % self.coordinate_reporter_interval
        ):
            return
        directory = self._directory(ensemble) / "coordinates"
        directory.mkdir(parents=True, exist_ok=True)
        trace = directory / "frames.csv"
        existing = set()
        if trace.exists():
            with trace.open(newline="") as handle:
                existing = {
                    (int(row["cycle"]), int(row["state_index"]))
                    for row in csv.DictReader(handle)
                }
        for state_index in self.coordinate_reporter_states:
            key = (self.cycle, state_index)
            if key in existing:
                continue
            walker = self.assignments.index(state_index)
            state = self._coordinate_state(walker)
            temperature = self.temperatures[state_index]
            temperature_label = f"{temperature:.2f}".rstrip("0").rstrip(".")
            trajectory = directory / (
                f"state_{state_index:02d}_{temperature_label}K.dcd"
            )
            append = trajectory.exists()
            with trajectory.open("r+b" if append else "wb") as handle:
                dcd = DCDFile(
                    handle,
                    self.topology,
                    self.exchange_interval * self.timestep_fs * unit.femtosecond,
                    firstStep=self.cycle * self.exchange_interval,
                    interval=self.coordinate_reporter_interval * self.exchange_interval,
                    append=append,
                )
                dcd.writeModel(
                    state.getPositions(),
                    periodicBoxVectors=state.getPeriodicBoxVectors(),
                )
            _append_csv(
                trace,
                [
                    "cycle",
                    "state_index",
                    "effective_temperature_k",
                    "walker",
                    "trajectory",
                ],
                [
                    self.cycle,
                    state_index,
                    temperature,
                    walker,
                    trajectory.name,
                ],
            )

    def _metadata_path(self, ensemble):
        return self._directory(ensemble) / "state.json"

    def has_bank(self, ensemble):
        path = self._metadata_path(ensemble)
        if not path.exists():
            return False
        _validate_bank_backend(path, self.sampler_backend)
        return True

    def _initialize_bank(self, ensemble):
        source_xml = Path(self.state_files[ensemble]).read_text()
        source = mm.XmlSerializer.deserialize(source_xml)
        temperature = (
            self.atm_states[ensemble]["temperature"]
            if self.atm_states is not None else self.physical_temperature * unit.kelvin
        )
        temperature_k = float(temperature.value_in_unit(unit.kelvin))
        for index in range(len(self.scales)):
            if self.execution == "serial":
                context = self.contexts[index]
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
            else:
                self.workers[index].request("set_state", source_xml)
                self._set_atm_state(index, ensemble)
            self._set_scale(index, self.scales[index])
            ensemble_offset = {"a": 1000, "m": 2000, "b": 3000}.get(ensemble, 4000)
            velocity_seed = 73000 + ensemble_offset + index
            if self.execution == "serial":
                self.contexts[index].setVelocitiesToTemperature(
                    temperature, velocity_seed
                )
            else:
                self.workers[index].request(
                    "set_velocities", (temperature_k, velocity_seed)
                )
        self.assignments = list(range(len(self.scales)))
        self.attempts = np.zeros(len(self.scales) - 1, dtype=int)
        self.accepts = np.zeros(len(self.scales) - 1, dtype=int)
        self.round_trips = np.zeros(len(self.scales), dtype=int)
        self.trip_phase = np.asarray([1, *([0] * (len(self.scales) - 1))], dtype=int)
        self.cycle = 0

    def _load_bank(self, ensemble):
        directory = self._directory(ensemble)
        metadata = _validate_bank_backend(
            self._metadata_path(ensemble), self.sampler_backend
        )
        for index in range(len(self.scales)):
            portable_state = directory / f"walker_{index}.xml"
            if portable_state.exists():
                state_xml = portable_state.read_text()
                if self.execution == "serial":
                    self.contexts[index].setState(mm.XmlSerializer.deserialize(state_xml))
                else:
                    self.workers[index].request("set_state", state_xml)
            else:
                checkpoint = (directory / f"walker_{index}.chk").read_bytes()
                if self.execution == "serial":
                    self.contexts[index].loadCheckpoint(checkpoint)
                else:
                    self.workers[index].request("load_checkpoint", checkpoint)
        self.assignments = [int(value) for value in metadata["assignments"]]
        self.attempts = np.asarray(metadata["attempts"], dtype=int)
        self.accepts = np.asarray(metadata["accepts"], dtype=int)
        self.round_trips = np.asarray(metadata.get("round_trips", [0] * len(self.scales)), dtype=int)
        self.trip_phase = np.asarray(metadata.get("trip_phase", [0] * len(self.scales)), dtype=int)
        self.cycle = int(metadata["cycle"])
        self.rng.bit_generator.state = metadata["rng_state"]
        for walker in range(len(self.scales)):
            self._set_atm_state(walker, ensemble)
            self._set_scale(walker, self.scales[self.assignments[walker]])

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
        for index in range(len(self.scales)):
            temporary = directory / f"walker_{index}.chk.tmp"
            checkpoint = (
                self.contexts[index].createCheckpoint()
                if self.execution == "serial"
                else self.workers[index].request("checkpoint")
            )
            temporary.write_bytes(checkpoint)
            os.replace(temporary, directory / f"walker_{index}.chk")
            portable = directory / f"walker_{index}.xml"
            portable_temporary = directory / f"walker_{index}.xml.tmp"
            state_xml = (
                mm.XmlSerializer.serialize(
                    self.contexts[index].getState(
                        getPositions=True,
                        getVelocities=True,
                        getParameters=True,
                    )
                )
                if self.execution == "serial"
                else self.workers[index].request("state")
            )
            portable_temporary.write_text(state_xml)
            os.replace(portable_temporary, portable)
        payload = {
            "schema_version": 1,
            "sampler_backend": self.sampler_backend,
            "openmm_version": mm.__version__,
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
        u_ii = self._energy(walker_i)
        u_jj = self._energy(walker_j)
        self._set_scale(walker_i, self.scales[upper_state])
        self._set_scale(walker_j, self.scales[lower_state])
        u_ij = self._energy(walker_i)
        u_ji = self._energy(walker_j)
        log_acceptance = exchange_log_acceptance(self.beta, u_ii, u_ij, u_jj, u_ji)
        accepted = math.log(self.rng.random()) < min(0.0, log_acceptance)
        self.attempts[lower_state] += 1
        if accepted:
            self.accepts[lower_state] += 1
            self.assignments[walker_i], self.assignments[walker_j] = upper_state, lower_state
        else:
            self._set_scale(walker_i, self.scales[lower_state])
            self._set_scale(walker_j, self.scales[upper_state])
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
        started = time.perf_counter()
        for _ in range(cycles):
            self._step_all()
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
            self._report_coordinates(ensemble)
            if self.cycle % self.checkpoint_interval == 0:
                self.save_bank()
        self.save_bank()
        elapsed = time.perf_counter() - started
        simulated_ns = steps * len(self.scales) * 1.0e-6 * self.timestep_fs
        self.last_run_performance = {
            "execution": self.execution,
            "elapsed_seconds": elapsed,
            "aggregate_ns_per_day": simulated_ns * 86400.0 / elapsed,
        }
        if self.logger:
            rates = np.divide(
                self.accepts, self.attempts,
                out=np.zeros_like(self.accepts, dtype=float), where=self.attempts > 0,
            )
            self.logger.info(
                "%s REST2 complete: %d steps per replica, cycle %d, neighbor acceptance %s, "
                "execution %s, aggregate %.3f ns/day",
                label or ensemble,
                steps,
                self.cycle,
                np.round(rates, 3).tolist(),
                self.execution,
                self.last_run_performance["aggregate_ns_per_day"],
            )

    def physical_state(self, ensemble):
        self.activate(ensemble)
        walker = self.assignments.index(0)
        if self.execution == "serial":
            return self.contexts[walker].getState(
                getPositions=True,
                getVelocities=True,
                getEnergy=True,
                enforcePeriodicBox=True,
            )
        return mm.XmlSerializer.deserialize(self.workers[walker].request("state"))

    def summary(self):
        rates = np.divide(
            self.accepts, self.attempts,
            out=np.zeros_like(self.accepts, dtype=float), where=self.attempts > 0,
        )
        return {
            "sampler_backend": self.sampler_backend,
            "openmm_version": mm.__version__,
            "execution": self.execution,
            "effective_temperatures_k": self.temperatures,
            "acceptance_rates": rates.tolist(),
            "attempts": self.attempts.tolist(),
            "accepts": self.accepts.tolist(),
            "round_trips": self.round_trips.tolist(),
            "performance": self.last_run_performance,
            "coordinate_reporter": {
                "enabled": self.coordinate_reporter_enabled,
                "interval_cycles": self.coordinate_reporter_interval,
                "state_indices": list(self.coordinate_reporter_states),
            },
        }

    def close(self):
        self.save_bank()
        for worker in self.workers:
            worker.close()
        self.workers.clear()
        self.contexts.clear()
        self.integrators.clear()


class OpenMMREST2ExchangeSampler:
    """Production adapter for OpenMM 8.6's native replica exchange sampler."""

    sampler_backend = "openmm_native"

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
        if not hasattr(app, "ReplicaExchangeSampler"):
            raise ValueError(
                "REST2 sampler_backend=openmm_native requires OpenMM 8.6 or newer"
            )
        execution = str(config.get("execution", "serial"))
        if execution != "serial":
            raise ValueError(
                "REST2 sampler_backend=openmm_native requires execution: serial"
            )
        device_indices = config.get("device_indices")
        if device_indices is not None and len(device_indices) > 1:
            raise ValueError(
                "REST2 sampler_backend=openmm_native accepts at most one device index"
            )

        self.system = system
        self.topology = topology
        self.base_integrator_xml = mm.XmlSerializer.serialize(base_integrator)
        self.ommsystem = ommsystem
        self.rest2_system = rest2_system or (
            None if ommsystem is None else ommsystem.rest2_system
        )
        self.state_files = {key: str(value) for key, value in state_files.items()}
        self.atm_states = atm_states
        self.config = config
        self.platform = platform
        self.platform_properties = dict(platform_properties)
        if device_indices:
            self.platform_properties["DeviceIndex"] = str(device_indices[0])
        self.output_dir = Path(output_dir)
        self.resume = bool(resume)
        self.logger = logger
        self.random_seed = int(random_seed)
        self.exchange_interval = int(config["exchange_interval_steps"])
        self.checkpoint_interval = int(config.get("checkpoint_interval_cycles", 10))
        self.temperatures = [
            float(value) for value in config["effective_temperatures_k"]
        ]
        self.physical_temperature = self.temperatures[0]
        self.scales = [self.physical_temperature / value for value in self.temperatures]
        self.timestep_fs = float(
            base_integrator.getStepSize().value_in_unit(unit.femtosecond)
        )
        reporter = config.get("coordinate_reporter") or {}
        self.coordinate_reporter_enabled = bool(reporter.get("enabled", False))
        self.coordinate_reporter_interval = int(reporter.get("interval_cycles", 10))
        state_indices = reporter.get("state_indices", "all")
        if state_indices == "all":
            state_indices = list(range(len(self.scales)))
        self.coordinate_reporter_states = tuple(int(value) for value in state_indices)
        if self.coordinate_reporter_enabled:
            if topology is None:
                raise ValueError("REST2 coordinate reporting requires a topology")
            if self.coordinate_reporter_interval < 1:
                raise ValueError(
                    "REST2 coordinate reporter interval_cycles must be positive"
                )
            if len(set(self.coordinate_reporter_states)) != len(
                self.coordinate_reporter_states
            ) or any(
                index < 0 or index >= len(self.scales)
                for index in self.coordinate_reporter_states
            ):
                raise ValueError("REST2 coordinate reporter state_indices are invalid")
        if not self.resume and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resources = {}
        self.active_ensemble = None
        self.last_run_performance = None

    def _directory(self, ensemble):
        return self.output_dir / str(ensemble)

    def _metadata_path(self, ensemble):
        return self._directory(ensemble) / "state.json"

    def has_bank(self, ensemble):
        path = self._metadata_path(ensemble)
        if not path.exists():
            return False
        _validate_bank_backend(path, self.sampler_backend)
        return True

    def _thermodynamic_states(self, ensemble):
        fixed = {}
        if self.atm_states is not None:
            fixed = _atm_parameter_values(self.ommsystem, self.atm_states[ensemble])
        return [
            {
                **fixed,
                self.rest2_system.scale_parameter: float(scale),
                self.rest2_system.sqrt_scale_parameter: math.sqrt(float(scale)),
            }
            for scale in self.scales
        ]

    def _source_temperature(self, ensemble):
        if self.atm_states is None:
            return self.physical_temperature * unit.kelvin
        return self.atm_states[ensemble]["temperature"]

    def _new_resource(self, ensemble):
        integrator = mm.XmlSerializer.deserialize(self.base_integrator_xml)
        if hasattr(integrator, "setRandomNumberSeed"):
            integrator.setRandomNumberSeed(self.random_seed + 72000)
        simulation = app.Simulation(
            self.topology,
            self.system,
            integrator,
            self.platform,
            self.platform_properties,
        )
        source = mm.XmlSerializer.deserialize(
            Path(self.state_files[ensemble]).read_text()
        )
        simulation.context.setState(source)
        states = self._thermodynamic_states(ensemble)
        sampler = app.ReplicaExchangeSampler(
            states, simulation, self.exchange_interval
        )
        ensemble_offset = {"a": 1000, "m": 2000, "b": 3000}.get(ensemble, 4000)
        conformations = []
        for index, scale in enumerate(self.scales):
            simulation.context.setState(source)
            if self.atm_states is not None:
                set_atm_state(
                    simulation.context, self.ommsystem, self.atm_states[ensemble]
                )
            set_rest2_scale(simulation.context, scale, self.rest2_system)
            simulation.context.setVelocitiesToTemperature(
                self._source_temperature(ensemble),
                self.random_seed + 73000 + ensemble_offset + index,
            )
            conformations.append(
                simulation.context.getState(
                    getPositions=True,
                    getVelocities=True,
                    getParameters=True,
                    integratorParameters=True,
                    enforcePeriodicBox=True,
                )
            )
        sampler.replicaConformation = conformations
        sampler.reporters.append(
            lambda current, name=ensemble: self._report_iteration(name, current)
        )
        resource = {
            "simulation": simulation,
            "integrator": integrator,
            "sampler": sampler,
            "round_trips": np.zeros(len(self.scales), dtype=int),
            "trip_phase": np.asarray(
                [1, *([0] * (len(self.scales) - 1))], dtype=int
            ),
            "rng_state": random.Random(
                self.random_seed + 71000 + ensemble_offset
            ).getstate(),
        }
        return resource

    @staticmethod
    def _encode_rng_state(state):
        return base64.b64encode(pickle.dumps(state)).decode("ascii")

    @staticmethod
    def _decode_rng_state(value):
        return pickle.loads(base64.b64decode(value.encode("ascii")))

    def _load_resource(self, ensemble, resource):
        metadata = _validate_bank_backend(
            self._metadata_path(ensemble), self.sampler_backend
        )
        if metadata.get("effective_temperatures_k") != self.temperatures:
            raise ValueError(
                f"REST2 bank {self._metadata_path(ensemble)} temperature ladder "
                "does not match the workflow"
            )
        if int(metadata.get("exchange_interval_steps", -1)) != self.exchange_interval:
            raise ValueError(
                f"REST2 bank {self._metadata_path(ensemble)} exchange interval "
                "does not match the workflow"
            )
        sampler = resource["sampler"]
        directory = self._directory(ensemble)
        sampler.replicaConformation = [
            mm.XmlSerializer.deserialize(
                (directory / f"walker_{index}.xml").read_text()
            )
            for index in range(len(self.scales))
        ]
        sampler.replicaStateIndex = [int(v) for v in metadata["assignments"]]
        sampler._previousReplicaStateIndex = [
            int(v) for v in metadata["previous_assignments"]
        ]
        sampler.currentIteration = int(metadata["cycle"])
        _truncate_csv_after_cycle(
            self._directory(ensemble) / "state_trace.csv",
            sampler.currentIteration,
        )
        resource["round_trips"] = np.asarray(
            metadata.get("round_trips", [0] * len(self.scales)), dtype=int
        )
        resource["trip_phase"] = np.asarray(
            metadata.get("trip_phase", [0] * len(self.scales)), dtype=int
        )
        resource["rng_state"] = self._decode_rng_state(metadata["rng_state"])

    def activate(self, ensemble):
        if ensemble not in self.resources:
            resource = self._new_resource(ensemble)
            self.resources[ensemble] = resource
            if self._metadata_path(ensemble).exists():
                self._load_resource(ensemble, resource)
                if self.logger:
                    self.logger.info(
                        "Native OpenMM REST2 resumed %s at exchange cycle %d",
                        ensemble,
                        resource["sampler"].currentIteration,
                    )
            else:
                self.save_bank(ensemble)
        self.active_ensemble = ensemble

    def _report_coordinates(self, ensemble, sampler, assignments):
        cycle = int(sampler.currentIteration)
        if (
            not self.coordinate_reporter_enabled
            or cycle % self.coordinate_reporter_interval
        ):
            return
        directory = self._directory(ensemble) / "coordinates"
        directory.mkdir(parents=True, exist_ok=True)
        trace = directory / "frames.csv"
        existing = set()
        if trace.exists():
            with trace.open(newline="") as handle:
                existing = {
                    (int(row["cycle"]), int(row["state_index"]))
                    for row in csv.DictReader(handle)
                }
        for state_index in self.coordinate_reporter_states:
            if (cycle, state_index) in existing:
                continue
            walker = assignments.index(state_index)
            state = sampler.replicaConformation[walker]
            temperature = self.temperatures[state_index]
            label = f"{temperature:.2f}".rstrip("0").rstrip(".")
            trajectory = directory / f"state_{state_index:02d}_{label}K.dcd"
            append = trajectory.exists()
            with trajectory.open("r+b" if append else "wb") as handle:
                dcd = DCDFile(
                    handle,
                    self.topology,
                    self.exchange_interval * self.timestep_fs * unit.femtosecond,
                    firstStep=cycle * self.exchange_interval,
                    interval=self.coordinate_reporter_interval * self.exchange_interval,
                    append=append,
                )
                dcd.writeModel(
                    state.getPositions(),
                    periodicBoxVectors=state.getPeriodicBoxVectors(),
                )
            _append_csv(
                trace,
                [
                    "cycle", "state_index", "effective_temperature_k",
                    "walker", "trajectory",
                ],
                [cycle, state_index, temperature, walker, trajectory.name],
            )

    def _report_iteration(self, ensemble, sampler):
        # OpenMM invokes reporters after propagation and before exchanging states.
        assignments = [int(value) for value in sampler.replicaStateIndex]
        resource = self.resources.get(ensemble)
        if resource is None:
            return
        for walker, state_index in enumerate(assignments):
            if state_index == 0:
                if resource["trip_phase"][walker] == 2:
                    resource["round_trips"][walker] += 1
                resource["trip_phase"][walker] = 1
            elif (
                state_index == len(self.scales) - 1
                and resource["trip_phase"][walker] == 1
            ):
                resource["trip_phase"][walker] = 2
        _append_csv(
            self._directory(ensemble) / "state_trace.csv",
            ["cycle", *[f"walker_{i}" for i in range(len(self.scales))]],
            [int(sampler.currentIteration), *assignments],
        )
        self._report_coordinates(ensemble, sampler, assignments)

    def save_bank(self, ensemble=None):
        ensemble = ensemble or self.active_ensemble
        if ensemble is None or ensemble not in self.resources:
            return
        resource = self.resources[ensemble]
        sampler = resource["sampler"]
        directory = self._directory(ensemble)
        directory.mkdir(parents=True, exist_ok=True)
        for index, state in enumerate(sampler.replicaConformation):
            temporary = directory / f"walker_{index}.xml.tmp"
            temporary.write_text(mm.XmlSerializer.serialize(state))
            os.replace(temporary, directory / f"walker_{index}.xml")
        payload = {
            "schema_version": 2,
            "sampler_backend": self.sampler_backend,
            "openmm_version": mm.__version__,
            "ensemble": ensemble,
            "cycle": int(sampler.currentIteration),
            "assignments": [int(v) for v in sampler.replicaStateIndex],
            "previous_assignments": [
                int(v) for v in sampler._previousReplicaStateIndex
            ],
            "round_trips": resource["round_trips"].tolist(),
            "trip_phase": resource["trip_phase"].tolist(),
            "rng_state": self._encode_rng_state(resource["rng_state"]),
            "effective_temperatures_k": self.temperatures,
            "exchange_interval_steps": self.exchange_interval,
        }
        temporary = directory / "state.json.tmp"
        temporary.write_text(json.dumps(payload) + "\n")
        os.replace(temporary, self._metadata_path(ensemble))

    def run_steps(self, ensemble, steps, label=None):
        self.activate(ensemble)
        steps = int(steps)
        if steps % self.exchange_interval:
            raise ValueError(
                "REST2 sampling steps must be divisible by exchange_interval_steps"
            )
        resource = self.resources[ensemble]
        sampler = resource["sampler"]
        cycles = steps // self.exchange_interval
        started = time.perf_counter()
        global_rng_state = random.getstate()
        try:
            random.setstate(resource["rng_state"])
            for _ in range(cycles):
                sampler.simulate(1)
                resource["rng_state"] = random.getstate()
                if sampler.currentIteration % self.checkpoint_interval == 0:
                    self.save_bank(ensemble)
        finally:
            resource["rng_state"] = random.getstate()
            random.setstate(global_rng_state)
        self.save_bank(ensemble)
        elapsed = time.perf_counter() - started
        simulated_ns = steps * len(self.scales) * 1.0e-6 * self.timestep_fs
        self.last_run_performance = {
            "execution": "openmm_native_serial",
            "elapsed_seconds": elapsed,
            "aggregate_ns_per_day": simulated_ns * 86400.0 / elapsed,
        }
        if self.logger:
            self.logger.info(
                "%s native OpenMM REST2 complete: %d steps per replica, cycle %d, "
                "round trips %s, aggregate %.3f ns/day",
                label or ensemble,
                steps,
                sampler.currentIteration,
                resource["round_trips"].tolist(),
                self.last_run_performance["aggregate_ns_per_day"],
            )

    def physical_state(self, ensemble):
        self.activate(ensemble)
        sampler = self.resources[ensemble]["sampler"]
        # Conformations are propagated before exchange, so the previous assignment
        # identifies the Hamiltonian that generated each stored conformation.
        walker = sampler._previousReplicaStateIndex.index(0)
        return sampler.replicaConformation[walker]

    def _transition_diagnostics(self, ensemble):
        return observed_transition_diagnostics(
            self._directory(ensemble) / "state_trace.csv", len(self.scales)
        )

    def summary(self):
        resource = self.resources.get(self.active_ensemble)
        round_trips = (
            resource["round_trips"].tolist() if resource is not None else []
        )
        return {
            "sampler_backend": self.sampler_backend,
            "openmm_version": mm.__version__,
            "execution": "openmm_native_serial",
            "effective_temperatures_k": self.temperatures,
            "acceptance_rates": [],
            "attempts": [],
            "accepts": [],
            "round_trips": round_trips,
            "exchange_diagnostics": (
                self._transition_diagnostics(self.active_ensemble)
                if self.active_ensemble is not None else None
            ),
            "performance": self.last_run_performance,
            "coordinate_reporter": {
                "enabled": self.coordinate_reporter_enabled,
                "interval_cycles": self.coordinate_reporter_interval,
                "state_indices": list(self.coordinate_reporter_states),
            },
        }

    def close(self):
        for ensemble in list(self.resources):
            self.save_bank(ensemble)
        self.resources.clear()
        self.active_ensemble = None
