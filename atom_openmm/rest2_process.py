"""Persistent process workers for concurrent REST2 replica propagation."""

from __future__ import annotations

import multiprocessing as mp
import traceback

import openmm as mm
from openmm import unit


def _worker_main(
    connection,
    system_xml,
    integrator_xml,
    platform_name,
    platform_properties,
    scale_parameter,
    sqrt_scale_parameter,
):
    try:
        system = mm.XmlSerializer.deserialize(system_xml)
        integrator = mm.XmlSerializer.deserialize(integrator_xml)
        platform = mm.Platform.getPlatformByName(platform_name)
        context = mm.Context(system, integrator, platform, platform_properties)
        connection.send({"ok": True, "result": "ready"})
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send({"ok": True, "result": None})
                break
            if command == "step":
                integrator.step(int(payload))
                result = None
            elif command == "energy":
                result = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )
            elif command == "set_scale":
                scale = float(payload)
                context.setParameter(scale_parameter, scale)
                context.setParameter(sqrt_scale_parameter, scale**0.5)
                result = None
            elif command == "set_parameters":
                for name, value in payload.items():
                    context.setParameter(name, float(value))
                result = None
            elif command == "set_state":
                context.setState(mm.XmlSerializer.deserialize(payload))
                result = None
            elif command == "set_velocities":
                temperature_k, seed = payload
                context.setVelocitiesToTemperature(
                    float(temperature_k) * unit.kelvin, int(seed)
                )
                result = None
            elif command == "load_checkpoint":
                context.loadCheckpoint(payload)
                result = None
            elif command == "checkpoint":
                result = context.createCheckpoint()
            elif command == "state":
                state = context.getState(
                    getPositions=True,
                    getVelocities=True,
                    getEnergy=True,
                    enforcePeriodicBox=True,
                )
                result = mm.XmlSerializer.serialize(state)
            elif command == "coordinate_state":
                state = context.getState(
                    getPositions=True,
                    enforcePeriodicBox=True,
                )
                result = mm.XmlSerializer.serialize(state)
            elif command == "parameter":
                result = context.getParameter(str(payload))
            else:
                raise ValueError(f"unknown REST2 worker command {command!r}")
            connection.send({"ok": True, "result": result})
    except BaseException as exc:
        try:
            connection.send(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        connection.close()


class REST2ReplicaProcess:
    def __init__(
        self,
        *,
        system_xml,
        integrator_xml,
        platform_name,
        platform_properties,
        scale_parameter,
        sqrt_scale_parameter,
    ):
        process_context = mp.get_context("spawn")
        parent, child = process_context.Pipe()
        self.connection = parent
        self.process = process_context.Process(
            target=_worker_main,
            args=(
                child,
                system_xml,
                integrator_xml,
                platform_name,
                platform_properties,
                scale_parameter,
                sqrt_scale_parameter,
            ),
        )
        self.process.start()
        child.close()

    def send(self, command, payload=None):
        self.connection.send((command, payload))

    def receive(self):
        try:
            response = self.connection.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"REST2 replica process {self.process.pid} terminated unexpectedly"
            ) from exc
        if not response["ok"]:
            raise RuntimeError(
                f"REST2 replica process failed: {response['error']}\n"
                f"{response.get('traceback', '')}"
            )
        return response["result"]

    def request(self, command, payload=None):
        self.send(command, payload)
        return self.receive()

    def wait_ready(self):
        if self.receive() != "ready":
            raise RuntimeError("REST2 replica process did not report ready")

    def close(self):
        if self.process.is_alive():
            try:
                self.request("close")
            except (EOFError, BrokenPipeError, RuntimeError):
                pass
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.connection.close()
