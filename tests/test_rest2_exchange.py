import json
import csv

import openmm as mm
import pytest
from openmm import unit
from openmm.app import Topology, element


def _simple_rest2_fixture(tmp_path):
    from atom_openmm.rest2 import create_rest2_system

    physical = mm.System()
    for _ in range(2):
        physical.addParticle(12.0)
    bonds = mm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.1, 100.0)
    physical.addForce(bonds)
    nonbonded = mm.NonbondedForce()
    for _ in range(2):
        nonbonded.addParticle(0.0, 0.3, 0.0)
    physical.addForce(nonbonded)
    rest2 = create_rest2_system(physical, [0, 1])
    topology = Topology()
    residue = topology.addResidue("LIG", topology.addChain())
    atoms = [
        topology.addAtom(f"C{index}", element.carbon, residue)
        for index in range(2)
    ]
    topology.addBond(*atoms)
    context = mm.Context(rest2.system, mm.VerletIntegrator(0.001))
    context.setPositions([[0, 0, 0], [0.2, 0, 0]])
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    state_file = tmp_path / "a.xml"
    state_file.write_text(
        mm.XmlSerializer.serialize(
            context.getState(getPositions=True, getVelocities=True)
        )
    )
    del context
    return rest2, topology, state_file


def _test_rest2_factory_defaults_to_custom_and_rejects_native_on_old_openmm(tmp_path):
    from atom_openmm.rest2_exchange import (
        REST2ExchangeSampler,
        create_rest2_exchange_sampler,
    )

    rest2, topology, state_file = _simple_rest2_fixture(tmp_path)
    kwargs = {
        "system": rest2.system,
        "topology": topology,
        "base_integrator": mm.LangevinMiddleIntegrator(300, 1, 0.001),
        "rest2_system": rest2,
        "state_files": {"a": state_file},
        "config": {
            "effective_temperatures_k": [300, 600],
            "exchange_interval_steps": 1,
            "execution": "serial",
        },
        "platform": mm.Platform.getPlatformByName("Reference"),
        "platform_properties": {},
        "output_dir": tmp_path / "factory",
        "resume": False,
    }
    sampler = create_rest2_exchange_sampler(**kwargs)
    assert isinstance(sampler, REST2ExchangeSampler)
    sampler.close()

    if not hasattr(__import__("openmm.app", fromlist=["app"]), "ReplicaExchangeSampler"):
        kwargs["config"] = {**kwargs["config"], "sampler_backend": "openmm_native"}
        with pytest.raises(ValueError, match="OpenMM 8.6"):
            create_rest2_exchange_sampler(**kwargs)


def _test_rest2_bank_backend_mismatch_is_rejected(tmp_path):
    from atom_openmm.rest2_exchange import _validate_bank_backend

    bank = tmp_path / "state.json"
    bank.write_text(json.dumps({"sampler_backend": "custom"}))
    with pytest.raises(ValueError, match="cannot be resumed"):
        _validate_bank_backend(bank, "openmm_native")


@pytest.mark.skipif(
    not hasattr(__import__("openmm.app", fromlist=["app"]), "ReplicaExchangeSampler"),
    reason="OpenMM native replica exchange requires OpenMM 8.6",
)
def _test_openmm_native_rest2_sampler_resumes_and_returns_physical_state(tmp_path):
    from atom_openmm.rest2_exchange import create_rest2_exchange_sampler

    rest2, topology, state_file = _simple_rest2_fixture(tmp_path)
    config = {
        "sampler_backend": "openmm_native",
        "effective_temperatures_k": [300, 600],
        "exchange_interval_steps": 1,
        "checkpoint_interval_cycles": 1,
        "execution": "serial",
        "coordinate_reporter": {
            "enabled": True,
            "interval_cycles": 1,
            "state_indices": "all",
        },
    }

    def sampler():
        return create_rest2_exchange_sampler(
            system=rest2.system,
            topology=topology,
            base_integrator=mm.LangevinMiddleIntegrator(300, 1, 0.001),
            rest2_system=rest2,
            state_files={"a": state_file},
            config=config,
            platform=mm.Platform.getPlatformByName("Reference"),
            platform_properties={},
            output_dir=tmp_path / "native",
            resume=True,
        )

    first = sampler()
    first.run_steps("a", 2)
    state = first.physical_state("a")
    assert state.getPositions() is not None
    assert first.summary()["sampler_backend"] == "openmm_native"
    first.close()
    metadata = json.loads((tmp_path / "native/a/state.json").read_text())
    assert metadata["cycle"] == 2
    assert "previous_assignments" in metadata

    resumed = sampler()
    resumed.run_steps("a", 1)
    assert resumed.resources["a"]["sampler"].currentIteration == 3
    resumed.close()
    with (tmp_path / "native/a/coordinates/frames.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {int(row["state_index"]) for row in rows} == {0, 1}


@pytest.mark.skipif(
    not hasattr(__import__("openmm.app", fromlist=["app"]), "ReplicaExchangeSampler"),
    reason="OpenMM native replica exchange requires OpenMM 8.6",
)
def _test_openmm_native_rest2_sampler_sets_temperature_for_custom_integrator(tmp_path):
    from atom_openmm.rest2_exchange import create_rest2_exchange_sampler

    rest2, topology, state_file = _simple_rest2_fixture(tmp_path)
    integrator = mm.CustomIntegrator(0.001)
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    sampler = create_rest2_exchange_sampler(
        system=rest2.system,
        topology=topology,
        base_integrator=integrator,
        rest2_system=rest2,
        state_files={"a": state_file},
        config={
            "sampler_backend": "openmm_native",
            "effective_temperatures_k": [300, 600],
            "exchange_interval_steps": 1,
            "execution": "serial",
        },
        platform=mm.Platform.getPlatformByName("Reference"),
        platform_properties={},
        output_dir=tmp_path / "native_custom_integrator",
        resume=False,
    )
    sampler.activate("a")
    states = sampler.resources["a"]["sampler"].states
    assert all(
        state["temperature"].value_in_unit(unit.kelvin) == pytest.approx(300)
        for state in states
    )
    sampler.run_steps("a", 1)
    sampler.close()


@pytest.mark.skipif(
    not hasattr(__import__("openmm.app", fromlist=["app"]), "ReplicaExchangeSampler"),
    reason="OpenMM native replica exchange requires OpenMM 8.6",
)
def _test_openmm_native_rest2_sampler_applies_fixed_atm_parameters(tmp_path):
    from atom_openmm.rest2_exchange import create_rest2_exchange_sampler

    rest2, topology, state_file = _simple_rest2_fixture(tmp_path)
    parameters = mm.CustomExternalForce("0")
    for name, value in (
        ("Lambda1", 0), ("Lambda2", 0), ("Alpha", 0.1), ("Uh", 0),
        ("W0", 0), ("Direction", 1), ("Umax", 200), ("Ubcore", 100),
        ("Acore", 0.0625), ("UOffset", 0),
    ):
        parameters.addGlobalParameter(name, value)
    parameters.addParticle(0, [])
    rest2.system.addForce(parameters)
    atm_state = {
        "lambda1": 0.25, "lambda2": 0.5,
        "alpha": 0.1 / unit.kilocalorie_per_mole,
        "uh": 0 * unit.kilocalorie_per_mole,
        "w0": 0 * unit.kilocalorie_per_mole,
        "atmdirection": 1.0,
        "Umax": 200 * unit.kilocalorie_per_mole,
        "Ubcore": 100 * unit.kilocalorie_per_mole,
        "Acore": 0.0625,
        "uoffset": 0 * unit.kilocalorie_per_mole,
        "temperature": 300 * unit.kelvin,
    }
    ommsystem = type("FakeSystem", (), {
        "atmforce": _ATMNames(), "multisoftplus": False, "rest2_system": rest2,
    })()
    sampler = create_rest2_exchange_sampler(
        system=rest2.system,
        topology=topology,
        base_integrator=mm.LangevinMiddleIntegrator(300, 1, 0.001),
        ommsystem=ommsystem,
        state_files={"a": state_file},
        atm_states={"a": atm_state},
        config={
            "sampler_backend": "openmm_native",
            "effective_temperatures_k": [300, 600],
            "exchange_interval_steps": 1,
            "execution": "serial",
        },
        platform=mm.Platform.getPlatformByName("Reference"),
        platform_properties={},
        output_dir=tmp_path / "native_atm",
        resume=False,
    )
    sampler.activate("a")
    states = sampler.resources["a"]["sampler"].states
    assert states[0]["Lambda1"] == pytest.approx(0.25)
    assert states[0]["Lambda2"] == pytest.approx(0.5)
    sampler.run_steps("a", 1)
    assert sampler.physical_state("a").getPositions() is not None
    sampler.close()


class _ATMNames:
    def Lambda1(self): return "Lambda1"
    def Lambda2(self): return "Lambda2"
    def Alpha(self): return "Alpha"
    def Uh(self): return "Uh"
    def W0(self): return "W0"
    def Direction(self): return "Direction"
    def Umax(self): return "Umax"
    def Ubcore(self): return "Ubcore"
    def Acore(self): return "Acore"


def _test_rest2_exchange_sampler_preserves_independent_atm_banks(tmp_path):
    from atom_openmm.rest2 import create_rest2_system
    from atom_openmm.rest2_exchange import REST2ExchangeSampler

    physical = mm.System()
    physical.addParticle(12.0)
    physical.addParticle(12.0)
    bonds = mm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.1, 100.0)
    physical.addForce(bonds)
    nonbonded = mm.NonbondedForce()
    nonbonded.addParticle(0.0, 0.3, 0.0)
    nonbonded.addParticle(0.0, 0.3, 0.0)
    physical.addForce(nonbonded)
    rest2 = create_rest2_system(physical, [0, 1])
    parameters = mm.CustomExternalForce("0")
    for name, value in (
        ("Lambda1", 0), ("Lambda2", 0), ("Alpha", 0.1), ("Uh", 0),
        ("W0", 0), ("Direction", 1), ("Umax", 200), ("Ubcore", 100),
        ("Acore", 0.0625), ("UOffset", 0),
    ):
        parameters.addGlobalParameter(name, value)
    parameters.addParticle(0, [])
    rest2.system.addForce(parameters)

    state_context = mm.Context(rest2.system, mm.VerletIntegrator(0.001))
    state_context.setPositions([[0, 0, 0], [0.2, 0, 0]])
    state_context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    state_xml = mm.XmlSerializer.serialize(
        state_context.getState(getPositions=True, getVelocities=True)
    )
    state_files = {}
    for ensemble in ("a", "m", "b"):
        path = tmp_path / f"{ensemble}.xml"
        path.write_text(state_xml)
        state_files[ensemble] = path

    atm_state = {
        "lambda1": 0.0, "lambda2": 0.0, "alpha": 0.1 / unit.kilocalorie_per_mole,
        "uh": 0 * unit.kilocalorie_per_mole, "w0": 0 * unit.kilocalorie_per_mole,
        "atmdirection": 1.0, "Umax": 200 * unit.kilocalorie_per_mole,
        "Ubcore": 100 * unit.kilocalorie_per_mole, "Acore": 0.0625,
        "uoffset": 0 * unit.kilocalorie_per_mole, "temperature": 300 * unit.kelvin,
    }
    ommsystem = type("FakeSystem", (), {
        "atmforce": _ATMNames(), "multisoftplus": False, "rest2_system": rest2,
    })()
    sampler = REST2ExchangeSampler(
        system=rest2.system,
        topology=None,
        base_integrator=mm.LangevinMiddleIntegrator(300, 1, 0.001),
        ommsystem=ommsystem,
        state_files=state_files,
        atm_states={name: dict(atm_state) for name in state_files},
        config={
            "effective_temperatures_k": [300, 600],
            "exchange_interval_steps": 1,
            "checkpoint_interval_cycles": 1,
            "execution": "process",
        },
        platform=mm.Platform.getPlatformByName("Reference"),
        platform_properties={},
        output_dir=tmp_path / "rest2",
        resume=False,
    )
    sampler.run_steps("a", 2)
    sampler.run_steps("b", 1)
    sampler.activate("a")
    assert json.loads((tmp_path / "rest2/a/state.json").read_text())["cycle"] == 2
    assert sampler.cycle == 2
    assert sampler.physical_state("a").getPositions() is not None
    sampler.close()


def _test_rest2_exchange_sampler_supports_fixed_native_hamiltonian(tmp_path):
    from atom_openmm.rest2 import create_rest2_system
    from atom_openmm.rest2_exchange import REST2ExchangeSampler

    physical = mm.System()
    physical.addParticle(12.0)
    physical.addParticle(12.0)
    bonds = mm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.1, 100.0)
    physical.addForce(bonds)
    nonbonded = mm.NonbondedForce()
    nonbonded.addParticle(0.0, 0.3, 0.0)
    nonbonded.addParticle(0.0, 0.3, 0.0)
    physical.addForce(nonbonded)
    rest2 = create_rest2_system(physical, [0, 1])

    state_context = mm.Context(rest2.system, mm.VerletIntegrator(0.001))
    state_context.setPositions([[0, 0, 0], [0.2, 0, 0]])
    state_context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    state_file = tmp_path / "a.xml"
    state_file.write_text(mm.XmlSerializer.serialize(
        state_context.getState(getPositions=True, getVelocities=True)
    ))

    process_config = {
        "effective_temperatures_k": [300, 600],
        "exchange_interval_steps": 1,
        "checkpoint_interval_cycles": 1,
        "execution": "process",
    }
    sampler = REST2ExchangeSampler(
        system=rest2.system,
        topology=None,
        base_integrator=mm.LangevinMiddleIntegrator(300, 1, 0.001),
        rest2_system=rest2,
        state_files={"a": state_file},
        config=process_config,
        platform=mm.Platform.getPlatformByName("Reference"),
        platform_properties={},
        output_dir=tmp_path / "native_rest2",
        resume=False,
    )
    sampler.run_steps("a", 2)

    assert sampler.cycle == 2
    assert sampler.workers[0].request("parameter", "REST2_SCALE") in (0.5, 1.0)
    assert sampler.physical_state("a").getPositions() is not None
    sampler.close()
    for checkpoint in (tmp_path / "native_rest2/a").glob("walker_*.chk"):
        checkpoint.unlink()

    resumed = REST2ExchangeSampler(
        system=rest2.system,
        topology=None,
        base_integrator=mm.LangevinMiddleIntegrator(300, 1, 0.001),
        rest2_system=rest2,
        state_files={"a": state_file},
        config=process_config,
        platform=mm.Platform.getPlatformByName("Reference"),
        platform_properties={},
        output_dir=tmp_path / "native_rest2",
        resume=True,
    )
    resumed.run_steps("a", 1)
    assert resumed.cycle == 3
    resumed.close()


def _test_rest2_coordinate_reporter_tracks_states_and_appends_on_resume(tmp_path):
    from atom_openmm.rest2 import create_rest2_system
    from atom_openmm.rest2_exchange import REST2ExchangeSampler

    physical = mm.System()
    for _ in range(2):
        physical.addParticle(12.0)
    bonds = mm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.1, 100.0)
    physical.addForce(bonds)
    nonbonded = mm.NonbondedForce()
    for _ in range(2):
        nonbonded.addParticle(0.0, 0.3, 0.0)
    physical.addForce(nonbonded)
    rest2 = create_rest2_system(physical, [0, 1])
    topology = Topology()
    residue = topology.addResidue("LIG", topology.addChain())
    atoms = [
        topology.addAtom(f"C{index}", element.carbon, residue)
        for index in range(2)
    ]
    topology.addBond(*atoms)

    context = mm.Context(rest2.system, mm.VerletIntegrator(0.001))
    context.setPositions([[0, 0, 0], [0.2, 0, 0]])
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    state_file = tmp_path / "a.xml"
    state_file.write_text(
        mm.XmlSerializer.serialize(
            context.getState(getPositions=True, getVelocities=True)
        )
    )
    del context
    config = {
        "effective_temperatures_k": [300, 600],
        "exchange_interval_steps": 1,
        "checkpoint_interval_cycles": 1,
        "execution": "serial",
        "coordinate_reporter": {
            "enabled": True,
            "interval_cycles": 1,
            "state_indices": "all",
        },
    }

    def sampler():
        return REST2ExchangeSampler(
            system=rest2.system,
            topology=topology,
            base_integrator=mm.LangevinMiddleIntegrator(300, 1, 0.001),
            rest2_system=rest2,
            state_files={"a": state_file},
            config=config,
            platform=mm.Platform.getPlatformByName("Reference"),
            platform_properties={},
            output_dir=tmp_path / "reported_rest2",
            resume=True,
        )

    first = sampler()
    first.run_steps("a", 2)
    first.close()
    trajectories = sorted(
        (tmp_path / "reported_rest2/a/coordinates").glob("*.dcd")
    )
    initial_sizes = [path.stat().st_size for path in trajectories]
    assert len(trajectories) == 2

    resumed = sampler()
    resumed.run_steps("a", 1)
    resumed.close()

    assert all(
        path.stat().st_size > size
        for path, size in zip(trajectories, initial_sizes)
    )
    with (tmp_path / "reported_rest2/a/coordinates/frames.csv").open(
        newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {int(row["state_index"]) for row in rows} == {0, 1}
