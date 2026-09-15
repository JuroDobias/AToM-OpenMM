import pytest


def _test_steps_from_section_rejects_duplicate_step_ids():
    from atom_openmm.equilibration import EquilibrationConfigError, _steps_from_section

    section = {
        "steps": [
            {"id": "endpoint_eq", "type": "md", "n_steps": 1},
            {"id": "endpoint_eq", "type": "md", "n_steps": 1},
        ]
    }

    with pytest.raises(
        EquilibrationConfigError,
        match="workflow\\.equilibration\\.neqti\\.endpoint\\.steps has duplicate step id 'endpoint_eq' "
        "at steps 1 and 2",
    ):
        _steps_from_section(section, "workflow.equilibration.neqti.endpoint")


def _test_normalize_equilibration_protocol_rejects_duplicate_step_ids():
    from atom_openmm.equilibration import EquilibrationConfigError, normalize_equilibration_protocol

    workflow = {
        "equilibration": {
            "neqti": {
                "endpoint": {
                    "steps": [
                        {"id": "repeat", "type": "md", "n_steps": 1},
                        {"id": "unique", "type": "md", "n_steps": 1},
                        {"id": "repeat", "type": "md", "n_steps": 1},
                    ]
                }
            }
        }
    }

    with pytest.raises(EquilibrationConfigError, match="duplicate step id 'repeat' at steps 1 and 3"):
        normalize_equilibration_protocol(workflow)


def _test_hybrid_endpoint_steps_use_environment_override_and_shared_fallback():
    from atom_openmm.equilibration import neqti_hybrid_endpoint_steps

    shared = [{"id": "shared", "type": "minimization"}]
    complex_steps = [{"id": "complex", "type": "minimization"}]
    protocol = {
        "neqti": {
            "endpoint": {"steps": shared},
            "complex_endpoint": {"steps": complex_steps},
            "solvent_endpoint": {"mode": "default"},
        }
    }

    assert neqti_hybrid_endpoint_steps(protocol, "complex") == complex_steps
    assert neqti_hybrid_endpoint_steps(protocol, "solvent") is None
    assert neqti_hybrid_endpoint_steps(
        {"neqti": {"endpoint": {"steps": shared}}}, "solvent"
    ) == shared


def _test_custom_equilibration_resumes_completed_steps(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import openmm as mm
    from openmm import app, unit

    from atom_openmm import equilibration

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("ONE", chain)
    topology.addAtom("Ar", app.Element.getByAtomicNumber(18), residue)
    system = mm.System()
    system.addParticle(39.9 * unit.dalton)
    positions = unit.Quantity([[0.0, 0.0, 0.0]], unit.nanometer)
    ommsystem = SimpleNamespace(
        topology=topology,
        positions=positions,
        system=system,
        boxvectors=None,
        keywords={"WORKDIR": str(tmp_path)},
        temperature=300.0 * unit.kelvin,
    )
    steps = [{"id": "min", "type": "minimization", "max_iterations": 1}]
    output = tmp_path / "custom"
    final_state = tmp_path / "final.xml"
    final_pdb = tmp_path / "final.pdb"
    platform = mm.Platform.getPlatformByName("Reference")

    equilibration.run_custom_equilibration(
        ommsystem=ommsystem,
        steps=steps,
        platform=platform,
        platform_properties={},
        output_dir=output,
        final_state_path=final_state,
        final_pdb_path=final_pdb,
    )

    def fail_minimize(*args, **kwargs):
        raise AssertionError("completed minimization was rerun")

    monkeypatch.setattr(mm.LocalEnergyMinimizer, "minimize", fail_minimize)
    equilibration.run_custom_equilibration(
        ommsystem=ommsystem,
        steps=steps,
        platform=platform,
        platform_properties={},
        output_dir=output,
        final_state_path=final_state,
        final_pdb_path=final_pdb,
    )


def _test_restraint_resolution_is_indexed_not_step_id(monkeypatch):
    from atom_openmm import equilibration

    class FakeResolver:
        def __init__(self, topology, positions, **kwargs):
            pass

        def resolve(self, mask, label):
            return [7]

    monkeypatch.setattr(equilibration, "AmberMaskResolver", FakeResolver)
    steps = [
        {
            "id": "repeat",
            "type": "md",
            "n_steps": 1,
            "positional_restraints": {"mask": "@CA", "k_kcal_mol_a2": 1.0},
        },
        {"id": "repeat", "type": "md", "n_steps": 1},
    ]

    resolved = equilibration._resolve_step_restraints(steps, topology=None, positions=None)

    assert resolved == [{"positional_atom_indices": [7]}, {}]


def _test_temperature_ramp_integrator_starts_at_initial_temperature():
    from openmm import unit

    from atom_openmm.equilibration import _build_integrator

    integrator = _build_integrator(
        {
            "type": "md",
            "n_steps": 100,
            "thermostat": {
                "initial_temperature_k": 50.0,
                "temperature_k": 310.0,
            },
        },
        300.0 * unit.kelvin,
    )

    assert integrator.getTemperature().value_in_unit(
        unit.kelvin
    ) == pytest.approx(50.0)


def _test_temperature_ramp_rejects_non_langevin_integrator():
    from atom_openmm.equilibration import (
        EquilibrationConfigError,
        _validate_step,
    )

    with pytest.raises(
        EquilibrationConfigError,
        match="requires a Langevin integrator",
    ):
        _validate_step(
            {
                "type": "md",
                "integrator": "verlet",
                "n_steps": 100,
                "thermostat": {
                    "initial_temperature_k": 50.0,
                    "temperature_k": 310.0,
                },
            },
            0,
        )


def _test_distance_restraint_resolves_closest_pair_and_has_flat_bottom(monkeypatch):
    import numpy as np
    import openmm as mm
    from openmm import app, unit

    from atom_openmm import equilibration

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("TST", chain)
    for index in range(3):
        topology.addAtom(f"A{index}", app.element.carbon, residue)
    positions = np.asarray([[0, 0, 0], [1, 0, 0], [0.25, 0, 0]]) * unit.nanometer

    class FakeResolver:
        def __init__(self, topology, positions, **kwargs):
            pass

        def resolve(self, mask, label):
            return {"ligand": [0, 1], "metal": [2]}[mask]

    monkeypatch.setattr(equilibration, "AmberMaskResolver", FakeResolver)
    cfg = {
        "type": "minimization",
        "distance_restraints": [{
            "atom1_mask": "ligand", "atom2_mask": "metal",
            "lower_bound_a": 1.8, "upper_bound_a": 3.0,
            "k_kcal_mol_a2": 25.0,
        }],
    }
    resolved = equilibration._resolve_step_restraints([cfg], topology, positions)[0]
    assert resolved["distance_pairs"] == [{"atom1": 0, "atom2": 2}]
    system = mm.System()
    for _ in range(3):
        system.addParticle(12.0)
    equilibration._apply_restraints(system, cfg, positions, resolved)
    context = mm.Context(system, mm.VerletIntegrator(0.001))
    context.setPositions(positions)
    assert context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    ) == pytest.approx(0.0)
