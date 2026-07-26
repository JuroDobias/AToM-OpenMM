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
