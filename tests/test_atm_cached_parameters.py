from types import SimpleNamespace

import numpy as np
import openmm as mm
from openmm import app, unit
import pytest

from atom_openmm.covalent_parameters import VirtualSiteParameter
from atom_openmm.make_atm_system_from_rcpt_lig import (
    _install_cached_ligand_parameters,
)
from atom_openmm.rbfe_workflow import WorkflowConfigError, normalize_setup_options


def _test_normalize_cached_resp_sigma_hole_setup():
    result = normalize_setup_options(
        {
            "setup": {
                "ligand_forcefield": "gaff-2.2.20",
                "ligand_charge_model": "resp-sigma-hole",
                "ligand_parameter_cache": "/cache/ligands",
                "ligand_parameter_protocol": "gaff2-resp-halogen-ep-v2",
                "metal_ions": {"model": "standard_12_6"},
            }
        },
        {},
    )
    assert result["ligandchargemodel"] == "resp-sigma-hole"
    assert result["ligandparametercache"] == "/cache/ligands"
    assert result["ligandparameterprotocol"] == "gaff2-resp-halogen-ep-v2"


def _test_normalize_atm_fixed_sigma_hole_defaults():
    result = normalize_setup_options(
        {
            "setup": {
                "ligand_forcefield": "espaloma-0.3.2",
                "ligand_charge_model": "nn",
                "ligand_sigma_holes": {"halogens": ["Cl", "Br"]},
            }
        },
        {},
    )
    assert result["ligandchargemodel"] == "nn"
    assert result["ligandsigmaholes"]["charges_e"] == {
        "Cl": 0.033, "Br": 0.039,
    }
    assert result["ligandsigmaholes"]["distances_a"] == {
        "Cl": 1.64, "Br": 1.89,
    }


def _test_cached_atm_rejects_panteva_until_force_is_alchemical():
    with pytest.raises(WorkflowConfigError, match="integrated into ATMForce"):
        normalize_setup_options(
            {
                "setup": {
                    "ligand_forcefield": "gaff-2.2.20",
                    "ligand_charge_model": "resp-sigma-hole",
                    "ligand_parameter_cache": "/cache/ligands",
                    "ligand_parameter_protocol": "gaff2-resp-halogen-ep-v2",
                    "metal_ions": {"model": "panteva_m12_6_4"},
                }
            },
            {},
        )


def _nonbonded_system(charges):
    system = mm.System()
    force = mm.NonbondedForce()
    for index, charge in enumerate(charges):
        system.addParticle((12.0 if index != 1 else 35.45) * unit.dalton)
        force.addParticle(
            charge * unit.elementary_charge,
            (0.30 + 0.01 * index) * unit.nanometer,
            (0.10 + 0.01 * index) * unit.kilojoules_per_mole,
        )
    force.addException(
        0, 1, 0.0 * unit.elementary_charge**2,
        1.0 * unit.nanometer, 0.0 * unit.kilojoules_per_mole,
    )
    force.addException(
        1, 2, 0.0 * unit.elementary_charge**2,
        1.0 * unit.nanometer, 0.0 * unit.kilojoules_per_mole,
    )
    force.addException(
        0, 2, charges[0] * charges[2] / 1.2 * unit.elementary_charge**2,
        0.25 * unit.nanometer, 0.02 * unit.kilojoules_per_mole,
    )
    system.addForce(force)
    return system


def _test_install_cached_parameters_appends_ordered_sigma_hole():
    topology = app.Topology()
    chain = topology.addChain("L")
    residue = topology.addResidue("L1", chain)
    topology.addAtom("C1", app.element.carbon, residue)
    topology.addAtom("Cl1", app.element.chlorine, residue)
    topology.addAtom("C2", app.element.carbon, residue)
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.18, 0.0, 0.0], [0.0, 0.15, 0.0]]
    ) * unit.nanometer
    system = _nonbonded_system([0.0, 0.0, 0.0])
    cached = _nonbonded_system([-0.10, -0.05, 0.10])
    parameters = SimpleNamespace(
        molecule=SimpleNamespace(n_atoms=3),
        system=cached,
        charges_e=np.asarray([-0.10, -0.05, 0.10]),
        virtual_sites=(
            VirtualSiteParameter(
                name="CL_EP_1",
                kind="sigma_hole",
                parent_atom_indices=(0, 1, 2),
                distance_a=1.64,
                charge_e=0.05,
            ),
        ),
    )

    output_positions, site_indices = _install_cached_ligand_parameters(
        system, topology, positions, parameters, "L1"
    )

    assert site_indices == [3]
    assert system.getNumParticles() == topology.getNumAtoms() == len(output_positions) == 4
    assert system.isVirtualSite(3)
    assert list(topology.residues())[-1].name == "E1"
    nonbonded = next(
        force for force in system.getForces() if isinstance(force, mm.NonbondedForce)
    )
    observed = [
        nonbonded.getParticleParameters(index)[0].value_in_unit(unit.elementary_charge)
        for index in range(4)
    ]
    assert np.allclose(observed, [-0.10, -0.05, 0.10, 0.05])
    assert np.isclose(sum(observed), 0.0)
    assert np.isclose(
        output_positions[3].value_in_unit(unit.nanometer)[0], 0.344
    )
    exceptions = {
        tuple(sorted(map(int, nonbonded.getExceptionParameters(index)[:2])))
        for index in range(nonbonded.getNumExceptions())
    }
    assert (1, 3) in exceptions
    exception_values = {
        tuple(sorted(map(int, nonbonded.getExceptionParameters(index)[:2]))):
        nonbonded.getExceptionParameters(index)[2]
        for index in range(nonbonded.getNumExceptions())
    }
    assert np.isclose(
        exception_values[(0, 2)].value_in_unit(unit.elementary_charge**2),
        -0.10 * 0.10 / 1.2,
    )
