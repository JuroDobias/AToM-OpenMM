import json

import numpy as np
import openmm as mm
from openff.toolkit import Molecule
from openff.units import unit as offunit

from atom_openmm import covalent_parameters


def _molecule():
    return Molecule.from_smiles("CO", hydrogens_are_explicit=False)


def _test_constrain_charge_sum_only_changes_adjustable_atoms():
    charges, correction = covalent_parameters.constrain_charge_sum(
        np.asarray([0.2, -0.1, -0.05]), [1, 2], 0.0
    )
    assert charges[0] == 0.2
    assert np.isclose(charges.sum(), 0.0)
    assert np.isclose(correction, -0.025)


def _test_espaloma_charge_cache_avoids_recalculation(tmp_path, monkeypatch):
    molecule = _molecule()
    expected = np.zeros(molecule.n_atoms)
    calls = []

    def predict(received, model):
        calls.append(model)
        return expected.copy()

    monkeypatch.setattr(covalent_parameters, "_predict_espaloma_charges", predict)
    first, key, first_hit = covalent_parameters.assign_espaloma_nn_charges(
        molecule, cache_dir=tmp_path
    )
    second, second_key, second_hit = covalent_parameters.assign_espaloma_nn_charges(
        molecule, cache_dir=tmp_path
    )
    assert calls == ["espaloma-0.3.2"]
    assert not first_hit and second_hit
    assert key == second_key
    assert np.array_equal(first, second)
    cached = json.loads((tmp_path / f"{key}.charges.json").read_text())
    assert cached["charge_model"] == "espaloma_nn"


def _test_openff_uses_supplied_charges():
    molecule = _molecule()
    charges = np.linspace(-0.1, 0.1, molecule.n_atoms)
    charges -= charges.sum() / molecule.n_atoms
    molecule.partial_charges = charges * offunit.elementary_charge
    system = covalent_parameters.ForceField("openff-2.2.1.offxml").create_openmm_system(
        molecule.to_topology(), charge_from_molecules=[molecule]
    )
    force = next(force for force in system.getForces() if isinstance(force, mm.NonbondedForce))
    observed = np.asarray(
        [
            force.getParticleParameters(index)[0].value_in_unit(mm.unit.elementary_charge)
            for index in range(force.getNumParticles())
        ]
    )
    assert np.allclose(observed, charges)
