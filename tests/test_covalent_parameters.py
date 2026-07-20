import json

import numpy as np
import openmm as mm
import pytest
from openff.toolkit import Molecule
from openff.units import unit as offunit
from rdkit import Chem
from rdkit.Chem import AllChem

from atom_openmm import covalent_parameters
from atom_openmm.covalent_dataset import build_capped_thiohemiacetal


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


def _test_modified_residue_charge_copies_fixed_backbone_and_is_neutral():
    ligand = Chem.AddHs(Chem.MolFromSmiles("CC=O"))
    assert AllChem.EmbedMolecule(ligand, randomSeed=7) == 0
    cys = {
        "N": np.asarray([0.0, 0.0, 0.0]),
        "CA": np.asarray([1.45, 0.0, 0.0]),
        "C": np.asarray([2.10, 1.30, 0.0]),
        "O": np.asarray([1.60, 2.40, 0.0]),
        "CB": np.asarray([1.90, -0.80, 1.20]),
        "SG": np.asarray([3.65, -0.75, 1.25]),
        "HG": np.asarray([4.25, -0.10, 1.25]),
    }
    product, metadata = build_capped_thiohemiacetal(ligand, cys)
    molecule = Molecule.from_rdkit(product, allow_undefined_stereo=True)
    charges = np.linspace(-0.2, 0.2, molecule.n_atoms)
    charges -= charges.sum() / molecule.n_atoms
    molecule.partial_charges = charges * offunit.elementary_charge
    system = covalent_parameters.ForceField("openff-2.2.1.offxml").create_openmm_system(
        molecule.to_topology(), charge_from_molecules=[molecule]
    )
    bundle = covalent_parameters.CovalentParameterBundle(molecule, system, charges, "test", {})
    fixed = {"N": -0.4157, "H": 0.2719, "CA": 0.0213, "HA": 0.1124, "C": 0.5973, "O": -0.5679}
    corrected = covalent_parameters.apply_modified_residue_charges(bundle, metadata, fixed)
    mapped = {int(key): int(value) for key, value in metadata["capped_cys_atom_map_indices"].items()}
    assert corrected.charges_e[mapped[4]] == fixed["N"]
    assert corrected.charges_e[mapped[5]] == fixed["CA"]
    assert corrected.charges_e[mapped[8]] == fixed["C"]
    assert corrected.charges_e[mapped[9]] == fixed["O"]
    assert corrected.provenance["modified_residue_charge_e"] == pytest.approx(0.0, abs=1e-10)
    assert corrected.charges_e.sum() == pytest.approx(0.0, abs=1e-10)
