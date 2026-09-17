from pathlib import Path
import shutil
import subprocess

import numpy as np
import openmm as mm
import pytest
import yaml
from openff.toolkit import Molecule
from openff.units import unit as offunit
from openmm import app, unit
from rdkit import Chem

from atom_openmm.covalent_hybrid import CovalentHybridMolecule, DummyBondedScales
from atom_openmm.covalent_parameters import (
    CovalentParameterBundle,
    VirtualSiteParameter,
)
from atom_openmm.covalent_softcore import create_softcore_hamiltonian
from atom_openmm.hybrid_virtual_sites import (
    HybridVirtualSiteError,
    add_alchemical_sigma_holes,
    add_common_sigma_holes,
)
from atom_openmm.hybrid_parameters import _apply_fixed_sigma_holes
from atom_openmm.ligand_parameterization import (
    LigandParameterizationError,
    _canonical_identity,
    _find_sigma_holes,
    _parse_resp_charges,
    _parse_amber_esp,
    _resp_input,
    _write_multi_esp,
    cache_identity,
    load_cached_parameters,
    normalize_protocol,
    normalize_sigma_hole_settings,
)


def _molecule(smiles="CCCl"):
    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=1)
    molecule.generate_unique_atom_names()
    return molecule


def _test_cache_identity_does_not_depend_on_atom_order():
    molecule = _molecule()
    rdkit = molecule.to_rdkit()
    order = list(reversed(range(rdkit.GetNumAtoms())))
    reordered = Molecule.from_rdkit(Chem.RenumberAtoms(rdkit, order))
    protocol = normalize_protocol(None)
    assert _canonical_identity(molecule) == _canonical_identity(reordered)
    assert cache_identity(molecule, protocol) == cache_identity(reordered, protocol)


def _test_cache_identity_ignores_execution_resources():
    molecule = _molecule()
    first = normalize_protocol(None)
    second = normalize_protocol({
        "qm": {
            "cores_per_conformer": 2,
            "memory_mb_per_conformer": 4000,
            "parallel_conformers": 1,
            "executable": "/opt/gaussian/g16",
        },
        "resp": {
            "executable": "/opt/amber/bin/resp",
            "espgen_executable": "/opt/amber/bin/espgen",
        },
    })
    assert cache_identity(molecule, first) == cache_identity(molecule, second)


def _test_protocol_rejects_unsupported_qm_engine():
    with pytest.raises(LigandParameterizationError, match="gaussian16"):
        normalize_protocol({"qm": {"engine": "orca"}})


def _test_sigma_hole_selector_is_normalized_and_validated():
    assert normalize_protocol(None)["sigma_holes"]["halogens"] == ["Cl"]
    assert normalize_sigma_hole_settings({"halogens": ["cl", "BR", "Cl"]})[
        "halogens"
    ] == ["Cl", "Br"]
    with pytest.raises(LigandParameterizationError, match="not both"):
        normalize_sigma_hole_settings({
            "halogens": ["Cl"],
            "smarts": "[#6]-[#17]",
        })
    with pytest.raises(LigandParameterizationError, match="unsupported"):
        normalize_sigma_hole_settings({"halogens": ["At"]})

    mixed = normalize_sigma_hole_settings({
        "halogens": ["Cl", "Br"],
        "distances_a": {"cl": 1.64, "BR": 1.89},
    })
    assert mixed["distances_a"] == {"Cl": 1.64, "Br": 1.89}
    with pytest.raises(LigandParameterizationError, match="missing selected"):
        normalize_sigma_hole_settings({
            "halogens": ["Cl", "Br"],
            "distances_a": {"Cl": 1.64},
        })


def _test_sigma_hole_selector_controls_eligible_elements():
    molecule = _molecule("FCCl")
    chlorine_only = normalize_protocol({"sigma_holes": {"halogens": ["Cl"]}})
    both = normalize_protocol({"sigma_holes": {"halogens": ["F", "Cl"]}})

    chlorine_sites = _find_sigma_holes(molecule, chlorine_only)
    both_sites = _find_sigma_holes(molecule, both)

    assert len(chlorine_sites) == 1
    assert molecule.atoms[chlorine_sites[0][1]].symbol == "Cl"
    assert {molecule.atoms[site[1]].symbol for site in both_sites} == {"F", "Cl"}


def _test_legacy_sigma_hole_smarts_remains_supported():
    molecule = _molecule("FCCl")
    protocol = normalize_protocol({
        "sigma_holes": {"smarts": "[#6:1]-[#9X1:2]"}
    })
    sites = _find_sigma_holes(molecule, protocol)
    assert len(sites) == 1
    assert molecule.atoms[sites[0][1]].symbol == "F"


def _test_resp_input_equivalences_repeated_conformers():
    text = _resp_input([6, 17], 1, 2, 0, 0.0005)
    assert "nmol=2" in text
    assert "conformer 1\n    0    3" in text
    assert "conformer 2\n    0    3" in text
    assert text.count(" 1.0") == 2
    assert "    2\n    1    2    2    2" in text
    assert text.rstrip().endswith("    1    3    2    3")
    assert "    0    0\n\n\n    2" in text
    assert "    0    0\n\n\n\n    2" not in text


def _test_parse_resp_charges_accepts_equivalent_repeated_values(tmp_path):
    path = tmp_path / "resp.chg"
    path.write_text(" 0.10 -0.15 0.05\n 0.10 -0.15 0.05\n")
    observed = _parse_resp_charges(path, 3, 2)
    assert np.allclose(observed, [0.10, -0.15, 0.05])


def _test_parse_resp_charges_accepts_adjacent_fixed_width_values(tmp_path):
    path = tmp_path / "resp.chg"
    path.write_text(" -8.606855-20.470074 29.076929\n")
    observed = _parse_resp_charges(path, 3, 1)
    assert np.allclose(observed, [-8.606855, -20.470074, 29.076929])


def _test_parse_resp_charges_rejects_fortran_overflow(tmp_path):
    path = tmp_path / "resp.chg"
    path.write_text(" -0.001044**********\n")
    with pytest.raises(LigandParameterizationError, match="overflowed"):
        _parse_resp_charges(path, 2, 1)


def _test_parse_amber_esp_accepts_adjacent_large_grid_count(tmp_path):
    centers = np.asarray([[0.0, 0.0, 0.0]])
    potentials = np.zeros(10000)
    points = np.ones((10000, 3))
    path = tmp_path / "large.esp"
    _write_multi_esp(path, [(centers, potentials, points)])

    parsed_centers, parsed_potentials, parsed_points = _parse_amber_esp(path)
    assert parsed_centers.shape == (1, 3)
    assert parsed_potentials.shape == (10000,)
    assert parsed_points.shape == (10000, 3)


@pytest.mark.skipif(shutil.which("resp") is None, reason="AmberTools RESP is unavailable")
def _test_multiconformer_input_is_accepted_by_amber_resp(tmp_path):
    centers = np.asarray(
        [[0.0, 0.0, 0.0], [3.4, 0.0, 0.0], [6.5, 0.0, 0.0]],
        dtype=float,
    )
    charges = np.asarray([0.15, -0.20, 0.05])
    points = np.asarray(
        [
            [x, y, z]
            for x in (-3.0, 1.5, 5.0, 9.0)
            for y in (-3.5, 3.5)
            for z in (-2.5, 2.5)
        ],
        dtype=float,
    )
    potentials = np.sum(
        charges[None, :] /
        np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2),
        axis=1,
    )
    shifted = centers + np.asarray([0.2, -0.1, 0.15])
    shifted_points = points + np.asarray([0.2, -0.1, 0.15])
    _write_multi_esp(
        tmp_path / "esp.dat",
        [
            (centers, potentials, points),
            (shifted, potentials, shifted_points),
        ],
    )
    (tmp_path / "resp.in").write_text(_resp_input([6, 17], 1, 2, 0, 0.0005))
    completed = subprocess.run(
        [
            shutil.which("resp"), "-O", "-i", "resp.in", "-o", "resp.out",
            "-p", "resp.pch", "-t", "resp.chg", "-e", "esp.dat",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    fitted = _parse_resp_charges(tmp_path / "resp.chg", 3, 2)
    assert fitted.sum() == pytest.approx(0.0, abs=1.0e-5)
    assert fitted[0] > 0.0
    assert fitted[1] < 0.0
    assert fitted[2] > 0.0


def _simple_system(charges):
    system = mm.System()
    for mass in (12.0, 35.45, 12.0):
        system.addParticle(mass * unit.dalton)
    bonds = mm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.18 * unit.nanometer, 100.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    bonds.addBond(0, 2, 0.14 * unit.nanometer, 100.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    system.addForce(bonds)
    nonbonded = mm.NonbondedForce()
    for charge in charges:
        nonbonded.addParticle(charge, 0.3, 0.1)
    nonbonded.addException(0, 1, 0.0, 1.0, 0.0)
    nonbonded.addException(1, 2, 0.0, 1.0, 0.0)
    system.addForce(nonbonded)
    return system


def _test_cached_bundle_is_remapped_to_requested_atom_order(tmp_path):
    stored = _molecule()
    protocol = normalize_protocol(None)
    molecule_key, artifact_key = cache_identity(stored, protocol)
    artifact = tmp_path / "artifacts" / artifact_key
    artifact.mkdir(parents=True)
    stored.to_file(str(artifact / "ligand.sdf"), file_format="SDF")
    charges = np.linspace(-0.2, 0.2, stored.n_atoms) + 1.0e-6
    expected_charges = charges - charges.sum() / stored.n_atoms
    system = mm.System()
    for atom in stored.atoms:
        system.addParticle(float(atom.mass.m_as(offunit.dalton)) * unit.dalton)
    force = mm.NonbondedForce()
    for charge in charges:
        force.addParticle(charge, 0.3, 0.0)
    system.addForce(force)
    (artifact / "system.xml").write_text(mm.XmlSerializer.serialize(system))
    (artifact / "manifest.yaml").write_text(yaml.safe_dump({
        "schema_version": 1,
        "status": "completed",
        "artifact_key": artifact_key,
        "molecule": _canonical_identity(stored),
        "protocol": protocol,
        "atomic_charges_e": charges.tolist(),
        "virtual_sites": [],
        "files": {"molecule": "ligand.sdf", "system": "system.xml"},
    }))
    index = tmp_path / "index" / molecule_key
    index.mkdir(parents=True)
    (index / f"{protocol['id']}.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "artifact_key": artifact_key,
    }))
    requested_path = tmp_path / "requested.sdf"
    rdkit = Chem.RenumberAtoms(stored.to_rdkit(), list(reversed(range(stored.n_atoms))))
    requested = Molecule.from_rdkit(rdkit)
    requested.to_file(str(requested_path), file_format="SDF")
    loaded = load_cached_parameters(
        requested_path, cache_dir=tmp_path, protocol_id=protocol["id"]
    )
    assert loaded.cache_key == artifact_key
    assert sorted(loaded.charges_e) == pytest.approx(sorted(expected_charges))
    assert loaded.total_charge_e == pytest.approx(0.0, abs=1.0e-12)
    observed = next(
        force for force in loaded.system.getForces()
        if isinstance(force, mm.NonbondedForce)
    )
    assert [
        observed.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in range(requested.n_atoms)
    ] == pytest.approx(loaded.charges_e)


def _hybrid_bundle(site_charge):
    topology = app.Topology()
    chain = topology.addChain("A")
    residue = topology.addResidue("HYB", chain)
    topology.addAtom("C1", app.element.carbon, residue)
    topology.addAtom("CL", app.element.chlorine, residue)
    topology.addAtom("C2", app.element.carbon, residue)
    positions = np.asarray([[0.0, 0.0, 0.0], [0.18, 0.0, 0.0], [0.0, 0.14, 0.0]]) * unit.nanometer
    return topology, positions, _simple_system([0.1, -0.15, 0.0]), VirtualSiteParameter(
        name="CL_EP_1", kind="sigma_hole", parent_atom_indices=(0, 1, 2),
        distance_a=1.64, charge_e=site_charge,
    )


def _test_common_sigma_hole_is_added_to_both_endpoints():
    topology, positions, endpoint_a, site_a = _hybrid_bundle(0.05)
    _, _, endpoint_b, site_b = _hybrid_bundle(0.06)
    hybrid = CovalentHybridMolecule(
        topology=topology, positions=positions, endpoint_a=endpoint_a,
        endpoint_b=endpoint_b, map_a_to_b={0: 0, 1: 1, 2: 2},
        map_a_to_hybrid={0: 0, 1: 1, 2: 2},
        map_b_to_hybrid={0: 0, 1: 1, 2: 2}, unique_a=(), unique_b=(),
        anchor_pairs=(), attachment_pairs=None,
        dummy_bonded_scales=DummyBondedScales(), dummy_core_nonbonded="off",
    )
    molecule = _molecule("CCl")
    parameters_a = CovalentParameterBundle(
        molecule, endpoint_a, np.asarray([0.1, -0.15, 0.0]), "a", {}, (site_a,)
    )
    parameters_b = CovalentParameterBundle(
        molecule, endpoint_b, np.asarray([0.1, -0.16, 0.0]), "b", {}, (site_b,)
    )
    augmented = add_common_sigma_holes(hybrid, parameters_a, parameters_b)
    assert augmented.topology.getNumAtoms() == 4
    assert augmented.endpoint_a.isVirtualSite(3)
    assert augmented.endpoint_b.isVirtualSite(3)
    assert augmented.endpoint_a.getParticleMass(3) == 0.0 * unit.dalton
    force_a = next(
        force for force in augmented.endpoint_a.getForces()
        if isinstance(force, mm.NonbondedForce)
    )
    assert force_a.getParticleParameters(3)[0].value_in_unit(unit.elementary_charge) == pytest.approx(0.05)
    assert force_a.getNumExceptions() > endpoint_a.getForce(1).getNumExceptions()
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(
        augmented.endpoint_a, integrator, mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(augmented.positions)
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    assert np.isfinite(energy.value_in_unit(unit.kilojoules_per_mole))


def _test_cl_to_br_uses_endpoint_specific_sigma_holes():
    topology, positions, endpoint_a, site_a = _hybrid_bundle(0.05)
    _, _, endpoint_b, site_b = _hybrid_bundle(0.07)
    site_b = VirtualSiteParameter(
        name="BR_EP_1", kind="sigma_hole", parent_atom_indices=(0, 1, 2),
        distance_a=1.90, charge_e=site_b.charge_e,
    )
    hybrid = CovalentHybridMolecule(
        topology=topology, positions=positions, endpoint_a=endpoint_a,
        endpoint_b=endpoint_b, map_a_to_b={0: 0, 1: 1, 2: 2},
        map_a_to_hybrid={0: 0, 1: 1, 2: 2},
        map_b_to_hybrid={0: 0, 1: 1, 2: 2}, unique_a=(), unique_b=(),
        anchor_pairs=(), attachment_pairs=None,
        dummy_bonded_scales=DummyBondedScales(), dummy_core_nonbonded="off",
    )
    parameters_a = CovalentParameterBundle(
        _molecule("CCl"), endpoint_a, np.asarray([0.1, -0.15, 0.0]),
        "a", {}, (site_a,),
    )
    parameters_b = CovalentParameterBundle(
        _molecule("CBr"), endpoint_b, np.asarray([0.1, -0.17, 0.0]),
        "b", {}, (site_b,),
    )

    augmented = add_alchemical_sigma_holes(hybrid, parameters_a, parameters_b)

    assert augmented.unique_particle_indices_a == ()
    assert augmented.unique_particle_indices_b == ()
    assert [site.role for site in augmented.alchemical_virtual_sites] == [
        "mapped_a", "mapped_b"
    ]
    force_a = next(
        force for force in augmented.endpoint_a.getForces()
        if isinstance(force, mm.NonbondedForce)
    )
    force_b = next(
        force for force in augmented.endpoint_b.getForces()
        if isinstance(force, mm.NonbondedForce)
    )
    charges_a = [
        force_a.getParticleParameters(index)[0].value_in_unit(unit.elementary_charge)
        for index in (3, 4)
    ]
    charges_b = [
        force_b.getParticleParameters(index)[0].value_in_unit(unit.elementary_charge)
        for index in (3, 4)
    ]
    assert charges_a == pytest.approx([0.05, 0.0])
    assert charges_b == pytest.approx([0.0, 0.07])

    switching = create_softcore_hamiltonian(
        augmented.endpoint_a,
        augmented.endpoint_b,
        (),
        (),
        total_steps=20,
        path_mode="concerted",
    )
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(
        switching.system, integrator, mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(augmented.positions)
    observed = []
    for position in (0, -1):
        for name, schedule in switching.parameter_values.items():
            context.setParameter(name, schedule[position])
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        observed.append(energy.value_in_unit(unit.kilojoules_per_mole))
    expected = []
    for system in (augmented.endpoint_a, augmented.endpoint_b):
        endpoint_integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
        endpoint_context = mm.Context(
            system, endpoint_integrator, mm.Platform.getPlatformByName("Reference")
        )
        endpoint_context.setPositions(augmented.positions)
        expected.append(
            endpoint_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
        )
        del endpoint_context, endpoint_integrator
    assert np.all(np.isfinite(observed))
    assert observed == pytest.approx(expected, abs=1.0e-5)


def _test_sigma_hole_on_mapped_h_to_cl_transmutation_uses_mapped_charge():
    topology, positions, endpoint_a, _ = _hybrid_bundle(0.0)
    _, _, endpoint_b, site_b = _hybrid_bundle(0.06)
    hybrid = CovalentHybridMolecule(
        topology=topology, positions=positions, endpoint_a=endpoint_a,
        endpoint_b=endpoint_b, map_a_to_b={0: 0, 1: 1, 2: 2},
        map_a_to_hybrid={0: 0, 1: 1, 2: 2},
        map_b_to_hybrid={0: 0, 1: 1, 2: 2}, unique_a=(), unique_b=(),
        anchor_pairs=(), attachment_pairs=None,
        dummy_bonded_scales=DummyBondedScales(), dummy_core_nonbonded="off",
    )
    parameters_a = CovalentParameterBundle(
        _molecule("C"), endpoint_a, np.asarray([0.1, -0.15, 0.0]), "a", {}, ()
    )
    parameters_b = CovalentParameterBundle(
        _molecule("CCl"), endpoint_b, np.asarray([0.1, -0.16, 0.0]),
        "b", {}, (site_b,),
    )

    augmented = add_alchemical_sigma_holes(hybrid, parameters_a, parameters_b)

    assert augmented.unique_particle_indices_a == ()
    assert augmented.unique_particle_indices_b == ()
    assert augmented.alchemical_virtual_sites[0].role == "mapped_b"
    force_a = next(
        force for force in augmented.endpoint_a.getForces()
        if isinstance(force, mm.NonbondedForce)
    )
    force_b = next(
        force for force in augmented.endpoint_b.getForces()
        if isinstance(force, mm.NonbondedForce)
    )
    assert force_a.getParticleParameters(3)[0].value_in_unit(
        unit.elementary_charge
    ) == pytest.approx(0.0)
    assert force_b.getParticleParameters(3)[0].value_in_unit(
        unit.elementary_charge
    ) == pytest.approx(0.06)


def _test_unchanged_halogen_rejects_missing_endpoint_site():
    topology, positions, endpoint_a, site_a = _hybrid_bundle(0.05)
    _, _, endpoint_b, _ = _hybrid_bundle(0.0)
    hybrid = CovalentHybridMolecule(
        topology=topology, positions=positions, endpoint_a=endpoint_a,
        endpoint_b=endpoint_b, map_a_to_b={0: 0, 1: 1, 2: 2},
        map_a_to_hybrid={0: 0, 1: 1, 2: 2},
        map_b_to_hybrid={0: 0, 1: 1, 2: 2}, unique_a=(), unique_b=(),
        anchor_pairs=(), attachment_pairs=None,
        dummy_bonded_scales=DummyBondedScales(), dummy_core_nonbonded="off",
    )
    parameters_a = CovalentParameterBundle(
        _molecule("CCl"), endpoint_a, np.asarray([0.1, -0.15, 0.0]),
        "a", {}, (site_a,),
    )
    parameters_b = CovalentParameterBundle(
        _molecule("CCl"), endpoint_b, np.asarray([0.1, -0.15, 0.0]), "b", {}, ()
    )

    with pytest.raises(HybridVirtualSiteError, match="only endpoint A"):
        add_alchemical_sigma_holes(hybrid, parameters_a, parameters_b)


def _test_fixed_sigma_hole_transfers_charge_from_chlorine():
    molecule = _molecule("CCl")
    system = mm.System()
    force = mm.NonbondedForce()
    charges = np.zeros(molecule.n_atoms)
    for atom in molecule.atoms:
        system.addParticle(atom.mass.m_as(offunit.dalton) * unit.dalton)
        force.addParticle(0.0, 0.3, 0.0)
    system.addForce(force)

    adjusted, sites = _apply_fixed_sigma_holes(
        molecule,
        system,
        charges,
        {"charge_e": 0.03, "distance_a": 1.64},
    )

    chlorine = next(
        index for index, atom in enumerate(molecule.atoms)
        if atom.atomic_number == 17
    )
    assert adjusted[chlorine] == pytest.approx(-0.03)
    assert len(sites) == 1
    assert sites[0].charge_e == pytest.approx(0.03)
    assert adjusted.sum() + sites[0].charge_e == pytest.approx(0.0)
    observed = force.getParticleParameters(chlorine)[0].value_in_unit(
        unit.elementary_charge
    )
    assert observed == pytest.approx(-0.03)


def _test_fixed_sigma_holes_are_a_noop_without_selected_halogen():
    molecule = _molecule("CC")
    system = mm.System()
    force = mm.NonbondedForce()
    charges = np.zeros(molecule.n_atoms)
    for atom in molecule.atoms:
        system.addParticle(atom.mass.m_as(offunit.dalton) * unit.dalton)
        force.addParticle(0.0, 0.3, 0.0)
    system.addForce(force)

    adjusted, sites = _apply_fixed_sigma_holes(
        molecule,
        system,
        charges,
        {"halogens": ["Cl"], "charge_e": 0.03, "distance_a": 1.64},
    )
    assert np.array_equal(adjusted, charges)
    assert sites == ()
