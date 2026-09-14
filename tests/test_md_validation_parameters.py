import csv

import numpy as np

import openmm as mm
from openmm import app, unit
import pytest

from atom_openmm.hu2024_atp import read_b3_cmap, read_frcmod, read_prepi
from atom_openmm.md_validation import (
    MDValidationError, _metric_context, _phase_protocol, _run_md_phase, task_spec,
)
from atom_openmm.metal_ions import (
    MetalIonParameterError, apply_panteva_m1264, c4_kcal_a4_to_kj_nm4,
)
from atom_openmm.receptor_normalization import normalize_legacy_pdb


def _minimal_topology():
    topology = app.Topology()
    chain = topology.addChain()
    atp = topology.addResidue("ATP", chain)
    topology.addAtom("O1A", app.element.oxygen, atp)
    topology.addAtom("N7", app.element.nitrogen, atp)
    mg = topology.addResidue("MG", chain)
    topology.addAtom("MG", app.element.magnesium, mg)
    water = topology.addResidue("HOH", chain)
    topology.addAtom("O", app.element.oxygen, water)
    return topology


def _minimal_system(count):
    system = mm.System()
    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(0.9 * unit.nanometer)
    for _ in range(count):
        system.addParticle(16.0)
        force.addParticle(0.0, 0.3, 0.1)
    system.addForce(force)
    return system


def _tip4p_topology_and_system():
    topology = _minimal_topology()
    water = list(topology.residues())[-1]
    topology.addAtom("M", None, water)
    system = _minimal_system(topology.getNumAtoms())
    system.setVirtualSite(4, mm.ThreeParticleAverageSite(3, 3, 3, 1.0, 0.0, 0.0))
    return topology, system


def _test_panteva_c4_units_and_atp_overrides(tmp_path):
    topology = _minimal_topology()
    system = _minimal_system(topology.getNumAtoms())
    table = tmp_path / "lj_1264_pol.dat"
    table.write_text("OW 1.444\nO2 0.569\nNB 1.090\n")
    result = apply_panteva_m1264(
        system, topology, atom_classes=["O2", "NB", "OW", "OW"],
        polarizability_table=table,
    )
    custom = next(force for force in system.getForces() if isinstance(force, mm.CustomNonbondedForce))
    assert result["magnesium_count"] == 1
    assert custom.getParticleParameters(0)[1] == pytest.approx(c4_kcal_a4_to_kj_nm4(21.25))
    assert custom.getParticleParameters(1)[1] == pytest.approx(c4_kcal_a4_to_kj_nm4(238.75))
    assert custom.getParticleParameters(3)[1] == pytest.approx(c4_kcal_a4_to_kj_nm4(180.5))


def _test_panteva_exclusions_match_base_nonbonded_force(tmp_path):
    topology = _minimal_topology()
    system = _minimal_system(topology.getNumAtoms())
    nonbonded = system.getForce(0)
    nonbonded.addException(0, 1, 0.0, 0.2, 0.0)
    nonbonded.addException(1, 2, 0.0, 0.2, 0.0)
    table = tmp_path / "lj_1264_pol.dat"
    table.write_text("OW 1.444\nO2 0.569\nNB 1.090\n")
    apply_panteva_m1264(
        system, topology, atom_classes=["O2", "NB", "OW", "OW"],
        polarizability_table=table,
    )
    custom = system.getForce(1)
    assert [tuple(custom.getExclusionParticles(i)) for i in range(custom.getNumExclusions())] == [
        (0, 1), (1, 2),
    ]
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3),
    )
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("CPU"))
    context.setPositions(np.asarray([[0, 0, 0], [0.2, 0, 0], [1, 0, 0], [1.5, 0, 0]]) * unit.nanometer)
    assert np.isfinite(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def _test_panteva_rejects_mismatched_water_model(tmp_path):
    table = tmp_path / "lj_1264_pol.dat"
    table.write_text("OW 1.444\nO2 0.569\nNB 1.090\n")
    with pytest.raises(MetalIonParameterError, match="TIP4P-Ew"):
        apply_panteva_m1264(
            _minimal_system(4), _minimal_topology(),
            atom_classes=["O2", "NB", "OW", "OW"],
            polarizability_table=table, water_model="tip3p",
        )


def _test_panteva_gives_tip4p_virtual_site_zero_c4(tmp_path):
    topology, system = _tip4p_topology_and_system()
    table = tmp_path / "lj_1264_pol.dat"
    table.write_text("OW 1.444\nO2 0.569\nNB 1.090\n")
    apply_panteva_m1264(
        system, topology, atom_classes=["O2", "NB", "OW", "OW", "EP"],
        polarizability_table=table,
    )
    custom = system.getForce(1)
    assert custom.getParticleParameters(3)[1] == pytest.approx(c4_kcal_a4_to_kj_nm4(180.5))
    assert custom.getParticleParameters(4)[1] == pytest.approx(0.0)


def _test_hu_parameter_readers_normalize_prime_names(tmp_path):
    prepi = tmp_path / "ATP.prepi"
    prepi.write_text(
        "   4  O5'   OY    M    3 2 1 1.0 100.0 180.0 -0.5\nLOOP\n"
    )
    frcmod = tmp_path / "ATP.frcmod"
    frcmod.write_text("BOND\nOY-P 230.0 1.61\nANGLE\nOY-P-O2 100.0 108.2\nDIHE\nP-OY-CT-CT 3 1.15 0.0 3.0\n")
    mod = tmp_path / "mod.py"
    mod.write_text("cmap_matrix_B3 = [" + ",".join(["0.0"] * 576) + "]\n")
    assert read_prepi(prepi)["O5*"]["type"] == "OY"
    assert len(read_frcmod(frcmod)["ANGLE"]) == 1
    assert len(read_b3_cmap(mod)) == 576


def _test_hu_torsion_fields_have_phase_and_periodicity(tmp_path):
    from atom_openmm.hu2024_atp import read_frcmod

    path = tmp_path / "test.frcmod"
    path.write_text("DIHE\nO3-P -OS-P    1       0.0948     51.5226      3.0000\n")
    assert read_frcmod(path)["DIHE"] == [["O3-P-OS-P", "1", "0.0948", "51.5226", "3.0000"]]


def _test_receptor_normalization_preserves_records_and_removes_epw(tmp_path):
    source = tmp_path / "source.pdb"
    source.write_text(
        "TER\n"
        "ATOM      1  SG  CY3 A   1       0.000   0.000   0.000  1.00  0.00          S\n"
        "TER\n"
        "HETATM    2 ZN   ZN3 A   2       1.000   0.000   0.000  1.00  0.00         ZN\n"
        "ATOM      3 EPW  WAT A   3       2.000   0.000   0.000  1.00  0.00          O\n"
        "END\n"
    )
    destination = tmp_path / "normalized.pdb"
    result = normalize_legacy_pdb(source, destination)
    text = destination.read_text()
    assert "CYS" in text and " ZN " in text
    assert "EPW" not in text
    assert "TER" in text
    assert result["removed_extra_particles"] == 1


def _test_matrix_task_mapping():
    config = {"replicates": 3}
    assert task_spec(config, 0)["variant"] == "legacy_atp_12_6"
    assert task_spec(config, 3)["variant"] == "legacy_atp_panteva_m12_6_4"
    assert task_spec(config, 11)["replicate"] == 3
    with pytest.raises(MDValidationError):
        task_spec(config, 12)


def _test_first_nvt_resets_velocities_after_minimization():
    phases = _phase_protocol({})
    assert [phase["id"] for phase in phases if phase.get("reset_velocities")] == ["restrained_nvt"]


def _test_explicit_validation_protocol_is_preserved():
    config = {"protocol": {"steps": [
        {"id": "ligand_min", "kind": "min", "max_iterations": 123},
        {"id": "npt", "kind": "md", "steps": 456, "npt": True,
         "restraint_selection": "protein_dna_heavy"},
    ]}}
    phases = _phase_protocol(config)
    assert phases[0]["max_iterations"] == 123
    assert phases[1]["steps"] == 456
    assert phases[1]["restraint_selection"] == "protein_dna_heavy"


def _test_short_cpu_md_writes_metrics_and_checkpoint(tmp_path):
    topology = app.Topology()
    chain = topology.addChain()
    for name, element in (
        ("MET", app.element.carbon), ("ATP", app.element.oxygen),
        ("UNK", app.element.carbon), ("MG", app.element.magnesium),
        ("HOH", app.element.oxygen),
    ):
        residue = topology.addResidue(name, chain)
        topology.addAtom("O" if element == app.element.oxygen else "C", element, residue)
    system = mm.System()
    for _ in topology.atoms():
        system.addParticle(12.0)
    vectors = (
        mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3)
    ) * unit.nanometer
    topology.setPeriodicBoxVectors(vectors)
    system.setDefaultPeriodicBoxVectors(*vectors)
    positions = np.asarray([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.5, 0.0, 0.0],
        [0.1, 0.2, 0.0], [0.2, 0.1, 0.0],
    ]) * unit.nanometer
    config = {"platform": "Reference", "timestep_fs": 1.0, "report_interval_steps": 1}
    phase = {"id": "production", "kind": "md", "steps": 3,
             "npt": False, "restrained": False, "production": True}
    final = _run_md_phase(
        config, topology, system, positions, None, phase, tmp_path, 123,
        _metric_context(topology, positions),
    )
    assert len(final.getPositions()) == topology.getNumAtoms()
    with (tmp_path / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert "mg1_coordination_0.25nm" in rows[0]
    assert "atp_protein_aligned_rmsd_nm" in rows[0]
    assert (tmp_path / "trajectory.dcd").is_file()
    assert (tmp_path / "production_progress.yaml").exists()
    (tmp_path / "production_running_state.xml").write_text(mm.XmlSerializer.serialize(final))
    (tmp_path / "production_progress.yaml").write_text("completed_steps: 3\ntarget_steps: 4\n")
    resumed = _run_md_phase(
        config, topology, system, positions, None,
        {**phase, "steps": 4}, tmp_path, 123,
        _metric_context(topology, positions),
    )
    assert len(resumed.getPositions()) == topology.getNumAtoms()
    with (tmp_path / "metrics.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    assert (tmp_path / "production_progress.yaml").exists()


def _test_segmented_checkpoint_metadata_for_short_md(tmp_path):
    topology = app.Topology()
    chain = topology.addChain()
    for residue_name, element in (("MET", app.element.carbon), ("ATP", app.element.oxygen)):
        residue = topology.addResidue(residue_name, chain)
        topology.addAtom("C" if residue_name == "MET" else "O", element, residue)
    vectors = (mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3)) * unit.nanometer
    topology.setPeriodicBoxVectors(vectors)
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*vectors)
    for _ in topology.atoms():
        system.addParticle(12.0)
    positions = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]) * unit.nanometer
    config = {"platform": "Reference", "timestep_fs": 1.0,
              "report_interval_steps": 1, "checkpoint_interval_steps": 2}
    phase = {"id": "production", "kind": "md", "steps": 2,
             "npt": False, "restrained": False, "production": True}
    _run_md_phase(config, topology, system, positions, None, phase, tmp_path, 42,
                  _metric_context(topology, positions))
    with (tmp_path / "metrics.csv").open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([3, 3.0, 999.0, 999.0])
    _run_md_phase(config, topology, system, positions, None,
                  {**phase, "steps": 4}, tmp_path, 42, _metric_context(topology, positions))
    import yaml

    segments = yaml.safe_load((tmp_path / "trajectory_segments.yaml").read_text())["segments"]
    assert [(entry["start_step"], entry["last_committed_step"]) for entry in segments] == [(0, 2), (2, 4)]
    with (tmp_path / "metrics.csv").open(newline="") as handle:
        assert [int(row["step"]) for row in csv.DictReader(handle)] == [1, 2, 3, 4]


def _test_panteva_c4_force_energy(tmp_path):
    topology = app.Topology()
    atp = topology.addResidue("ATP", topology.addChain())
    topology.addAtom("O1A", app.element.oxygen, atp)
    magnesium = topology.addResidue("MG", topology.addChain())
    topology.addAtom("MG", app.element.magnesium, magnesium)
    system = mm.System()
    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(0.9 * unit.nanometer)
    for _ in range(2):
        system.addParticle(16.0)
        force.addParticle(0.0, 0.3, 0.0)
    system.addForce(force)
    vectors = (mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3)) * unit.nanometer
    system.setDefaultPeriodicBoxVectors(*vectors)
    table = tmp_path / "lj_1264_pol.dat"
    table.write_text("OW 1.444\nO2 0.569\nMg2+ 0.048\n")
    apply_panteva_m1264(system, topology, atom_classes=["O2", "Mg2+"], polarizability_table=table)
    context = mm.Context(system, mm.VerletIntegrator(0.001 * unit.picosecond),
                         mm.Platform.getPlatformByName("Reference"))
    context.setPositions(np.asarray([[0, 0, 0], [0.2, 0, 0]]) * unit.nanometer)
    energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert energy == pytest.approx(-c4_kcal_a4_to_kj_nm4(21.25) / 0.2**4, abs=1e-4)
