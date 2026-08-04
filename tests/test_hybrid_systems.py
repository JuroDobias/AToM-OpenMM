from openmm import Vec3, app, unit

from atom_openmm.hybrid_systems import (
    HybridSystemError,
    _forcefield_files,
    _repair_template_bonds,
    _solvation_box_options,
)


def test_openmmforcefields_resources_are_resolved():
    files = _forcefield_files(
        ["openmmforcefields:amber/phosaa14SB.xml"], []
    )

    assert files[0].endswith("openmmforcefields/ffxml/amber/phosaa14SB.xml")


def test_missing_openmmforcefields_resource_is_rejected():
    try:
        _forcefield_files(["openmmforcefields:amber/not-present.xml"], [])
    except HybridSystemError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("missing force-field resource was accepted")


def test_supplemental_residue_bonds_are_repaired_from_template():
    files = _forcefield_files(
        [
            "openmmforcefields:amber/ff14SB.xml",
            "openmmforcefields:amber/phosaa14SB.xml",
        ],
        [],
    )
    forcefield = app.ForceField(*files)
    template = forcefield._templates["TPO"]
    topology = app.Topology()
    residue = topology.addResidue("TPO", topology.addChain())
    for atom in template.atoms:
        topology.addAtom(atom.name, atom.element, residue)
    positions = [Vec3(0, 0, 0) for _ in template.atoms] * unit.nanometer

    _repair_template_bonds(topology, positions, forcefield)

    assert len(list(topology.bonds())) == len(template.bonds)


def test_rectangular_solvation_box_uses_axis_extents_plus_padding():
    positions = [Vec3(-1, 0, 2), Vec3(3, 5, 4)] * unit.nanometer

    options = _solvation_box_options(positions, 10.0, "rectangular")

    dimensions = options["boxSize"].value_in_unit(unit.nanometer)
    assert list(dimensions) == [6.0, 7.0, 4.0]
