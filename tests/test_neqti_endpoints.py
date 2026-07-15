import numpy as np
import openmm as mm
import pytest
from openmm import unit


def _keywords():
    return {
        "LIGAND1_ATOMS": [0, 1, 2],
        "LIGAND2_ATOMS": [3, 4, 5],
        "LIGAND1_VAR_ATOMS": [1, 2],
        "LIGAND2_VAR_ATOMS": [4, 5],
        "LIGAND1_COMMON_ATOMS": [0],
        "LIGAND2_COMMON_ATOMS": [3],
        "LIGAND1_ATTACH_ATOM": 0,
        "LIGAND2_ATTACH_ATOM": 3,
        "LIGAND1_CM_ATOMS": [0],
        "LIGAND2_CM_ATOMS": [3],
        "ALIGN_LIGAND1_REF_ATOMS": [0, 1, 2],
        "ALIGN_LIGAND2_REF_ATOMS": [2, 1, 0],
        "VSITE_KFTHETA_LIG1": 1.0,
        "VSITE_KFTHETA_LIG2": 2.0,
        "EXCLUSION_POT_MOL1_INDEXES": [6, 7],
        "EXCLUSION_POT_MOL2_INDEXES": [3, 4, 5],
    }


def _positions():
    return np.asarray([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [1.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [2.0, 0.0, 0.0],
        [2.1, 0.0, 0.0],
    ]) * unit.nanometer


def _test_endpoint_roles_preserve_current_a_b_convention():
    from atom_openmm.neqti_endpoints import endpoint_ligand_roles

    roles_a = endpoint_ligand_roles(_keywords(), "a")
    roles_b = endpoint_ligand_roles(_keywords(), "b")

    assert (roles_a.bound_ligand, roles_a.unbound_ligand) == ("L1", "L2")
    assert roles_a.bound_atoms == (0, 1, 2)
    assert (roles_b.bound_ligand, roles_b.unbound_ligand) == ("L2", "L1")
    assert roles_b.unbound_atoms == (0, 1, 2)


def _test_native_b_keywords_swap_ligand_restraint_roles():
    from atom_openmm.neqti_endpoints import native_endpoint_keywords

    result = native_endpoint_keywords(_keywords(), "b")

    assert result["LIGAND1_ATOMS"] == [3, 4, 5]
    assert result["LIGAND2_ATOMS"] == [0, 1, 2]
    assert result["ALIGN_LIGAND1_REF_ATOMS"] == [2, 1, 0]
    assert result["VSITE_KFTHETA_LIG1"] == 2.0
    assert result["EXCLUSION_POT_MOL2_INDEXES"] == [0, 1, 2]
    assert result["NATIVE_BOUND_LIGAND"] == "L2"


def _test_b_position_mapping_is_involutive_and_preserves_internal_geometry():
    from atom_openmm.neqti_endpoints import (
        map_atm_to_native_positions,
        map_native_to_atm_positions,
    )

    positions = _positions()
    native = map_atm_to_native_positions(positions, "b", _keywords())
    restored = map_native_to_atm_positions(native, "b", _keywords())
    native_nm = native.value_in_unit(unit.nanometer)
    original_nm = positions.value_in_unit(unit.nanometer)

    assert np.asarray(restored.value_in_unit(unit.nanometer)) == pytest.approx(original_nm)
    assert native_nm[0] == pytest.approx(original_nm[3])
    assert native_nm[1] - native_nm[0] == pytest.approx(original_nm[1] - original_nm[0])
    assert native_nm[4] - native_nm[3] == pytest.approx(original_nm[4] - original_nm[3])


def _test_a_position_mapping_is_identity():
    from atom_openmm.neqti_endpoints import map_atm_to_native_positions

    positions = _positions()
    assert map_atm_to_native_positions(positions, "a", _keywords()) is positions


def _test_state_transfer_preserves_velocities_while_mapping_b_positions():
    from atom_openmm.neqti_endpoints import transfer_state_to_context

    system = mm.System()
    for _ in range(8):
        system.addParticle(12.0)
    source_context = mm.Context(system, mm.VerletIntegrator(0.001))
    target_context = mm.Context(system, mm.VerletIntegrator(0.001))
    velocities = np.arange(24, dtype=float).reshape(8, 3) * 0.001
    source_context.setPositions(_positions())
    source_context.setVelocities(velocities * unit.nanometer / unit.picosecond)
    source = source_context.getState(getPositions=True, getVelocities=True)

    transfer_state_to_context(
        source,
        target_context,
        endpoint="b",
        keywords=_keywords(),
        to_native=True,
    )
    target = target_context.getState(getPositions=True, getVelocities=True)

    assert target.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    ) == pytest.approx(velocities)
    assert target.getPositions(asNumpy=True).value_in_unit(unit.nanometer)[0] == pytest.approx(
        [1.0, 0.0, 0.0]
    )
