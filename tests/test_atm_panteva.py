import logging
from types import SimpleNamespace

import numpy as np
import openmm as mm
from openmm import app, unit
import pytest

from atom_openmm.metal_ions import (
    apply_panteva_m1264, PANTEVA_FORCE_NAME, forcefield_atom_classes,
)
from atom_openmm.ommsystem import OMMSystem
from atom_openmm.rest2 import create_rest2_system, create_multi_rest2_system
from atom_openmm.rbfe_workflow import normalize_setup_options, WorkflowConfigError


def model(tmp_path):
    table = tmp_path / 'pol.dat'
    table.write_text('OW 1.444\nO2 0.569\no 0.569\nMg2+ 0.12\n')
    topology = app.Topology()
    chain = topology.addChain()
    for name, symbol in [('MG', 'Mg'), ('L1', 'O'), ('L2', 'O')]:
        residue = topology.addResidue(name, chain)
        topology.addAtom(symbol, app.element.get_by_symbol(symbol), residue)
    system = mm.System()
    nb = mm.NonbondedForce()
    for _ in range(3):
        system.addParticle(16)
        nb.addParticle(0, .2, 0)
    system.addForce(nb)
    apply_panteva_m1264(system, topology, atom_classes=['Mg2+', 'o', 'o'],
                       polarizability_table=table)
    return system


def energy(system, positions, params=None, groups=-1):
    integrator = mm.VerletIntegrator(.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions(positions)
    for name, value in (params or {}).items():
        context.setParameter(name, value)
    state = context.getState(energy=True, forces=True, groups=groups)
    return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))


@pytest.mark.parametrize('var_regions,group', [(True, None), (False, None), (False, 3)])
def _test_panteva_inside_atm_matches_both_coordinate_states(tmp_path, var_regions, group):
    system = model(tmp_path)
    positions = np.array([[0, 0, 0], [.25, 0, 0], [1, 0, 0]])
    swapped = positions.copy()
    swapped[1, 0] += .75
    swapped[2, 0] -= .75
    # Different C4 coefficients make accidental use of u0 for both states visible.
    c4 = system.getForce(1)
    c4.setParticleParameters(2, [0, .01])
    expected_a = energy(system, positions)
    expected_b = energy(system, swapped)
    if group is not None:
        system.getForce(0).setForceGroup(group)
    atm = mm.ATMForce('u0')
    wrapper = SimpleNamespace(system=system, atmforce=atm, var_force_group=group,
                              var_regions=var_regions, logger=logging.getLogger(__name__))
    OMMSystem.add_forces_to_atmforce(wrapper)
    assert sum(atm.getForce(i).getName() == PANTEVA_FORCE_NAME for i in range(atm.getNumForces())) == 1
    assert not any(f.getName() == PANTEVA_FORCE_NAME for f in system.getForces())
    for displacement in [(0, 0, 0), (.75, 0, 0), (-.75, 0, 0)]:
        atm.addParticle(mm.Vec3(*displacement))
    system.addForce(atm)
    actual_a = energy(system, positions)
    atm.setEnergyFunction('u1')
    actual_b = energy(system, positions)
    for observed, expected in [(actual_a, expected_a), (actual_b, expected_b)]:
        assert observed[0] == pytest.approx(expected[0], abs=1e-8)
        np.testing.assert_allclose(observed[1], expected[1], atol=1e-7)


@pytest.mark.parametrize('hot,scale', [([1, 2], .5), ([0, 1, 2], .25)])
def _test_rest2_c4_scaling(tmp_path, hot, scale):
    system = model(tmp_path)
    system.getForce(1).setForceGroup(1)
    xyz = [[0, 0, 0], [.25, 0, 0], [1, 0, 0]]
    original = energy(system, xyz, groups=2)
    transformed = create_rest2_system(system, hot)
    physical = energy(transformed.system, xyz, groups=2)
    heated = energy(transformed.system, xyz, {'REST2_SCALE': .25, 'REST2_SQRT_SCALE': .5}, groups=2)
    assert physical[0] == pytest.approx(original[0])
    assert heated[0] == pytest.approx(original[0]*scale)
    np.testing.assert_allclose(heated[1], original[1]*scale, atol=1e-8)


def _test_multi_rest2_panteva_identity(tmp_path):
    system = model(tmp_path)
    xyz = [[0, 0, 0], [.25, 0, 0], [1, 0, 0]]
    transformed = create_multi_rest2_system(system, {'a': [1], 'b': [2]})
    assert energy(transformed.system, xyz)[0] == pytest.approx(energy(system, xyz)[0])


def _test_panteva_configuration():
    setup = dict(ligand_forcefield='espaloma', ligand_charge_model='nn',
                 solvent_model='tip4pew',
                 metal_ions=dict(model='panteva_m12_6_4', polarizability_table='pol.dat'))
    assert normalize_setup_options({'setup': setup}, {})['metalions'] == setup['metal_ions']
    setup['metal_ions'].pop('polarizability_table')
    with pytest.raises(WorkflowConfigError, match='polarizability_table'):
        normalize_setup_options({'setup': setup}, {})


def _test_panteva_rejects_ambertools_bypass():
    with pytest.raises(WorkflowConfigError, match='openmmforcefields'):
        normalize_setup_options({'setup': dict(mode='ambertools',
            metal_ions={'model': 'panteva_m12_6_4'})}, {})


def _test_prefixed_ion_class_is_normalized():
    topology = app.Topology()
    r = topology.addResidue('MG', topology.addChain())
    topology.addAtom('MG', app.element.magnesium, r)
    assert forcefield_atom_classes(app.ForceField('amber14/tip4pew.xml'), topology) == ['Mg2+']


def _test_actual_template_water_classes():
    topology = app.Topology()
    r = topology.addResidue('HOH', topology.addChain())
    o = topology.addAtom('O', app.element.oxygen, r)
    for name in ['H1', 'H2']:
        h = topology.addAtom(name, app.element.hydrogen, r)
        topology.addBond(o, h)
    topology.addAtom('M', None, r)
    assert forcefield_atom_classes(app.ForceField('amber14/tip4pew.xml'), topology) == ['OW', 'HW', 'HW', 'EP']


def _test_panteva_appended_site_uses_particle_order(tmp_path):
    table = tmp_path / 'pol.dat'
    table.write_text('OW 1.444\nO2 0.569\no 0.569\nMg2+ 0.12\n')
    top = app.Topology()
    chain = top.addChain()
    ligand = top.addResidue('L1', chain)
    top.addAtom('O', app.element.oxygen, ligand)
    metal = top.addResidue('MG', chain)
    top.addAtom('Mg', app.element.magnesium, metal)
    top.addAtom('EP', None, top.addResidue('E1', chain))
    system = mm.System()
    nb = mm.NonbondedForce()
    for mass in [16, 24, 0]:
        system.addParticle(mass)
        nb.addParticle(0, .2, 0)
    system.setVirtualSite(2, mm.TwoParticleAverageSite(0, 1, 1, 0))
    system.addForce(nb)
    apply_panteva_m1264(system, top, atom_classes=['o', 'Mg2+', 'EP'],
                       polarizability_table=table)
    force = system.getForce(1)
    assert list(force.getParticleParameters(2)) == [0, 0]
    assert force.getParticleParameters(1)[0] == 1
    assert force.getParticleParameters(0)[1] == pytest.approx(180.5*.569/1.444*4.184e-4)
