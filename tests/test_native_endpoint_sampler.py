import logging

import openmm as mm
from openmm import app, unit

from atom_openmm.neqti import _NativeEndpointSampler


def _test_native_endpoint_sampler_resumes_portable_state(tmp_path):
    topology = app.Topology()
    residue = topology.addResidue('MOL', topology.addChain())
    topology.addAtom('Ar', app.element.argon, residue)
    system = mm.System()
    system.addParticle(40)
    initial_integrator = mm.VerletIntegrator(.001)
    context = mm.Context(system, initial_integrator,
                         mm.Platform.getPlatformByName('Reference'))
    context.setPositions([[0, 0, 0]])
    context.setVelocities([[1, 0, 0]])
    initial = tmp_path/'initial.xml'
    initial.write_text(mm.XmlSerializer.serialize(context.getState(
        positions=True, velocities=True)))
    del context, initial_integrator

    def make():
        native = type('Native', (), dict(
            topology=topology, system=system,
            integrator=mm.VerletIntegrator(.001*unit.picoseconds)))()
        return _NativeEndpointSampler(
            native, initial, tmp_path/'checkpoint.xml',
            mm.Platform.getPlatformByName('Reference'), {}, logging.getLogger(__name__))

    first = make()
    assert not first.has_bank('a')
    first.run_steps('a', 2, 'first')
    x1 = first.physical_state('a').getPositions(asNumpy=True)[0][0]
    first.close()
    second = make()
    assert second.has_bank('a')
    second.run_steps('a', 2, 'second')
    x2 = second.physical_state('a').getPositions(asNumpy=True)[0][0]
    assert x2 > x1
    second.close()
