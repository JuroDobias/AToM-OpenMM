from types import SimpleNamespace

import numpy as np
import openmm as mm
from openmm import unit
import pytest

from atom_openmm.soft_bond_screen import path_variants, _audit_endpoint, OPTIMIZER
from atom_openmm.covalent_softcore import (
    EXPLICIT_ENDPOINT_A_CONTROLS, EXPLICIT_ENDPOINT_B_CONTROLS,
    create_softcore_hamiltonian, resolve_softcore_path,
)
from atom_openmm.covalent_workflow import (
    _softcore_switch_context, _apply_state, _run_segmented_protocol,
)
from atom_openmm.neqti import _allocate_segment_steps
from atom_openmm.soft_bond_core_screen import variants as core_variants, split_mapped_forces


@pytest.mark.parametrize("variant", core_variants(), ids=lambda v: v["name"])
def _test_core_ordering_endpoints_and_switch(variant):
    prepared = SimpleNamespace(endpoint_a=_endpoint("a"), endpoint_b=_endpoint("b"))
    prepared.endpoint_b.getForce(0).setBondParameters(0, 0, 1, 0.16, 120)
    for end in ("a", "b"):
        system = getattr(prepared, 'endpoint_' + end)
        angles = mm.HarmonicAngleForce()
        angles.addAngle(0, 1, 2, 1.5 if end == 'a' else 1.7, 10)
        system.addForce(angles)
        torsions = mm.PeriodicTorsionForce()
        torsions.addTorsion(0, 1, 2, 3, 2, 0, 1 if end == 'a' else 2)
        system.addForce(torsions)
    h = create_softcore_hamiltonian(
        prepared.endpoint_a, prepared.endpoint_b, [], [],
        soft_bond_pairs=[], control_nodes=variant['nodes'],
        total_steps=50000, segments_per_interval=variant['segments_per_interval'])
    counts = split_mapped_forces(h, set(), variant)
    assert counts['CovalentInterpolatedBonds'] >= 1
    assert counts['CovalentInterpolatedAngles'] == 1
    assert counts['CovalentInterpolatedTorsionsA'] == 1
    assert all(len(v) == len(h.segment_steps)+1 for v in h.parameter_values.values())
    plat = mm.Platform.getPlatformByName('Reference')
    context = mm.Context(prepared.endpoint_a, mm.VerletIntegrator(.001), plat)
    context.setPositions([[0, 0, 0], [.15, 0, 0], [.15, .15, 0], [0, .15, .1]])
    context.setVelocitiesToTemperature(300, 17)
    state = context.getState(positions=True, velocities=True)
    del context
    for end in ('a', 'b'):
        _audit_endpoint(prepared, h, end, state, plat, {})
        context, integrator, _ = _softcore_switch_context(
            h, start=end, timestep_fs=1, temperature_k=300, platform=plat,
            properties={}, seed=17)
        steps = [1]*len(h.segment_steps)
        integrator.set_segment_steps(steps)
        _apply_state(context, state)
        work, _ = _run_segmented_protocol(integrator, steps)
        assert np.isfinite(work)
        del context, integrator


def _test_screen_allocations_and_endpoint_controls():
    variants = path_variants()
    assert len({v["name"] for v in variants}) == 24
    for variant in variants:
        nodes = variant["nodes"]
        assert nodes[0]["controls"] == EXPLICIT_ENDPOINT_A_CONTROLS
        assert nodes[-1]["controls"] == EXPLICIT_ENDPOINT_B_CONTROLS
        resolved = resolve_softcore_path(control_nodes=nodes, total_steps=50000,
                                        segments_per_interval=variant["segments_per_interval"])
        assert sum(resolved["interval_steps"]) == 50000
        steps = []
        for total, count in zip(resolved["interval_steps"], variant["segments_per_interval"]):
            steps.extend([total // count + (i < total % count) for i in range(count)])
        scores = np.arange(1, len(steps)+1)
        for _ in range(10):
            steps = _allocate_segment_steps(scores, 50000, steps, OPTIMIZER)
            assert sum(steps) == 50000
            assert min(steps) >= 250
        change = next(n for n in nodes if n["label"] == "exchange_topology_pairs")
        assert all(change["controls"][key] == 0 for key in
                   ("charge_a", "charge_b", "sterics_a", "sterics_b"))


def _endpoint(end):
    system = mm.System()
    for _ in range(4):
        system.addParticle(12)
    bonds = mm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, 100)
    bonds.addBond(1, 2, 0.15, 100)
    bonds.addBond(2, 3, 0.15, 100)
    bonds.addBond(*( (0, 2) if end == "a" else (1, 3)), 0.2, 100)
    system.addForce(bonds)
    nb = mm.NonbondedForce()
    for _ in range(4):
        nb.addParticle(0, 0.3, 0)
    system.addForce(nb)
    return system


@pytest.mark.parametrize("variant", path_variants(), ids=lambda v: v["name"])
def _test_each_path_supports_both_soft_bonds_and_reverse_switch(variant):
    prepared = SimpleNamespace(endpoint_a=_endpoint("a"), endpoint_b=_endpoint("b"))
    h = create_softcore_hamiltonian(
        prepared.endpoint_a, prepared.endpoint_b, [], [],
        soft_bond_pairs=[(0, 2), (1, 3)], control_nodes=variant["nodes"],
        total_steps=50000, segments_per_interval=variant["segments_per_interval"])
    plat = mm.Platform.getPlatformByName("Reference")
    initial = mm.Context(prepared.endpoint_a, mm.VerletIntegrator(0.001), plat)
    initial.setPositions([[0, 0, 0], [0.15, 0, 0], [0.15, 0.15, 0], [0, 0.15, 0]])
    initial.setVelocitiesToTemperature(300, 17)
    state = initial.getState(positions=True, velocities=True)
    del initial
    for end in ("a", "b"):
        _audit_endpoint(prepared, h, end, state, plat, {})
        context, integrator, _ = _softcore_switch_context(
            h, start=end, timestep_fs=1, temperature_k=300, platform=plat,
            properties={}, seed=17)
        steps = [1] * len(h.segment_steps)
        integrator.set_segment_steps(steps)
        _apply_state(context, state)
        progress = []
        work, segments = _run_segmented_protocol(
            integrator, steps, segment_callback=lambda **kw: progress.append(kw))
        assert np.isfinite(work)
        assert len(segments) == len(progress) == len(steps)
        assert progress[-1]["completed_steps"] == sum(steps)
        del context, integrator
