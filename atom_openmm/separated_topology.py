from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openmm as mm
from openmm import app, unit

from atom_openmm.covalent_hybrid import HybridBondedScales, build_hybrid_molecule
from atom_openmm.covalent_systems import PreparedCovalentHybrid, solvate_capped_reference_hybrid
from atom_openmm.utils.AtomUtils import AtomUtils


class SeparatedTopologyError(ValueError):
    pass


@dataclass(frozen=True)
class FrameRestraintSettings:
    translation_k_kcal_mol_a2: float = 10.0
    orientation_k_kcal_mol: float = 25.0
    roll_k_kcal_mol: float = 25.0
    thermalization_steps: int = 5000


def clone_system(system):
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))


def build_separated_ligands(parameters_a, parameters_b):
    """Build two complete ligand copies with no common particles."""
    return build_hybrid_molecule(
        parameters_a,
        parameters_b,
        atom_map={},
        dummy_bonded_scales=HybridBondedScales(),
        dummy_core_nonbonded="off",
    )


def _add_frame_restraint(system, anchors_a, anchors_b, settings):
    if len(anchors_a) != 3 or len(anchors_b) != 3:
        raise SeparatedTopologyError("each ligand frame requires three atoms")
    if len(set(anchors_a)) != 3 or len(set(anchors_b)) != 3:
        raise SeparatedTopologyError("ligand frame atoms must be distinct")
    AtomUtils(system, fix_zero_LJparams=False).addAlignmentForce(
        liga_ref_particles=list(anchors_a),
        ligb_ref_particles=list(anchors_b),
        kfdispl=(
            float(settings.translation_k_kcal_mol_a2)
            * unit.kilocalorie_per_mole
            / unit.angstrom**2
        ),
        ktheta=float(settings.orientation_k_kcal_mol) * unit.kilocalorie_per_mole,
        kpsi=float(settings.roll_k_kcal_mol) * unit.kilocalorie_per_mole,
        offset=mm.Vec3(0.0, 0.0, 0.0) * unit.angstrom,
    )


def prepare_separated_environment(
    parameters_a,
    parameters_b,
    physical_environment,
    *,
    anchors_a,
    anchors_b,
    restraint_settings=None,
) -> PreparedCovalentHybrid:
    restraint_settings = restraint_settings or FrameRestraintSettings()
    dual = build_separated_ligands(parameters_a, parameters_b)
    prepared = solvate_capped_reference_hybrid(dual, physical_environment)
    endpoint_a = clone_system(prepared.endpoint_a)
    endpoint_b = clone_system(prepared.endpoint_b)
    mapped_a = tuple(int(dual.map_a_to_hybrid[index]) for index in anchors_a)
    mapped_b = tuple(int(dual.map_b_to_hybrid[index]) for index in anchors_b)
    _add_frame_restraint(endpoint_a, mapped_a, mapped_b, restraint_settings)
    _add_frame_restraint(endpoint_b, mapped_a, mapped_b, restraint_settings)
    provenance = dict(prepared.provenance)
    provenance.update(
        {
            "alchemy_model": "separated_topology",
            "ligand_a_particle_indices": [
                int(dual.map_a_to_hybrid[index])
                for index in range(parameters_a.molecule.n_atoms)
            ],
            "ligand_b_particle_indices": [
                int(dual.map_b_to_hybrid[index])
                for index in range(parameters_b.molecule.n_atoms)
            ],
            "frame_anchors_a": list(mapped_a),
            "frame_anchors_b": list(mapped_b),
            "frame_restraint": dict(restraint_settings.__dict__),
            "cross_ligand_nonbonded": "excluded",
            "inactive_ligand_internal_interactions": "full",
        }
    )
    return PreparedCovalentHybrid(
        prepared.topology,
        prepared.positions,
        endpoint_a,
        endpoint_b,
        prepared.solute_atom_count,
        tuple(range(prepared.solute_atom_count)),
        provenance,
    )


def environment_signature(topology, ligand_atom_count):
    atoms = list(topology.atoms())[int(ligand_atom_count) :]
    return tuple(
        (
            atom.residue.name,
            atom.name,
            None if atom.element is None else atom.element.symbol,
        )
        for atom in atoms
    )


def _coordinates(quantity, target_unit):
    return np.asarray(quantity.value_in_unit(target_unit), dtype=float)


def _kabsch_transform(mobile, target):
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    translation = target_center - mobile_center @ rotation
    return rotation, translation


def assemble_positions_and_velocities(
    *,
    endpoint,
    active_state,
    inactive_state,
    ligand_a_count,
    ligand_b_count,
    anchors_a,
    anchors_b,
):
    """Combine a physical node state and an independently sampled vacuum state."""
    endpoint = str(endpoint).lower()
    if endpoint not in {"a", "b"}:
        raise SeparatedTopologyError("endpoint must be 'a' or 'b'")
    active_count = ligand_a_count if endpoint == "a" else ligand_b_count
    inactive_count = ligand_b_count if endpoint == "a" else ligand_a_count
    active_positions = _coordinates(active_state.getPositions(), unit.nanometer)
    inactive_positions = _coordinates(inactive_state.getPositions(), unit.nanometer)
    if len(inactive_positions) != inactive_count:
        raise SeparatedTopologyError("vacuum state particle count differs from inactive ligand")
    active_anchors = anchors_a if endpoint == "a" else anchors_b
    inactive_anchors = anchors_b if endpoint == "a" else anchors_a
    rotation, translation = _kabsch_transform(
        inactive_positions[list(inactive_anchors)],
        active_positions[list(active_anchors)],
    )
    aligned_inactive = inactive_positions @ rotation + translation
    environment = active_positions[active_count:]
    if endpoint == "a":
        positions = np.concatenate(
            [active_positions[:active_count], aligned_inactive, environment], axis=0
        )
    else:
        positions = np.concatenate(
            [aligned_inactive, active_positions[:active_count], environment], axis=0
        )

    velocities = None
    try:
        active_velocities = _coordinates(
            active_state.getVelocities(), unit.nanometer / unit.picosecond
        )
        inactive_velocities = _coordinates(
            inactive_state.getVelocities(), unit.nanometer / unit.picosecond
        ) @ rotation
        if endpoint == "a":
            velocities = np.concatenate(
                [active_velocities[:active_count], inactive_velocities,
                 active_velocities[active_count:]], axis=0
            )
        else:
            velocities = np.concatenate(
                [inactive_velocities, active_velocities[:active_count],
                 active_velocities[active_count:]], axis=0
            )
    except Exception:
        velocities = None
    return {
        "positions": positions * unit.nanometer,
        "velocities": (
            None if velocities is None
            else velocities * unit.nanometer / unit.picosecond
        ),
        "box_vectors": active_state.getPeriodicBoxVectors(),
        "anchor_rmsd_a": float(np.sqrt(np.mean(np.sum(
            (aligned_inactive[list(inactive_anchors)]
             - active_positions[list(active_anchors)]) ** 2,
            axis=1,
        ))) * 10.0),
    }


def apply_assembled_state(context, assembled, *, temperature_k, seed):
    box = assembled.get("box_vectors")
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    context.setPositions(assembled["positions"])
    if assembled.get("velocities") is None:
        context.setVelocitiesToTemperature(
            float(temperature_k) * unit.kelvin, int(seed)
        )
    else:
        context.setVelocities(assembled["velocities"])


def thermalize_inactive_ligand(
    system,
    assembled,
    inactive_indices,
    *,
    platform,
    properties,
    temperature_k=300.0,
    timestep_fs=2.0,
    steps=5000,
    seed=2026,
):
    """Sample the restrained inactive pose while the physical snapshot is fixed."""
    frozen = clone_system(system)
    inactive = {int(index) for index in inactive_indices}
    for index in range(frozen.getNumParticles()):
        if index not in inactive:
            frozen.setParticleMass(index, 0.0)
    integrator = mm.LangevinMiddleIntegrator(
        float(temperature_k) * unit.kelvin,
        1.0 / unit.picosecond,
        float(timestep_fs) * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(int(seed))
    context = mm.Context(frozen, integrator, platform, properties)
    apply_assembled_state(context, assembled, temperature_k=temperature_k, seed=seed + 1)
    if int(steps):
        integrator.step(int(steps))
    state = context.getState(
        getPositions=True, getVelocities=True, enforcePeriodicBox=False
    )
    result = dict(assembled)
    result["positions"] = state.getPositions()
    result["velocities"] = state.getVelocities()
    del context, integrator
    return result


class InactivePoseThermalizer:
    """Reusable context for inexpensive restrained-vacuum pose sampling."""

    def __init__(
        self,
        system,
        inactive_indices,
        *,
        platform,
        properties,
        temperature_k=300.0,
        timestep_fs=2.0,
        seed=2026,
    ):
        frozen = clone_system(system)
        inactive = {int(index) for index in inactive_indices}
        for index in range(frozen.getNumParticles()):
            if index not in inactive:
                frozen.setParticleMass(index, 0.0)
        self.temperature_k = float(temperature_k)
        self.integrator = mm.LangevinMiddleIntegrator(
            self.temperature_k * unit.kelvin,
            1.0 / unit.picosecond,
            float(timestep_fs) * unit.femtosecond,
        )
        self.integrator.setRandomNumberSeed(int(seed))
        self.context = mm.Context(frozen, self.integrator, platform, properties)

    def sample(self, assembled, steps, seed):
        apply_assembled_state(
            self.context,
            assembled,
            temperature_k=self.temperature_k,
            seed=seed,
        )
        if int(steps):
            self.integrator.step(int(steps))
        state = self.context.getState(
            getPositions=True, getVelocities=True, enforcePeriodicBox=False
        )
        result = dict(assembled)
        result["positions"] = state.getPositions()
        result["velocities"] = state.getVelocities()
        return result

    def close(self):
        del self.context, self.integrator
