from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random

import numpy as np
import openmm as mm
from openmm import app, unit
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from atom_openmm.covalent_parameters import CovalentParameterBundle, CovalentParameterError
from atom_openmm.covalent_hybrid import CovalentHybridMolecule


@dataclass(frozen=True)
class PreparedCovalentSystem:
    topology: app.Topology
    positions: unit.Quantity
    system: mm.System
    solute_atom_count: int
    provenance: dict[str, object]


@dataclass(frozen=True)
class PreparedCovalentHybrid:
    topology: app.Topology
    positions: unit.Quantity
    endpoint_a: mm.System
    endpoint_b: mm.System
    solute_atom_count: int
    hot_atom_indices: tuple[int, ...]
    provenance: dict[str, object]


@contextmanager
def _seeded_python_random(seed):
    state = random.getstate()
    random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(state)


def add_deterministic_ions(
    modeller,
    forcefield,
    *,
    ionic_strength_molar,
    seed,
):
    waters = {}
    for residue in modeller.topology.residues():
        if residue.name not in {"HOH", "WAT"}:
            continue
        oxygen = next(
            (
                atom
                for atom in residue.atoms()
                if atom.element is not None and atom.element.symbol == "O"
            ),
            None,
        )
        if oxygen is not None:
            waters[residue] = modeller.positions[oxygen.index]
    with _seeded_python_random(seed):
        modeller._addIons(
            forcefield,
            len(waters),
            waters,
            ionicStrength=float(ionic_strength_molar) * unit.molar,
            neutralize=True,
        )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_prepared_hybrid_bundle(prepared, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "topology": directory / f"{name}_topology.cif",
        "endpoint_a": directory / f"{name}_endpoint_a.xml",
        "endpoint_b": directory / f"{name}_endpoint_b.xml",
    }
    with paths["topology"].open("w") as handle:
        app.PDBxFile.writeFile(
            prepared.topology, prepared.positions, handle, keepIds=False
        )
    paths["endpoint_a"].write_text(mm.XmlSerializer.serialize(prepared.endpoint_a))
    paths["endpoint_b"].write_text(mm.XmlSerializer.serialize(prepared.endpoint_b))
    particle_count = prepared.endpoint_a.getNumParticles()
    if prepared.endpoint_b.getNumParticles() != particle_count:
        raise CovalentParameterError("prepared endpoint systems have different particle counts")
    if prepared.topology.getNumAtoms() != particle_count:
        raise CovalentParameterError(
            "prepared topology atom count does not match endpoint particle count"
        )
    return {
        "artifacts": {
            key: {"file": path.name, "sha256": _sha256(path)}
            for key, path in paths.items()
        },
        "particle_count": particle_count,
        "solute_atom_count": int(prepared.solute_atom_count),
        "hot_atom_indices": [int(value) for value in prepared.hot_atom_indices],
        "provenance": prepared.provenance,
    }


def load_prepared_hybrid_bundle(directory, payload):
    directory = Path(directory)
    paths = {}
    for key in ("topology", "endpoint_a", "endpoint_b"):
        artifact = payload["artifacts"][key]
        path = directory / artifact["file"]
        if not path.is_file():
            raise CovalentParameterError(f"prepared artifact is missing: {path}")
        if _sha256(path) != artifact["sha256"]:
            raise CovalentParameterError(f"prepared artifact checksum differs: {path}")
        paths[key] = path
    topology_file = (
        app.PDBxFile(str(paths["topology"]))
        if paths["topology"].suffix.lower() in {".cif", ".mmcif"}
        else app.PDBFile(str(paths["topology"]))
    )
    endpoint_a = mm.XmlSerializer.deserialize(paths["endpoint_a"].read_text())
    endpoint_b = mm.XmlSerializer.deserialize(paths["endpoint_b"].read_text())
    expected = int(payload["particle_count"])
    observed = {
        "topology": topology_file.topology.getNumAtoms(),
        "endpoint_a": endpoint_a.getNumParticles(),
        "endpoint_b": endpoint_b.getNumParticles(),
    }
    if any(value != expected for value in observed.values()):
        raise CovalentParameterError(
            f"prepared particle counts differ from manifest {expected}: {observed}"
        )
    return PreparedCovalentHybrid(
        topology_file.topology,
        topology_file.positions,
        endpoint_a,
        endpoint_b,
        int(payload["solute_atom_count"]),
        tuple(int(value) for value in payload["hot_atom_indices"]),
        dict(payload["provenance"]),
    )


def _nonbonded_force(system: mm.System) -> mm.NonbondedForce:
    forces = [force for force in system.getForces() if isinstance(force, mm.NonbondedForce)]
    if len(forces) != 1:
        raise CovalentParameterError(f"expected one NonbondedForce, found {len(forces)}")
    return forces[0]


def create_solvated_capped_reference(
    parameters: CovalentParameterBundle,
    *,
    padding_a: float = 10.0,
    ionic_strength_molar: float = 0.15,
    nonbonded_cutoff_a: float = 9.0,
    template_cache: Path | None = None,
    solvation_seed: int = 2026,
) -> PreparedCovalentSystem:
    molecule = parameters.molecule
    topology = molecule.to_topology().to_openmm()
    positions = molecule.conformers[0].to_openmm()
    modeller = app.Modeller(topology, positions)
    forcefield = app.ForceField("amber19/opc.xml")
    generator = SMIRNOFFTemplateGenerator(
        molecules=[molecule],
        forcefield="openff-2.2.1",
        cache=None if template_cache is None else str(template_cache),
    )
    forcefield.registerTemplateGenerator(generator.generator)
    # OpenMM supplies TIP4P-Ew coordinates as the generic four-site water box.
    # OPC parameters and virtual-site positions are assigned by amber19/opc.xml.
    with _seeded_python_random(solvation_seed):
        modeller.addSolvent(
            forcefield,
            model="tip4pew",
            padding=float(padding_a) * unit.angstrom,
            ionicStrength=float(ionic_strength_molar) * unit.molar,
            neutralize=True,
        )
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=float(nonbonded_cutoff_a) * unit.angstrom,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True,
    )
    force = _nonbonded_force(system)
    observed = np.asarray(
        [
            force.getParticleParameters(index)[0].value_in_unit(unit.elementary_charge)
            for index in range(molecule.n_atoms)
        ]
    )
    if not np.allclose(observed, parameters.charges_e, atol=1.0e-7):
        raise CovalentParameterError(
            "solvated reference system does not preserve Espaloma NN solute charges"
        )
    provenance = dict(parameters.provenance)
    provenance.update(
        {
            "protein_forcefield": "amber19/protein.ff19SB.xml",
            "water_forcefield": "amber19/opc.xml",
            "water_coordinate_template": "tip4pew",
            "padding_a": float(padding_a),
            "ionic_strength_molar": float(ionic_strength_molar),
            "solvation_seed": int(solvation_seed),
            "solute_atom_count": molecule.n_atoms,
            "total_particle_count": system.getNumParticles(),
        }
    )
    return PreparedCovalentSystem(
        modeller.topology,
        modeller.positions,
        system,
        molecule.n_atoms,
        provenance,
    )


def write_prepared_system(prepared: PreparedCovalentSystem, prefix: Path):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.with_suffix(".xml")).write_text(mm.XmlSerializer.serialize(prepared.system))
    with prefix.with_suffix(".pdb").open("w") as handle:
        app.PDBFile.writeFile(prepared.topology, prepared.positions, handle, keepIds=True)


def write_prepared_hybrid(prepared: PreparedCovalentHybrid, prefix: Path):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.parent / f"{prefix.name}_endpoint_a.xml").write_text(
        mm.XmlSerializer.serialize(prepared.endpoint_a)
    )
    (prefix.parent / f"{prefix.name}_endpoint_b.xml").write_text(
        mm.XmlSerializer.serialize(prepared.endpoint_b)
    )
    for endpoint in ("a", "b"):
        with (prefix.parent / f"{prefix.name}_endpoint_{endpoint}.pdb").open("w") as handle:
            app.PDBFile.writeFile(prepared.topology, prepared.positions, handle, keepIds=True)


def _single_force(system, cls):
    forces = [force for force in system.getForces() if isinstance(force, cls)]
    if len(forces) > 1:
        raise CovalentParameterError(f"multiple {cls.__name__} instances are not supported")
    return forces[0] if forces else None


def _copy_virtual_site(site, shift):
    particles = [int(site.getParticle(index)) + shift for index in range(site.getNumParticles())]
    if isinstance(site, mm.TwoParticleAverageSite):
        return mm.TwoParticleAverageSite(
            particles[0], particles[1], site.getWeight(0), site.getWeight(1)
        )
    if isinstance(site, mm.ThreeParticleAverageSite):
        return mm.ThreeParticleAverageSite(
            particles[0], particles[1], particles[2],
            site.getWeight(0), site.getWeight(1), site.getWeight(2),
        )
    if isinstance(site, mm.OutOfPlaneSite):
        return mm.OutOfPlaneSite(
            particles[0], particles[1], particles[2],
            site.getWeight12(), site.getWeight13(), site.getWeightCross(),
        )
    if isinstance(site, mm.LocalCoordinatesSite):
        return mm.LocalCoordinatesSite(
            particles,
            site.getOriginWeights(),
            site.getXWeights(),
            site.getYWeights(),
            site.getLocalPosition(),
        )
    raise CovalentParameterError(f"unsupported solvent virtual site {type(site).__name__}")


def _merge_hybrid_with_environment(
    hybrid_system: mm.System,
    environment_system: mm.System,
    environment_solute_atoms: int,
) -> mm.System:
    hybrid_atoms = hybrid_system.getNumParticles()
    shift = hybrid_atoms - int(environment_solute_atoms)
    output = mm.System()
    for index in range(hybrid_atoms):
        new_index = output.addParticle(hybrid_system.getParticleMass(index))
        if hybrid_system.isVirtualSite(index):
            output.setVirtualSite(
                new_index, _copy_virtual_site(hybrid_system.getVirtualSite(index), 0)
            )
    for index in range(environment_solute_atoms, environment_system.getNumParticles()):
        new_index = output.addParticle(environment_system.getParticleMass(index))
        if environment_system.isVirtualSite(index):
            output.setVirtualSite(new_index, _copy_virtual_site(environment_system.getVirtualSite(index), shift))
    for index in range(hybrid_system.getNumConstraints()):
        p1, p2, distance = hybrid_system.getConstraintParameters(index)
        output.addConstraint(p1, p2, distance)
    for index in range(environment_system.getNumConstraints()):
        p1, p2, distance = environment_system.getConstraintParameters(index)
        if int(p1) >= environment_solute_atoms and int(p2) >= environment_solute_atoms:
            output.addConstraint(int(p1) + shift, int(p2) + shift, distance)
    output.setDefaultPeriodicBoxVectors(*environment_system.getDefaultPeriodicBoxVectors())

    for cls, count_name, params_name, add_name, width in (
        (mm.HarmonicBondForce, "getNumBonds", "getBondParameters", "addBond", 2),
        (mm.HarmonicAngleForce, "getNumAngles", "getAngleParameters", "addAngle", 3),
        (mm.PeriodicTorsionForce, "getNumTorsions", "getTorsionParameters", "addTorsion", 4),
    ):
        target = cls()
        source_hybrid = _single_force(hybrid_system, cls)
        source_environment = _single_force(environment_system, cls)
        if source_hybrid is not None:
            for index in range(getattr(source_hybrid, count_name)()):
                getattr(target, add_name)(*getattr(source_hybrid, params_name)(index))
        if source_environment is not None:
            for index in range(getattr(source_environment, count_name)()):
                parameters = list(getattr(source_environment, params_name)(index))
                if all(int(particle) >= environment_solute_atoms for particle in parameters[:width]):
                    parameters[:width] = [int(particle) + shift for particle in parameters[:width]]
                    getattr(target, add_name)(*parameters)
        count = getattr(target, count_name)()
        if count:
            output.addForce(target)

    for force in hybrid_system.getForces():
        if isinstance(force, mm.CustomBondForce):
            output.addForce(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(force)))

    hybrid_nonbonded = _single_force(hybrid_system, mm.NonbondedForce)
    environment_nonbonded = _single_force(environment_system, mm.NonbondedForce)
    if hybrid_nonbonded is None or environment_nonbonded is None:
        raise CovalentParameterError("hybrid and environment require NonbondedForce instances")
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(environment_nonbonded.getNonbondedMethod())
    nonbonded.setCutoffDistance(environment_nonbonded.getCutoffDistance())
    nonbonded.setEwaldErrorTolerance(environment_nonbonded.getEwaldErrorTolerance())
    nonbonded.setUseDispersionCorrection(environment_nonbonded.getUseDispersionCorrection())
    for index in range(hybrid_atoms):
        nonbonded.addParticle(*hybrid_nonbonded.getParticleParameters(index))
    for index in range(environment_solute_atoms, environment_nonbonded.getNumParticles()):
        nonbonded.addParticle(*environment_nonbonded.getParticleParameters(index))
    for index in range(hybrid_nonbonded.getNumExceptions()):
        nonbonded.addException(*hybrid_nonbonded.getExceptionParameters(index))
    for index in range(environment_nonbonded.getNumExceptions()):
        parameters = list(environment_nonbonded.getExceptionParameters(index))
        if int(parameters[0]) >= environment_solute_atoms and int(parameters[1]) >= environment_solute_atoms:
            parameters[0] = int(parameters[0]) + shift
            parameters[1] = int(parameters[1]) + shift
            nonbonded.addException(*parameters)
    output.addForce(nonbonded)
    output.addForce(mm.CMMotionRemover())
    return output


def _hybrid_environment_topology(hybrid, environment):
    topology = app.Topology()
    atom_map = {}
    for source_chain in hybrid.topology.chains():
        chain = topology.addChain(source_chain.id)
        for source_residue in source_chain.residues():
            residue = topology.addResidue(source_residue.name, chain, source_residue.id)
            for source_atom in source_residue.atoms():
                atom_map[source_atom.index] = topology.addAtom(source_atom.name, source_atom.element, residue)
    for bond in hybrid.topology.bonds():
        topology.addBond(atom_map[bond[0].index], atom_map[bond[1].index])

    environment_atoms = list(environment.topology.atoms())
    offset = hybrid.topology.getNumAtoms() - environment.solute_atom_count
    environment_map = {}
    for source_chain in environment.topology.chains():
        chain = None
        for source_residue in source_chain.residues():
            atoms = list(source_residue.atoms())
            if not atoms or all(atom.index < environment.solute_atom_count for atom in atoms):
                continue
            if chain is None:
                chain = topology.addChain(source_chain.id)
            residue = topology.addResidue(source_residue.name, chain, source_residue.id)
            for source_atom in atoms:
                environment_map[source_atom.index] = topology.addAtom(
                    source_atom.name, source_atom.element, residue
                )
    for bond in environment.topology.bonds():
        if bond[0].index in environment_map and bond[1].index in environment_map:
            topology.addBond(environment_map[bond[0].index], environment_map[bond[1].index])
    topology.setPeriodicBoxVectors(environment.topology.getPeriodicBoxVectors())
    return topology, offset


def solvate_capped_reference_hybrid(
    hybrid: CovalentHybridMolecule,
    physical_a: PreparedCovalentSystem,
) -> PreparedCovalentHybrid:
    if physical_a.solute_atom_count != len(hybrid.map_a_to_hybrid):
        raise CovalentParameterError("physical reference A does not match the hybrid A molecule")
    endpoint_a = _merge_hybrid_with_environment(
        hybrid.endpoint_a, physical_a.system, physical_a.solute_atom_count
    )
    endpoint_b = _merge_hybrid_with_environment(
        hybrid.endpoint_b, physical_a.system, physical_a.solute_atom_count
    )
    topology, _ = _hybrid_environment_topology(hybrid, physical_a)
    solvent_positions = physical_a.positions[physical_a.solute_atom_count:]
    positions = unit.Quantity(
        list(hybrid.positions.value_in_unit(unit.nanometer))
        + list(solvent_positions.value_in_unit(unit.nanometer)),
        unit.nanometer,
    )
    provenance = dict(physical_a.provenance)
    common_virtual_sites = [
        index
        for index in range(hybrid.topology.getNumAtoms())
        if hybrid.endpoint_a.isVirtualSite(index)
    ]
    provenance.update(
        {
            "hybrid_solute_atom_count": hybrid.topology.getNumAtoms(),
            "mapped_atom_count": len(hybrid.map_a_to_b),
            "unique_a_count": len(hybrid.unique_a),
            "unique_b_count": len(hybrid.unique_b),
            "unique_a_particle_indices": [
                int(hybrid.map_a_to_hybrid[index]) for index in hybrid.unique_a
            ],
            "unique_b_particle_indices": [
                int(hybrid.map_b_to_hybrid[index]) for index in hybrid.unique_b
            ],
            "ligand_a_system_atom_indices": [
                int(hybrid.map_a_to_hybrid[index])
                for index in range(len(hybrid.map_a_to_hybrid))
            ] + common_virtual_sites,
            "ligand_b_system_atom_indices": [
                int(hybrid.map_b_to_hybrid[index])
                for index in range(len(hybrid.map_b_to_hybrid))
            ] + common_virtual_sites,
            "anchor_pairs": [list(pair) for pair in hybrid.anchor_pairs],
            "attachment_pairs": None if hybrid.attachment_pairs is None else [
                list(pair) for pair in hybrid.attachment_pairs
            ],
            "dummy_bonded_scales": dict(hybrid.dummy_bonded_scales.__dict__),
            "dummy_nonbonded": "unique_branch_vacuum",
            "dummy_core_nonbonded": hybrid.dummy_core_nonbonded,
            "cross_branch_nonbonded": "excluded",
            "solvent_source": "endpoint_a_single_solvation",
        }
    )
    return PreparedCovalentHybrid(
        topology,
        positions,
        endpoint_a,
        endpoint_b,
        hybrid.topology.getNumAtoms(),
        tuple(range(hybrid.topology.getNumAtoms())),
        provenance,
    )


PreparedHybridSystem = PreparedCovalentHybrid
merge_hybrid_with_environment = solvate_capped_reference_hybrid
