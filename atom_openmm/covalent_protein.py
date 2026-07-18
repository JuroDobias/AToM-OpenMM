from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit

from atom_openmm.covalent_alchemy import CovalentAlchemyError
from atom_openmm.covalent_hybrid import CovalentHybridMolecule
from atom_openmm.covalent_parameters import CovalentParameterBundle
from atom_openmm.covalent_systems import PreparedCovalentHybrid, _copy_virtual_site


def _force(system, cls):
    matches = [item for item in system.getForces() if isinstance(item, cls)]
    if len(matches) > 1:
        raise CovalentAlchemyError(f"multiple {cls.__name__} forces are unsupported")
    return matches[0] if matches else None


def _residue_atoms(topology, residue_id):
    matches = [
        residue for residue in topology.residues()
        if residue.name == "CYS" and residue.id == str(residue_id)
    ]
    if len(matches) != 1:
        raise CovalentAlchemyError(f"expected one CYS {residue_id}, found {len(matches)}")
    return {atom.name: atom.index for atom in matches[0].atoms()}


def _build_base_receptor(
    receptor_pdb: Path,
    ligand_positions_a: np.ndarray,
    ligand_positions_b: np.ndarray,
    *,
    padding_a: float,
    ionic_strength_molar: float,
    clash_cutoff_a: float = 2.2,
):
    pdb = app.PDBFile(str(receptor_pdb))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    forcefield = app.ForceField("amber19/protein.ff19SB.xml", "amber19/opc.xml")
    modeller.addExtraParticles(forcefield)
    modeller.addSolvent(
        forcefield,
        model="tip4pew",
        padding=float(padding_a) * unit.angstrom,
        ionicStrength=float(ionic_strength_molar) * unit.molar,
        neutralize=True,
    )
    ligand_positions = np.concatenate((ligand_positions_a, ligand_positions_b), axis=0)
    modeller_nm = np.asarray(modeller.positions.value_in_unit(unit.nanometer))
    ligand_nm = ligand_positions * 0.1
    delete = []
    for residue in modeller.topology.residues():
        if residue.name not in {"HOH", "WAT", "NA", "CL", "K"}:
            continue
        representative = next(
            (atom for atom in residue.atoms() if atom.element and atom.element.symbol == "O"),
            next(iter(residue.atoms()), None),
        )
        if representative is None:
            continue
        if (
            np.min(np.linalg.norm(ligand_nm - modeller_nm[representative.index], axis=1))
            < clash_cutoff_a * 0.1
        ):
            delete.append(residue)
    if delete:
        modeller.delete(delete)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=9.0 * unit.angstrom,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True,
    )
    return modeller, system


def _product_cys_map(bundle, metadata, receptor_atoms):
    molecule = bundle.molecule.to_rdkit()
    mapped = {
        int(number): int(index)
        for number, index in metadata["capped_cys_atom_map_indices"].items()
    }
    names = {4: "N", 5: "CA", 6: "CB", 7: "SG", 8: "C", 9: "O"}
    result = {mapped[number]: receptor_atoms[name] for number, name in names.items()}
    hydrogen_names = {4: ["H"], 5: ["HA"], 6: ["HB2", "HB3"]}
    for map_number, names_for_atom in hydrogen_names.items():
        source = molecule.GetAtomWithIdx(mapped[map_number])
        hydrogens = sorted(
            [neighbor.GetIdx() for neighbor in source.GetNeighbors() if neighbor.GetAtomicNum() == 1]
        )
        if len(hydrogens) != len(names_for_atom):
            raise CovalentAlchemyError(f"unexpected capped-Cys hydrogens around atom map {map_number}")
        for hydrogen, name in zip(hydrogens, names_for_atom):
            result[hydrogen] = receptor_atoms[name]
    result[int(metadata["transferred_hydrogen_atom_index"])] = receptor_atoms["HG"]
    return result


def _clone_base_without_thiol_constraint(base, sulfur, hydrogen):
    output = mm.System()
    for index in range(base.getNumParticles()):
        output.addParticle(base.getParticleMass(index))
    for index in range(base.getNumParticles()):
        if base.isVirtualSite(index):
            output.setVirtualSite(index, _copy_virtual_site(base.getVirtualSite(index), 0))
    for index in range(base.getNumConstraints()):
        p1, p2, distance = base.getConstraintParameters(index)
        if {int(p1), int(p2)} == {sulfur, hydrogen}:
            continue
        output.addConstraint(p1, p2, distance)
    output.setDefaultPeriodicBoxVectors(*base.getDefaultPeriodicBoxVectors())
    for force in base.getForces():
        if isinstance(force, (mm.CMMotionRemover, mm.MonteCarloBarostat)):
            continue
        output.addForce(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(force)))
    return output


def _zero_base_thiol_terms(system, sulfur, hydrogen):
    bond_force = _force(system, mm.HarmonicBondForce)
    if bond_force:
        for index in range(bond_force.getNumBonds()):
            p1, p2, length, k = bond_force.getBondParameters(index)
            if sulfur in {int(p1), int(p2)} or hydrogen in {int(p1), int(p2)}:
                bond_force.setBondParameters(index, p1, p2, length, 0 * k)
    angle_force = _force(system, mm.HarmonicAngleForce)
    if angle_force:
        for index in range(angle_force.getNumAngles()):
            p1, p2, p3, angle, k = angle_force.getAngleParameters(index)
            if sulfur in {int(p1), int(p2), int(p3)} or hydrogen in {int(p1), int(p2), int(p3)}:
                angle_force.setAngleParameters(index, p1, p2, p3, angle, 0 * k)
    torsion_force = _force(system, mm.PeriodicTorsionForce)
    if torsion_force:
        for index in range(torsion_force.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = torsion_force.getTorsionParameters(index)
            if sulfur in {int(p1), int(p2), int(p3), int(p4)} or hydrogen in {
                int(p1), int(p2), int(p3), int(p4)
            }:
                torsion_force.setTorsionParameters(index, p1, p2, p3, p4, periodicity, phase, 0 * k)


def _source_terms(target, source_system, source_to_global, include):
    for cls, count, get, add, width in (
        (mm.HarmonicBondForce, "getNumBonds", "getBondParameters", "addBond", 2),
        (mm.HarmonicAngleForce, "getNumAngles", "getAngleParameters", "addAngle", 3),
        (mm.PeriodicTorsionForce, "getNumTorsions", "getTorsionParameters", "addTorsion", 4),
    ):
        source = _force(source_system, cls)
        destination = _force(target, cls)
        if source is None:
            continue
        if destination is None:
            destination = cls()
            target.addForce(destination)
        for index in range(getattr(source, count)()):
            parameters = list(getattr(source, get)(index))
            atoms = [int(value) for value in parameters[:width]]
            if all(atom in source_to_global for atom in atoms) and include(atoms):
                parameters[:width] = [source_to_global[atom] for atom in atoms]
                getattr(destination, add)(*parameters)


def _add_source_constraints(target, sources):
    existing = {}
    for index in range(target.getNumConstraints()):
        p1, p2, distance = target.getConstraintParameters(index)
        existing[tuple(sorted((int(p1), int(p2))))] = distance.value_in_unit(unit.nanometer)
    for system, mapping, active_atoms in sources:
        for index in range(system.getNumConstraints()):
            p1, p2, distance = system.getConstraintParameters(index)
            source_pair = (int(p1), int(p2))
            if not all(atom in mapping for atom in source_pair):
                continue
            if not any(atom in active_atoms for atom in source_pair):
                continue
            pair = tuple(sorted(mapping[atom] for atom in source_pair))
            distance_nm = distance.value_in_unit(unit.nanometer)
            if pair in existing:
                if not np.isclose(existing[pair], distance_nm, atol=1.0e-8):
                    raise CovalentAlchemyError(
                        f"mapped constraint {pair} conflicts with the receptor or other endpoint"
                    )
                continue
            target.addConstraint(*pair, distance)
            existing[pair] = distance_nm


def _graft_endpoint(
    base_system,
    parameters_a,
    parameters_b,
    hybrid,
    source_to_global_a,
    source_to_global_b,
    receptor_atoms,
    unique_a,
    unique_b,
    active_atoms_a,
    active_atoms_b,
    state,
):
    sulfur = receptor_atoms["SG"]
    hydrogen = receptor_atoms["HG"]
    output = _clone_base_without_thiol_constraint(base_system, sulfur, hydrogen)
    _zero_base_thiol_terms(output, sulfur, hydrogen)
    base_particles = base_system.getNumParticles()
    reverse_a = {global_index: source for source, global_index in source_to_global_a.items() if global_index >= base_particles}
    reverse_b = {global_index: source for source, global_index in source_to_global_b.items() if global_index >= base_particles}
    ligand_particles = max(max(reverse_a), max(reverse_b)) - base_particles + 1
    for offset in range(ligand_particles):
        global_index = base_particles + offset
        if global_index in reverse_a:
            output.addParticle(parameters_a.system.getParticleMass(reverse_a[global_index]))
        else:
            output.addParticle(parameters_b.system.getParticleMass(reverse_b[global_index]))
    _add_source_constraints(
        output,
        (
            (parameters_a.system, source_to_global_a, active_atoms_a),
            (parameters_b.system, source_to_global_b, active_atoms_b),
        ),
    )

    if state == "a":
        _source_terms(
            output, parameters_a.system, source_to_global_a,
            lambda atoms: any(atom in active_atoms_a for atom in atoms),
        )
        _source_terms(
            output, parameters_b.system, source_to_global_b,
            lambda atoms: any(atom in unique_b for atom in atoms),
        )
        active_parameters = parameters_a
        active_map = source_to_global_a
        active_reverse = reverse_a
        inactive_parameters = parameters_b
        inactive_reverse = reverse_b
    else:
        _source_terms(
            output, parameters_b.system, source_to_global_b,
            lambda atoms: any(atom in active_atoms_b for atom in atoms),
        )
        _source_terms(
            output, parameters_a.system, source_to_global_a,
            lambda atoms: any(atom in unique_a for atom in atoms),
        )
        active_parameters = parameters_b
        active_map = source_to_global_b
        active_reverse = reverse_b
        inactive_parameters = parameters_a
        inactive_reverse = reverse_a

    nonbonded = _force(output, mm.NonbondedForce)
    source_nb = _force(active_parameters.system, mm.NonbondedForce)
    inactive_nb = _force(inactive_parameters.system, mm.NonbondedForce)
    for global_index in range(base_particles, base_particles + ligand_particles):
        if global_index in active_reverse:
            nonbonded.addParticle(*source_nb.getParticleParameters(active_reverse[global_index]))
        else:
            _, sigma, _ = inactive_nb.getParticleParameters(inactive_reverse[global_index])
            nonbonded.addParticle(0.0, sigma, 0.0)
    for source_atom, global_index in active_map.items():
        if global_index in {sulfur, hydrogen}:
            nonbonded.setParticleParameters(global_index, *source_nb.getParticleParameters(source_atom))

    existing = {}
    for index in range(nonbonded.getNumExceptions()):
        p1, p2, *_ = nonbonded.getExceptionParameters(index)
        existing[tuple(sorted((int(p1), int(p2))))] = index
    for pair, index in list(existing.items()):
        if sulfur in pair or hydrogen in pair:
            p1, p2, _, sigma, _ = nonbonded.getExceptionParameters(index)
            nonbonded.setExceptionParameters(index, p1, p2, 0.0, sigma, 0.0)
    for index in range(source_nb.getNumExceptions()):
        p1, p2, charge, sigma, epsilon = source_nb.getExceptionParameters(index)
        active_atoms = active_atoms_a if state == "a" else active_atoms_b
        if int(p1) not in active_atoms and int(p2) not in active_atoms:
            continue
        if int(p1) not in active_map or int(p2) not in active_map:
            continue
        global_pair = tuple(sorted((active_map[int(p1)], active_map[int(p2)])))
        if global_pair in existing:
            nonbonded.setExceptionParameters(existing[global_pair], *global_pair, charge, sigma, epsilon)
        else:
            existing[global_pair] = nonbonded.addException(*global_pair, charge, sigma, epsilon)
    output.addForce(mm.CMMotionRemover())
    return output


def _protein_hybrid_topology(base_topology, hybrid, ligand_global_start, cys_atoms, metadata_a, metadata_b):
    topology = app.Topology()
    atom_map = {}
    cys_residue = None
    for source_chain in base_topology.chains():
        chain = topology.addChain(source_chain.id)
        for source_residue in source_chain.residues():
            residue = topology.addResidue(source_residue.name, chain, source_residue.id)
            if source_residue.name == "CYS" and source_residue.id == "147":
                cys_residue = residue
            for source_atom in source_residue.atoms():
                atom_map[source_atom.index] = topology.addAtom(source_atom.name, source_atom.element, residue)
    sulfur = cys_atoms["SG"]
    hydrogen = cys_atoms["HG"]
    for bond in base_topology.bonds():
        if {bond[0].index, bond[1].index} == {sulfur, hydrogen}:
            continue
        topology.addBond(atom_map[bond[0].index], atom_map[bond[1].index])
    ligand_atoms = []
    hybrid_atoms = list(hybrid.topology.atoms())
    for index in range(hybrid.topology.getNumAtoms()):
        source_atom = hybrid_atoms[index]
        ligand_atoms.append(topology.addAtom(source_atom.name, source_atom.element, cys_residue))
    for bond in hybrid.topology.bonds():
        topology.addBond(ligand_atoms[bond[0].index], ligand_atoms[bond[1].index])
    sulfur_product = int(metadata_a["cys_sulfur_atom_index"])
    oxygen_product = int(metadata_a["product_oxygen_atom_index"])
    hydrogen_product = int(metadata_a["transferred_hydrogen_atom_index"])
    # The hybrid topology supplied here contains only ligand atoms; these bonds are added by caller metadata.
    return topology


def prepare_protein_covalent_hybrid(
    receptor_pdb: Path,
    parameters_a: CovalentParameterBundle,
    parameters_b: CovalentParameterBundle,
    hybrid: CovalentHybridMolecule,
    metadata_a: dict,
    metadata_b: dict,
    *,
    residue_id: int = 147,
    padding_a: float = 10.0,
    ionic_strength_molar: float = 0.15,
) -> PreparedCovalentHybrid:
    offset_a = int(metadata_a["ligand_atom_offset"])
    offset_b = int(metadata_b["ligand_atom_offset"])
    rdkit_a = parameters_a.molecule.to_rdkit()
    rdkit_b = parameters_b.molecule.to_rdkit()
    ligand_positions_a = np.asarray(
        [rdkit_a.GetConformer().GetAtomPosition(i) for i in range(offset_a, rdkit_a.GetNumAtoms())]
    )
    ligand_positions_b = np.asarray(
        [rdkit_b.GetConformer().GetAtomPosition(i) for i in range(offset_b, rdkit_b.GetNumAtoms())]
    )
    modeller, base_system = _build_base_receptor(
        receptor_pdb, ligand_positions_a, ligand_positions_b,
        padding_a=padding_a, ionic_strength_molar=ionic_strength_molar,
    )
    receptor_atoms = _residue_atoms(modeller.topology, residue_id)
    cys_map_a = _product_cys_map(parameters_a, metadata_a, receptor_atoms)
    cys_map_b = _product_cys_map(parameters_b, metadata_b, receptor_atoms)
    base_particles = base_system.getNumParticles()

    ligand_a_to_global = {}
    for source in range(offset_a, rdkit_a.GetNumAtoms()):
        full_hybrid = hybrid.map_a_to_hybrid[source]
        ligand_a_to_global[source] = base_particles + full_hybrid
    ligand_b_to_global = {}
    for source in range(offset_b, rdkit_b.GetNumAtoms()):
        full_hybrid = hybrid.map_b_to_hybrid[source]
        ligand_b_to_global[source] = base_particles + full_hybrid
    # Remove the capped-Cys prefix from hybrid numbering before appending ligand particles.
    cap_hybrid_indices = sorted(
        set(hybrid.map_a_to_hybrid[i] for i in range(offset_a))
        | set(hybrid.map_b_to_hybrid[i] for i in range(offset_b))
    )
    ligand_hybrid_indices = sorted(
        set(ligand_a_to_global.values()) | set(ligand_b_to_global.values())
    )
    renumber = {old: base_particles + index for index, old in enumerate(ligand_hybrid_indices)}
    ligand_a_to_global = {source: renumber[value] for source, value in ligand_a_to_global.items()}
    ligand_b_to_global = {source: renumber[value] for source, value in ligand_b_to_global.items()}
    source_to_global_a = {**cys_map_a, **ligand_a_to_global}
    source_to_global_b = {**cys_map_b, **ligand_b_to_global}
    unique_a = {index for index in hybrid.unique_a if index >= offset_a}
    unique_b = {index for index in hybrid.unique_b if index >= offset_b}
    active_atoms_a = set(range(offset_a, rdkit_a.GetNumAtoms())) | {
        int(metadata_a["cys_sulfur_atom_index"]),
        int(metadata_a["transferred_hydrogen_atom_index"]),
    }
    active_atoms_b = set(range(offset_b, rdkit_b.GetNumAtoms())) | {
        int(metadata_b["cys_sulfur_atom_index"]),
        int(metadata_b["transferred_hydrogen_atom_index"]),
    }
    endpoint_a = _graft_endpoint(
        base_system, parameters_a, parameters_b, hybrid,
        source_to_global_a, source_to_global_b, receptor_atoms, unique_a, unique_b,
        active_atoms_a, active_atoms_b, "a",
    )
    endpoint_b = _graft_endpoint(
        base_system, parameters_a, parameters_b, hybrid,
        source_to_global_a, source_to_global_b, receptor_atoms, unique_a, unique_b,
        active_atoms_a, active_atoms_b, "b",
    )

    # Clone the base topology and append the ligand-union residue atoms and bonds.
    topology = app.Topology()
    cloned_atoms = {}
    cys_residue = None
    for source_chain in modeller.topology.chains():
        chain = topology.addChain(source_chain.id)
        for source_residue in source_chain.residues():
            residue = topology.addResidue(source_residue.name, chain, source_residue.id)
            if source_residue.name == "CYS" and source_residue.id == str(residue_id):
                cys_residue = residue
            for source_atom in source_residue.atoms():
                cloned_atoms[source_atom.index] = topology.addAtom(source_atom.name, source_atom.element, residue)
    for bond in modeller.topology.bonds():
        if {bond[0].index, bond[1].index} == {receptor_atoms["SG"], receptor_atoms["HG"]}:
            continue
        topology.addBond(cloned_atoms[bond[0].index], cloned_atoms[bond[1].index])
    ligand_chain = topology.addChain("L")
    ligand_residue = topology.addResidue("CVL", ligand_chain, "1")
    global_to_atom = {}
    for source, global_index in sorted(ligand_a_to_global.items(), key=lambda item: item[1]):
        atom = rdkit_a.GetAtomWithIdx(source)
        global_to_atom[global_index] = topology.addAtom(
            f"A{atom.GetSymbol()}{source-offset_a+1}",
            app.Element.getByAtomicNumber(atom.GetAtomicNum()), ligand_residue,
        )
    for source, global_index in sorted(ligand_b_to_global.items(), key=lambda item: item[1]):
        if global_index in global_to_atom:
            continue
        atom = rdkit_b.GetAtomWithIdx(source)
        global_to_atom[global_index] = topology.addAtom(
            f"B{atom.GetSymbol()}{source-offset_b+1}",
            app.Element.getByAtomicNumber(atom.GetAtomicNum()), ligand_residue,
        )
    bonds = set()
    for molecule, mapping in ((rdkit_a, ligand_a_to_global), (rdkit_b, ligand_b_to_global)):
        for bond in molecule.GetBonds():
            p1, p2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if p1 in mapping and p2 in mapping:
                pair = tuple(sorted((mapping[p1], mapping[p2])))
                if pair not in bonds:
                    topology.addBond(global_to_atom[pair[0]], global_to_atom[pair[1]])
                    bonds.add(pair)
    for mapping, metadata in ((source_to_global_a, metadata_a), (source_to_global_b, metadata_b)):
        sulfur = cloned_atoms[receptor_atoms["SG"]]
        carbon = global_to_atom[mapping[int(metadata["electrophile_carbon_atom_index"])]]
        oxygen = global_to_atom[mapping[int(metadata["product_oxygen_atom_index"])]]
        hydrogen = cloned_atoms[receptor_atoms["HG"]]
        for pair in ((sulfur, carbon), (oxygen, hydrogen)):
            key = tuple(sorted((pair[0].index, pair[1].index)))
            if key not in bonds:
                topology.addBond(*pair)
                bonds.add(key)
    topology.setPeriodicBoxVectors(modeller.topology.getPeriodicBoxVectors())

    base_positions = list(modeller.positions.value_in_unit(unit.nanometer))
    appended_positions = [None] * len(global_to_atom)
    for source, global_index in ligand_a_to_global.items():
        appended_positions[global_index - base_particles] = np.asarray(
            rdkit_a.GetConformer().GetAtomPosition(source)
        ) * 0.1
    for source, global_index in ligand_b_to_global.items():
        if appended_positions[global_index - base_particles] is None:
            appended_positions[global_index - base_particles] = np.asarray(
                rdkit_b.GetConformer().GetAtomPosition(source)
            ) * 0.1
    positions = unit.Quantity(base_positions + appended_positions, unit.nanometer)
    provenance = {
        "environment": "protein",
        "protein_forcefield": "amber19/protein.ff19SB.xml",
        "water_forcefield": "amber19/opc.xml",
        "covalent_residue_id": residue_id,
        "hybrid_solute_atom_count": len(global_to_atom),
        "mapped_product_atom_count": len(hybrid.map_a_to_b),
        "unique_a_count": len(unique_a),
        "unique_b_count": len(unique_b),
    }
    hot_atoms = tuple(sorted({receptor_atoms["CB"], receptor_atoms["SG"], *global_to_atom.keys()}))
    return PreparedCovalentHybrid(
        topology, positions, endpoint_a, endpoint_b, len(global_to_atom), hot_atoms, provenance
    )
