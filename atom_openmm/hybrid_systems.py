from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit
from openff.toolkit import Molecule
from openff.units import unit as offunit
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator

from atom_openmm.covalent_parameters import CovalentParameterError
from atom_openmm.covalent_systems import PreparedCovalentSystem, _nonbonded_force


PreparedPhysicalEnvironment = PreparedCovalentSystem
HybridSystemError = CovalentParameterError


def _forcefield_files(value, default):
    if value is None:
        values = list(default)
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = list(value)
    else:
        raise HybridSystemError("force-field settings must be strings or lists of strings")
    return [_resolve_forcefield_file(item) for item in values]


def _resolve_forcefield_file(value: str) -> str:
    prefix = "openmmforcefields:"
    if not value.startswith(prefix):
        return value
    relative = Path(value[len(prefix) :])
    if relative.is_absolute() or ".." in relative.parts:
        raise HybridSystemError(
            f"invalid openmmforcefields resource path: {value}"
        )
    resource = resources.files("openmmforcefields").joinpath("ffxml", *relative.parts)
    if not resource.is_file():
        raise HybridSystemError(
            f"openmmforcefields resource does not exist: {value}"
        )
    return str(resource)


def _repair_template_bonds(topology, positions, forcefield):
    """Add bonds omitted when PDBFile does not recognize a supplemental residue."""
    existing = {
        frozenset((atom1.index, atom2.index)) for atom1, atom2 in topology.bonds()
    }
    repaired = set()
    for residue in topology.residues():
        template = forcefield._templates.get(residue.name)
        if template is None:
            continue
        atoms = {atom.name: atom for atom in residue.atoms()}
        if set(atoms) != set(template.atomIndices):
            continue
        missing = []
        for first, second in template.bonds:
            atom1 = atoms[template.atoms[first].name]
            atom2 = atoms[template.atoms[second].name]
            pair = frozenset((atom1.index, atom2.index))
            if pair not in existing:
                missing.append((atom1, atom2, pair))
        if not missing:
            continue
        for atom1, atom2, pair in missing:
            topology.addBond(atom1, atom2)
            existing.add(pair)
        repaired.add(residue)

    for chain in topology.chains():
        residues = list(chain.residues())
        for first, second in zip(residues, residues[1:]):
            if first not in repaired and second not in repaired:
                continue
            atoms1 = {atom.name: atom for atom in first.atoms()}
            atoms2 = {atom.name: atom for atom in second.atoms()}
            if "C" not in atoms1 or "N" not in atoms2:
                continue
            atom1, atom2 = atoms1["C"], atoms2["N"]
            pair = frozenset((atom1.index, atom2.index))
            if pair in existing:
                continue
            delta = positions[atom1.index] - positions[atom2.index]
            distance_nm = np.linalg.norm(
                np.asarray(delta.value_in_unit(unit.nanometer))
            )
            if distance_nm <= 0.2:
                topology.addBond(atom1, atom2)
                existing.add(pair)


def _solvation_box_options(positions, padding_a: float, box_shape: str):
    allowed = {"cube", "dodecahedron", "octahedron", "rectangular"}
    if box_shape not in allowed:
        raise HybridSystemError(
            f"solvent_box_shape must be one of {sorted(allowed)}"
        )
    if box_shape != "rectangular":
        return {
            "padding": padding_a * unit.angstrom,
            "boxShape": box_shape,
        }
    coordinates = np.asarray(
        [position.value_in_unit(unit.nanometer) for position in positions]
    )
    dimensions_nm = np.ptp(coordinates, axis=0) + 2.0 * padding_a / 10.0
    return {"boxSize": dimensions_nm * unit.nanometer}


def create_physical_ligand_environment(
    parameters,
    *,
    receptor: Path | None,
    setup: dict,
    solvation_seed: int,
    record_atom_classes: bool = False,
) -> PreparedPhysicalEnvironment:
    molecule = parameters.molecule
    template_molecule = Molecule(molecule)
    if parameters.virtual_sites:
        solvation_charges = np.asarray(parameters.charges_e, dtype=float).copy()
        for site in parameters.virtual_sites:
            if site.kind != "sigma_hole" or len(site.parent_atom_indices) != 3:
                raise HybridSystemError("unsupported cached ligand virtual site")
            chlorine = int(site.parent_atom_indices[1])
            solvation_charges[chlorine] += float(site.charge_e)
        template_molecule.partial_charges = (
            solvation_charges * offunit.elementary_charge
        )
    topology = template_molecule.to_topology().to_openmm()
    positions = template_molecule.conformers[0].to_openmm()
    modeller = app.Modeller(topology, positions)
    protein_files = _forcefield_files(
        setup.get("protein_forcefield"), ["amber14-all.xml"]
    )
    solvent_files = _forcefield_files(
        setup.get("solvent_forcefield"), ["amber14/tip3p.xml"]
    )
    forcefield = app.ForceField(*(protein_files + solvent_files))
    if receptor is not None:
        receptor_pdb = app.PDBFile(str(receptor))
        _repair_template_bonds(
            receptor_pdb.topology, receptor_pdb.positions, forcefield
        )
        modeller.add(receptor_pdb.topology, receptor_pdb.positions)
    ligand_forcefield = setup.get("ligand_forcefield", "espaloma-0.3.2")
    if ligand_forcefield.startswith("espaloma"):
        generator = EspalomaTemplateGenerator(
            molecules=[template_molecule],
            forcefield=ligand_forcefield,
            template_generator_kwargs={"charge_method": "from-molecule"},
        )
    elif ligand_forcefield.startswith("gaff-"):
        generator = GAFFTemplateGenerator(
            molecules=[template_molecule], forcefield=ligand_forcefield
        )
    else:
        raise HybridSystemError(
            "ligand_forcefield must be an Espaloma or GAFF force field"
        )
    forcefield.registerTemplateGenerator(generator.generator)
    modeller.addExtraParticles(forcefield)
    solvent_model = setup.get("solvent_model")
    if solvent_model is None:
        solvent_model = (
            "tip4pew"
            if any("opc.xml" in item for item in solvent_files)
            else "tip3p"
        )
    padding_a = float(setup.get("solvent_padding_a", 10.0))
    box_shape = str(setup.get("solvent_box_shape", "cube"))
    ionic_strength = float(setup.get("ionic_strength_molar", 0.15))
    from atom_openmm.covalent_systems import _seeded_python_random

    canonical_box_vectors = setup.get("_canonical_box_vectors_nm")
    canonical_box_size = setup.get("_canonical_box_size_nm")
    if canonical_box_vectors is not None:
        vectors = np.asarray(canonical_box_vectors, dtype=float)
        if vectors.shape != (3, 3):
            raise HybridSystemError("canonical box vectors must have shape (3, 3)")
        box_options = {
            "boxVectors": tuple(
                mm.Vec3(*row) * unit.nanometer for row in vectors
            )
        }
    elif canonical_box_size is not None:
        box_options = {
            "boxSize": np.asarray(canonical_box_size, dtype=float) * unit.nanometer
        }
    else:
        box_options = _solvation_box_options(modeller.positions, padding_a, box_shape)
    with _seeded_python_random(solvation_seed):
        modeller.addSolvent(
            forcefield,
            model=solvent_model,
            ionicStrength=ionic_strength * unit.molar,
            neutralize=True,
            **box_options,
        )
    target_water_count = setup.get("_target_water_count")
    if target_water_count is not None:
        target_water_count = int(target_water_count)
        waters = [
            residue for residue in modeller.topology.residues()
            if residue.name.upper() in {"HOH", "WAT", "TIP3", "TIP4", "OPC"}
        ]
        if len(waters) < target_water_count:
            raise HybridSystemError(
                f"canonical solvent requires {target_water_count} waters but only "
                f"{len(waters)} were generated"
            )
        if len(waters) > target_water_count:
            solute_positions = np.asarray([
                position.value_in_unit(unit.nanometer)
                for atom, position in zip(modeller.topology.atoms(), modeller.positions)
                if atom.residue not in waters
                and atom.residue.name.upper() not in {"NA", "CL", "K", "CA"}
            ])
            center = np.mean(solute_positions, axis=0)
            box_vectors = np.asarray([
                vector.value_in_unit(unit.nanometer)
                for vector in modeller.topology.getPeriodicBoxVectors()
            ])
            inverse_box = np.linalg.inv(box_vectors)
            ranked = []
            for residue in waters:
                oxygen = next(
                    atom for atom in residue.atoms()
                    if atom.element is not None and atom.element.symbol == "O"
                )
                position = modeller.positions[oxygen.index].value_in_unit(unit.nanometer)
                displacement = position - center
                fractional = displacement @ inverse_box
                fractional -= np.rint(fractional)
                minimum_image = fractional @ box_vectors
                ranked.append((float(np.linalg.norm(minimum_image)), residue))
            modeller.delete([
                residue for _, residue in sorted(ranked, reverse=True)[
                    : len(waters) - target_water_count
                ]
            ])
    cutoff_a = float(setup.get("nonbonded_cutoff_a", 9.0))
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_a * unit.angstrom,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True,
    )
    if parameters.virtual_sites:
        cached_nonbonded = _nonbonded_force(parameters.system)
        environment_nonbonded = _nonbonded_force(system)
        for index in range(molecule.n_atoms):
            environment_nonbonded.setParticleParameters(
                index, *cached_nonbonded.getParticleParameters(index)
            )
    atom_classes = None
    if record_atom_classes:
        atom_classes = []
        for residue in modeller.topology.residues():
            template = forcefield._templates.get(residue.name)
            template_atoms = {} if template is None else {atom.name: atom for atom in template.atoms}
            residue_atom_names = {atom.name for atom in residue.atoms()}
            if set(template_atoms) != residue_atom_names and residue.name in {"HIS", "DG", "DC", "DA", "DT", "MET", "ALA"}:
                candidates = (
                    candidate for name, candidate in forcefield._templates.items()
                    if name.startswith(residue.name[:2]) or name.endswith(residue.name)
                    or (residue.name == "HIS" and name in {"HID", "HIE", "HIP"})
                )
                for candidate in candidates:
                    mapped = {atom.name: atom for atom in candidate.atoms}
                    if set(mapped) == residue_atom_names:
                        template_atoms = mapped
                        break
            for atom in residue.atoms():
                template_atom = template_atoms.get(atom.name)
                atom_type = None if template_atom is None else forcefield._atomTypes.get(template_atom.type)
                atom_class = None if atom_type is None else atom_type.atomClass
                if residue.name in {"HOH", "WAT"}:
                    if atom.element is None:
                        atom_class = "EP"
                    else:
                        atom_class = "OW" if atom.element.symbol == "O" else "HW"
                elif atom_class is None and atom.name in {"H2", "H3"} and atom.element.symbol == "H":
                    atom_class = "H"
                elif atom_class is None and atom.name == "OXT" and atom.element.symbol == "O":
                    atom_class = "O2"
                elif atom.element and atom.element.symbol in {"Na", "Cl", "Mg", "Zn"}:
                    atom_class = {"Na": "Na+", "Cl": "Cl-", "Mg": "Mg2+", "Zn": "Zn2+"}[atom.element.symbol]
                atom_classes.append(atom_class)
    force = _nonbonded_force(system)
    observed = np.asarray(
        [
            force.getParticleParameters(index)[0].value_in_unit(unit.elementary_charge)
            for index in range(molecule.n_atoms)
        ]
    )
    if not np.allclose(observed, parameters.charges_e, atol=1.0e-6):
        raise HybridSystemError(
            "physical environment does not preserve the parameterized ligand charges"
        )
    provenance = {
        **parameters.provenance,
        "environment": "complex" if receptor is not None else "solvent",
        "receptor": None if receptor is None else str(Path(receptor).resolve()),
        "protein_forcefield": protein_files,
        "solvent_forcefield": solvent_files,
        "solvent_model": solvent_model,
        "padding_a": padding_a,
        "solvent_box_shape": box_shape,
        "ionic_strength_molar": ionic_strength,
        "nonbonded_cutoff_a": cutoff_a,
        "solvation_seed": int(solvation_seed),
        "solute_atom_count": molecule.n_atoms,
        "total_particle_count": system.getNumParticles(),
        "water_count": sum(
            residue.name.upper() in {"HOH", "WAT", "TIP3", "TIP4", "OPC"}
            for residue in modeller.topology.residues()
        ),
        "ion_counts": {
            name: sum(residue.name.upper() == name for residue in modeller.topology.residues())
            for name in ("NA", "CL", "K", "CA")
        },
        "canonical_box_size_nm": (
            None if canonical_box_size is None
            else [float(value) for value in canonical_box_size]
        ),
        "canonical_box_vectors_nm": (
            None if canonical_box_vectors is None
            else np.asarray(canonical_box_vectors, dtype=float).tolist()
        ),
    }
    if record_atom_classes:
        provenance["amber_atom_classes"] = atom_classes
    return PreparedPhysicalEnvironment(
        modeller.topology,
        modeller.positions,
        system,
        molecule.n_atoms,
        provenance,
    )
