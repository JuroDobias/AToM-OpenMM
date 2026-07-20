from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit
from openff.toolkit import ForceField, Molecule
from openff.units import unit as offunit


class CovalentParameterError(RuntimeError):
    pass


@dataclass(frozen=True)
class CovalentParameterBundle:
    molecule: Molecule
    system: mm.System
    charges_e: np.ndarray
    cache_key: str
    provenance: dict[str, object]


def _canonical_cache_key(molecule: Molecule, model: str, forcefield: str) -> str:
    payload = {
        "mapped_isomeric_smiles": molecule.to_smiles(
            isomeric=True, explicit_hydrogens=True, mapped=True
        ),
        "formal_charge": int(round(molecule.total_charge.m_as(offunit.elementary_charge))),
        "charge_model": model,
        "forcefield": forcefield,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _predict_espaloma_charges(molecule: Molecule, model: str) -> np.ndarray:
    try:
        import espaloma as esp
        from openmmforcefields.generators import EspalomaTemplateGenerator
    except ImportError as exc:
        raise CovalentParameterError(
            "Espaloma NN charges were requested, but espaloma and openmmforcefields are not installed"
        ) from exc

    generator = EspalomaTemplateGenerator(
        forcefield=model,
        template_generator_kwargs={
            "reference_forcefield": "openff_unconstrained-2.2.1",
            "charge_method": "nn",
        },
    )
    graph = esp.Graph(molecule)
    generator.espaloma_model(graph.heterograph)
    try:
        charges = graph.nodes["n1"].data["q"].flatten().detach().cpu().numpy()
    except Exception as exc:
        raise CovalentParameterError("Espaloma did not produce atomic NN charges") from exc
    return np.asarray(charges, dtype=np.float64)


def constrain_charge_sum(
    charges_e: np.ndarray,
    adjustable_indices: list[int] | np.ndarray,
    target_total_e: float,
) -> tuple[np.ndarray, float]:
    charges = np.asarray(charges_e, dtype=np.float64).copy()
    adjustable = np.asarray(adjustable_indices, dtype=int)
    if adjustable.size == 0:
        raise CovalentParameterError("charge correction requires at least one adjustable atom")
    correction = (float(target_total_e) - float(charges.sum())) / float(adjustable.size)
    charges[adjustable] += correction
    if not np.isclose(charges.sum(), target_total_e, atol=1.0e-10):
        raise CovalentParameterError("charge correction did not reach the requested total")
    return charges, correction


def assign_espaloma_nn_charges(
    molecule: Molecule,
    *,
    cache_dir: Path | None = None,
    model: str = "espaloma-0.3.2",
    forcefield: str = "openff-2.2.1.offxml",
) -> tuple[np.ndarray, str, bool]:
    cache_key = _canonical_cache_key(molecule, model, forcefield)
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{cache_key}.charges.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            charges = np.asarray(data["charges_e"], dtype=np.float64)
            if charges.shape != (molecule.n_atoms,):
                raise CovalentParameterError(f"cached charges in {cache_file} have the wrong size")
            return charges, cache_key, True

    charges = _predict_espaloma_charges(molecule, model)
    if charges.shape != (molecule.n_atoms,):
        raise CovalentParameterError(
            f"Espaloma returned {charges.size} charges for {molecule.n_atoms} atoms"
        )
    formal_charge = float(molecule.total_charge.m_as(offunit.elementary_charge))
    charges, _ = constrain_charge_sum(charges, np.arange(molecule.n_atoms), formal_charge)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cache_key": cache_key,
                    "charge_model": "espaloma_nn",
                    "espaloma_model": model,
                    "forcefield": forcefield,
                    "charges_e": charges.tolist(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return charges, cache_key, False


def _system_charges(system: mm.System) -> np.ndarray:
    forces = [force for force in system.getForces() if isinstance(force, mm.NonbondedForce)]
    if len(forces) != 1:
        raise CovalentParameterError(
            f"expected one OpenFF NonbondedForce, found {len(forces)}"
        )
    force = forces[0]
    return np.asarray(
        [
            force.getParticleParameters(index)[0].value_in_unit(mm.unit.elementary_charge)
            for index in range(force.getNumParticles())
        ],
        dtype=np.float64,
    )


def ff19sb_backbone_charges(receptor_pdb: Path, residue_id: int) -> dict[str, float]:
    pdb = app.PDBFile(str(receptor_pdb))
    forcefield = app.ForceField("amber19/protein.ff19SB.xml", "amber19/opc.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addExtraParticles(forcefield)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    force = next(item for item in system.getForces() if isinstance(item, mm.NonbondedForce))
    residues = [
        residue for residue in modeller.topology.residues()
        if residue.name == "CYS" and residue.id == str(residue_id)
    ]
    if len(residues) != 1:
        raise CovalentParameterError(
            f"expected one ff19SB CYS {residue_id}, found {len(residues)}"
        )
    by_name = {atom.name: atom.index for atom in residues[0].atoms()}
    required = ("N", "H", "CA", "HA", "C", "O")
    missing = [name for name in required if name not in by_name]
    if missing:
        raise CovalentParameterError(
            f"ff19SB CYS {residue_id} is missing fixed backbone atoms: {', '.join(missing)}"
        )
    return {
        name: force.getParticleParameters(by_name[name])[0].value_in_unit(unit.elementary_charge)
        for name in required
    }


def apply_modified_residue_charges(
    bundle: CovalentParameterBundle,
    metadata: dict[str, object],
    backbone_charges_e: dict[str, float],
    *,
    forcefield: str = "openff-2.2.1.offxml",
) -> CovalentParameterBundle:
    molecule = Molecule(bundle.molecule)
    rdkit = molecule.to_rdkit()
    mapped = {
        int(number): int(index)
        for number, index in metadata["capped_cys_atom_map_indices"].items()
    }

    fixed = {
        mapped[4]: backbone_charges_e["N"],
        mapped[5]: backbone_charges_e["CA"],
        mapped[8]: backbone_charges_e["C"],
        mapped[9]: backbone_charges_e["O"],
    }
    for map_number, hydrogen_name in ((4, "H"), (5, "HA")):
        hydrogens = [
            atom.GetIdx() for atom in rdkit.GetAtomWithIdx(mapped[map_number]).GetNeighbors()
            if atom.GetAtomicNum() == 1
        ]
        if len(hydrogens) != 1:
            raise CovalentParameterError(
                f"capped product requires one {hydrogen_name} attached to mapped atom {map_number}"
            )
        fixed[hydrogens[0]] = backbone_charges_e[hydrogen_name]

    forcefield_templates = app.ForceField("amber19/protein.ff19SB.xml")._templates
    ace = {atom.name: float(atom.parameters["charge"]) for atom in forcefield_templates["ACE"].atoms}
    nme = {atom.name: float(atom.parameters["charge"]) for atom in forcefield_templates["NME"].atoms}
    cap_fixed = {
        mapped[1]: ace["CH3"],
        mapped[2]: ace["C"],
        mapped[3]: ace["O"],
        mapped[10]: nme["N"],
        mapped[11]: nme["C"],
    }
    cap_hydrogen_parameters = ((1, (ace["H1"], ace["H2"], ace["H3"])),
                               (10, (nme["H"],)),
                               (11, (nme["H1"], nme["H2"], nme["H3"])))
    for map_number, values in cap_hydrogen_parameters:
        hydrogens = sorted(
            atom.GetIdx() for atom in rdkit.GetAtomWithIdx(mapped[map_number]).GetNeighbors()
            if atom.GetAtomicNum() == 1
        )
        if len(hydrogens) != len(values):
            raise CovalentParameterError(
                f"capped product has an unexpected number of hydrogens at mapped cap atom {map_number}"
            )
        cap_fixed.update(zip(hydrogens, values))

    residue_indices = set(fixed)
    for map_number in (6, 7):
        center = mapped[map_number]
        residue_indices.add(center)
        residue_indices.update(
            atom.GetIdx() for atom in rdkit.GetAtomWithIdx(center).GetNeighbors()
            if atom.GetAtomicNum() == 1
        )
    residue_indices.add(int(metadata["transferred_hydrogen_atom_index"]))
    residue_indices.update(range(int(metadata["ligand_atom_offset"]), molecule.n_atoms))
    adjustable = sorted(residue_indices - set(fixed))

    charges = np.asarray(bundle.charges_e, dtype=np.float64).copy()
    for index, charge in fixed.items():
        charges[index] = float(charge)
    for index, charge in cap_fixed.items():
        charges[index] = float(charge)
    residue_charge = float(charges[list(fixed)].sum()) + float(charges[adjustable].sum())
    correction = -residue_charge / len(adjustable)
    charges[adjustable] += correction

    cap_indices = sorted(set(range(molecule.n_atoms)) - residue_indices)
    if set(cap_indices) != set(cap_fixed):
        raise CovalentParameterError("capped product contains unclassified ACE/NME atoms")
    if not np.isclose(charges[cap_indices].sum(), 0.0, atol=1.0e-10):
        raise CovalentParameterError("ff19SB ACE/NME cap charge is not zero")
    if not np.isclose(charges[list(residue_indices)].sum(), 0.0, atol=1.0e-10):
        raise CovalentParameterError("modified Cys-ligand residue charge is not zero")
    if not np.isclose(charges.sum(), 0.0, atol=1.0e-10):
        raise CovalentParameterError("corrected capped product charge is not zero")

    molecule.partial_charges = charges * offunit.elementary_charge
    system = ForceField(forcefield).create_openmm_system(
        molecule.to_topology(),
        charge_from_molecules=[molecule],
        allow_nonintegral_charges=False,
    )
    observed = _system_charges(system)
    if not np.allclose(observed, charges, atol=1.0e-7):
        raise CovalentParameterError("corrected charges were not preserved by OpenFF")
    provenance = dict(bundle.provenance)
    provenance.update(
        {
            "charge_constraint": "ff19SB_fixed_backbone_modified_residue_zero",
            "fixed_backbone_atoms": sorted(backbone_charges_e),
            "adjustable_modified_residue_atom_count": len(adjustable),
            "modified_residue_charge_e": float(charges[list(residue_indices)].sum()),
            "modified_residue_uniform_correction_e": float(correction),
            "cap_charge_model": "ff19SB_ACE_NME",
            "net_charge_e": float(charges.sum()),
        }
    )
    return CovalentParameterBundle(
        molecule=molecule,
        system=system,
        charges_e=charges,
        cache_key=f"{bundle.cache_key}-ff19sb-backbone",
        provenance=provenance,
    )


def parameterize_capped_product(
    sdf: Path,
    *,
    cache_dir: Path | None = None,
    forcefield: str = "openff-2.2.1.offxml",
    espaloma_model: str = "espaloma-0.3.2",
) -> CovalentParameterBundle:
    molecule = Molecule.from_file(str(sdf), allow_undefined_stereo=False)
    charges, cache_key, cache_hit = assign_espaloma_nn_charges(
        molecule,
        cache_dir=cache_dir,
        model=espaloma_model,
        forcefield=forcefield,
    )
    molecule.partial_charges = charges * offunit.elementary_charge
    openff = ForceField(forcefield)
    system = openff.create_openmm_system(
        molecule.to_topology(),
        charge_from_molecules=[molecule],
        allow_nonintegral_charges=False,
    )
    observed = _system_charges(system)
    if not np.allclose(observed, charges, atol=1.0e-7):
        raise CovalentParameterError(
            "OpenFF system charges differ from the supplied Espaloma NN charges"
        )
    provenance = {
        "charge_model": "espaloma_nn",
        "espaloma_model": espaloma_model,
        "forcefield": forcefield,
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "net_charge_e": float(charges.sum()),
        "max_charge_correction_e": float(
            np.max(np.abs(charges - observed)) if charges.size else 0.0
        ),
    }
    return CovalentParameterBundle(molecule, system, charges, cache_key, provenance)
