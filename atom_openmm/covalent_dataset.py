from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem


R_KJ_MOL_K = 0.00831446261815324
DEFAULT_CORE_SMARTS = (
    "[#6]1:[#6]:[#6]:[#6]:[#7](-[#6]2(-[#6](=[#8])-[#7]-[#6@@H](-[#6]-"
    "[#6@@H]3-[#6]-[#6]-[#7]-[#6]-3=[#8])-[#6](=[#8])-[H])-[#6]-[#6]3:"
    "[#6]:[#6]:[#6]:[#6]:[#6]:3-[#6]-2):[#6]:1=[#8]"
)
ALDEHYDE_SMARTS = "[CX3H1:1](=[O:2])"
CAPPED_CYS_SMILES = (
    "[CH3:1][C:2](=[O:3])[NH:4][C@@H:5]([CH2:6][SH:7])"
    "[C:8](=[O:9])[NH:10][CH3:11]"
)


class CovalentDatasetError(ValueError):
    pass


def _read_csv(path: Path, *, encoding: str = "utf-8") -> list[dict[str, str]]:
    with path.open(encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise CovalentDatasetError(f"cannot derive a file name from {value!r}")
    return cleaned


def _load_sdf(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    molecule = supplier[0] if len(supplier) else None
    if molecule is None:
        raise CovalentDatasetError(f"could not read molecule from {path}")
    if molecule.GetNumConformers() != 1:
        raise CovalentDatasetError(f"{path} must contain one 3D conformer")
    return molecule


def _pdb_atom_record(line: str):
    if not line.startswith(("ATOM  ", "HETATM")):
        return None
    return {
        "name": line[12:16].strip(),
        "resname": line[17:20].strip(),
        "chain": line[21:22],
        "resid": line[22:26].strip(),
        "icode": line[26:27],
        "element": (line[76:78].strip() or line[12:16].strip()[0]).upper(),
        "xyz": np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
        ),
    }


def receptor_atom_coordinates(receptor: Path, residue_id: int) -> dict[str, np.ndarray]:
    coordinates = {}
    for line in receptor.read_text().splitlines():
        atom = _pdb_atom_record(line)
        if atom and atom["resname"] == "CYS" and atom["resid"] == str(residue_id):
            coordinates[atom["name"]] = atom["xyz"]
    required = {"N", "CA", "C", "O", "CB", "SG", "HG"}
    missing = sorted(required - coordinates.keys())
    if missing:
        raise CovalentDatasetError(
            f"receptor CYS {residue_id} is missing required atoms: {', '.join(missing)}"
        )
    return coordinates


def retain_protein_waters(receptor: Path, output: Path, cutoff_a: float = 5.0) -> int:
    lines = receptor.read_text().splitlines(keepends=True)
    protein_heavy = []
    water_oxygen = {}
    for line in lines:
        atom = _pdb_atom_record(line)
        if atom is None:
            continue
        key = (atom["chain"], atom["resid"], atom["icode"], atom["resname"])
        if atom["resname"] in {"HOH", "WAT"}:
            if atom["element"] == "O":
                water_oxygen[key] = atom["xyz"]
        elif atom["element"] != "H":
            protein_heavy.append(atom["xyz"])
    if not protein_heavy:
        raise CovalentDatasetError(f"no protein heavy atoms found in {receptor}")
    protein_heavy = np.asarray(protein_heavy)
    retained = {
        key
        for key, oxygen in water_oxygen.items()
        if float(np.min(np.linalg.norm(protein_heavy - oxygen, axis=1))) <= float(cutoff_a)
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    previous_atom_key = None
    with output.open("w") as handle:
        for line in lines:
            atom = _pdb_atom_record(line)
            if atom is not None:
                key = (atom["chain"], atom["resid"], atom["icode"], atom["resname"])
                previous_atom_key = key
                if atom["resname"] in {"HOH", "WAT"} and key not in retained:
                    continue
                handle.write(line)
            elif line.startswith("TER"):
                if previous_atom_key is None or previous_atom_key[3] not in {"HOH", "WAT"} or previous_atom_key in retained:
                    handle.write(line)
            elif line.startswith("END"):
                handle.write(line)
            elif not line.startswith(("CONECT", "MASTER")):
                handle.write(line)
    return len(retained)


def _mapped_atoms(molecule: Chem.Mol) -> dict[int, int]:
    return {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum()
    }


def _kabsch_transform(source: np.ndarray, target: np.ndarray):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    translation = target_center - source_center @ rotation
    return rotation, translation


def build_capped_thiohemiacetal(
    ligand: Chem.Mol,
    cys_coordinates: dict[str, np.ndarray],
) -> tuple[Chem.Mol, dict[str, object]]:
    aldehyde = Chem.MolFromSmarts(ALDEHYDE_SMARTS)
    matches = ligand.GetSubstructMatches(aldehyde, uniquify=True)
    if len(matches) != 1:
        raise CovalentDatasetError(f"expected one aldehyde match, found {len(matches)}")
    aldehyde_carbon, aldehyde_oxygen = matches[0]

    capped_cys = Chem.AddHs(Chem.MolFromSmiles(CAPPED_CYS_SMILES))
    if AllChem.EmbedMolecule(capped_cys, randomSeed=20260718) != 0:
        raise CovalentDatasetError("could not generate capped-cysteine coordinates")
    AllChem.MMFFOptimizeMolecule(capped_cys, maxIters=500)
    maps = _mapped_atoms(capped_cys)
    atom_names = {4: "N", 5: "CA", 8: "C", 9: "O", 6: "CB", 7: "SG"}
    source = np.asarray(
        [list(capped_cys.GetConformer().GetAtomPosition(maps[number])) for number in atom_names]
    )
    target = np.asarray([cys_coordinates[atom_names[number]] for number in atom_names])
    rotation, translation = _kabsch_transform(source, target)
    conformer = capped_cys.GetConformer()
    for atom in capped_cys.GetAtoms():
        position = np.asarray(conformer.GetAtomPosition(atom.GetIdx())) @ rotation + translation
        conformer.SetAtomPosition(atom.GetIdx(), position)

    sulfur = maps[7]
    sulfur_hydrogens = [
        neighbor.GetIdx()
        for neighbor in capped_cys.GetAtomWithIdx(sulfur).GetNeighbors()
        if neighbor.GetAtomicNum() == 1
    ]
    if len(sulfur_hydrogens) != 1:
        raise CovalentDatasetError("capped cysteine must contain one thiol hydrogen")
    transferred_hydrogen = sulfur_hydrogens[0]
    conformer.SetAtomPosition(transferred_hydrogen, cys_coordinates["HG"])

    combined = Chem.CombineMols(capped_cys, ligand)
    editable = Chem.RWMol(combined)
    ligand_offset = capped_cys.GetNumAtoms()
    product_carbon = ligand_offset + aldehyde_carbon
    product_oxygen = ligand_offset + aldehyde_oxygen
    editable.RemoveBond(sulfur, transferred_hydrogen)
    editable.RemoveBond(product_carbon, product_oxygen)
    editable.AddBond(product_carbon, product_oxygen, Chem.BondType.SINGLE)
    editable.AddBond(sulfur, product_carbon, Chem.BondType.SINGLE)
    editable.AddBond(product_oxygen, transferred_hydrogen, Chem.BondType.SINGLE)
    product = editable.GetMol()
    Chem.SanitizeMol(product)
    Chem.AssignAtomChiralTagsFromStructure(product, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(product, cleanIt=True, force=True)

    carbon_atom = product.GetAtomWithIdx(product_carbon)
    cip = carbon_atom.GetProp("_CIPCode") if carbon_atom.HasProp("_CIPCode") else None
    metadata = {
        "capped_cys_atom_count": capped_cys.GetNumAtoms(),
        "capped_cys_atom_map_indices": {
            str(number): index for number, index in sorted(maps.items())
        },
        "ligand_atom_offset": ligand_offset,
        "ligand_to_product_atom_indices": {
            str(index): ligand_offset + index for index in range(ligand.GetNumAtoms())
        },
        "cys_sulfur_atom_index": sulfur,
        "transferred_hydrogen_atom_index": transferred_hydrogen,
        "electrophile_carbon_atom_index": product_carbon,
        "product_oxygen_atom_index": product_oxygen,
        "product_cip": cip,
        "initial_sulfur_carbon_distance_a": float(
            np.linalg.norm(
                np.asarray(product.GetConformer().GetAtomPosition(sulfur))
                - np.asarray(product.GetConformer().GetAtomPosition(product_carbon))
            )
        ),
    }
    product.SetProp("covalent_reaction", "cysteine_aldehyde_thiohemiacetal")
    product.SetProp("product_cip", cip or "unknown")
    return product, metadata


def _write_sdf(molecule: Chem.Mol, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_rhino_dataset(
    source: Path,
    output: Path,
    *,
    core_smarts: str = DEFAULT_CORE_SMARTS,
    residue_id: int = 147,
    water_cutoff_a: float = 5.0,
    temperature_k: float = 298.15,
) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    data_rows = _read_csv(source / "data.csv", encoding="utf-8-sig")
    manifest_rows = _read_csv(source / "manifest.csv")
    by_name = {row["Name"]: row for row in data_rows}
    query = Chem.MolFromSmarts(core_smarts)
    if query is None:
        raise CovalentDatasetError("core SMARTS is invalid")
    receptor = source / "protein_annealed.pdb"
    cys_coordinates = receptor_atom_coordinates(receptor, residue_id)
    retained_waters = retain_protein_waters(
        receptor, output / "receptor" / "protein_retained_waters.pdb", water_cutoff_a
    )

    assay_columns = {
        "A2": "Property: HRVA-A2                Ki  (mM)",
        "B14": "Property: HRVB-14              Ki  (mM)",
    }
    assay_rows = []
    ligand_manifest = []
    assay_by_ligand = defaultdict(dict)
    for record in manifest_rows:
        name = record["entity_name"]
        if name not in by_name:
            raise CovalentDatasetError(f"manifest ligand {name} is missing from data.csv")
        source_sdf = source / record["archive_path"]
        ligand = _load_sdf(source_sdf)
        matches = ligand.GetSubstructMatches(query, uniquify=True)
        if len(matches) != 1:
            raise CovalentDatasetError(f"{name}: core SMARTS matched {len(matches)} times")
        formal_charge = sum(atom.GetFormalCharge() for atom in ligand.GetAtoms())
        if formal_charge != 0:
            raise CovalentDatasetError(f"{name}: expected neutral ligand, found charge {formal_charge}")
        ligand_dir = output / "ligands" / _safe_name(name)
        ligand_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_sdf, ligand_dir / "aldehyde.sdf")
        product, product_metadata = build_capped_thiohemiacetal(ligand, cys_coordinates)
        _write_sdf(product, ligand_dir / "ace_cys_product_nme.sdf")
        (ligand_dir / "product_metadata.yaml").write_text(
            yaml.safe_dump(product_metadata, sort_keys=False)
        )

        row = by_name[name]
        for serotype, column in assay_columns.items():
            value = row.get(column, "").strip()
            if not value:
                continue
            ki_mm = float(value)
            assay_by_ligand[name][serotype] = ki_mm
            assay_rows.append(
                {
                    "ligand_id": name,
                    "entity_id": record["entity_id"],
                    "serotype": serotype,
                    "ki_value": ki_mm,
                    "ki_unit": "mM",
                    "temperature_k": temperature_k,
                    "qualifier": "=",
                }
            )
        ligand_manifest.append(
            {
                "ligand_id": name,
                "entity_id": record["entity_id"],
                "compound_id": record["compound_id"],
                "microstate_id": record["microstate_id"],
                "aldehyde_sdf": str((ligand_dir / "aldehyde.sdf").relative_to(output)),
                "capped_product_sdf": str(
                    (ligand_dir / "ace_cys_product_nme.sdf").relative_to(output)
                ),
                "formal_charge": formal_charge,
                "core_match_atom_indices_1based": [index + 1 for index in matches[0]],
                "product_cip": product_metadata["product_cip"],
            }
        )

    reference = "I79DJ_543"
    pilot_targets = ["LIBA225", "I79DJ_735", "I79DJ_820"]
    pilot_edges = []
    for target in pilot_targets:
        if "A2" not in assay_by_ligand[reference] or "A2" not in assay_by_ligand[target]:
            raise CovalentDatasetError(f"missing A2 Ki for pilot edge {reference} -> {target}")
        ddg = R_KJ_MOL_K * temperature_k * math.log(
            assay_by_ligand[target]["A2"] / assay_by_ligand[reference]["A2"]
        )
        pilot_edges.append(
            {
                "ligand_a": reference,
                "ligand_b": target,
                "serotype": "A2",
                "experimental_ddg_kj_per_mol": ddg,
            }
        )

    _write_csv(
        output / "assays.csv",
        assay_rows,
        ["ligand_id", "entity_id", "serotype", "ki_value", "ki_unit", "temperature_k", "qualifier"],
    )
    dataset = {
        "schema_version": 1,
        "dataset": "rhino_covalent_rbfe",
        "source": str(source),
        "receptor": "receptor/protein_retained_waters.pdb",
        "covalent_residue": {"name": "CYS", "id": residue_id, "sulfur_atom": "SG"},
        "reaction": "cysteine_aldehyde_thiohemiacetal",
        "core_smarts": core_smarts,
        "water_retention": {
            "selection": "oxygen within cutoff of protein heavy atoms",
            "cutoff_a": water_cutoff_a,
            "retained_count": retained_waters,
        },
        "assay_temperature_k": temperature_k,
        "ligands": ligand_manifest,
        "pilot_edges": pilot_edges,
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False, width=120))
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Normalize and prepare a covalent RBFE dataset")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--core-smarts", default=DEFAULT_CORE_SMARTS)
    parser.add_argument("--residue-id", type=int, default=147)
    parser.add_argument("--water-cutoff-a", type=float, default=5.0)
    parser.add_argument("--temperature-k", type=float, default=298.15)
    args = parser.parse_args(argv)
    result = prepare_rhino_dataset(
        args.source,
        args.output,
        core_smarts=args.core_smarts,
        residue_id=args.residue_id,
        water_cutoff_a=args.water_cutoff_a,
        temperature_k=args.temperature_k,
    )
    print(yaml.safe_dump(result, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
