#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

from atom_openmm.covalent_hybrid import normalize_mapping_aromaticity
from atom_openmm.hybrid_mapping import _direct_rmsd, _smarts_constrained_map
try:
    from .generate_hybrid_cdk2_cohort import MAPPING_SMARTS, _run_script, _workflow
except ImportError:
    from generate_hybrid_cdk2_cohort import MAPPING_SMARTS, _run_script, _workflow


GENERATED_LIGAND = "mOMe_pCONH2"
NODES = {
    "1h1q": "naked phenyl",
    "1oiy": "para-amide",
    "1h1s": "para-sulfonamide",
    "21": "meta-methoxy",
    "32": "meta-methoxy plus para-sulfonamide",
    GENERATED_LIGAND: "meta-methoxy plus para-amide",
}
EDGES = (
    ("1h1q", "1oiy"),
    ("1h1q", "1h1s"),
    ("1h1q", "21"),
    ("21", GENERATED_LIGAND),
    ("21", "32"),
    ("1oiy", GENERATED_LIGAND),
    ("1h1s", "32"),
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_sdf(path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() != 1:
        raise ValueError(f"could not read one 3D molecule from {path}")
    return normalize_mapping_aromaticity(molecule)


def _source_ligand(source_cohort, ligand):
    matches = sorted(Path(source_cohort).glob(f"*/alignment_structures/{ligand}-p.sdf"))
    if not matches:
        raise FileNotFoundError(f"no aligned structure found for {ligand}")
    hashes = {_sha256(path) for path in matches}
    if len(hashes) != 1:
        raise ValueError(f"aligned structures differ across edges for {ligand}")
    return matches[0]


def _remove_hydrogens_with_index_map(molecule):
    indexed = Chem.Mol(molecule)
    for atom in indexed.GetAtoms():
        atom.SetIntProp("_source_atom_index", atom.GetIdx())
    heavy = Chem.RemoveHs(indexed)
    index_map = {
        atom.GetIntProp("_source_atom_index"): atom.GetIdx()
        for atom in heavy.GetAtoms()
    }
    return heavy, index_map


def generate_methoxy_amide(amide_path, disubstituted_path, output_path):
    amide = _load_sdf(amide_path)
    disubstituted = _load_sdf(disubstituted_path)
    mapping = _smarts_constrained_map(
        amide, disubstituted, MAPPING_SMARTS
    )
    mapped_b = set(mapping.values())
    methoxy = Chem.MolFromSmarts("[c:1]-[O:2]-[CH3:3]")
    matches = [
        match
        for match in disubstituted.GetSubstructMatches(methoxy)
        if match[0] in mapped_b and match[1] not in mapped_b and match[2] not in mapped_b
    ]
    if len(matches) != 1:
        raise ValueError("could not identify one unique methoxy group in ligand 32")
    target_ring, target_oxygen, target_methyl = matches[0]
    source_ring = {right: left for left, right in mapping.items()}[target_ring]

    amide_heavy, amide_index_map = _remove_hydrogens_with_index_map(amide)
    target_heavy, target_index_map = _remove_hydrogens_with_index_map(disubstituted)
    source_ring_heavy = amide_index_map[source_ring]
    target_oxygen_heavy = target_index_map[target_oxygen]
    target_methyl_heavy = target_index_map[target_methyl]
    editor = Chem.RWMol(amide_heavy)
    oxygen = editor.AddAtom(Chem.Atom(target_heavy.GetAtomWithIdx(target_oxygen_heavy)))
    methyl = editor.AddAtom(Chem.Atom(target_heavy.GetAtomWithIdx(target_methyl_heavy)))
    editor.AddBond(source_ring_heavy, oxygen, Chem.BondType.SINGLE)
    editor.AddBond(oxygen, methyl, Chem.BondType.SINGLE)
    product_heavy = editor.GetMol()
    Chem.SanitizeMol(product_heavy)

    conformer = Chem.Conformer(product_heavy.GetNumAtoms())
    source_conformer = amide_heavy.GetConformer()
    target_conformer = target_heavy.GetConformer()
    for index in range(amide_heavy.GetNumAtoms()):
        conformer.SetAtomPosition(index, source_conformer.GetAtomPosition(index))
    conformer.SetAtomPosition(oxygen, target_conformer.GetAtomPosition(target_oxygen_heavy))
    conformer.SetAtomPosition(methyl, target_conformer.GetAtomPosition(target_methyl_heavy))
    product_heavy.RemoveAllConformers()
    product_heavy.AddConformer(conformer, assignId=True)
    product = Chem.AddHs(product_heavy, addCoords=True)
    product.SetProp("_Name", GENERATED_LIGAND)

    forcefield = AllChem.UFFGetMoleculeForceField(product, confId=0)
    if forcefield is None:
        raise ValueError("UFF could not parameterize generated methoxy-amide ligand")
    for index in range(amide_heavy.GetNumAtoms()):
        forcefield.AddFixedPoint(index)
    forcefield.Initialize()
    forcefield.Minimize(maxIts=500)

    amide_pattern = Chem.MolFromSmarts("[c]-[C](=[O])-[N]")
    sulfonamide_pattern = Chem.MolFromSmarts("[S](=[O])(=[O])-[N]")
    methoxy_matches = product.GetSubstructMatches(methoxy)
    amide_matches = product.GetSubstructMatches(amide_pattern)
    if len(methoxy_matches) != 1 or len(amide_matches) != 1:
        raise ValueError("generated ligand does not contain one methoxy and one amide")
    if product.HasSubstructMatch(sulfonamide_pattern):
        raise ValueError("generated methoxy-amide still contains a sulfonamide")
    methoxy_ring = methoxy_matches[0][0]
    amide_ring = amide_matches[0][0]
    if len(rdmolops.GetShortestPath(product, methoxy_ring, amide_ring)) - 1 != 1:
        raise ValueError("generated methoxy and amide are not ortho to each other")

    output_path = Path(output_path)
    writer = Chem.SDWriter(str(output_path))
    writer.write(product)
    writer.close()
    return {
        "ligand_id": GENERATED_LIGAND,
        "source_amide": str(Path(amide_path).resolve()),
        "source_disubstituted": str(Path(disubstituted_path).resolve()),
        "source_amide_sha256": _sha256(amide_path),
        "source_disubstituted_sha256": _sha256(disubstituted_path),
        "mapping_smarts": MAPPING_SMARTS,
        "mapped_heavy_atom_count": sum(
            amide.GetAtomWithIdx(index).GetAtomicNum() != 1 for index in mapping
        ),
        "mapped_direct_rmsd_angstrom": _direct_rmsd(amide, disubstituted, mapping),
        "canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(product), canonical=True),
        "construction": "1oiy coordinates plus the unique meta-methoxy group from 32",
    }


def _network_payload():
    def edge(a, b):
        return {"ligand_a": a, "ligand_b": b, "directory": f"{a}--{b}"}

    return {
        "schema_version": 1,
        "reference_node": "1h1q",
        "target": ["1oiy", "32"],
        "target_references": {
            "published_atm_kcal_per_mol": 1.547461,
            "published_atm_error_kcal_per_mol": 0.52554,
            "experimental_kcal_per_mol": 0.035,
        },
        "nodes": [{"id": node, "chemistry": chemistry} for node, chemistry in NODES.items()],
        "edges": [edge(a, b) for a, b in EDGES],
        "cycles": [
            {
                "id": "amide_additivity",
                "terms": [
                    {"edge": "1h1q--1oiy", "coefficient": 1},
                    {"edge": f"1oiy--{GENERATED_LIGAND}", "coefficient": 1},
                    {"edge": f"21--{GENERATED_LIGAND}", "coefficient": -1},
                    {"edge": "1h1q--21", "coefficient": -1},
                ],
            },
            {
                "id": "sulfonamide_additivity",
                "terms": [
                    {"edge": "1h1q--1h1s", "coefficient": 1},
                    {"edge": "1h1s--32", "coefficient": 1},
                    {"edge": "21--32", "coefficient": -1},
                    {"edge": "1h1q--21", "coefficient": -1},
                ],
            },
        ],
    }


def generate(source_cohort, benchmark_root, output, source_dir_name):
    source_cohort = Path(source_cohort).resolve()
    benchmark_root = Path(benchmark_root).resolve()
    output = Path(output).resolve()
    receptor = benchmark_root / "ATM_Validation/CDK2/receptor/CDK2_new_2_edit.pdb"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    sources = {
        ligand: _source_ligand(source_cohort, ligand)
        for ligand in NODES if ligand != GENERATED_LIGAND
    }
    generated = output / f"{GENERATED_LIGAND}-p.sdf"
    provenance = generate_methoxy_amide(sources["1oiy"], sources["32"], generated)
    sources[GENERATED_LIGAND] = generated

    for ligand_a, ligand_b in EDGES:
        edge = f"{ligand_a}--{ligand_b}"
        target = output / edge
        (target / "ligands").mkdir(parents=True)
        shutil.copy2(receptor, target / "receptor.pdb")
        for ligand in (ligand_a, ligand_b):
            shutil.copy2(sources[ligand], target / "ligands" / f"{ligand}-p.sdf")
        workflow = _workflow(
            ligand_a,
            ligand_b,
            n_snapshots=100,
            switch_time_ps=100.0,
            adaptive_switching=True,
            convergence=True,
            dummy_core_nonbonded="off",
        )
        (target / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False))
        run = target / "run.sh"
        run.write_text(_run_script(edge, source_dir_name=source_dir_name))
        run.chmod(0o755)

    (output / "generated_ligand.yaml").write_text(
        yaml.safe_dump(provenance, sort_keys=False)
    )
    (output / "network.yaml").write_text(
        yaml.safe_dump(_network_payload(), sort_keys=False)
    )
    analyze = output / "analyze_network.sh"
    analyze.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f'PYTHONPATH="$HOME/myAToM/{source_dir_name}${{PYTHONPATH:+:$PYTHONPATH}}" '
        f'python -m atom_openmm.rbfe_network "$(dirname "$0")/network.yaml"\n'
    )
    analyze.chmod(0o755)
    submit = output / "submit_all.sh"
    submit.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(f"(cd {a}--{b} && sbatch run.sh)" for a, b in EDGES)
        + "\n"
    )
    submit.chmod(0o755)
    (output / "README.md").write_text(
        "# CDK2 point-mutation hybrid network\n\n"
        "Seven small hybrid-topology transformations replace the direct 1oiy -> 32 "
        "edge. Ligand 21 supplies the meta-methoxy parent; mOMe_pCONH2 is generated "
        "from aligned 1oiy and 32 coordinates. Two cycles test amide and sulfonamide "
        "substituent additivity. Each edge uses REST2 endpoint sampling, 20 reusable "
        "adaptive pilot samples at 100/300/1000 ps, up to 100 samples, and BAR "
        "convergence stopping. Run ./analyze_network.sh after or during the cohort.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cohort", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir-name", default="AToM-OpenMM-hybrid-dummy-core")
    args = parser.parse_args()
    generate(
        args.source_cohort,
        args.benchmark_root,
        args.output,
        args.source_dir_name,
    )


if __name__ == "__main__":
    main()
