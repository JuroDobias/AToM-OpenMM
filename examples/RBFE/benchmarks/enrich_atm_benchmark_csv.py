#!/usr/bin/env python
"""Enrich ATM benchmark DDG CSV rows with alignment atoms from *_asyncre.cntl files."""

import argparse
import csv
import re
from pathlib import Path


EXTRA_COLUMNS = [
    "cntl_file",
    "align_ligand1_ref_atoms",
    "align_ligand2_ref_atoms",
    "displacement",
]


def _parse_assignment(text, name):
    match = re.search(rf"^{re.escape(name)}\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return ""
    value = match.group(1).strip()
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        value = value[1:-1].strip()
    return re.sub(r"\s+", " ", value)


def _protein_root(validation_root, protein):
    candidates = [validation_root / protein, validation_root / protein.upper(), validation_root / protein.lower()]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return validation_root / protein


def _cntl_pair_key(cntl_file):
    text = cntl_file.read_text(errors="replace")
    basename = _parse_assignment(text, "BASENAME") or cntl_file.parent.name
    if "_edit-" in basename:
        pair_part = basename.rsplit("_edit-", 1)[1]
    else:
        pair_part = basename.rsplit("-", 2)[-2:]
        pair_part = "-".join(pair_part) if isinstance(pair_part, list) else pair_part
    pieces = pair_part.split("-")
    if len(pieces) < 2:
        raise ValueError(f"could not parse ligand pair from BASENAME={basename!r} in {cntl_file}")
    ligand1 = pieces[-2]
    ligand2 = pieces[-1]
    return ligand1, ligand2, text


def load_cntl_index(validation_root, proteins=None):
    proteins = {protein.upper() for protein in proteins} if proteins else None
    index = {}
    for protein_dir in sorted(path for path in validation_root.iterdir() if path.is_dir()):
        if proteins and protein_dir.name.upper() not in proteins:
            continue
        complexes = protein_dir / "complexes"
        if not complexes.is_dir():
            continue
        for cntl_file in sorted(complexes.glob("*/*_asyncre.cntl")):
            ligand1, ligand2, text = _cntl_pair_key(cntl_file)
            key = (protein_dir.name.upper(), ligand1.lower(), ligand2.lower())
            if key in index:
                raise ValueError(f"duplicate cntl match for {key}: {index[key]['cntl_file']} and {cntl_file}")
            index[key] = {
                "cntl_file": str(cntl_file),
                "align_ligand1_ref_atoms": _parse_assignment(text, "ALIGN_LIGAND1_REF_ATOMS"),
                "align_ligand2_ref_atoms": _parse_assignment(text, "ALIGN_LIGAND2_REF_ATOMS"),
                "displacement": _parse_assignment(text, "DISPLACEMENT"),
            }
    return index


def enrich_csv(input_csv, validation_root, output_csv, proteins=None):
    proteins = {protein.upper() for protein in proteins} if proteins else None
    cntl_index = load_cntl_index(validation_root, proteins=proteins)
    rows = []
    missing = []
    with open(input_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"empty CSV: {input_csv}")
        required = {"Protein", "Ligand1", "Ligand2"}
        missing_columns = required - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing_columns))}")
        fieldnames = list(reader.fieldnames)
        for column in EXTRA_COLUMNS:
            if column not in fieldnames:
                fieldnames.append(column)
        for row in reader:
            protein = row["Protein"].strip()
            if proteins and protein.upper() not in proteins:
                for column in EXTRA_COLUMNS:
                    row.setdefault(column, "")
                rows.append(row)
                continue
            key = (protein.upper(), row["Ligand1"].strip().lower(), row["Ligand2"].strip().lower())
            match = cntl_index.get(key)
            if match is None:
                missing.append(f"{protein}:{row['Ligand1']}:{row['Ligand2']}")
                for column in EXTRA_COLUMNS:
                    row[column] = ""
            else:
                row.update(match)
            rows.append(row)
    if missing:
        raise ValueError("missing exact-orientation cntl files for: " + ", ".join(missing))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--protein", action="append", help="only enrich this protein; can be repeated")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = enrich_csv(
        args.input_csv.resolve(),
        args.validation_root.resolve(),
        args.output_csv.resolve(),
        proteins=args.protein,
    )
    print(f"Wrote {len(rows)} enriched rows to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
