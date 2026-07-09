#!/usr/bin/env python
"""Collect generated ATM benchmark NEQTI result.yaml files into one CSV."""

import argparse
import csv
from pathlib import Path

import yaml


OUTPUT_COLUMNS = [
    "system",
    "ligand_a",
    "ligand_b",
    "reference_ddg",
    "neqti_ddg",
    "neqti_error",
    "overlap_score",
    "status",
    "warnings",
    "workdir",
]


def _read_yaml(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def collect_results(index_csv, output_csv):
    rows = []
    with open(index_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"system", "ligand_a", "ligand_b", "reference_ddg", "workdir"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"index CSV is missing columns: {', '.join(sorted(missing))}")
        for index_row in reader:
            workdir = Path(index_row["workdir"])
            result_files = sorted(workdir.glob("*/result.yaml"))
            if len(result_files) > 1:
                raise ValueError(f"multiple result.yaml files under {workdir}")
            if result_files:
                result = _read_yaml(result_files[0])
                result_block = result.get("result") or {}
                quality = result.get("quality") or {}
                rows.append(
                    {
                        "system": index_row["system"],
                        "ligand_a": index_row["ligand_a"],
                        "ligand_b": index_row["ligand_b"],
                        "reference_ddg": index_row["reference_ddg"],
                        "neqti_ddg": result_block.get("ddg_kcal_per_mol"),
                        "neqti_error": result_block.get("ddg_error_kcal_per_mol"),
                        "overlap_score": quality.get("overlap_score"),
                        "status": result.get("status", "unknown"),
                        "warnings": "; ".join(quality.get("warnings") or []),
                        "workdir": str(workdir),
                    }
                )
            else:
                rows.append(
                    {
                        "system": index_row["system"],
                        "ligand_a": index_row["ligand_a"],
                        "ligand_b": index_row["ligand_b"],
                        "reference_ddg": index_row["reference_ddg"],
                        "neqti_ddg": None,
                        "neqti_error": None,
                        "overlap_score": None,
                        "status": "missing",
                        "warnings": "result.yaml missing",
                        "workdir": str(workdir),
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-csv", required=True, type=Path)
    parser.add_argument("--output-csv", default=Path("benchmark_results.csv"), type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = collect_results(args.index_csv.resolve(), args.output_csv.resolve())
    print(f"Wrote {len(rows)} rows to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
