from pathlib import Path
import csv

import numpy as np
import yaml

from atom_openmm.rest2_validation import distribution_metrics


RUN_DIR = Path(__file__).resolve().parent
CUSTOM_DIR = RUN_DIR.parent / "run_openmm86_production"
CUSTOM_RESULT = CUSTOM_DIR / "result.yaml"
NATIVE_RESULT = RUN_DIR / "result.yaml"
OUTPUT = RUN_DIR / "comparison_custom_vs_native.yaml"
BIN_WIDTH_DEG = 5.0
BURN_IN_STEPS = 1000000


def intervals_overlap(first, second):
    return max(first[0], second[0]) <= min(first[1], second[1])


def angles(directory, ensemble):
    path = directory / "rest2" / ensemble / "physical_torsion.csv"
    with path.open(newline="") as handle:
        return np.asarray([
            float(row["torsion_deg"])
            for row in csv.DictReader(handle)
            if int(row["step"]) > BURN_IN_STEPS
        ])


def compare_ensemble(custom, native, custom_angles, native_angles):
    custom_ci = [float(value) for value in custom["basin_a_fraction_95ci"]]
    native_ci = [float(value) for value in native["basin_a_fraction_95ci"]]
    edges = np.arange(-180.0, 180.0 + BIN_WIDTH_DEG, BIN_WIDTH_DEG)
    custom_histogram, _ = np.histogram(custom_angles, bins=edges)
    native_histogram, _ = np.histogram(native_angles, bins=edges)
    return {
        "custom_basin_a_fraction": float(custom["basin_a_fraction"]),
        "native_basin_a_fraction": float(native["basin_a_fraction"]),
        "native_minus_custom_basin_a_fraction": float(
            native["basin_a_fraction"] - custom["basin_a_fraction"]
        ),
        "custom_basin_a_fraction_95ci": custom_ci,
        "native_basin_a_fraction_95ci": native_ci,
        "confidence_intervals_overlap": intervals_overlap(custom_ci, native_ci),
        "torsion_distribution": distribution_metrics(
            custom_histogram, native_histogram
        ),
        "custom_round_trips": int(custom["round_trips"]),
        "native_round_trips": int(native["round_trips"]),
        "custom_basin_transitions": int(custom["basin_transitions"]),
        "native_basin_transitions": int(native["basin_transitions"]),
        "native_exchange_diagnostics": native.get("exchange_diagnostics"),
        "native_performance": native.get("performance"),
    }


def main():
    custom = yaml.safe_load(CUSTOM_RESULT.read_text())
    native = yaml.safe_load(NATIVE_RESULT.read_text())
    ensemble_ids = ("a_started", "b_started")
    comparisons = {}
    custom_pooled = []
    native_pooled = []
    for ensemble in ensemble_ids:
        custom_angles = angles(CUSTOM_DIR, ensemble)
        native_angles = angles(RUN_DIR, ensemble)
        custom_pooled.append(custom_angles)
        native_pooled.append(native_angles)
        comparisons[ensemble] = compare_ensemble(
            custom["rest2"]["ensembles"][ensemble],
            native["rest2"]["ensembles"][ensemble],
            custom_angles, native_angles,
        )
    comparison = {
        "custom_result": str(CUSTOM_RESULT),
        "native_result": str(NATIVE_RESULT),
        "custom_backend": custom["rest2"].get("sampler_backend", "custom"),
        "native_backend": native["rest2"].get("sampler_backend"),
        "combined": compare_ensemble(
            custom["rest2"], native["rest2"],
            np.concatenate(custom_pooled), np.concatenate(native_pooled),
        ),
        "ensembles": comparisons,
    }
    OUTPUT.write_text(yaml.safe_dump(comparison, sort_keys=False))
    print(yaml.safe_dump(comparison, sort_keys=False))


if __name__ == "__main__":
    main()
