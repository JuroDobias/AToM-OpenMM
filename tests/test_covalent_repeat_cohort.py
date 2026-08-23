import importlib.util
from pathlib import Path

import yaml


SCRIPT = (
    Path(__file__).parents[1]
    / "examples/RBFE/benchmarks/generate_covalent_rhino_repeat_graph.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("rhino_repeat_generator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _test_rhino_repeat_graph_has_ten_edges_and_balanced_mapping_modes():
    module = _module()
    graph = yaml.safe_load(
        (
            SCRIPT.parent / "rhino_covalent_repeat_graph.yaml"
        ).read_text()
    )
    edges = [f"{left}--{right}" for left, right in graph["simulated_edges"]]
    methods = [module._mapping(edge)["method"] for edge in edges]

    assert len(edges) == 10
    assert methods.count("mcs_core_smarts") == 5
    assert methods.count("paired_smarts_transmutation") == 2
    assert methods.count("dataset_core") == 3
    assert "I79DJ_733--I79DJ_734" in module.PAIRED_MAPPINGS
    assert "I79DJ_820--I79DJ_866" in module.PAIRED_MAPPINGS


def _test_rhino_workflow_enables_mature_sampling_controls():
    module = _module()
    payload = module._workflow(
        "/dataset.yaml", "I79DJ_733", "I79DJ_734", 0.1, 1234
    )["workflow"]
    neqti = payload["neqti"]

    assert neqti["n_snapshots"] == 100
    assert neqti["schedule_optimization"]["pilot_samples"] == 10
    assert neqti["adaptive_switching"]["candidate_times_ps"] == [100, 300, 1000]
    assert neqti["convergence"]["min_samples_per_direction"] == 50
    assert neqti["convergence"]["stationarity"]["enabled"]
    assert payload["alchemy"]["mapping"]["method"] == (
        "paired_smarts_transmutation"
    )
