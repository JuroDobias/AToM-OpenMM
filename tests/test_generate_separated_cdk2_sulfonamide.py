import importlib.util
from pathlib import Path


GENERATOR = (
    Path(__file__).resolve().parents[1]
    / "examples/RBFE/benchmarks/generate_separated_cdk2_sulfonamide.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_separated_cdk2_sulfonamide", GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _test_resumable_scripts_use_stable_submission_paths():
    generator = _load_generator()

    edge = generator._slurm_script(
        "run-command", "edge", "result.yaml", "source"
    )
    node = generator._bank_node_script("source", "1h1q")
    finalize = generator._bank_finalize_script("source")

    assert 'SCRIPT="$RUN_DIR/run.sh"' in edge
    assert 'SCRIPT="$RUN_DIR/prepare_node_1h1q.sh"' in node
    assert 'sbatch --begin=now+20minutes "$RUN_DIR/finalize_nodes.sh"' in finalize
    assert 'basename "$0"' not in edge
    assert 'basename "$0"' not in node
    assert 'basename "$0"' not in finalize
