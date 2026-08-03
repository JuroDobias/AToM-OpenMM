import pytest

from atom_openmm.workflow_schema import WorkflowAxesError, normalize_workflow_axes


def _workflow(chemistry, model, cycle, method):
    return {
        "chemistry": chemistry,
        "alchemy": {"model": model, "cycle": cycle},
        "sampling": {"method": method},
    }


@pytest.mark.parametrize(
    "values",
    [
        ("noncovalent", "atm", "transfer", "async_re"),
        ("noncovalent", "atm", "transfer", "neqti"),
        ("noncovalent", "atm", "transfer", "awh"),
        ("noncovalent", "hybrid_topology", "complex_solvent", "neqti"),
        ("covalent", "hybrid_topology", "complex_solvent", "neqti"),
    ],
)
def _test_supported_workflow_axes(values):
    axes = normalize_workflow_axes(_workflow(*values))
    assert (
        axes.chemistry,
        axes.alchemy_model,
        axes.thermodynamic_cycle,
        axes.sampling_method,
    ) == values


@pytest.mark.parametrize(
    "values",
    [
        ("covalent", "atm", "transfer", "neqti"),
        ("noncovalent", "hybrid_topology", "complex_solvent", "awh"),
        ("noncovalent", "hybrid_topology", "transfer", "neqti"),
    ],
)
def _test_unsupported_workflow_axes_fail_early(values):
    with pytest.raises(WorkflowAxesError, match="unsupported RBFE combination"):
        normalize_workflow_axes(_workflow(*values))


def _test_legacy_dispatch_keys_are_rejected():
    workflow = _workflow("noncovalent", "atm", "transfer", "neqti")
    workflow["production_method"] = "neqti"
    with pytest.raises(WorkflowAxesError, match="legacy workflow dispatch keys"):
        normalize_workflow_axes(workflow)
