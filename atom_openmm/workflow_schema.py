from __future__ import annotations

from dataclasses import dataclass


class WorkflowAxesError(ValueError):
    pass


@dataclass(frozen=True)
class WorkflowAxes:
    chemistry: str
    alchemy_model: str
    thermodynamic_cycle: str
    sampling_method: str


SUPPORTED_COMBINATIONS = {
    ("noncovalent", "atm", "transfer", "async_re"),
    ("noncovalent", "atm", "transfer", "neqti"),
    ("noncovalent", "atm", "transfer", "awh"),
    ("noncovalent", "hybrid_topology", "complex_solvent", "neqti"),
    ("covalent", "hybrid_topology", "complex_solvent", "neqti"),
}


def normalize_workflow_axes(workflow) -> WorkflowAxes:
    if not isinstance(workflow, dict):
        raise WorkflowAxesError("workflow must be a mapping")
    legacy = [key for key in ("mode", "production_method") if key in workflow]
    if legacy:
        raise WorkflowAxesError(
            "legacy workflow dispatch keys are not supported: "
            + ", ".join(f"workflow.{key}" for key in legacy)
            + "; use workflow.chemistry, workflow.alchemy, and workflow.sampling"
        )
    chemistry = workflow.get("chemistry")
    if chemistry not in {"noncovalent", "covalent"}:
        raise WorkflowAxesError(
            "workflow.chemistry must be 'noncovalent' or 'covalent'"
        )
    alchemy = workflow.get("alchemy")
    if not isinstance(alchemy, dict):
        raise WorkflowAxesError("workflow.alchemy must be a mapping")
    model = alchemy.get("model")
    if model not in {"atm", "hybrid_topology"}:
        raise WorkflowAxesError(
            "workflow.alchemy.model must be 'atm' or 'hybrid_topology'"
        )
    cycle = alchemy.get("cycle")
    if cycle not in {"transfer", "complex_solvent"}:
        raise WorkflowAxesError(
            "workflow.alchemy.cycle must be 'transfer' or 'complex_solvent'"
        )
    sampling = workflow.get("sampling")
    if not isinstance(sampling, dict):
        raise WorkflowAxesError("workflow.sampling must be a mapping")
    method = sampling.get("method")
    if method not in {"async_re", "neqti", "awh"}:
        raise WorkflowAxesError(
            "workflow.sampling.method must be 'async_re', 'neqti', or 'awh'"
        )
    axes = WorkflowAxes(chemistry, model, cycle, method)
    combination = (chemistry, model, cycle, method)
    if combination not in SUPPORTED_COMBINATIONS:
        raise WorkflowAxesError(
            "unsupported RBFE combination: chemistry={!r}, alchemy.model={!r}, "
            "alchemy.cycle={!r}, sampling.method={!r}".format(*combination)
        )
    return axes
