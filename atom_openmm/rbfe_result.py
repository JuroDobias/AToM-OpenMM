from __future__ import annotations

import math
import os
from pathlib import Path

import yaml


KCAL_TO_KJ = 4.184


class RBFEResultWriter:
    def __init__(self, *, pair_plan, receptor_file, workflow_yaml, method, requested_samples):
        self.jobname = pair_plan["jobname"]
        self.workdir = Path(pair_plan["jobdir"]).resolve()
        self.method = method
        self.requested_samples = int(requested_samples) if requested_samples is not None else None
        self.path = self.workdir / "result.yaml"
        self.data = {
            "schema_version": 1,
            "tool": "atom_openmm_rbfe",
            "jobname": self.jobname,
            "status": "running",
            "method": method,
            "ligand_a": pair_plan["lig1_name"],
            "ligand_b": pair_plan["lig2_name"],
            "workdir": str(self.workdir),
            "result": self._empty_result(),
            "quality": {
                "convergence_status": "unknown",
                "overlap_score": None,
                "cycle_closure_error": None,
                "warnings": [],
            },
            "error": None,
            "inputs": {
                "receptor": str(Path(receptor_file).resolve()),
                "ligand_a_file": str(Path(pair_plan["lig1_file"]).resolve()),
                "ligand_b_file": str(Path(pair_plan["lig2_file"]).resolve()),
                "workflow_yaml": str(Path(workflow_yaml).resolve()),
                "final_pair_yaml": str((self.workdir / f"{self.jobname}.yaml").resolve()),
            },
            "artifacts": {},
        }

    def _empty_result(self):
        return {
            "ddg_kcal_per_mol": None,
            "ddg_error_kcal_per_mol": None,
            "ddg_kj_per_mol": None,
            "ddg_error_kj_per_mol": None,
            "estimator": "BAR" if self.method == "neqti" else "UWHAM",
            "samples_forward": None,
            "samples_reverse": None,
            "samples_per_replica": None,
            "components": None,
        }

    def _relative_if_exists(self, name):
        path = self.workdir / name
        return name if path.exists() else None

    def _artifacts(self):
        job = self.jobname
        return {
            "prepared_complex": self._relative_if_exists(f"{job}.pdb"),
            "equilibrated_complex": self._relative_if_exists(f"{job}_equil.pdb"),
            "endpoint_a": self._relative_if_exists("neqti_endpoint_A.pdb"),
            "endpoint_b": self._relative_if_exists("neqti_endpoint_B.pdb"),
            "endpoint_a_swapped": self._relative_if_exists("neqti_endpoint_A_swapped.pdb"),
            "endpoint_b_swapped": self._relative_if_exists("neqti_endpoint_B_swapped.pdb"),
            "midpoint_plus": self._relative_if_exists("neqti_midpoint_plus.pdb"),
            "midpoint_minus": self._relative_if_exists("neqti_midpoint_minus.pdb"),
            "leg_a_forward_work_csv": self._relative_if_exists("neqti_leg_a_forward.csv"),
            "leg_a_reverse_work_csv": self._relative_if_exists("neqti_leg_a_reverse.csv"),
            "leg_b_forward_work_csv": self._relative_if_exists("neqti_leg_b_forward.csv"),
            "leg_b_reverse_work_csv": self._relative_if_exists("neqti_leg_b_reverse.csv"),
            "midpoint_bridge_csv": self._relative_if_exists("neqti_midpoint_bridge.csv"),
            "leg_a_forward_integrated_work": self._relative_if_exists("integ_leg_a_forward.dat"),
            "leg_a_reverse_integrated_work": self._relative_if_exists("integ_leg_a_reverse.dat"),
            "leg_b_forward_integrated_work": self._relative_if_exists("integ_leg_b_forward.dat"),
            "leg_b_reverse_integrated_work": self._relative_if_exists("integ_leg_b_reverse.dat"),
            "neqti_summary": self._relative_if_exists("neqti_summary.yaml"),
            "neqti_protocol": self._relative_if_exists("neqti_protocol.yaml"),
            "neqti_switch_validation": self._relative_if_exists("neqti_switch_validation.yaml"),
            "async_re_log": self._relative_if_exists(f"{job}.log"),
            "async_re_replica_output_pattern": f"r*/{job}.out" if any(self.workdir.glob(f"r*/{job}.out")) else None,
            "plot": self._relative_if_exists(f"{job}.png"),
        }

    @staticmethod
    def _finite(value):
        return value is not None and math.isfinite(float(value))

    def _set_analysis(self, analysis):
        result = self._empty_result()
        if self.method == "neqti":
            values = (analysis or {}).get("analysis") or {}
            result["ddg_kcal_per_mol"] = values.get("bar_dg_kcal_per_mol")
            result["ddg_error_kcal_per_mol"] = values.get("bar_bootstrap_std_kcal_per_mol")
            result["samples_forward"] = int((analysis or {}).get("forward_samples", 0))
            result["samples_reverse"] = int((analysis or {}).get("reverse_samples", 0))
            result["components"] = values.get("components")
            self.data["quality"]["overlap_score"] = values.get("overlap_score")
        else:
            result["ddg_kcal_per_mol"] = (analysis or {}).get("ddg")
            result["ddg_error_kcal_per_mol"] = (analysis or {}).get("ddg_std")
            samples = (analysis or {}).get("samples")
            result["samples_per_replica"] = None if samples is None else int(samples)
        if self._finite(result["ddg_kcal_per_mol"]):
            result["ddg_kcal_per_mol"] = float(result["ddg_kcal_per_mol"])
            result["ddg_kj_per_mol"] = result["ddg_kcal_per_mol"] * KCAL_TO_KJ
        if self._finite(result["ddg_error_kcal_per_mol"]):
            result["ddg_error_kcal_per_mol"] = float(result["ddg_error_kcal_per_mol"])
            result["ddg_error_kj_per_mol"] = result["ddg_error_kcal_per_mol"] * KCAL_TO_KJ
        self.data["result"] = result

    def update(self, status, *, analysis=None, error=None, warning=None):
        self.data["status"] = status
        self.data["error"] = error
        if analysis is not None:
            self._set_analysis(analysis)

        result = self.data["result"]
        warnings = []
        if warning:
            warnings.append(str(warning))
        finite_result = self._finite(result["ddg_kcal_per_mol"])
        if finite_result and result["ddg_error_kcal_per_mol"] is None:
            warnings.append("Free-energy uncertainty is unavailable.")
        if status == "partial":
            warnings.append("Sampling or overlap quality requirements were not met.")
        if status == "failed" and error:
            warnings.append(f"{error['stage']} failed: {error['message']}")

        if status == "failed":
            convergence = "failed"
        elif status == "completed" and finite_result:
            convergence = "usable"
        elif status == "partial" and finite_result:
            convergence = "partial"
        else:
            convergence = "unknown"
        self.data["quality"]["convergence_status"] = convergence
        self.data["quality"]["warnings"] = warnings
        self.data["artifacts"] = self._artifacts()
        self._write_atomic()

    def _write_atomic(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".yaml.tmp")
        with open(temporary, "w") as handle:
            yaml.safe_dump(self.data, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
