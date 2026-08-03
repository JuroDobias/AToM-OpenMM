from __future__ import annotations

import math
import os
from pathlib import Path
from datetime import datetime, timezone

import yaml


KCAL_TO_KJ = 4.184


CONVENTION = {
    "edge_direction": "ligand_a_to_ligand_b",
    "ddg_definition": "G(ligand_b) - G(ligand_a)",
    "positive_value_meaning": "ligand_b binds weaker than ligand_a",
}


class RBFEResultWriter:
    def __init__(
        self,
        *,
        pair_plan,
        receptor_file,
        workflow_yaml,
        method,
        requested_samples,
        chemistry="noncovalent",
        alchemy_model="atm",
        thermodynamic_cycle="transfer",
        pair_index=1,
        total_pairs=1,
    ):
        self.jobname = pair_plan["jobname"]
        self.workdir = Path(pair_plan["jobdir"]).resolve()
        self.method = method
        self.requested_samples = int(requested_samples) if requested_samples is not None else None
        self.pair_index = int(pair_index)
        self.total_pairs = int(total_pairs)
        self.path = self.workdir / "result.yaml"
        self.data = {
            "schema_version": 1,
            "tool": "atom_openmm_rbfe",
            "jobname": self.jobname,
            "status": "running",
            "method": method,
            "chemistry": chemistry,
            "alchemy_model": alchemy_model,
            "thermodynamic_cycle": thermodynamic_cycle,
            "ligand_a": pair_plan["lig1_name"],
            "ligand_b": pair_plan["lig2_name"],
            "workdir": str(self.workdir),
            "convention": CONVENTION.copy(),
            "external_metadata": pair_plan.get("external_metadata") or {},
            "termination_reason": None,
            "result": self._empty_result(),
            "quality": {
                "convergence_status": "unknown",
                "overlap_score": None,
                "cycle_closure_error": None,
                "warnings": [],
                "rest2": None,
                "finite_sample_counts": None,
                "counted_infinite_work_counts": None,
                "convergence": None,
                "schedule_optimization": None,
            },
            "error": None,
            "inputs": {
                "receptor": str(Path(receptor_file).resolve()),
                "ligand_a_file": str(Path(pair_plan["lig1_file"]).resolve()),
                "ligand_b_file": str(Path(pair_plan["lig2_file"]).resolve()),
                "workflow_yaml": str(Path(workflow_yaml).resolve()),
                "final_pair_yaml": str((self.workdir / f"{self.jobname}.yaml").resolve()),
            },
            "artifacts": self._empty_artifacts(),
            "progress": self._progress("setup"),
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
            "estimator_variants": None,
        }

    def _relative_if_exists(self, name):
        path = self.workdir / name
        return name if path.exists() else None

    def _empty_artifacts(self):
        return {
            "prepared_complex": None,
            "equilibrated_complex": None,
            "endpoint_a": None,
            "endpoint_b": None,
            "endpoint_a_swapped": None,
            "endpoint_b_swapped": None,
            "endpoint_a_atm_annealed": None,
            "endpoint_b_atm_annealed": None,
            "endpoint_a_native_state": None,
            "endpoint_b_native_state": None,
            "midpoint": None,
            "midpoint_swapped": None,
            "leg_a_forward_work_csv": None,
            "leg_a_reverse_work_csv": None,
            "leg_b_forward_work_csv": None,
            "leg_b_reverse_work_csv": None,
            "leg_a_forward_integrated_work": None,
            "leg_a_reverse_integrated_work": None,
            "leg_b_forward_integrated_work": None,
            "leg_b_reverse_integrated_work": None,
            "neqti_summary": None,
            "neqti_protocol": None,
            "neqti_switch_validation": None,
            "neqti_rest2": None,
            "neqti_schedule_optimization": None,
            "neqti_convergence": None,
            "async_re_log": None,
            "async_re_replica_output_pattern": None,
            "awh_summary": None,
            "awh_protocol": None,
            "awh_state_trace": None,
            "awh_bias_history": None,
            "awh_reduced_energies": None,
            "awh_checkpoint": None,
            "awh_diagnostics": None,
            "awh_diagnostics_plot": None,
            "awh_trajectory": None,
            "awh_trajectory_topology": None,
            "awh_trajectory_frames": None,
            "awh_friction": None,
            "awh_friction_samples": None,
            "plot": None,
        }

    def _artifacts(self):
        job = self.jobname
        artifacts = self._empty_artifacts()
        artifacts.update(
            {
                "prepared_complex": self._relative_if_exists(f"{job}.pdb"),
                "equilibrated_complex": self._relative_if_exists(f"{job}_equil.pdb"),
                "endpoint_a": self._relative_if_exists("neqti_endpoint_A.pdb"),
                "endpoint_b": self._relative_if_exists("neqti_endpoint_B.pdb"),
                "endpoint_a_swapped": self._relative_if_exists("neqti_endpoint_A_swapped.pdb"),
                "endpoint_b_swapped": self._relative_if_exists("neqti_endpoint_B_swapped.pdb"),
                "endpoint_a_atm_annealed": self._relative_if_exists("neqti_endpoint_A_annealed.pdb"),
                "endpoint_b_atm_annealed": self._relative_if_exists("neqti_endpoint_B_annealed.pdb"),
                "endpoint_a_native_state": self._relative_if_exists("neqti_endpoint_A.xml"),
                "endpoint_b_native_state": self._relative_if_exists("neqti_endpoint_B.xml"),
                "midpoint": self._relative_if_exists("neqti_midpoint.pdb"),
                "midpoint_swapped": self._relative_if_exists("neqti_midpoint_swapped.pdb"),
                "leg_a_forward_work_csv": self._relative_if_exists("neqti_leg_a_forward.csv"),
                "leg_a_reverse_work_csv": self._relative_if_exists("neqti_leg_a_reverse.csv"),
                "leg_b_forward_work_csv": self._relative_if_exists("neqti_leg_b_forward.csv"),
                "leg_b_reverse_work_csv": self._relative_if_exists("neqti_leg_b_reverse.csv"),
                "leg_a_forward_integrated_work": self._relative_if_exists("integ_leg_a_forward.dat"),
                "leg_a_reverse_integrated_work": self._relative_if_exists("integ_leg_a_reverse.dat"),
                "leg_b_forward_integrated_work": self._relative_if_exists("integ_leg_b_forward.dat"),
                "leg_b_reverse_integrated_work": self._relative_if_exists("integ_leg_b_reverse.dat"),
                "neqti_summary": self._relative_if_exists("neqti_summary.yaml"),
                "neqti_protocol": self._relative_if_exists("neqti_protocol.yaml"),
                "neqti_switch_validation": self._relative_if_exists("neqti_switch_validation.yaml"),
                "neqti_rest2": "neqti_rest2" if (self.workdir / "neqti_rest2").is_dir() else None,
                "neqti_schedule_optimization": self._relative_if_exists("neqti_schedule_optimization.yaml"),
                "neqti_convergence": self._relative_if_exists("neqti_convergence.yaml"),
                "async_re_log": self._relative_if_exists(f"{job}.log"),
                "async_re_replica_output_pattern": f"r*/{job}.out" if any(self.workdir.glob(f"r*/{job}.out")) else None,
                "awh_summary": self._relative_if_exists("awh_summary.yaml"),
                "awh_protocol": self._relative_if_exists("awh_protocol.yaml"),
                "awh_state_trace": self._relative_if_exists("awh_state_trace.csv"),
                "awh_bias_history": self._relative_if_exists("awh_bias_history.csv"),
                "awh_reduced_energies": self._relative_if_exists("awh_reduced_energies.csv"),
                "awh_validation_reduced_energies": self._relative_if_exists(
                    "awh_validation_reduced_energies.csv"
                ),
                "awh_checkpoint": self._relative_if_exists("awh_checkpoint.yaml"),
                "awh_diagnostics": self._relative_if_exists("awh_diagnostics.yaml"),
                "awh_diagnostics_plot": self._relative_if_exists("awh_diagnostics.png"),
                "awh_trajectory": self._relative_if_exists("awh_trajectory.xtc"),
                "awh_trajectory_topology": self._relative_if_exists(
                    "awh_trajectory_topology.pdb"
                ),
                "awh_trajectory_frames": self._relative_if_exists(
                    "awh_trajectory_frames.csv"
                ),
                "awh_friction": self._relative_if_exists("awh_friction.yaml"),
                "awh_friction_samples": self._relative_if_exists(
                    "awh_friction_samples.csv"
                ),
                "plot": self._relative_if_exists(f"{job}.png"),
            }
        )
        return artifacts

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
            variants = (analysis or {}).get("work_estimator_analyses")
            if variants:
                result["estimator_variants"] = {
                    name: None if variant is None else {
                        "ddg_kcal_per_mol": variant.get("bar_dg_kcal_per_mol"),
                        "ddg_error_kcal_per_mol": variant.get("bar_bootstrap_std_kcal_per_mol"),
                        "ddg_kj_per_mol": variant.get("bar_dg_kj_per_mol"),
                        "ddg_error_kj_per_mol": variant.get("bar_bootstrap_std_kj_per_mol"),
                        "overlap_score": variant.get("overlap_score"),
                        "difference_from_exact_kcal_per_mol": variant.get(
                            "difference_from_exact_kcal_per_mol", 0.0 if name == "exact" else None
                        ),
                        "paired_bootstrap_difference_std_kcal_per_mol": variant.get(
                            "paired_bootstrap_difference_std_kcal_per_mol"
                        ),
                    }
                    for name, variant in variants.items()
                }
            self.data["quality"]["overlap_score"] = values.get("overlap_score")
            self.data["quality"]["rest2"] = (analysis or {}).get("rest2")
            self.data["quality"]["finite_sample_counts"] = (analysis or {}).get("finite_sample_counts")
            self.data["quality"]["counted_infinite_work_counts"] = (
                (analysis or {}).get("counted_infinite_work_counts")
            )
            self.data["quality"]["convergence"] = (analysis or {}).get("convergence")
            self.data["quality"]["schedule_optimization"] = (
                (analysis or {}).get("schedule_optimization")
            )
            self.data["termination_reason"] = (analysis or {}).get("termination_reason")
        elif self.method == "awh":
            values = (analysis or {}).get("analysis") or {}
            diagnostics = (analysis or {}).get("diagnostics") or {}
            result["ddg_kcal_per_mol"] = values.get("uwham_ddg_kcal_per_mol")
            result["ddg_error_kcal_per_mol"] = values.get(
                "uwham_bootstrap_std_kcal_per_mol"
            )
            result["estimator_variants"] = {
                "awh_bias": {
                    "ddg_kcal_per_mol": values.get("awh_bias_ddg_kcal_per_mol"),
                    "ddg_error_kcal_per_mol": None,
                    "ddg_kj_per_mol": None
                    if values.get("awh_bias_ddg_kcal_per_mol") is None
                    else float(values["awh_bias_ddg_kcal_per_mol"]) * KCAL_TO_KJ,
                    "ddg_error_kj_per_mol": None,
                },
                "fixed_bias_uwham": {
                    "ddg_kcal_per_mol": values.get("uwham_ddg_kcal_per_mol"),
                    "ddg_error_kcal_per_mol": values.get(
                        "uwham_bootstrap_std_kcal_per_mol"
                    ),
                    "ddg_kj_per_mol": None
                    if values.get("uwham_ddg_kcal_per_mol") is None
                    else float(values["uwham_ddg_kcal_per_mol"]) * KCAL_TO_KJ,
                    "ddg_error_kj_per_mol": None
                    if values.get("uwham_bootstrap_std_kcal_per_mol") is None
                    else float(values["uwham_bootstrap_std_kcal_per_mol"]) * KCAL_TO_KJ,
                },
            }
            for name, variant in (
                values.get("uwham_estimators") or {}
            ).items():
                ddg = variant.get("ddg_kcal_per_mol")
                error = variant.get("bootstrap_std_kcal_per_mol")
                result["estimator_variants"][f"fixed_bias_uwham_{name}"] = {
                    "ddg_kcal_per_mol": ddg,
                    "ddg_error_kcal_per_mol": error,
                    "ddg_kj_per_mol": (
                        None if ddg is None else float(ddg) * KCAL_TO_KJ
                    ),
                    "ddg_error_kj_per_mol": (
                        None if error is None else float(error) * KCAL_TO_KJ
                    ),
                    "samples": variant.get("samples"),
                }
            self.data["quality"]["convergence"] = {
                "stage": (analysis or {}).get("stage"),
                "round_trips": (analysis or {}).get("round_trips"),
                "minimum_visits": (analysis or {}).get("minimum_visits"),
                "state_visits": (analysis or {}).get("state_visits"),
                "production": diagnostics.get("production"),
                "endpoint_effective_samples": (
                    diagnostics.get("uwham") or {}
                ).get("endpoint_effective_samples"),
                "bias_stability": diagnostics.get("bias_stability"),
            }
            self.data["quality"]["overlap_score"] = (analysis or {}).get(
                "overlap_score"
            )
            self.data["quality"]["rest2"] = (analysis or {}).get("rest2")
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

    def _progress(self, stage, analysis=None):
        leg_names = ("leg_a_forward", "leg_a_reverse", "leg_b_forward", "leg_b_reverse")
        progress = {
            "stage": stage,
            "current_pair_index": self.pair_index,
            "total_pairs": self.total_pairs,
            "forward_samples": None,
            "reverse_samples": None,
            "target_forward_samples": None,
            "target_reverse_samples": None,
            "sample_counts": None,
            "finite_sample_counts": None,
            "counted_infinite_work_counts": None,
            "target_sample_counts": None,
            "completed_snapshot_cycles": None,
            "target_snapshot_cycles": None,
            "last_update": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if self.method == "neqti":
            target_work_rows = None if self.requested_samples is None else 2 * self.requested_samples
            progress["target_forward_samples"] = target_work_rows
            progress["target_reverse_samples"] = target_work_rows
            progress["target_sample_counts"] = {
                name: self.requested_samples for name in leg_names
            }
            progress["target_snapshot_cycles"] = self.requested_samples
            if analysis:
                progress["forward_samples"] = int((analysis or {}).get("forward_samples", 0))
                progress["reverse_samples"] = int((analysis or {}).get("reverse_samples", 0))
                raw_counts = (analysis or {}).get("sample_counts") or {}
                sample_counts = {
                    name: None if raw_counts.get(name) is None else int(raw_counts.get(name, 0))
                    for name in leg_names
                }
                progress["sample_counts"] = sample_counts
                progress["finite_sample_counts"] = (analysis or {}).get("finite_sample_counts")
                progress["counted_infinite_work_counts"] = (
                    (analysis or {}).get("counted_infinite_work_counts")
                )
                if all(value is not None for value in sample_counts.values()):
                    progress["completed_snapshot_cycles"] = min(sample_counts.values())
        elif self.method == "awh":
            progress.update(
                {
                    "md_steps": None,
                    "current_awh_stage": None,
                    "round_trips": None,
                    "minimum_state_visits": None,
                }
            )
            if analysis:
                progress["md_steps"] = int((analysis or {}).get("total_steps", 0))
                progress["current_awh_stage"] = (analysis or {}).get("stage")
                progress["round_trips"] = int((analysis or {}).get("round_trips", 0))
                progress["minimum_state_visits"] = int(
                    (analysis or {}).get("minimum_visits", 0)
                )
        elif analysis:
            samples = (analysis or {}).get("samples")
            progress["forward_samples"] = None if samples is None else int(samples)
            progress["reverse_samples"] = None
            progress["target_forward_samples"] = self.requested_samples
        return progress

    def update(self, status, *, analysis=None, error=None, warning=None, stage=None):
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
        rest2_quality = self.data["quality"].get("rest2") or {}
        warnings.extend(str(value) for value in rest2_quality.get("warnings", []))
        warnings.extend(str(value) for value in (analysis or {}).get("warnings", []))

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
        self.data["progress"] = self._progress(stage or status, analysis=analysis)
        self._write_atomic()

    def _write_atomic(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".yaml.tmp")
        with open(temporary, "w") as handle:
            yaml.safe_dump(self.data, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
