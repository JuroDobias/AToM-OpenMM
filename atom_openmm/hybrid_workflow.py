from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import yaml
from openmm import unit
from openff.toolkit import Molecule
from openff.units import unit as offunit

from atom_openmm.covalent_hybrid import (
    HybridBondedScales,
    build_hybrid_molecule,
    inactive_branch_metadata,
    inactive_z_matrix_metadata,
    vacuum_nonbonded_pair_counts,
)
from atom_openmm.covalent_softcore import resolve_softcore_path
from atom_openmm.covalent_systems import (
    load_prepared_hybrid_bundle,
    solvate_capped_reference_hybrid,
    write_prepared_hybrid_bundle,
)
from atom_openmm.covalent_workflow import (
    _ensure_equilibration_protocol,
    _normalized_settings,
    _ensure_switch_protocol,
    _iter_environment,
    _platform,
    _read_work,
    _rewrite_work,
    _run_environment,
    _validate_prepared_endpoint_charges,
    _write_yaml_atomic,
)
from atom_openmm.hybrid_mapping import (
    _strict_explicit_pairs,
    _strict_nonnegative_indices,
    build_hybrid_atom_map,
)
from atom_openmm.hybrid_parameters import parameterize_ligand
from atom_openmm.hybrid_systems import create_physical_ligand_environment
from atom_openmm.neqti import (
    _bar_overlap_score,
    _convergence_reached,
    _convergence_record,
    analyze_neqti_work,
    analyze_two_leg_work,
)


class HybridWorkflowError(ValueError):
    pass


PREPARATION_SCHEMA_VERSION = 3


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_settings(workflow):
    settings = dict((workflow.get("alchemy") or {}).get("mapping") or {})
    method = settings.get("method", "mcs")
    if method not in {"mcs", "mcs_core_smarts", "paired_smarts_transmutation", "explicit_pairs"}:
        raise HybridWorkflowError(
            "workflow.alchemy.mapping.method must be 'mcs', 'mcs_core_smarts', "
            "'paired_smarts_transmutation', or 'explicit_pairs'"
        )
    settings["method"] = method
    if method == "mcs_core_smarts" and not settings.get("smarts"):
        raise HybridWorkflowError(
            "mcs_core_smarts requires workflow.alchemy.mapping.smarts"
        )
    if method == "paired_smarts_transmutation":
        if not settings.get("ligand_a_smarts") or not settings.get("ligand_b_smarts"):
            raise HybridWorkflowError(
                "paired_smarts_transmutation requires ligand_a_smarts and ligand_b_smarts"
            )
    if method == "explicit_pairs":
        try:
            pairs = _strict_explicit_pairs(settings.get("pairs_0based"))
            inactive_a = _strict_nonnegative_indices(
                settings.get("inactive_bonded_atoms_a_0based", []),
                "workflow.alchemy.mapping.inactive_bonded_atoms_a_0based",
            )
            inactive_b = _strict_nonnegative_indices(
                settings.get("inactive_bonded_atoms_b_0based", []),
                "workflow.alchemy.mapping.inactive_bonded_atoms_b_0based",
            )
        except ValueError as exc:
            raise HybridWorkflowError(str(exc)) from exc
        settings["pairs_0based"] = [list(pair) for pair in pairs]
        settings["inactive_bonded_atoms_a_0based"] = inactive_a
        settings["inactive_bonded_atoms_b_0based"] = inactive_b
    geometry = str(settings.get("inactive_bonded_geometry", "bond_only")).lower()
    if geometry not in {"bond_only", "terminal_z_matrix"}:
        raise HybridWorkflowError(
            "workflow.alchemy.mapping.inactive_bonded_geometry must be "
            "'bond_only' or 'terminal_z_matrix'"
        )
    if geometry == "terminal_z_matrix" and method not in {"paired_smarts_transmutation", "explicit_pairs"}:
        raise HybridWorkflowError(
            "terminal_z_matrix requires paired_smarts_transmutation or explicit_pairs"
        )
    if (
        method == "explicit_pairs"
        and geometry == "terminal_z_matrix"
        and not (
            settings["inactive_bonded_atoms_a_0based"]
            or settings["inactive_bonded_atoms_b_0based"]
        )
    ):
        raise HybridWorkflowError(
            "explicit_pairs terminal_z_matrix requires at least one inactive bonded atom"
        )
    settings["inactive_bonded_geometry"] = geometry
    if "max_mapped_rmsd_a" in settings:
        settings["max_mapped_rmsd_a"] = float(settings["max_mapped_rmsd_a"])
    return settings


def _formal_charge(path, *, allow_undefined_stereo=False):
    molecule = Molecule.from_file(
        str(path), allow_undefined_stereo=bool(allow_undefined_stereo)
    )
    return int(round(molecule.total_charge.m_as(offunit.elementary_charge)))


def _validate_mapping_sampling(mapping_payload, workflow):
    if not mapping_payload.get("transmuted_pairs_0based"):
        return
    if (workflow.get("sampling") or {}).get("method") != "neqti":
        raise HybridWorkflowError(
            "mapped-atom element transmutations currently support only NEQTI sampling"
        )


def _validate_settings(workflow):
    setup = workflow.get("setup") or {}
    if not isinstance(setup.get("allow_undefined_stereo", False), bool):
        raise HybridWorkflowError(
            "workflow.setup.allow_undefined_stereo must be true or false"
        )
    if not str(setup.get("ligand_forcefield", "espaloma-0.3.2")).startswith("espaloma"):
        raise HybridWorkflowError(
            "noncovalent hybrid topology currently supports Espaloma ligand force fields"
        )
    if setup.get("ligand_charge_model", "nn") != "nn":
        raise HybridWorkflowError(
            "noncovalent hybrid topology currently supports ligand_charge_model: nn"
        )
    config = _normalized_settings(workflow)
    if any(value < 0.0 for value in config["dummy_bonded_scales"].values()):
        raise HybridWorkflowError(
            "workflow.setup.dummy_bonded_scales values cannot be negative"
        )
    if config["dummy_core_nonbonded"] not in {"off", "retain"}:
        raise HybridWorkflowError(
            "workflow.alchemy.dummy_core_nonbonded must be 'off' or 'retain'"
        )
    if config["interpolation"] != "softcore_linear":
        raise HybridWorkflowError(
            "noncovalent hybrid topology requires workflow.neqti.interpolation: softcore_linear"
        )
    if config["softcore"]["stage_interpolation"] not in {
        "linear",
        "smoothstep2",
    }:
        raise HybridWorkflowError(
            "workflow.neqti.softcore.stage_interpolation must be "
            "'linear' or 'smoothstep2'"
        )
    if config["softcore"]["function"] not in {
        "beutler",
        "gapsys",
        "amber_ssc2",
        "effective_distance_ssc2",
    }:
        raise HybridWorkflowError(
            "workflow.neqti.softcore.function must be 'beutler', 'gapsys', "
            "'amber_ssc2', or 'effective_distance_ssc2'"
        )
    if config["softcore"]["coulomb_function"] not in {
        "linear_pme",
        "amber_ssc2",
        "effective_distance_ssc2",
    }:
        raise HybridWorkflowError(
            "workflow.neqti.softcore.coulomb_function must be "
            "'linear_pme', 'amber_ssc2', or 'effective_distance_ssc2'"
        )
    if (
        config["softcore"]["ssc2_alpha_lj"] <= 0.0
        or config["softcore"]["ssc2_alpha_coul"] <= 0.0
        or config["softcore"]["ssc2_beta_coul"] <= 0.0
        or config["softcore"]["ssc2_switch_width_nm"] <= 0.0
    ):
        raise HybridWorkflowError("Amber SSC(2) LJ parameters must be positive")
    if config["softcore"]["coulomb_function"] in {
        "amber_ssc2",
        "effective_distance_ssc2",
    }:
        if config["softcore"]["function"] != config["softcore"]["coulomb_function"]:
            raise HybridWorkflowError(
                "SSC(2) Coulomb function must match softcore.function"
            )
        if config["softcore"]["stage_interpolation"] != "linear":
            raise HybridWorkflowError(
                "Amber SSC(2) Coulomb requires stage_interpolation: linear"
            )
        if config["softcore"].get("path_mode") != "concerted":
            raise HybridWorkflowError(
                "Amber SSC(2) Coulomb requires softcore.path.mode: concerted"
            )
    if config["failed_switch_policy"] not in {"abort", "count_as_infinite"}:
        raise HybridWorkflowError(
            "workflow.neqti.failed_switch_policy must be 'abort' or 'count_as_infinite'"
        )
    try:
        resolved = resolve_softcore_path(**{
            key: value
            for key, value in config["softcore"].items()
            if key in {
                "charge_steps_per_stage", "sterics_steps", "subdivisions_per_stage",
                "total_steps", "path_nodes", "vdw_a", "charge_a",
                "segments_per_interval", "path_mode",
            }
        })
    except Exception as exc:
        raise HybridWorkflowError(str(exc)) from exc
    if config["n_snapshots"] < 1 or config["switch_steps"] < 1:
        raise HybridWorkflowError("NEQTI snapshots and switching steps must be positive")
    if config["schedule_optimization"]["enabled"]:
        segments = sum(resolved["segments_per_interval"])
        optimization = config["schedule_optimization"]
        if optimization["min_segment_steps"] * segments > config["switch_steps"]:
            raise HybridWorkflowError("schedule optimizer minimum segment allocation is too large")
        if optimization["max_segment_steps"] * segments < config["switch_steps"]:
            raise HybridWorkflowError("schedule optimizer maximum segment allocation is too small")
    adaptive = config["adaptive_switching"]
    if adaptive["enabled"]:
        candidates = adaptive["candidate_times_ps"]
        candidate_steps = adaptive["candidate_total_steps"]
        if not candidates or any(value <= 0 for value in candidates):
            raise HybridWorkflowError(
                "adaptive_switching.candidate_times_ps must contain positive values"
            )
        if any(right <= left for left, right in zip(candidates, candidates[1:])):
            raise HybridWorkflowError(
                "adaptive_switching.candidate_times_ps must be strictly increasing"
            )
        if candidate_steps[0] != config["switch_steps"]:
            base_ps = config["switch_steps"] * config["timestep_fs"] / 1000.0
            raise HybridWorkflowError(
                "the first adaptive switching candidate must match the base softcore "
                f"protocol duration ({base_ps:g} ps)"
            )
        for time_ps, steps in zip(candidates, candidate_steps):
            represented = steps * config["timestep_fs"] / 1000.0
            if not np.isclose(represented, time_ps, atol=1.0e-9, rtol=0.0):
                raise HybridWorkflowError(
                    f"adaptive switching time {time_ps:g} ps is not divisible by the "
                    f"{config['timestep_fs']:g} fs timestep"
                )
        if adaptive["pilot_samples_per_direction"] < 2:
            raise HybridWorkflowError(
                "adaptive_switching.pilot_samples_per_direction must be at least 2"
            )
        if adaptive["pilot_samples_per_direction"] > config["n_snapshots"]:
            raise HybridWorkflowError(
                "adaptive switching pilot samples cannot exceed n_snapshots"
            )
        if adaptive["min_overlap_score_per_leg"] <= 0:
            raise HybridWorkflowError(
                "adaptive switching overlap threshold must be positive"
            )
        failed_fraction = adaptive["max_failed_fraction_per_direction"]
        if not 0 <= failed_fraction < 1:
            raise HybridWorkflowError(
                "adaptive switching failed fraction must be in [0, 1)"
            )
        if adaptive["on_exhausted"] != "use_longest":
            raise HybridWorkflowError(
                "adaptive_switching.on_exhausted currently supports only use_longest"
            )
        if not adaptive["reuse_selected_pilot_samples"]:
            raise HybridWorkflowError(
                "hybrid adaptive switching currently requires "
                "reuse_selected_pilot_samples: true"
            )
    convergence = config["convergence"]
    if convergence["enabled"]:
        if convergence["min_samples_per_direction"] < 2:
            raise HybridWorkflowError(
                "convergence.min_samples_per_direction must be at least 2"
            )
        if convergence["min_samples_per_direction"] > config["n_snapshots"]:
            raise HybridWorkflowError(
                "convergence.min_samples_per_direction cannot exceed n_snapshots"
            )
        if convergence["min_overlap_score_per_leg"] <= 0:
            raise HybridWorkflowError("convergence overlap threshold must be positive")
        if convergence["max_dg_error_kcal_per_mol"] <= 0:
            raise HybridWorkflowError("convergence uncertainty threshold must be positive")
        if convergence["consecutive_checks"] < 1:
            raise HybridWorkflowError("convergence.consecutive_checks must be positive")
        if convergence["max_dg_range_kcal_per_mol"] < 0:
            raise HybridWorkflowError("convergence DG range must be non-negative")
        if convergence["check_interval_samples"] < 1:
            raise HybridWorkflowError(
                "convergence.check_interval_samples must be positive"
            )
        stationarity = convergence["stationarity"]
        if stationarity["enabled"]:
            if not 0 < stationarity["discard_fraction"] < 0.5:
                raise HybridWorkflowError(
                    "convergence.stationarity.discard_fraction must be in (0, 0.5)"
                )
            if stationarity["min_discard_samples"] < 1:
                raise HybridWorkflowError(
                    "convergence.stationarity.min_discard_samples must be positive"
                )
            if (
                stationarity["max_discard_first_shift_kcal_per_mol"] < 0
                or stationarity["max_discard_last_shift_kcal_per_mol"] < 0
            ):
                raise HybridWorkflowError(
                    "convergence stationarity shift thresholds must be non-negative"
                )
    return config


def validate_noncovalent_hybrid_workflow(path):
    from atom_openmm.rbfe_workflow import (
        build_small_molecule_plan,
        load_workflow_config,
        normalize_workflow_axes,
    )

    config = load_workflow_config(path)
    plan = build_small_molecule_plan(config)
    allow_undefined_stereo = bool(
        (config["workflow"].get("setup") or {}).get(
            "allow_undefined_stereo", False
        )
    )
    mapping = _mapping_settings(config["workflow"])
    if mapping["method"] == "explicit_pairs" and len(plan["pairs"]) != 1:
        raise HybridWorkflowError(
            "explicit_pairs mapping requires a workflow containing exactly one edge"
        )
    if (
        mapping["method"] == "paired_smarts_transmutation"
        and normalize_workflow_axes(config["workflow"]).sampling_method != "neqti"
    ):
        raise HybridWorkflowError(
            "paired_smarts_transmutation currently supports only NEQTI sampling"
        )
    _validate_settings(config["workflow"])
    for pair in plan["pairs"]:
        charge_a = _formal_charge(
            pair["lig1_file"], allow_undefined_stereo=allow_undefined_stereo
        )
        charge_b = _formal_charge(
            pair["lig2_file"], allow_undefined_stereo=allow_undefined_stereo
        )
        if charge_a != charge_b:
            raise HybridWorkflowError(
                f"hybrid topology requires equal endpoint formal charges in this release: "
                f"{pair['lig1_name']}={charge_a}, {pair['lig2_name']}={charge_b}"
            )
    return True


def plan_noncovalent_hybrid_workflow(path):
    from atom_openmm.rbfe_workflow import build_small_molecule_plan, load_workflow_config

    validate_noncovalent_hybrid_workflow(path)
    config = load_workflow_config(path)
    plan = build_small_molecule_plan(config)
    return {
        "schema_version": 1,
        "chemistry": "noncovalent",
        "alchemy_model": "hybrid_topology",
        "thermodynamic_cycle": "complex_solvent",
        "sampling_method": "neqti",
        "dummy_core_nonbonded": _normalized_settings(config["workflow"])[
            "dummy_core_nonbonded"
        ],
        "receptor": str(plan["receptor_file"]),
        "workdir": str(plan["workdir"]),
        "mapping": _mapping_settings(config["workflow"]),
        "pairs": [
            {
                "ligand_a": pair["lig1_name"],
                "ligand_b": pair["lig2_name"],
                "workdir": str(pair["jobdir"]),
                "environments": ["complex", "solvent"],
            }
            for pair in plan["pairs"]
        ],
    }


def _preparation_fingerprint(pair, receptor, workflow, mapping):
    files = {
        "receptor": receptor,
        "ligand_a": pair["lig1_file"],
        "ligand_b": pair["lig2_file"],
    }
    payload = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "files": {
            name: {"path": str(Path(value).resolve()), "sha256": _sha256(value)}
            for name, value in files.items()
        },
        "setup": workflow.get("setup") or {},
        "mapping": mapping,
    }
    digest = hashlib.sha256(yaml.safe_dump(payload, sort_keys=True).encode()).hexdigest()
    return digest, payload


def _write_bundle(workdir, complex_system, solvent_system, manifest):
    target = workdir / "prepared"
    temporary = workdir / f".prepared.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    manifest = {
        **manifest,
        "environments": {
            "complex": write_prepared_hybrid_bundle(complex_system, temporary, "complex"),
            "solvent": write_prepared_hybrid_bundle(solvent_system, temporary, "solvent"),
        },
    }
    _write_yaml_atomic(temporary / "manifest.yaml", manifest)
    if target.exists():
        raise HybridWorkflowError(f"prepared bundle already exists: {target}")
    os.replace(temporary, target)
    return manifest


def _load_bundle(workdir, fingerprint):
    directory = workdir / "prepared"
    manifest_path = directory / "manifest.yaml"
    if not manifest_path.is_file():
        return None
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    if manifest.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        raise HybridWorkflowError("prepared bundle schema differs; use a new workdir")
    if manifest.get("fingerprint") != fingerprint:
        raise HybridWorkflowError("prepared bundle inputs differ; use a new workdir")
    return (
        load_prepared_hybrid_bundle(directory, manifest["environments"]["complex"]),
        load_prepared_hybrid_bundle(directory, manifest["environments"]["solvent"]),
        manifest,
    )


def _runtime_artifacts_exist(workdir):
    return any(
        any(Path(workdir).glob(pattern))
        for pattern in (
            "*_endpoint_*_state*.xml",
            "*_forward.csv",
            "*_reverse.csv",
            "*_rest2_*",
            "neqti_adaptive_switching*",
            "neqti_convergence.yaml",
        )
    )


def _prepare_pair(pair, receptor, workflow, workdir):
    mapping_settings = _mapping_settings(workflow)
    fingerprint, fingerprint_inputs = _preparation_fingerprint(
        pair, receptor, workflow, mapping_settings
    )
    loaded = _load_bundle(workdir, fingerprint)
    if loaded is not None:
        return (*loaded[:2], loaded[2], fingerprint)
    if _runtime_artifacts_exist(workdir):
        raise HybridWorkflowError(
            "resume_incompatible: runtime artifacts exist without the current prepared bundle; "
            "use a new workdir"
        )
    setup = workflow.get("setup") or {}
    ligand_forcefield = setup.get("ligand_forcefield", "espaloma-0.3.2")
    charge_model = setup.get("ligand_charge_model", "nn")
    allow_undefined_stereo = bool(setup.get("allow_undefined_stereo", False))
    parameters_a = parameterize_ligand(
        pair["lig1_file"], ligand_forcefield=ligand_forcefield,
        ligand_charge_model=charge_model,
        allow_undefined_stereo=allow_undefined_stereo,
    )
    parameters_b = parameterize_ligand(
        pair["lig2_file"], ligand_forcefield=ligand_forcefield,
        ligand_charge_model=charge_model,
        allow_undefined_stereo=allow_undefined_stereo,
    )
    if not np.isclose(parameters_a.charges_e.sum(), parameters_b.charges_e.sum(), atol=1e-6):
        raise HybridWorkflowError("parameterized endpoint ligand charges differ")
    atom_map, mapping_payload = build_hybrid_atom_map(
        parameters_a, parameters_b, mapping_settings
    )
    _validate_mapping_sampling(mapping_payload, workflow)
    mapping_payload["transmutations"] = [
        {
            "ligand_a_atom_0based": int(atom_a),
            "ligand_b_atom_0based": int(atom_b),
            "ligand_a_element": parameters_a.molecule.atoms[atom_a].symbol,
            "ligand_b_element": parameters_b.molecule.atoms[atom_b].symbol,
            "ligand_a_mass_da": float(
                parameters_a.system.getParticleMass(atom_a).value_in_unit(
                    unit.dalton
                )
            ),
            "ligand_b_mass_da": float(
                parameters_b.system.getParticleMass(atom_b).value_in_unit(
                    unit.dalton
                )
            ),
            "switching_mass_da": max(
                parameters_a.system.getParticleMass(atom_a).value_in_unit(unit.dalton),
                parameters_b.system.getParticleMass(atom_b).value_in_unit(unit.dalton),
            ),
        }
        for atom_a, atom_b in (
            tuple(pair) for pair in mapping_payload["transmuted_pairs_0based"]
        )
    ]
    hybrid = build_hybrid_molecule(
        parameters_a,
        parameters_b,
        atom_map=atom_map,
        dummy_bonded_scales=HybridBondedScales(
            **_normalized_settings(workflow)["dummy_bonded_scales"]
        ),
        dummy_core_nonbonded=_normalized_settings(workflow)[
            "dummy_core_nonbonded"
        ],
        transmuted_pairs={
            tuple(pair) for pair in mapping_payload["transmuted_pairs_0based"]
        },
        inactive_bonded_atoms_a=set(
            mapping_payload["inactive_bonded_atoms_a_0based"]
        ),
        inactive_bonded_atoms_b=set(
            mapping_payload["inactive_bonded_atoms_b_0based"]
        ),
        inactive_bonded_geometry=mapping_payload["inactive_bonded_geometry"],
    )
    mapping_payload["inactive_z_matrix_terms"] = inactive_z_matrix_metadata(hybrid)
    mapping_payload["inactive_bonded_branches"] = inactive_branch_metadata(hybrid)
    mapping_payload["resolved_inactive_bonded_atoms_a_0based"] = list(
        hybrid.inactive_bonded_atoms_a
    )
    mapping_payload["resolved_inactive_bonded_atoms_b_0based"] = list(
        hybrid.inactive_bonded_atoms_b
    )
    seed = int(setup.get("solvation_seed", _normalized_settings(workflow)["random_seed"]))
    physical_complex = create_physical_ligand_environment(
        parameters_a, receptor=receptor, setup=setup, solvation_seed=seed
    )
    physical_solvent = create_physical_ligand_environment(
        parameters_a, receptor=None, setup=setup, solvation_seed=seed + 1
    )
    complex_system = solvate_capped_reference_hybrid(hybrid, physical_complex)
    solvent_system = solvate_capped_reference_hybrid(hybrid, physical_solvent)
    selection_metadata = {
        "ligand_a": {
            "structure_file": str(Path(pair["lig1_file"]).resolve()),
            "system_atom_indices": [
                int(hybrid.map_a_to_hybrid[index])
                for index in range(parameters_a.molecule.n_atoms)
            ],
        },
        "ligand_b": {
            "structure_file": str(Path(pair["lig2_file"]).resolve()),
            "system_atom_indices": [
                int(hybrid.map_b_to_hybrid[index])
                for index in range(parameters_b.molecule.n_atoms)
            ],
        },
    }
    endpoint_elements = {
        "a": {
            int(hybrid.map_a_to_hybrid[index]): int(atom.atomic_number)
            for index, atom in enumerate(parameters_a.molecule.atoms)
        },
        "b": {
            int(hybrid.map_b_to_hybrid[index]): int(atom.atomic_number)
            for index, atom in enumerate(parameters_b.molecule.atoms)
        },
    }
    complex_system.provenance["SELECTION_METADATA"] = selection_metadata
    solvent_system.provenance["SELECTION_METADATA"] = selection_metadata
    complex_system.provenance["ENDPOINT_ELEMENTS"] = endpoint_elements
    solvent_system.provenance["ENDPOINT_ELEMENTS"] = endpoint_elements
    _validate_prepared_endpoint_charges(complex_system, "complex")
    _validate_prepared_endpoint_charges(solvent_system, "solvent")
    manifest = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_inputs,
        "mapping": mapping_payload,
        "dummy_core_nonbonded": hybrid.dummy_core_nonbonded,
        "inactive_bonded_geometry": hybrid.inactive_bonded_geometry,
        "vacuum_nonbonded_pair_counts": vacuum_nonbonded_pair_counts(hybrid),
        "parameterization": {
            "ligand_a": parameters_a.provenance,
            "ligand_b": parameters_b.provenance,
        },
        "mass_policy": {
            "endpoint_equilibration": "physical",
            "switching": "heavier_endpoint",
            "switch_velocities": "maxwell_boltzmann_resampled",
        },
    }
    manifest = _write_bundle(workdir, complex_system, solvent_system, manifest)
    _write_yaml_atomic(workdir / "hybrid_mapping.yaml", mapping_payload)
    return complex_system, solvent_system, manifest, fingerprint


def _analysis_payload(work, config):
    return analyze_two_leg_work(
        work, config["temperature_k"], config["bootstrap_samples"], config["random_seed"]
    )


def _convergence_state(workdir, config):
    path = workdir / "neqti_convergence.yaml"
    if path.exists():
        state = yaml.safe_load(path.read_text()) or {}
        settings = state.get("settings") or {}
        expected = config["convergence"]
        legacy_expected = {
            **expected,
            "max_ddg_error_kcal_per_mol": expected["max_dg_error_kcal_per_mol"],
            "max_ddg_range_kcal_per_mol": expected["max_dg_range_kcal_per_mol"],
        }
        legacy_expected.pop("max_dg_error_kcal_per_mol")
        legacy_expected.pop("max_dg_range_kcal_per_mol")
        if settings != expected and settings != legacy_expected:
            if not expected.get("reopen_on_settings_change", False):
                raise HybridWorkflowError(
                    "existing hybrid convergence state uses different settings"
                )
            return {
                "schema_version": 2,
                "settings": expected,
                "environments": {
                    environment: {"history": [], "termination_reason": None}
                    for environment in ("complex", "solvent")
                },
                "combined_history": [],
                "termination_reason": None,
                "reopened_from": {
                    "settings": settings,
                    "termination_reason": state.get("termination_reason"),
                    "environment_termination_reasons": {
                        name: payload.get("termination_reason")
                        for name, payload in (state.get("environments") or {}).items()
                    },
                },
            }
        return state
    return {
        "schema_version": 2,
        "settings": config["convergence"],
        "environments": {
            environment: {"history": [], "termination_reason": None}
            for environment in ("complex", "solvent")
        },
        "combined_history": [],
        "termination_reason": None,
    }


def _environment_convergence_record(forward, reverse, config, environment):
    analysis = analyze_neqti_work(
        forward,
        reverse,
        config["temperature_k"],
        config["bootstrap_samples"],
        config["random_seed"] + (0 if environment == "complex" else 100000),
    )
    dg = None if analysis is None else analysis["bar_dg_kcal_per_mol"]
    error = (
        None if analysis is None
        else analysis["bar_bootstrap_std_kcal_per_mol"]
    )
    overlap = _bar_overlap_score(
        forward, reverse, dg, config["temperature_k"]
    )
    settings = config["convergence"]
    sample_count = min(len(forward), len(reverse))
    stationarity_settings = settings["stationarity"]
    stationarity = {
        "enabled": stationarity_settings["enabled"],
        "discard_fraction": stationarity_settings["discard_fraction"],
        "discard_samples": 0,
        "discard_first_dg_kcal_per_mol": None,
        "discard_last_dg_kcal_per_mol": None,
        "discard_first_shift_kcal_per_mol": None,
        "discard_last_shift_kcal_per_mol": None,
        "passed": not stationarity_settings["enabled"],
    }
    if stationarity_settings["enabled"] and dg is not None:
        discard = int(sample_count * stationarity_settings["discard_fraction"])
        stationarity["discard_samples"] = discard
        if discard >= stationarity_settings["min_discard_samples"]:
            first_analysis = analyze_neqti_work(
                forward[discard:],
                reverse[discard:],
                config["temperature_k"],
                0,
                config["random_seed"],
            )
            last_analysis = analyze_neqti_work(
                forward[:-discard],
                reverse[:-discard],
                config["temperature_k"],
                0,
                config["random_seed"],
            )
            if first_analysis is not None and last_analysis is not None:
                first_dg = first_analysis["bar_dg_kcal_per_mol"]
                last_dg = last_analysis["bar_dg_kcal_per_mol"]
                first_shift = abs(first_dg - dg)
                last_shift = abs(last_dg - dg)
                stationarity.update(
                    {
                        "discard_first_dg_kcal_per_mol": first_dg,
                        "discard_last_dg_kcal_per_mol": last_dg,
                        "discard_first_shift_kcal_per_mol": first_shift,
                        "discard_last_shift_kcal_per_mol": last_shift,
                        "passed": bool(
                            first_shift
                            <= stationarity_settings[
                                "max_discard_first_shift_kcal_per_mol"
                            ]
                            and last_shift
                            <= stationarity_settings[
                                "max_discard_last_shift_kcal_per_mol"
                            ]
                        ),
                    }
                )
    return {
        "sample_count_per_direction": sample_count,
        "dg_kcal_per_mol": dg,
        "dg_error_kcal_per_mol": error,
        "overlap_score": overlap,
        "stationarity": stationarity,
        "thresholds_pass": bool(
            sample_count >= settings["min_samples_per_direction"]
            and overlap is not None
            and overlap >= settings["min_overlap_score_per_leg"]
            and error is not None
            and error <= settings["max_dg_error_kcal_per_mol"]
            and stationarity["passed"]
        ),
    }


def _environment_convergence_reached(history, settings):
    required = settings["consecutive_checks"]
    if len(history) < required:
        return False
    recent = history[-required:]
    if not all(record["thresholds_pass"] for record in recent):
        return False
    estimates = [record["dg_kcal_per_mol"] for record in recent]
    return max(estimates) - min(estimates) <= settings["max_dg_range_kcal_per_mol"]


def _append_combined_convergence_record(state, workdir, config):
    work = {
        "leg_a_forward": _read_work(workdir / "complex_forward.csv"),
        "leg_a_reverse": _read_work(workdir / "complex_reverse.csv"),
        "leg_b_forward": _read_work(workdir / "solvent_forward.csv"),
        "leg_b_reverse": _read_work(workdir / "solvent_reverse.csv"),
    }
    if any(not values for values in work.values()):
        return
    counts = {
        "complex": min(len(work["leg_a_forward"]), len(work["leg_a_reverse"])),
        "solvent": min(len(work["leg_b_forward"]), len(work["leg_b_reverse"])),
    }
    previous = state.get("combined_history") or []
    if previous and previous[-1].get("sample_counts") == counts:
        return
    analysis = analyze_two_leg_work(
        work,
        config["temperature_k"],
        bootstrap_samples=0,
        random_seed=config["random_seed"],
    )
    state.setdefault("combined_history", []).append({
        "sample_counts": counts,
        "ddg_kcal_per_mol": (
            None if analysis is None else analysis["bar_dg_kcal_per_mol"]
        ),
        "ddg_error_kcal_per_mol": (
            None if analysis is None
            else analysis["bar_bootstrap_std_kcal_per_mol"]
        ),
    })


def _legacy_hybrid_convergence_callback(workdir, config, state):
    path = workdir / "neqti_convergence.yaml"
    legacy_settings = state["settings"]

    def callback(environment, _sample, _forward, _reverse):
        if environment != "complex":
            return False
        if state.get("termination_reason") == "converged":
            return True
        work = {
            "leg_a_forward": _read_work(workdir / "complex_forward.csv"),
            "leg_a_reverse": _read_work(workdir / "complex_reverse.csv"),
            "leg_b_forward": _read_work(workdir / "solvent_forward.csv"),
            "leg_b_reverse": _read_work(workdir / "solvent_reverse.csv"),
        }
        available = min(len(values) for values in work.values())
        previous = max(
            (int(record["sample_count_per_direction"]) for record in state.get("history", [])),
            default=0,
        )
        first = max(previous + 1, legacy_settings["min_samples_per_direction"])
        for sample_count in range(first, available + 1):
            prefix = {name: values[:sample_count] for name, values in work.items()}
            analysis = _analysis_payload(prefix, config)
            record = _convergence_record(
                analysis, {name: sample_count for name in prefix}, legacy_settings
            )
            state["history"].append(record)
            reached = _convergence_reached(state["history"], legacy_settings)
            if reached:
                state["termination_reason"] = "converged"
            _write_yaml_atomic(path, state)
            if reached:
                for name, values in work.items():
                    environment_name, direction = name.removeprefix("leg_").split("_", 1)
                    environment_name = {"a": "complex", "b": "solvent"}[environment_name]
                    _rewrite_work(
                        workdir / f"{environment_name}_{direction}.csv",
                        values[:sample_count],
                    )
                return True
        return False

    return callback


def _hybrid_convergence_callback(workdir, config):
    path = workdir / "neqti_convergence.yaml"
    state = _convergence_state(workdir, config)
    settings = config["convergence"]

    if state.get("reopened_from") is not None:
        _write_yaml_atomic(path, state)

    if int(state.get("schema_version", 1)) < 2:
        return _legacy_hybrid_convergence_callback(workdir, config, state), state

    def callback(environment, _sample, _forward, _reverse):
        environment_state = state["environments"][environment]
        if environment_state.get("termination_reason") is not None:
            return True
        forward = _read_work(workdir / f"{environment}_forward.csv")
        reverse = _read_work(workdir / f"{environment}_reverse.csv")
        available = min(len(forward), len(reverse))
        previous = max(
            (
                int(record["sample_count_per_direction"])
                for record in environment_state.get("history", [])
            ),
            default=0,
        )
        first = max(previous + 1, settings["min_samples_per_direction"])
        if (
            state.get("reopened_from") is not None
            and not environment_state.get("history")
            and available >= settings["min_samples_per_direction"]
        ):
            interval = settings["check_interval_samples"]
            latest = settings["min_samples_per_direction"] + (
                (available - settings["min_samples_per_direction"]) // interval
            ) * interval
            first = max(
                settings["min_samples_per_direction"],
                latest - (settings["consecutive_checks"] - 1) * interval,
            )
        for sample_count in range(first, available + 1):
            if (
                sample_count - settings["min_samples_per_direction"]
            ) % settings["check_interval_samples"]:
                continue
            record = _environment_convergence_record(
                forward[:sample_count], reverse[:sample_count], config, environment
            )
            environment_state["history"].append(record)
            reached = _environment_convergence_reached(
                environment_state["history"], settings
            )
            if reached:
                environment_state["termination_reason"] = "converged"
            _append_combined_convergence_record(state, workdir, config)
            reasons = [
                item.get("termination_reason")
                for item in state["environments"].values()
            ]
            if all(reason is not None for reason in reasons):
                state["termination_reason"] = (
                    "converged" if set(reasons) == {"converged"} else "max_samples"
                )
            _write_yaml_atomic(path, state)
            if reached:
                return True
        return False

    return callback, state


def _normalize_converged_work_prefix(workdir, convergence_state):
    if int(convergence_state.get("schema_version", 1)) >= 2:
        return
    if convergence_state.get("termination_reason") != "converged":
        return
    history = convergence_state.get("history") or []
    if not history:
        raise HybridWorkflowError(
            "converged hybrid state is missing convergence history"
        )
    sample_count = int(history[-1]["sample_count_per_direction"])
    for environment in ("complex", "solvent"):
        for direction in ("forward", "reverse"):
            path = workdir / f"{environment}_{direction}.csv"
            values = _read_work(path)
            if len(values) < sample_count:
                raise HybridWorkflowError(
                    f"converged work is incomplete: {path} contains {len(values)} "
                    f"of {sample_count} required samples"
                )
            if len(values) != sample_count:
                _rewrite_work(path, values[:sample_count])


def _run_independent_environment_iterators(iterators, convergence_callback):
    completed = {
        environment: bool(convergence_callback(environment, 0, None, None))
        for environment in iterators
    }
    summaries = {environment: None for environment in iterators}

    def advance(environment):
        try:
            latest = next(iterators[environment])
        except StopIteration as stopped:
            completed[environment] = True
            latest = stopped.value
        if latest is not None:
            summaries[environment] = latest[2]
        return convergence_callback(environment, 0, None, None)

    try:
        while not all(completed.values()):
            for environment in iterators:
                if not completed[environment] and advance(environment):
                    completed[environment] = True
    finally:
        for iterator in iterators.values():
            iterator.close()
    return summaries


def _finalize_independent_convergence(workdir, config, state):
    if int(state.get("schema_version", 1)) < 2:
        return state
    for environment, environment_state in state["environments"].items():
        if environment_state.get("termination_reason") is not None:
            continue
        count = min(
            len(_read_work(workdir / f"{environment}_forward.csv")),
            len(_read_work(workdir / f"{environment}_reverse.csv")),
        )
        if count >= config["n_snapshots"]:
            environment_state["termination_reason"] = "max_samples"
    reasons = [
        item.get("termination_reason") for item in state["environments"].values()
    ]
    if all(reason is not None for reason in reasons):
        state["termination_reason"] = (
            "converged" if set(reasons) == {"converged"} else "max_samples"
        )
    _append_combined_convergence_record(state, workdir, config)
    _write_yaml_atomic(workdir / "neqti_convergence.yaml", state)
    return state


def _result(
    pair,
    workflow_path,
    receptor,
    workdir,
    analysis,
    manifest,
    config,
    status,
    rest2=None,
):
    components = None if analysis is None else {
        "complex": analysis["components"]["leg_a"],
        "solvent": analysis["components"]["leg_b"],
    }
    counts = {
        "leg_a_forward": len(_read_work(workdir / "complex_forward.csv")),
        "leg_a_reverse": len(_read_work(workdir / "complex_reverse.csv")),
        "leg_b_forward": len(_read_work(workdir / "solvent_forward.csv")),
        "leg_b_reverse": len(_read_work(workdir / "solvent_reverse.csv")),
    }
    work_values = {
        "leg_a_forward": _read_work(workdir / "complex_forward.csv"),
        "leg_a_reverse": _read_work(workdir / "complex_reverse.csv"),
        "leg_b_forward": _read_work(workdir / "solvent_forward.csv"),
        "leg_b_reverse": _read_work(workdir / "solvent_reverse.csv"),
    }
    finite_counts = {
        name: int(np.count_nonzero(np.isfinite(values)))
        for name, values in work_values.items()
    }
    infinite_counts = {
        name: counts[name] - finite_counts[name] for name in counts
    }
    optimizer_path = workdir / "covalent_schedule_optimization.yaml"
    optimizer = (
        yaml.safe_load(optimizer_path.read_text()) if optimizer_path.exists() else None
    )
    adaptive_path = workdir / "neqti_adaptive_switching.yaml"
    adaptive = yaml.safe_load(adaptive_path.read_text()) if adaptive_path.exists() else None
    convergence_path = workdir / "neqti_convergence.yaml"
    convergence = (
        yaml.safe_load(convergence_path.read_text())
        if convergence_path.exists()
        else None
    )
    warnings = []
    if adaptive is not None:
        if any(
            (item.get("selected") or {}).get("selection_reason")
            == "candidate_list_exhausted"
            for item in adaptive.get("environments", {}).values()
        ):
            warnings.append(
                "At least one hybrid NEQTI environment exhausted all adaptive "
                "switching candidates; production used the longest duration."
            )
        if any(
            (item.get("selected") or {}).get("pilot_samples_reused")
            for item in adaptive.get("environments", {}).values()
        ):
            warnings.append(
                "Selected adaptive pilot work was reused in the production BAR estimate."
            )
    convergence_status = (
        "usable"
        if config["convergence"]["enabled"]
        and convergence is not None
        and convergence.get("termination_reason") == "converged"
        else (
            "partial"
            if config["convergence"]["enabled"]
            else ("usable" if analysis and analysis["overlap_score"] >= 0.01 else "partial")
        )
    )
    return {
        "schema_version": 1,
        "tool": "atom_openmm_rbfe",
        "jobname": pair["jobname"],
        "status": status,
        "method": "neqti",
        "chemistry": "noncovalent",
        "alchemy_model": "hybrid_topology",
        "thermodynamic_cycle": "complex_solvent",
        "dummy_core_nonbonded": config.get("dummy_core_nonbonded", "off"),
        "ligand_a": pair["lig1_name"],
        "ligand_b": pair["lig2_name"],
        "workdir": str(workdir.resolve()),
        "external_metadata": pair.get("external_metadata") or {},
        "termination_reason": (
            None if convergence is None else convergence.get("termination_reason")
        ),
        "convention": {
            "edge_direction": "ligand_a_to_ligand_b",
            "ddg_definition": "G(ligand_b) - G(ligand_a)",
            "positive_value_meaning": "ligand_b binds weaker than ligand_a",
        },
        "result": {
            "ddg_kcal_per_mol": None if analysis is None else analysis["bar_dg_kcal_per_mol"],
            "ddg_error_kcal_per_mol": None if analysis is None else analysis["bar_bootstrap_std_kcal_per_mol"],
            "ddg_kj_per_mol": None if analysis is None else analysis["bar_dg_kj_per_mol"],
            "ddg_error_kj_per_mol": None if analysis is None else analysis["bar_bootstrap_std_kj_per_mol"],
            "estimator": "BAR",
            "samples_forward": counts["leg_a_forward"] + counts["leg_b_forward"],
            "samples_reverse": counts["leg_a_reverse"] + counts["leg_b_reverse"],
            "samples_per_replica": None,
            "components": components,
            "estimator_variants": None,
        },
        "quality": {
            "convergence_status": convergence_status,
            "overlap_score": None if analysis is None else analysis["overlap_score"],
            "cycle_closure_error": None,
            "warnings": warnings,
            "rest2": rest2,
            "finite_sample_counts": finite_counts,
            "counted_infinite_work_counts": infinite_counts,
            "convergence": convergence,
            "schedule_optimization": optimizer,
            "adaptive_switching": adaptive,
        },
        "error": None,
        "inputs": {
            "receptor": str(Path(receptor).resolve()),
            "ligand_a_file": str(pair["lig1_file"]),
            "ligand_b_file": str(pair["lig2_file"]),
            "workflow_yaml": str(Path(workflow_path).resolve()),
            "final_pair_yaml": None,
        },
        "artifacts": {
            "prepared_manifest": "prepared/manifest.yaml",
            "hybrid_mapping": "hybrid_mapping.yaml",
            "complex_forward_work_csv": "complex_forward.csv",
            "complex_reverse_work_csv": "complex_reverse.csv",
            "solvent_forward_work_csv": "solvent_forward.csv",
            "solvent_reverse_work_csv": "solvent_reverse.csv",
            "switch_protocol": "switch_protocol.yaml",
            "equilibration_protocol": "equilibration_protocol.yaml",
            "equilibration_manifests": {
                f"{environment}_endpoint_{endpoint}": (
                    str(
                        Path("equilibration")
                        / environment
                        / f"endpoint_{endpoint}"
                        / "manifest.json"
                    )
                    if (
                        workdir
                        / "equilibration"
                        / environment
                        / f"endpoint_{endpoint}"
                        / "manifest.json"
                    ).exists()
                    else None
                )
                for environment in ("complex", "solvent")
                for endpoint in ("a", "b")
            },
            "schedule_optimization": (
                "covalent_schedule_optimization.yaml" if optimizer is not None else None
            ),
            "adaptive_switching": (
                "neqti_adaptive_switching.yaml" if adaptive is not None else None
            ),
            "neqti_convergence": (
                "neqti_convergence.yaml" if convergence is not None else None
            ),
        },
        "parameterization": manifest.get("parameterization"),
        "progress": {
            "stage": status if status in {"prepared", "completed"} else "production",
            "current_pair_index": pair.get("pair_index", 1),
            "total_pairs": pair.get("total_pairs", 1),
            "forward_samples": counts["leg_a_forward"] + counts["leg_b_forward"],
            "reverse_samples": counts["leg_a_reverse"] + counts["leg_b_reverse"],
            "target_forward_samples": 2 * config["n_snapshots"],
            "target_reverse_samples": 2 * config["n_snapshots"],
            "sample_counts": counts,
            "target_sample_counts": {
                name: config["n_snapshots"] for name in counts
            },
            "completed_snapshot_cycles": min(counts.values()),
            "target_snapshot_cycles": config["n_snapshots"],
        },
    }


def run_noncovalent_hybrid_workflow(path):
    from atom_openmm.rbfe_workflow import build_small_molecule_plan, load_workflow_config

    validate_noncovalent_hybrid_workflow(path)
    loaded = load_workflow_config(path)
    workflow = loaded["workflow"]
    plan = build_small_molecule_plan(loaded)
    config = _validate_settings(workflow)
    platform, properties = _platform(workflow)
    results = []
    for pair in plan["pairs"]:
        workdir = pair["jobdir"]
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            _ensure_switch_protocol(
                workdir,
                config,
                _mapping_settings(workflow),
                environments=("complex", "solvent"),
                mapping_label="hybrid_mapping",
            )
            _ensure_equilibration_protocol(workdir, config)
            complex_system, solvent_system, manifest, _ = _prepare_pair(
                pair, plan["receptor_file"], workflow, workdir
            )
            if workflow.get("prepare_only", False) or not workflow.get("run", True):
                payload = _result(
                    pair,
                    path,
                    plan["receptor_file"],
                    workdir,
                    None,
                    manifest,
                    config,
                    "prepared",
                )
                _write_yaml_atomic(workdir / "result.yaml", payload)
                results.append(
                    {
                        "jobname": pair["jobname"],
                        "status": "prepared",
                        "workdir": str(workdir),
                    }
                )
                continue
            previous_result = (
                yaml.safe_load((workdir / "result.yaml").read_text()) or {}
                if (workdir / "result.yaml").exists()
                else {}
            )
            _write_yaml_atomic(
                workdir / "result.yaml",
                _result(pair, path, plan["receptor_file"], workdir, None, manifest, config, "running"),
            )
            convergence_callback = None
            convergence_state = None
            if config["convergence"]["enabled"]:
                convergence_callback, convergence_state = _hybrid_convergence_callback(
                    workdir, config
                )
            terminal = (
                convergence_state is not None
                and convergence_state.get("termination_reason")
                in {"converged", "max_samples"}
            )
            if terminal and convergence_state is not None:
                _normalize_converged_work_prefix(workdir, convergence_state)
            complex_rest2 = None
            solvent_rest2 = None
            if not terminal and config["convergence"]["enabled"]:
                iterators = {
                    "complex": _iter_environment(
                        "complex", complex_system, config, workdir, platform,
                        properties, config["random_seed"],
                    ),
                    "solvent": _iter_environment(
                        "solvent", solvent_system, config, workdir, platform,
                        properties, config["random_seed"] + 100000,
                    ),
                }
                summaries = _run_independent_environment_iterators(
                    iterators, convergence_callback
                )
                complex_forward = _read_work(workdir / "complex_forward.csv")
                complex_reverse = _read_work(workdir / "complex_reverse.csv")
                solvent_forward = _read_work(workdir / "solvent_forward.csv")
                solvent_reverse = _read_work(workdir / "solvent_reverse.csv")
                existing_rest2 = (
                    previous_result.get("quality", {}).get("rest2") or {}
                )
                complex_rest2 = (
                    summaries["complex"] or existing_rest2.get("complex")
                )
                solvent_rest2 = (
                    summaries["solvent"] or existing_rest2.get("solvent")
                )
                convergence_state = _convergence_state(workdir, config)
                if int(convergence_state.get("schema_version", 1)) >= 2:
                    convergence_state = _finalize_independent_convergence(
                        workdir, config, convergence_state
                    )
                elif convergence_state.get("termination_reason") is None:
                    counts = [
                        len(_read_work(workdir / f"{environment}_{direction}.csv"))
                        for environment in ("complex", "solvent")
                        for direction in ("forward", "reverse")
                    ]
                    if min(counts) >= config["n_snapshots"]:
                        convergence_state["termination_reason"] = "max_samples"
                        _write_yaml_atomic(
                            workdir / "neqti_convergence.yaml", convergence_state
                        )
            elif not terminal:
                complex_forward, complex_reverse, complex_rest2 = _run_environment(
                    "complex", complex_system, config, workdir, platform, properties,
                    config["random_seed"],
                )
                solvent_forward, solvent_reverse, solvent_rest2 = _run_environment(
                    "solvent", solvent_system, config, workdir, platform, properties,
                    config["random_seed"] + 100000,
                )
            else:
                complex_forward = _read_work(workdir / "complex_forward.csv")
                complex_reverse = _read_work(workdir / "complex_reverse.csv")
                solvent_forward = _read_work(workdir / "solvent_forward.csv")
                solvent_reverse = _read_work(workdir / "solvent_reverse.csv")
                existing_rest2 = (
                    previous_result.get("quality", {}).get("rest2") or {}
                )
                complex_rest2 = existing_rest2.get("complex")
                solvent_rest2 = existing_rest2.get("solvent")
                if complex_rest2 is None or solvent_rest2 is None:
                    adaptive_path = workdir / "neqti_adaptive_switching.yaml"
                    adaptive_state = (
                        yaml.safe_load(adaptive_path.read_text()) or {}
                        if adaptive_path.exists()
                        else {}
                    )
                    environments = adaptive_state.get("environments", {})
                    complex_rest2 = complex_rest2 or environments.get(
                        "complex", {}
                    ).get("rest2")
                    solvent_rest2 = solvent_rest2 or environments.get(
                        "solvent", {}
                    ).get("rest2")
            work = {
                "leg_a_forward": _read_work(workdir / "complex_forward.csv"),
                "leg_a_reverse": _read_work(workdir / "complex_reverse.csv"),
                "leg_b_forward": _read_work(workdir / "solvent_forward.csv"),
                "leg_b_reverse": _read_work(workdir / "solvent_reverse.csv"),
            }
            analysis = _analysis_payload(work, config)
            status = "completed" if analysis is not None else "partial"
            if (
                config["convergence"]["enabled"]
                and _convergence_state(workdir, config).get("termination_reason")
                == "max_samples"
            ):
                status = "partial"
            payload = _result(
                pair,
                path,
                plan["receptor_file"],
                workdir,
                analysis,
                manifest,
                config,
                status,
                rest2={"complex": complex_rest2, "solvent": solvent_rest2},
            )
            _write_yaml_atomic(workdir / "result.yaml", payload)
            results.append({
                "jobname": pair["jobname"], "status": status,
                "workdir": str(workdir),
                "ddg": None if analysis is None else analysis["bar_dg_kcal_per_mol"],
                "ddg_std": None if analysis is None else analysis["bar_bootstrap_std_kcal_per_mol"],
            })
        except (Exception, KeyboardInterrupt) as exc:
            failure = {
                "schema_version": 1, "tool": "atom_openmm_rbfe",
                "jobname": pair["jobname"], "status": "failed", "method": "neqti",
                "chemistry": "noncovalent", "alchemy_model": "hybrid_topology",
                "thermodynamic_cycle": "complex_solvent",
                "ligand_a": pair["lig1_name"], "ligand_b": pair["lig2_name"],
                "workdir": str(workdir.resolve()),
                "error": {"type": type(exc).__name__, "message": str(exc), "stage": "production"},
            }
            _write_yaml_atomic(workdir / "result.yaml", failure)
            raise
    return results


def analyze_noncovalent_hybrid_workflow(path):
    from atom_openmm.rbfe_workflow import build_small_molecule_plan, load_workflow_config

    validate_noncovalent_hybrid_workflow(path)
    loaded = load_workflow_config(path)
    workflow = loaded["workflow"]
    plan = build_small_molecule_plan(loaded)
    config = _validate_settings(workflow)
    results = []
    for pair in plan["pairs"]:
        workdir = pair["jobdir"]
        work = {
            "leg_a_forward": _read_work(workdir / "complex_forward.csv"),
            "leg_a_reverse": _read_work(workdir / "complex_reverse.csv"),
            "leg_b_forward": _read_work(workdir / "solvent_forward.csv"),
            "leg_b_reverse": _read_work(workdir / "solvent_reverse.csv"),
        }
        analysis = _analysis_payload(work, config)
        manifest = yaml.safe_load((workdir / "prepared" / "manifest.yaml").read_text())
        status = "completed" if analysis is not None else "partial"
        _write_yaml_atomic(
            workdir / "result.yaml",
            _result(pair, path, plan["receptor_file"], workdir, analysis, manifest, config, status),
        )
        results.append({
            "jobname": pair["jobname"], "status": status, "workdir": str(workdir),
            "ddg": None if analysis is None else analysis["bar_dg_kcal_per_mol"],
            "ddg_std": None if analysis is None else analysis["bar_bootstrap_std_kcal_per_mol"],
        })
    return results
