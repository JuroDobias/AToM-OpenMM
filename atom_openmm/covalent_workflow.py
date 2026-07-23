from __future__ import annotations

import csv
import hashlib
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import openmm as mm
import yaml
from openmm import app, unit
from rdkit import Chem
from rdkit.Chem import rdFMCS

from atom_openmm.covalent_alchemy import create_endpoint_hamiltonian
from atom_openmm.covalent_hybrid import (
    DummyBondedScales,
    build_covalent_hybrid_molecule,
    complete_covalent_atom_map,
)
from atom_openmm.covalent_parameters import (
    CovalentParameterError,
    apply_modified_residue_charges,
    ff19sb_backbone_charges,
    parameterize_capped_product,
)
from atom_openmm.covalent_protein import prepare_protein_covalent_hybrid
from atom_openmm.covalent_softcore import (
    SOFTCORE_NONBONDED_FORCE_GROUP,
    create_softcore_hamiltonian,
)
from atom_openmm.covalent_systems import (
    create_solvated_capped_reference,
    load_prepared_hybrid_bundle,
    solvate_capped_reference_hybrid,
    write_prepared_hybrid_bundle,
)
from atom_openmm.neqti import _is_numerical_switch_failure, analyze_two_leg_work
from atom_openmm.neqti_integrator import ATMNonequilibriumLangevinIntegrator
from atom_openmm.rest2 import create_rest2_system
from atom_openmm.rest2_exchange import REST2ExchangeSampler


LOGGER = logging.getLogger("atom_openmm.covalent_workflow")
KCAL_TO_KJ = 4.184
LRC_CORRECTION_VERSION = 2


class CovalentWorkflowError(ValueError):
    pass


class CovalentResumeError(CovalentWorkflowError):
    pass


PREPARATION_SCHEMA_VERSION = 1


def _resolve(path, base):
    value = Path(path)
    return (value if value.is_absolute() else base / value).resolve()


def _mapping_settings(workflow, pair):
    default = workflow.get("mapping")
    override = pair.get("mapping")
    if default is not None and not isinstance(default, dict):
        raise CovalentWorkflowError("workflow.mapping must be a mapping")
    if override is not None and not isinstance(override, dict):
        raise CovalentWorkflowError("workflow.pairs[].mapping must be a mapping")
    settings = dict(default or {})
    settings.update(override or {})
    method = settings.get("method")
    if method is None:
        method = "mcs_core_smarts" if settings.get("smarts") else "dataset_core"
    method = str(method)
    if method not in {"dataset_core", "mcs_core_smarts"}:
        raise CovalentWorkflowError(
            "covalent mapping.method must be 'dataset_core' or 'mcs_core_smarts'"
        )
    normalized = {"method": method}
    if method == "mcs_core_smarts":
        smarts = settings.get("smarts")
        if not isinstance(smarts, str) or not smarts.strip():
            raise CovalentWorkflowError(
                "covalent mcs_core_smarts mapping requires a non-empty smarts string"
            )
        if Chem.MolFromSmarts(smarts) is None:
            raise CovalentWorkflowError("covalent mapping.smarts is invalid")
        normalized["smarts"] = smarts.strip()
    return normalized


def load_covalent_workflow(path):
    workflow_path = Path(path).resolve()
    if not workflow_path.exists():
        raise CovalentWorkflowError(f"workflow does not exist: {workflow_path}")
    config = yaml.safe_load(workflow_path.read_text()) or {}
    workflow = config.get("workflow") or {}
    if workflow.get("type") != "rbfe" or workflow.get("mode") != "covalent":
        raise CovalentWorkflowError("covalent workflow requires workflow.type='rbfe' and mode='covalent'")
    if not workflow.get("dataset"):
        raise CovalentWorkflowError("workflow.dataset is required for covalent mode")
    dataset_path = _resolve(workflow["dataset"], workflow_path.parent)
    if not dataset_path.exists():
        raise CovalentWorkflowError(f"covalent dataset does not exist: {dataset_path}")
    dataset = yaml.safe_load(dataset_path.read_text()) or {}
    pairs = workflow.get("pairs") or dataset.get("pilot_edges") or []
    if not pairs:
        raise CovalentWorkflowError("covalent workflow contains no pairs")
    settings = {
        "config_path": workflow_path,
        "dataset_path": dataset_path,
        "dataset_root": dataset_path.parent,
        "dataset": dataset,
        "workflow": workflow,
        "pairs": pairs,
    }
    return config, settings


def plan_covalent_workflow(path):
    _, settings = load_covalent_workflow(path)
    workflow = settings["workflow"]
    workdir = _resolve(workflow.get("workdir", "run"), settings["config_path"].parent)
    return {
        "mode": "covalent",
        "dataset": str(settings["dataset_path"]),
        "receptor": str(settings["dataset_root"] / settings["dataset"]["receptor"]),
        "workdir": str(workdir),
        "pairs": [
            {
                "ligand_a": pair["ligand_a"],
                "ligand_b": pair["ligand_b"],
                "mapping": _mapping_settings(workflow, pair),
                "workdir": str(workdir / f"{pair['ligand_a']}--{pair['ligand_b']}"),
            }
            for pair in settings["pairs"]
        ],
    }


def validate_covalent_workflow(path):
    _, settings = load_covalent_workflow(path)
    ligands = {item["ligand_id"]: item for item in settings["dataset"].get("ligands", [])}
    for pair in settings["pairs"]:
        mapping = _mapping_settings(settings["workflow"], pair)
        for key in ("ligand_a", "ligand_b"):
            name = pair.get(key)
            if name not in ligands:
                raise CovalentWorkflowError(f"unknown dataset ligand {name!r}")
            product = settings["dataset_root"] / ligands[name]["capped_product_sdf"]
            if not product.exists():
                raise CovalentWorkflowError(f"missing capped product: {product}")
            if mapping["method"] == "mcs_core_smarts":
                aldehyde = settings["dataset_root"] / ligands[name].get(
                    "aldehyde_sdf", ""
                )
                if not aldehyde.is_file():
                    raise CovalentWorkflowError(f"missing aldehyde ligand: {aldehyde}")
    receptor = settings["dataset_root"] / settings["dataset"]["receptor"]
    if not receptor.exists():
        raise CovalentWorkflowError(f"missing receptor: {receptor}")
    workflow = settings["workflow"]
    setup = workflow.get("setup") or {}
    required_setup = {
        "protein_forcefield": "amber19/protein.ff19SB.xml",
        "water_forcefield": "amber19/opc.xml",
        "ligand_forcefield": "openff-2.2.1.offxml",
        "ligand_charge_model": "espaloma_nn",
        "espaloma_model": "espaloma-0.3.2",
    }
    for key, expected in required_setup.items():
        if key in setup and setup[key] != expected:
            raise CovalentWorkflowError(
                f"workflow.setup.{key} must be {expected!r} in the initial covalent implementation"
            )
    if float(settings["dataset"].get("assay_temperature_k", 298.15)) <= 0.0:
        raise CovalentWorkflowError("dataset assay_temperature_k must be positive")
    config = _normalized_settings(workflow)
    if config["interpolation"] == "softcore_linear":
        for pair in settings["pairs"]:
            charge_a = ligands[pair["ligand_a"]].get("formal_charge")
            charge_b = ligands[pair["ligand_b"]].get("formal_charge")
            if charge_a is not None and charge_b is not None and int(charge_a) != int(charge_b):
                raise CovalentWorkflowError(
                    "softcore_linear currently requires equal endpoint formal charge"
                )
    if config["failed_switch_policy"] not in {"abort", "count_as_infinite"}:
        raise CovalentWorkflowError(
            "covalent NEQTI failed_switch_policy must be 'abort' or 'count_as_infinite'"
        )
    if config["n_snapshots"] < 1 or config["switch_steps"] < 1:
        raise CovalentWorkflowError("NEQTI n_snapshots and switching stage steps must be positive")
    if config["interpolation"] not in {"envelope", "linear", "softcore_linear"}:
        raise CovalentWorkflowError(
            "covalent NEQTI interpolation must be 'envelope', 'linear', or 'softcore_linear'"
        )
    if config["interpolation"] == "softcore_linear" and config["legacy_switch_steps_set"]:
        raise CovalentWorkflowError(
            "workflow.neqti.switch_steps cannot be combined with softcore_linear; "
            "set softcore.charge_steps_per_stage and softcore.sterics_steps"
        )
    if config["softcore"]["long_range_correction"] not in {
        "dynamic",
        "endpoint_correction",
    }:
        raise CovalentWorkflowError(
            "workflow.neqti.softcore.long_range_correction must be "
            "'dynamic' or 'endpoint_correction'"
        )
    if any(value < 0.0 for value in config["dummy_bonded_scales"].values()):
        raise CovalentWorkflowError("workflow.setup.dummy_bonded_scales values cannot be negative")
    equilibration = config["endpoint_equilibration"]
    if config["decorrelation_steps"] < 0 or any(
        equilibration[key] < 0
        for key in ("minimization_max_iterations", "nvt_steps", "npt_steps")
    ):
        raise CovalentWorkflowError("NEQTI equilibration and decorrelation steps cannot be negative")
    if (
        equilibration["minimization_tolerance_kj_mol_nm"] <= 0
        or equilibration["nvt_timestep_fs"] <= 0
        or equilibration["npt_timestep_fs"] <= 0
    ):
        raise CovalentWorkflowError("covalent endpoint equilibration tolerances and timesteps must be positive")
    rest2 = config["rest2"]
    if rest2["enabled"]:
        if rest2["execution"] not in {"serial", "process"}:
            raise CovalentWorkflowError(
                "REST2 execution must be 'serial' or 'process'"
            )
        if rest2["device_indices"] is not None and not isinstance(
            rest2["device_indices"], list
        ):
            raise CovalentWorkflowError("REST2 device_indices must be a list")
        temperatures = rest2["effective_temperatures_k"]
        if len(temperatures) < 2 or temperatures[0] != config["temperature_k"]:
            raise CovalentWorkflowError(
                "REST2 requires at least two temperatures and its first temperature must be physical"
            )
        if any(right <= left for left, right in zip(temperatures, temperatures[1:])):
            raise CovalentWorkflowError("REST2 effective temperatures must increase strictly")
        interval = rest2["exchange_interval_steps"]
        if interval < 1 or config["decorrelation_steps"] % interval:
            raise CovalentWorkflowError(
                "NEQTI decorrelation_steps must be divisible by REST2 exchange_interval_steps"
            )
    return True


def _platform(config):
    requested = str(config.get("platform", "CUDA"))
    properties = {str(key): str(value) for key, value in (config.get("platform_properties") or {}).items()}
    names = [requested]
    if requested == "CUDA":
        names.append("OpenCL")
    names.append("CPU")
    for name in names:
        try:
            platform = mm.Platform.getPlatformByName(name)
        except Exception:
            continue
        supported = set(platform.getPropertyNames())
        return platform, {key: value for key, value in properties.items() if key in supported}
    raise CovalentWorkflowError("no usable OpenMM platform is available")


def _clone_system(system):
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))


def _write_state(path, state):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(mm.XmlSerializer.serialize(state))
    os.replace(temporary, path)


def _write_yaml_atomic(path, payload):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False))
    os.replace(temporary, path)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preparation_fingerprint(settings, pair, inputs, mapping_settings, solvation_seed):
    receptor = settings["dataset_root"] / settings["dataset"]["receptor"]
    files = {
        "dataset": settings["dataset_path"],
        "receptor": receptor,
        "ligand_a_product": inputs["ligand_a"]["product"],
        "ligand_b_product": inputs["ligand_b"]["product"],
    }
    for side in ("ligand_a", "ligand_b"):
        aldehyde = inputs[side].get("aldehyde")
        if aldehyde is not None:
            files[f"{side}_aldehyde"] = aldehyde
    workflow = settings["workflow"]
    payload = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "pair": {
            "ligand_a": pair["ligand_a"],
            "ligand_b": pair["ligand_b"],
        },
        "files": {
            name: {"path": str(Path(path).resolve()), "sha256": _file_sha256(path)}
            for name, path in sorted(files.items())
        },
        "mapping": mapping_settings,
        "setup": workflow.get("setup") or {},
        "covalent_residue": settings["dataset"]["covalent_residue"],
        "solvation_seed": int(solvation_seed),
    }
    serialized = yaml.safe_dump(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), payload


def _runtime_artifacts_exist(workdir):
    workdir = Path(workdir)
    patterns = (
        "*_endpoint_*_state*.xml",
        "*_forward.csv",
        "*_reverse.csv",
        "*_rest2_*",
    )
    return any(any(workdir.glob(pattern)) for pattern in patterns)


def _write_prepared_pair_bundle(
    workdir,
    protein,
    reference,
    *,
    fingerprint,
    fingerprint_inputs,
    mapping_payload,
    parameterization,
):
    workdir = Path(workdir)
    target = workdir / "prepared"
    temporary = workdir / f".prepared.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        manifest = {
            "schema_version": PREPARATION_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "fingerprint_inputs": fingerprint_inputs,
            "environments": {
                "protein": write_prepared_hybrid_bundle(
                    protein, temporary, "protein"
                ),
                "reference": write_prepared_hybrid_bundle(
                    reference, temporary, "reference"
                ),
            },
            "mapping": mapping_payload,
            "parameterization": parameterization,
        }
        _write_yaml_atomic(temporary / "manifest.yaml", manifest)
        if target.exists():
            raise CovalentResumeError(
                f"prepared-system bundle already exists and will not be overwritten: {target}"
            )
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def _load_prepared_pair_bundle(workdir, expected_fingerprint):
    directory = Path(workdir) / "prepared"
    manifest_path = directory / "manifest.yaml"
    if not manifest_path.is_file():
        raise CovalentResumeError(
            f"prepared-system manifest is missing: {manifest_path}"
        )
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    if manifest.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        raise CovalentResumeError(
            "prepared-system schema is incompatible; use a new workdir"
        )
    if manifest.get("fingerprint") != expected_fingerprint:
        raise CovalentResumeError(
            "prepared-system fingerprint differs from current inputs or setup; "
            "use a new workdir or restore the original inputs"
        )
    try:
        protein = load_prepared_hybrid_bundle(
            directory, manifest["environments"]["protein"]
        )
        reference = load_prepared_hybrid_bundle(
            directory, manifest["environments"]["reference"]
        )
    except (KeyError, ValueError, CovalentParameterError) as exc:
        raise CovalentResumeError(f"prepared-system bundle is invalid: {exc}") from exc
    return protein, reference, manifest


def _state_particle_count(state):
    try:
        return len(state.getPositions())
    except Exception as exc:
        raise CovalentResumeError("saved state does not contain positions") from exc


def _validate_state_system_compatibility(state, system, label):
    positions = _state_particle_count(state)
    particles = system.getNumParticles()
    if positions != particles:
        raise CovalentResumeError(
            f"{label} contains {positions} positions but its prepared system has "
            f"{particles} particles"
        )


def _system_total_charge(system):
    force = next(
        (
            force
            for force in system.getForces()
            if isinstance(force, mm.NonbondedForce)
        ),
        None,
    )
    if force is None:
        raise CovalentWorkflowError("prepared endpoint lacks a NonbondedForce")
    return float(
        sum(
            force.getParticleParameters(index)[0].value_in_unit(
                unit.elementary_charge
            )
            for index in range(force.getNumParticles())
        )
    )


def _validate_prepared_endpoint_charges(prepared, label):
    charges = [
        _system_total_charge(prepared.endpoint_a),
        _system_total_charge(prepared.endpoint_b),
    ]
    if not np.isclose(charges[0], charges[1], atol=1.0e-6):
        raise CovalentWorkflowError(
            f"{label} endpoint charges differ: {charges[0]:.8f} and {charges[1]:.8f} e"
        )
    if not all(np.isclose(value, 0.0, atol=1.0e-5) for value in charges):
        raise CovalentWorkflowError(
            f"{label} endpoint systems are not neutral: {charges[0]:.8f} and "
            f"{charges[1]:.8f} e"
        )


def _semantic_components(analysis):
    return {
        "protein": analysis["components"]["leg_a"],
        "reference": analysis["components"]["leg_b"],
    }


def _load_state(path):
    return mm.XmlSerializer.deserialize(Path(path).read_text())


def _write_state_pdb(path, topology, state):
    with Path(path).open("w") as handle:
        app.PDBFile.writeFile(topology, state.getPositions(), handle, keepIds=True)


def _write_switch_pdb(path, topology, state, *, endpoint, dummy_atom_indices):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w") as handle:
        app.PDBFile.writeFile(topology, state.getPositions(), handle, keepIds=True)
    dummy = {int(index) for index in dummy_atom_indices}
    output = [
        f"REMARK 900 COVALENT HYBRID ENDPOINT {str(endpoint).upper()}\n",
        "REMARK 900 DUMMY ATOMS HAVE OCCUPANCY 0.00; ACTIVE/COMMON ATOMS HAVE 1.00\n",
        "REMARK 901 DUMMY PARTICLE INDICES (1-BASED): "
        + (" ".join(str(index + 1) for index in sorted(dummy)) or "NONE")
        + "\n",
    ]
    atom_index = 0
    for line in temporary.read_text().splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")):
            occupancy = 0.0 if atom_index in dummy else 1.0
            line = f"{line[:54]}{occupancy:6.2f}{line[60:]}"
            atom_index += 1
        output.append(line)
    path.write_text("".join(output))
    temporary.unlink()


def _dummy_particles(prepared, endpoint):
    inactive = "unique_b_particle_indices" if endpoint == "a" else "unique_a_particle_indices"
    try:
        return tuple(int(index) for index in prepared.provenance[inactive])
    except KeyError as exc:
        raise CovalentWorkflowError(
            f"prepared covalent system lacks {inactive} diagnostic metadata"
        ) from exc


def _equilibrate_endpoint(
    system,
    positions,
    state_file,
    *,
    protocol,
    temperature_k,
    pressure_bar,
    platform,
    properties,
    seed,
    label,
):
    state_file = Path(state_file)
    if state_file.exists():
        LOGGER.info("%s equilibration already complete; resuming from %s", label, state_file)
        state = _load_state(state_file)
        _validate_state_system_compatibility(state, system, label)
        return state

    minimized_file = state_file.with_name(f"{state_file.stem}_minimized.xml")
    nvt_file = state_file.with_name(f"{state_file.stem}_nvt.xml")

    if minimized_file.exists():
        minimized = _load_state(minimized_file)
        _validate_state_system_compatibility(minimized, system, f"{label} minimization")
        LOGGER.info("%s minimization already complete; resuming from %s", label, minimized_file)
    else:
        integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
        context = mm.Context(system, integrator, platform, properties)
        context.setPositions(positions)
        started = time.perf_counter()
        LOGGER.info(
            "%s minimization started: tolerance %.3f kJ/mol/nm, max %d iterations",
            label,
            protocol["minimization_tolerance_kj_mol_nm"],
            protocol["minimization_max_iterations"],
        )
        mm.LocalEnergyMinimizer.minimize(
            context,
            float(protocol["minimization_tolerance_kj_mol_nm"]),
            int(protocol["minimization_max_iterations"]),
        )
        minimized = context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
        _write_state(minimized_file, minimized)
        LOGGER.info("%s minimization complete in %.3f s", label, time.perf_counter() - started)
        del context

    if nvt_file.exists():
        nvt_state = _load_state(nvt_file)
        _validate_state_system_compatibility(nvt_state, system, f"{label} NVT")
        LOGGER.info("%s NVT equilibration already complete; resuming from %s", label, nvt_file)
    else:
        nvt_steps = int(protocol["nvt_steps"])
        nvt_timestep_fs = float(protocol["nvt_timestep_fs"])
        integrator = mm.LangevinMiddleIntegrator(
            float(temperature_k) * unit.kelvin,
            1.0 / unit.picosecond,
            nvt_timestep_fs * unit.femtosecond,
        )
        integrator.setRandomNumberSeed(int(seed))
        context = mm.Context(system, integrator, platform, properties)
        context.setPositions(minimized.getPositions())
        context.setVelocitiesToTemperature(float(temperature_k) * unit.kelvin, int(seed) + 1)
        started = time.perf_counter()
        LOGGER.info(
            "%s NVT equilibration started: %d steps at %.3f fs",
            label, nvt_steps, nvt_timestep_fs,
        )
        if nvt_steps:
            integrator.step(nvt_steps)
        nvt_state = context.getState(
            getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True
        )
        _write_state(nvt_file, nvt_state)
        LOGGER.info("%s NVT equilibration complete in %.3f s", label, time.perf_counter() - started)
        del context

    equilibrium_system = _clone_system(system)
    equilibrium_system.addForce(
        mm.MonteCarloBarostat(float(pressure_bar) * unit.bar, float(temperature_k) * unit.kelvin)
    )
    npt_steps = int(protocol["npt_steps"])
    npt_timestep_fs = float(protocol["npt_timestep_fs"])
    integrator = mm.LangevinMiddleIntegrator(
        float(temperature_k) * unit.kelvin,
        1.0 / unit.picosecond,
        npt_timestep_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(int(seed) + 2)
    context = mm.Context(equilibrium_system, integrator, platform, properties)
    _apply_state(context, nvt_state)
    started = time.perf_counter()
    LOGGER.info(
        "%s NPT equilibration started: %d steps at %.3f fs and %.3f bar",
        label, npt_steps, npt_timestep_fs, pressure_bar,
    )
    if npt_steps:
        integrator.step(npt_steps)
    state = context.getState(
        getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True
    )
    _write_state(state_file, state)
    LOGGER.info("%s NPT equilibration complete in %.3f s", label, time.perf_counter() - started)
    del context
    return state


def _append_work(path, sample, work_kj):
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["sample", "work_kj_per_mol", "work_kcal_per_mol"])
        writer.writerow([sample, work_kj, work_kj / KCAL_TO_KJ])


def _append_switch_timing(path, environment, direction, sample, steps, timestep_fs, elapsed):
    path = Path(path)
    new = not path.exists()
    simulated_ns = float(steps) * float(timestep_fs) * 1.0e-6
    ns_per_day = simulated_ns * 86400.0 / elapsed
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(
                [
                    "environment", "direction", "sample", "steps", "timestep_fs",
                    "elapsed_seconds", "ns_per_day",
                ]
            )
        writer.writerow(
            [environment, direction, sample, steps, timestep_fs, elapsed, ns_per_day]
        )
    return ns_per_day


def _append_lrc_diagnostic(
    path,
    environment,
    direction,
    sample,
    raw_work_kj,
    initial_correction_kj,
    final_correction_kj,
    volume_nm3,
    correction_version=None,
    evaluation_platform=None,
):
    path = Path(path)
    new = not path.exists()
    delta = final_correction_kj - initial_correction_kj
    corrected = raw_work_kj + delta
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(
                [
                    "environment",
                    "direction",
                    "sample",
                    "raw_work_kj_per_mol",
                    "initial_lrc_kj_per_mol",
                    "final_lrc_kj_per_mol",
                    "lrc_delta_kj_per_mol",
                    "corrected_work_kj_per_mol",
                    "volume_nm3",
                    "correction_version",
                    "evaluation_platform",
                ]
            )
        writer.writerow(
            [
                environment,
                direction,
                sample,
                raw_work_kj,
                initial_correction_kj,
                final_correction_kj,
                delta,
                corrected,
                volume_nm3,
                correction_version,
                evaluation_platform,
            ]
        )
    return corrected


def _switch_timing_summary(path):
    path = Path(path)
    if not path.exists():
        return {}
    grouped = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = f"{row['environment']}_{row['direction']}"
            grouped.setdefault(key, []).append(float(row["ns_per_day"]))
    return {
        key: {
            "samples": len(values),
            "mean_ns_per_day": float(np.mean(values)),
            "min_ns_per_day": float(np.min(values)),
            "max_ns_per_day": float(np.max(values)),
        }
        for key, values in grouped.items()
    }


def _read_work(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [float(row["work_kcal_per_mol"]) for row in csv.DictReader(handle)]


def _switch_context(
    endpoint_a,
    endpoint_b,
    *,
    start,
    steps,
    timestep_fs,
    temperature_k,
    platform,
    properties,
    seed,
    interpolation,
):
    hamiltonian = create_endpoint_hamiltonian(
        endpoint_a, endpoint_b,
        interpolation=interpolation,
        temperature_k=temperature_k,
    )
    schedule = [0.0, 1.0] if start == "a" else [1.0, 0.0]
    integrator = ATMNonequilibriumLangevinIntegrator(
        temperature=float(temperature_k) * unit.kelvin,
        collision_rate=1.0 / unit.picosecond,
        timestep=float(timestep_fs) * unit.femtosecond,
        parameter_values={hamiltonian.lambda_parameter: schedule},
        steps_per_segment=int(steps),
        random_seed=int(seed),
    )
    context = mm.Context(hamiltonian.system, integrator, platform, properties)
    return context, integrator


def _softcore_switch_context(
    hamiltonian,
    *,
    start,
    timestep_fs,
    temperature_k,
    platform,
    properties,
    seed,
):
    forward = start == "a"
    values = {
        name: list(schedule if forward else reversed(schedule))
        for name, schedule in hamiltonian.parameter_values.items()
    }
    steps = list(
        hamiltonian.segment_steps if forward else reversed(hamiltonian.segment_steps)
    )
    integrator = ATMNonequilibriumLangevinIntegrator(
        temperature=float(temperature_k) * unit.kelvin,
        collision_rate=1.0 / unit.picosecond,
        timestep=float(timestep_fs) * unit.femtosecond,
        parameter_values=values,
        steps_per_segment=steps,
        random_seed=int(seed),
    )
    context = mm.Context(hamiltonian.system, integrator, platform, properties)
    return context, integrator, values


def _set_softcore_parameters(context, parameter_values, node):
    for name, values in parameter_values.items():
        context.setParameter(name, float(values[node]))


def _softcore_group_energy(context):
    return context.getState(
        getEnergy=True,
        groups={SOFTCORE_NONBONDED_FORCE_GROUP},
    ).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


def _lrc_correction_platform():
    try:
        platform = mm.Platform.getPlatformByName("CPU")
        properties = {}
        if "Threads" in set(platform.getPropertyNames()):
            properties["Threads"] = "1"
        return platform, properties
    except Exception:
        return mm.Platform.getPlatformByName("Reference"), {}


class _EndpointLRCCorrectionEvaluator:
    """Evaluate endpoint LRC values in paired, identically configured contexts."""

    def __init__(self, no_lrc_hamiltonian, lrc_hamiltonian):
        platform, properties = _lrc_correction_platform()
        self.platform_name = platform.getName()
        self.no_lrc_integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
        self.lrc_integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
        self.no_lrc_context = mm.Context(
            no_lrc_hamiltonian.system,
            self.no_lrc_integrator,
            platform,
            properties,
        )
        self.lrc_context = mm.Context(
            lrc_hamiltonian.system,
            self.lrc_integrator,
            platform,
            properties,
        )

    def correction(self, state, parameter_values, node):
        for context in (self.no_lrc_context, self.lrc_context):
            _apply_state(context, state)
            _set_softcore_parameters(context, parameter_values, node)
        no_lrc = _softcore_group_energy(self.no_lrc_context)
        with_lrc = _softcore_group_energy(self.lrc_context)
        correction = with_lrc - no_lrc
        if not np.isfinite(correction):
            raise CovalentWorkflowError("non-finite softcore endpoint LRC correction")
        return float(correction)

    def close(self):
        self.no_lrc_context = None
        self.lrc_context = None
        self.no_lrc_integrator = None
        self.lrc_integrator = None


def _precompute_endpoint_lrc_corrections(evaluator, state_a, state_b, parameter_values):
    try:
        return {
            "forward": {
                "initial": evaluator.correction(state_a, parameter_values, 0),
                "final": evaluator.correction(state_a, parameter_values, -1),
                "volume_nm3": _state_volume_nm3(state_a),
            },
            "reverse": {
                "initial": evaluator.correction(state_b, parameter_values, -1),
                "final": evaluator.correction(state_b, parameter_values, 0),
                "volume_nm3": _state_volume_nm3(state_b),
            },
        }
    finally:
        evaluator.close()


def _state_volume_nm3(state):
    vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    return float(abs(np.linalg.det(np.asarray(vectors, dtype=float))))


def _assert_fixed_volume_switch(system):
    if any(isinstance(force, mm.MonteCarloBarostat) for force in system.getForces()):
        raise CovalentWorkflowError(
            "softcore endpoint LRC correction requires fixed-volume switching"
        )


def _reset_softcore_context(context, integrator, parameter_values):
    integrator.reset_protocol()
    for name, values in parameter_values.items():
        context.setParameter(name, float(values[0]))


def _apply_state(context, state):
    box = state.getPeriodicBoxVectors()
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    context.setPositions(state.getPositions())
    try:
        velocities = state.getVelocities()
    except Exception:
        velocities = None
    if velocities is not None:
        context.setVelocities(velocities)


def _sample_endpoint(
    system,
    topology,
    state_file,
    hot_atoms,
    *,
    ensemble,
    steps,
    rest2_config,
    output_dir,
    platform,
    properties,
    temperature_k,
    timestep_fs,
    seed,
):
    if rest2_config.get("enabled", False):
        rest2 = create_rest2_system(system, hot_atoms)
        base_integrator = mm.LangevinMiddleIntegrator(
            float(temperature_k) * unit.kelvin,
            1.0 / unit.picosecond,
            float(timestep_fs) * unit.femtosecond,
        )
        sampler = REST2ExchangeSampler(
            system=rest2.system,
            topology=topology,
            base_integrator=base_integrator,
            rest2_system=rest2,
            state_files={ensemble: state_file},
            config=rest2_config,
            platform=platform,
            platform_properties=properties,
            output_dir=output_dir,
            resume=True,
            random_seed=seed,
            logger=LOGGER,
        )
        sampler.run_steps(ensemble, int(steps), label=ensemble)
        state = sampler.physical_state(ensemble)
        _write_state(state_file, state)
        summary = sampler.summary()
        sampler.close()
        return state, summary
    integrator = mm.LangevinMiddleIntegrator(
        float(temperature_k) * unit.kelvin,
        1.0 / unit.picosecond,
        float(timestep_fs) * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(int(seed))
    context = mm.Context(system, integrator, platform, properties)
    state = _load_state(state_file)
    _apply_state(context, state)
    integrator.step(int(steps))
    state = context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True)
    _write_state(state_file, state)
    del context
    return state, None


def _run_environment(
    name,
    prepared,
    config,
    workdir,
    platform,
    properties,
    seed,
):
    temperature = float(config["temperature_k"])
    timestep = float(config["timestep_fs"])
    state_files = {
        "a": workdir / f"{name}_endpoint_a_state.xml",
        "b": workdir / f"{name}_endpoint_b_state.xml",
    }
    _equilibrate_endpoint(
        prepared.endpoint_a, prepared.positions, state_files["a"],
        protocol=config["endpoint_equilibration"], temperature_k=temperature,
        pressure_bar=config["pressure_bar"], platform=platform, properties=properties, seed=seed,
        label=f"{name} endpoint A",
    )
    _equilibrate_endpoint(
        prepared.endpoint_b, prepared.positions, state_files["b"],
        protocol=config["endpoint_equilibration"], temperature_k=temperature,
        pressure_bar=config["pressure_bar"], platform=platform, properties=properties, seed=seed + 10,
        label=f"{name} endpoint B",
    )
    _write_state_pdb(
        workdir / f"{name}_endpoint_a_equilibrated.pdb",
        prepared.topology,
        _load_state(state_files["a"]),
    )
    _write_state_pdb(
        workdir / f"{name}_endpoint_b_equilibrated.pdb",
        prepared.topology,
        _load_state(state_files["b"]),
    )
    files = {
        "forward": workdir / f"{name}_forward.csv",
        "reverse": workdir / f"{name}_reverse.csv",
    }
    timing_file = workdir / "switch_timing.csv"
    lrc_diagnostics_file = workdir / "switch_lrc_diagnostics.csv"
    forward = _read_work(files["forward"])
    reverse = _read_work(files["reverse"])
    rest2_summaries = {}
    softcore = None
    softcore_contexts = {}
    lrc_corrections = None
    lrc_evaluation_platform = None
    if config["interpolation"] == "softcore_linear":
        unique_a = prepared.provenance["unique_a_particle_indices"]
        unique_b = prepared.provenance["unique_b_particle_indices"]
        softcore_options = {
            key: value
            for key, value in config["softcore"].items()
            if key != "long_range_correction"
        }
        lrc_mode = config["softcore"]["long_range_correction"]
        softcore = create_softcore_hamiltonian(
            prepared.endpoint_a,
            prepared.endpoint_b,
            unique_a,
            unique_b,
            use_long_range_correction=lrc_mode == "dynamic",
            **softcore_options,
        )
        _assert_fixed_volume_switch(softcore.system)
        if lrc_mode == "endpoint_correction":
            lrc_hamiltonian = create_softcore_hamiltonian(
                prepared.endpoint_a,
                prepared.endpoint_b,
                unique_a,
                unique_b,
                use_long_range_correction=True,
                **softcore_options,
            )
            lrc_evaluator = _EndpointLRCCorrectionEvaluator(
                softcore,
                lrc_hamiltonian,
            )
            lrc_evaluation_platform = lrc_evaluator.platform_name
            lrc_corrections = _precompute_endpoint_lrc_corrections(
                lrc_evaluator,
                _load_state(state_files["a"]),
                _load_state(state_files["b"]),
                softcore.parameter_values,
            )
            LOGGER.info(
                "%s endpoint LRC corrections precomputed in paired %s contexts: "
                "forward delta %.6f kJ/mol at %.6f nm^3; reverse delta %.6f "
                "kJ/mol at %.6f nm^3",
                name,
                lrc_evaluation_platform,
                lrc_corrections["forward"]["final"]
                - lrc_corrections["forward"]["initial"],
                lrc_corrections["forward"]["volume_nm3"],
                lrc_corrections["reverse"]["final"]
                - lrc_corrections["reverse"]["initial"],
                lrc_corrections["reverse"]["volume_nm3"],
            )
        LOGGER.info(
            "%s softcore switching system ready: charge %d + sterics %d + charge %d "
            "= %d steps",
            name,
            config["softcore"]["charge_steps_per_stage"],
            config["softcore"]["sterics_steps"],
            config["softcore"]["charge_steps_per_stage"],
            softcore.total_steps,
        )
    for sample in range(int(config["n_snapshots"])):
        for direction, endpoint, system, offset in (
            ("forward", "a", prepared.endpoint_a, 0),
            ("reverse", "b", prepared.endpoint_b, 100),
        ):
            existing = forward if direction == "forward" else reverse
            if len(existing) > sample:
                continue
            state, rest2_summary = _sample_endpoint(
                system, prepared.topology, state_files[endpoint], prepared.hot_atom_indices,
                ensemble=endpoint,
                steps=config["decorrelation_steps"],
                rest2_config=config["rest2"],
                output_dir=workdir / f"{name}_rest2_{endpoint}",
                platform=platform, properties=properties,
                temperature_k=temperature, timestep_fs=timestep,
                seed=seed + sample * 1000 + offset,
            )
            if rest2_summary is not None:
                rest2_summaries[endpoint] = rest2_summary
            _write_state_pdb(
                workdir / f"{name}_endpoint_{endpoint}_equilibrated.pdb",
                prepared.topology,
                state,
            )
            pre_switch_path = workdir / (
                f"{name}_{direction}_sample_{sample + 1:03d}_pre_switch.pdb"
            )
            post_switch_path = workdir / (
                f"{name}_{direction}_sample_{sample + 1:03d}_post_switch.pdb"
            )
            _write_switch_pdb(
                pre_switch_path,
                prepared.topology,
                state,
                endpoint=endpoint,
                dummy_atom_indices=_dummy_particles(prepared, endpoint),
            )
            context = None
            post_switch_written = False
            switch_elapsed = None
            switch_ns_per_day = None
            raw_work_kj = None
            initial_lrc_kj = None
            final_lrc_kj = None
            volume_nm3 = None
            try:
                if softcore is None:
                    context, integrator = _switch_context(
                        prepared.endpoint_a, prepared.endpoint_b,
                        start=endpoint,
                        steps=config["switch_steps"], timestep_fs=timestep,
                        temperature_k=temperature, platform=platform, properties=properties,
                        seed=seed + sample * 1000 + offset + 1,
                        interpolation=config["interpolation"],
                    )
                else:
                    cached = softcore_contexts.get(direction)
                    if cached is None:
                        cached = _softcore_switch_context(
                            softcore,
                            start=endpoint,
                            timestep_fs=timestep,
                            temperature_k=temperature,
                            platform=platform,
                            properties=properties,
                            seed=seed + offset + 1,
                        )
                        softcore_contexts[direction] = cached
                    context, integrator, parameter_values = cached
                    _reset_softcore_context(context, integrator, parameter_values)
                _apply_state(context, state)
                if lrc_corrections is not None:
                    correction = lrc_corrections[direction]
                    initial_lrc_kj = correction["initial"]
                    final_lrc_kj = correction["final"]
                    volume_nm3 = _state_volume_nm3(state)
                    if not np.isclose(
                        volume_nm3,
                        correction["volume_nm3"],
                        atol=1.0e-8,
                        rtol=0.0,
                    ):
                        raise CovalentWorkflowError(
                            f"{name} {direction} endpoint volume changed after LRC "
                            "precomputation"
                        )
                started = time.perf_counter()
                integrator.step(int(config["switch_steps"]))
                switch_elapsed = time.perf_counter() - started
                simulated_ns = config["switch_steps"] * timestep * 1.0e-6
                switch_ns_per_day = simulated_ns * 86400.0 / switch_elapsed
                LOGGER.info(
                    "%s %s sample %d switch complete: %d steps in %.3f s, %.3f ns/day",
                    name,
                    direction,
                    sample + 1,
                    config["switch_steps"],
                    switch_elapsed,
                    switch_ns_per_day,
                )
                final_endpoint = "b" if endpoint == "a" else "a"
                post_switch_state = context.getState(
                    getPositions=True, enforcePeriodicBox=True
                )
                _write_switch_pdb(
                    post_switch_path,
                    prepared.topology,
                    post_switch_state,
                    endpoint=final_endpoint,
                    dummy_atom_indices=_dummy_particles(prepared, final_endpoint),
                )
                post_switch_written = True
                raw_work_kj = integrator.get_protocol_work().value_in_unit(
                    unit.kilojoules_per_mole
                )
                work_kj = raw_work_kj
                if lrc_corrections is not None:
                    final_volume_nm3 = _state_volume_nm3(post_switch_state)
                    if not np.isclose(final_volume_nm3, volume_nm3, atol=1.0e-8):
                        raise CovalentWorkflowError(
                            "box volume changed during fixed-volume LRC-corrected switch"
                        )
                    work_kj = _append_lrc_diagnostic(
                        lrc_diagnostics_file,
                        name,
                        direction,
                        sample + 1,
                        raw_work_kj,
                        initial_lrc_kj,
                        final_lrc_kj,
                        volume_nm3,
                        correction_version=LRC_CORRECTION_VERSION,
                        evaluation_platform=lrc_evaluation_platform,
                    )
                    LOGGER.info(
                        "%s %s sample %d LRC correction: raw %.6f + delta %.6f "
                        "= %.6f kJ/mol",
                        name,
                        direction,
                        sample + 1,
                        raw_work_kj,
                        final_lrc_kj - initial_lrc_kj,
                        work_kj,
                    )
            except Exception as exc:
                if (
                    config["failed_switch_policy"] == "count_as_infinite"
                    and _is_numerical_switch_failure(exc)
                ):
                    work_kj = float("inf")
                    LOGGER.warning(
                        "%s %s sample %d failed numerically and was recorded as +inf work: %s",
                        name, direction, sample + 1, exc,
                    )
                    if softcore is not None:
                        softcore_contexts.pop(direction, None)
                else:
                    raise
            finally:
                if context is not None and not post_switch_written:
                    try:
                        final_endpoint = "b" if endpoint == "a" else "a"
                        post_switch_state = context.getState(
                            getPositions=True, enforcePeriodicBox=True
                        )
                        _write_switch_pdb(
                            post_switch_path,
                            prepared.topology,
                            post_switch_state,
                            endpoint=final_endpoint,
                            dummy_atom_indices=_dummy_particles(prepared, final_endpoint),
                        )
                    except Exception as diagnostic_exc:
                        LOGGER.warning(
                            "Could not write %s post-switch diagnostic: %s",
                            post_switch_path,
                            diagnostic_exc,
                        )
            _append_work(files[direction], sample + 1, work_kj)
            if switch_elapsed is not None:
                _append_switch_timing(
                    timing_file,
                    name,
                    direction,
                    sample + 1,
                    config["switch_steps"],
                    timestep,
                    switch_elapsed,
                )
            if context is not None and softcore is None:
                del context
            if direction == "forward":
                forward = _read_work(files[direction])
            else:
                reverse = _read_work(files[direction])
        forward = _read_work(files["forward"])
        reverse = _read_work(files["reverse"])
        LOGGER.info(
            "%s sample %d/%d complete: forward %.3f, reverse %.3f kJ/mol",
            name, sample + 1, config["n_snapshots"],
            forward[-1] * KCAL_TO_KJ, reverse[-1] * KCAL_TO_KJ,
        )
    return forward, reverse, rest2_summaries


def _normalized_settings(workflow):
    setup = workflow.get("setup") or {}
    dummy = setup.get("dummy_bonded_scales") or {}
    neqti = workflow.get("neqti") or {}
    equilibration = neqti.get("endpoint_equilibration") or {}
    rest2 = neqti.get("rest2") or {}
    softcore = neqti.get("softcore") or {}
    interpolation = str(neqti.get("interpolation", "envelope"))
    enabled = bool(rest2.get("enabled", True))
    temperatures = rest2.get(
        "effective_temperatures_k",
        [300.0, 344.6, 395.9, 454.7, 522.3, 600.0],
    )
    softcore_settings = {
        "alpha": float(softcore.get("alpha", 0.3)),
        "sigma_nm": float(softcore.get("sigma_nm", 0.25)),
        "power": int(softcore.get("power", 1)),
        "charge_steps_per_stage": int(softcore.get("charge_steps_per_stage", 10000)),
        "sterics_steps": int(softcore.get("sterics_steps", 30000)),
        "long_range_correction": str(
            softcore.get("long_range_correction", "dynamic")
        ),
    }
    switch_steps = (
        2 * softcore_settings["charge_steps_per_stage"] + softcore_settings["sterics_steps"]
        if interpolation == "softcore_linear"
        else int(neqti.get("switch_steps", 50000))
    )
    return {
        "temperature_k": float(neqti.get("temperature_k", 300.0)),
        "pressure_bar": float(neqti.get("pressure_bar", 1.0)),
        "timestep_fs": float(neqti.get("timestep_fs", 2.0)),
        "endpoint_equilibration": {
            "minimization_tolerance_kj_mol_nm": float(
                equilibration.get("minimization_tolerance_kj_mol_nm", 10.0)
            ),
            "minimization_max_iterations": int(
                equilibration.get("minimization_max_iterations", 2000)
            ),
            "nvt_steps": int(equilibration.get("nvt_steps", 50000)),
            "nvt_timestep_fs": float(equilibration.get("nvt_timestep_fs", 1.0)),
            "npt_steps": int(
                equilibration.get(
                    "npt_steps", neqti.get("initial_equilibration_steps", 50000)
                )
            ),
            "npt_timestep_fs": float(
                equilibration.get("npt_timestep_fs", neqti.get("timestep_fs", 2.0))
            ),
        },
        "decorrelation_steps": int(neqti.get("decorrelation_steps", 100000)),
        "switch_steps": switch_steps,
        "n_snapshots": int(neqti.get("n_snapshots", 10)),
        "bootstrap_samples": int(neqti.get("bootstrap_samples", 500)),
        "random_seed": int(neqti.get("random_seed", 2026)),
        "interpolation": interpolation,
        "legacy_switch_steps_set": "switch_steps" in neqti,
        "softcore": softcore_settings,
        "failed_switch_policy": str(
            neqti.get("failed_switch_policy", "count_as_infinite")
        ),
        "dummy_bonded_scales": {
            "bond": float(dummy.get("bond", 1.0)),
            "angle": float(dummy.get("angle", 1.0)),
            "proper_torsion": float(dummy.get("proper_torsion", 1.0)),
            "junction_angle": float(dummy.get("junction_angle", 1.0)),
            "junction_proper_torsion": float(
                dummy.get("junction_proper_torsion", 0.0)
            ),
        },
        "rest2": {
            "enabled": enabled,
            "effective_temperatures_k": [float(value) for value in temperatures],
            "exchange_interval_steps": int(rest2.get("exchange_interval_steps", 500)),
            "checkpoint_interval_cycles": int(rest2.get("checkpoint_interval_cycles", 10)),
            "execution": str(rest2.get("execution", "serial")),
            "device_indices": rest2.get("device_indices"),
        },
    }


def _pair_inputs(settings, pair):
    ligands = {item["ligand_id"]: item for item in settings["dataset"]["ligands"]}
    root = settings["dataset_root"]
    inputs = {}
    for key in ("ligand_a", "ligand_b"):
        name = pair[key]
        info = ligands[name]
        metadata = yaml.safe_load((root / Path(info["capped_product_sdf"]).parent / "product_metadata.yaml").read_text())
        inputs[key] = {
            "name": name,
            "info": info,
            "metadata": metadata,
            "product": root / info["capped_product_sdf"],
            "aldehyde": root / info["aldehyde_sdf"] if info.get("aldehyde_sdf") else None,
        }
    return inputs


def _load_covalent_sdf(path, label):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() != 1:
        raise CovalentWorkflowError(
            f"could not read one 3D molecule for covalent mapping from {label}: {path}"
        )
    return molecule


def _direct_match_rmsd(molecule_a, match_a, molecule_b, match_b):
    conformer_a = molecule_a.GetConformer()
    conformer_b = molecule_b.GetConformer()
    squared = 0.0
    for atom_a, atom_b in zip(match_a, match_b):
        difference = np.asarray(conformer_a.GetAtomPosition(atom_a)) - np.asarray(
            conformer_b.GetAtomPosition(atom_b)
        )
        squared += float(difference @ difference)
    return float(np.sqrt(squared / len(match_a)))


def _constrained_ligand_atom_map(inputs, smarts, timeout_seconds=30):
    molecule_a = _load_covalent_sdf(
        inputs["ligand_a"]["aldehyde"], inputs["ligand_a"]["name"]
    )
    molecule_b = _load_covalent_sdf(
        inputs["ligand_b"]["aldehyde"], inputs["ligand_b"]["name"]
    )
    core = Chem.MolFromSmarts(smarts)
    result = rdFMCS.FindMCS(
        [molecule_a, molecule_b, core],
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        timeout=int(timeout_seconds),
    )
    if result.canceled or result.numAtoms == 0:
        raise CovalentWorkflowError("SMARTS-constrained covalent MCS failed or timed out")
    query = result.queryMol
    matches_a = molecule_a.GetSubstructMatches(
        query, uniquify=False, useChirality=True
    )
    matches_b = molecule_b.GetSubstructMatches(
        query, uniquify=False, useChirality=True
    )
    if not matches_a or not matches_b:
        raise CovalentWorkflowError(
            "SMARTS-constrained covalent MCS did not match both ligands"
        )
    candidates = []
    for index_a, match_a in enumerate(matches_a, start=1):
        for index_b, match_b in enumerate(matches_b, start=1):
            candidates.append(
                (
                    _direct_match_rmsd(molecule_a, match_a, molecule_b, match_b),
                    index_a,
                    index_b,
                    match_a,
                    match_b,
                )
            )
    rmsd, index_a, index_b, match_a, match_b = min(
        candidates, key=lambda item: item[:3]
    )
    return dict(zip(match_a, match_b)), {
        "core_smarts": smarts,
        "mcs_smarts": Chem.MolToSmarts(query),
        "mcs_atom_count": int(result.numAtoms),
        "mcs_heavy_atom_count": sum(
            atom.GetAtomicNum() != 1 for atom in query.GetAtoms()
        ),
        "selected_match_a": int(index_a),
        "selected_match_b": int(index_b),
        "selected_direct_rmsd_angstrom": float(rmsd),
        "candidate_matches_a": len(matches_a),
        "candidate_matches_b": len(matches_b),
    }


def _prepare_covalent_atom_map(inputs, mapping_settings):
    meta_a = inputs["ligand_a"]["metadata"]
    meta_b = inputs["ligand_b"]["metadata"]
    cap_count_a = int(meta_a["capped_cys_atom_count"])
    cap_count_b = int(meta_b["capped_cys_atom_count"])
    if cap_count_a != cap_count_b:
        raise CovalentWorkflowError("capped cysteine atom counts differ between ligands")
    required = [(index, index) for index in range(cap_count_a)]
    provenance = {"mapping_mode": mapping_settings["method"]}
    atom_map = None
    if mapping_settings["method"] == "dataset_core":
        core_a = inputs["ligand_a"]["info"]["core_match_atom_indices_1based"]
        core_b = inputs["ligand_b"]["info"]["core_match_atom_indices_1based"]
        if len(core_a) != len(core_b):
            raise CovalentWorkflowError(
                "dataset core atom lists must contain the same number of atoms"
            )
        required.extend(
            (
                int(meta_a["ligand_atom_offset"]) + atom_a - 1,
                int(meta_b["ligand_atom_offset"]) + atom_b - 1,
            )
            for atom_a, atom_b in zip(
                core_a,
                core_b,
            )
        )
    else:
        ligand_map, constrained = _constrained_ligand_atom_map(
            inputs, mapping_settings["smarts"]
        )
        product_a = {
            int(atom): int(product)
            for atom, product in meta_a["ligand_to_product_atom_indices"].items()
        }
        product_b = {
            int(atom): int(product)
            for atom, product in meta_b["ligand_to_product_atom_indices"].items()
        }
        transferred = {
            product_a[atom_a]: product_b[atom_b]
            for atom_a, atom_b in ligand_map.items()
        }
        atom_map = dict(required)
        if set(atom_map).intersection(transferred):
            raise CovalentWorkflowError(
                "SMARTS-constrained ligand map overlaps capped cysteine atoms"
            )
        atom_map.update(transferred)
        provenance.update(constrained)
        provenance["transferred_ligand_pairs_0based"] = {
            int(atom_a): int(atom_b) for atom_a, atom_b in sorted(transferred.items())
        }
    attachment_pairs = (
        (
            int(meta_a["cys_sulfur_atom_index"]),
            int(meta_b["cys_sulfur_atom_index"]),
        ),
        (
            int(meta_a["electrophile_carbon_atom_index"]),
            int(meta_b["electrophile_carbon_atom_index"]),
        ),
    )
    for pair_to_require in attachment_pairs:
        if pair_to_require not in required:
            required.append(pair_to_require)
    if atom_map is not None:
        if attachment_pairs[1] not in atom_map.items():
            raise CovalentWorkflowError(
                "SMARTS-constrained MCS must contain the ligand electrophile carbon"
            )
        product_molecule_a = _load_covalent_sdf(
            inputs["ligand_a"]["product"], inputs["ligand_a"]["name"]
        )
        product_molecule_b = _load_covalent_sdf(
            inputs["ligand_b"]["product"], inputs["ligand_b"]["name"]
        )
        atom_map = complete_covalent_atom_map(
            product_molecule_a,
            product_molecule_b,
            atom_map,
            required_pairs=required,
        )
    return required, attachment_pairs, atom_map, provenance


def _validate_softcore_endpoint_charge(config, parameters_a, parameters_b):
    if config["interpolation"] != "softcore_linear":
        return
    charge_a = float(np.sum(parameters_a.charges_e))
    charge_b = float(np.sum(parameters_b.charges_e))
    if not np.isclose(charge_a, charge_b, atol=1.0e-6):
        raise CovalentWorkflowError(
            "softcore_linear currently requires equal endpoint total charge; "
            f"observed {charge_a:.8f} e and {charge_b:.8f} e"
        )


def _switch_protocol(config, mapping_settings=None):
    protocol = {
        "schema_version": 1,
        "interpolation": config["interpolation"],
        "timestep_fs": config["timestep_fs"],
        "total_steps": config["switch_steps"],
    }
    if config["interpolation"] == "softcore_linear":
        protocol["softcore"] = dict(config["softcore"])
        if config["softcore"]["long_range_correction"] == "endpoint_correction":
            protocol["softcore_endpoint_correction"] = {
                "version": LRC_CORRECTION_VERSION,
                "evaluation_platform": "CPU",
                "evaluation": "precomputed_per_endpoint_and_switch_volume",
            }
        protocol["stages"] = [
            {
                "name": "discharge_a",
                "steps": config["softcore"]["charge_steps_per_stage"],
            },
            {"name": "sterics_a_to_b", "steps": config["softcore"]["sterics_steps"]},
            {
                "name": "charge_b",
                "steps": config["softcore"]["charge_steps_per_stage"],
            },
        ]
    if mapping_settings and mapping_settings["method"] != "dataset_core":
        protocol["covalent_mapping"] = dict(mapping_settings)
    serialized = yaml.safe_dump(protocol, sort_keys=True)
    protocol["fingerprint"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return protocol


def _ensure_switch_protocol(workdir, config, mapping_settings=None):
    workdir = Path(workdir)
    path = workdir / "switch_protocol.yaml"
    expected = _switch_protocol(config, mapping_settings)
    existing_work = any(
        (workdir / f"{environment}_{direction}.csv").exists()
        for environment in ("protein", "reference")
        for direction in ("forward", "reverse")
    )
    if path.exists():
        observed = yaml.safe_load(path.read_text()) or {}
        if observed != expected:
            raise CovalentWorkflowError(
                "existing covalent work uses a different switching protocol; "
                "use a new workdir or restore the original workflow settings"
            )
    elif existing_work and config["interpolation"] == "softcore_linear":
        raise CovalentWorkflowError(
            "cannot resume softcore covalent work without switch_protocol.yaml; use a new workdir"
        )
    else:
        if existing_work:
            LOGGER.warning(
                "Accepting legacy %s work without protocol provenance and recording current settings",
                config["interpolation"],
            )
        _write_yaml_atomic(path, expected)
    return expected


def run_covalent_pair(settings, pair):
    workflow = settings["workflow"]
    config = _normalized_settings(workflow)
    if config["failed_switch_policy"] not in {"abort", "count_as_infinite"}:
        raise CovalentWorkflowError(
            "covalent NEQTI failed_switch_policy must be 'abort' or 'count_as_infinite'"
        )
    workroot = _resolve(workflow.get("workdir", "run"), settings["config_path"].parent)
    jobname = f"{pair['ligand_a']}--{pair['ligand_b']}"
    workdir = workroot / jobname
    workdir.mkdir(parents=True, exist_ok=True)
    mapping_settings = _mapping_settings(workflow, pair)
    switch_protocol = _ensure_switch_protocol(workdir, config, mapping_settings)
    _write_yaml_atomic(
        workdir / "result.yaml",
        {
            "schema_version": 1,
            "tool": "atom_openmm_rbfe",
            "jobname": jobname,
            "status": "running",
            "method": "neqti",
            "system_mode": "covalent",
            "ligand_a": pair["ligand_a"],
            "ligand_b": pair["ligand_b"],
            "workdir": str(workdir.resolve()),
            "progress": {"stage": "setup"},
        },
    )
    inputs = _pair_inputs(settings, pair)
    setup = workflow.get("setup") or {}
    padding = float(setup.get("solvent_padding_a", 10.0))
    ionic_strength = float(setup.get("ionic_strength_molar", 0.15))
    solvation_seed = int(setup.get("solvation_seed", config["random_seed"]))
    preparation_fingerprint, fingerprint_inputs = _preparation_fingerprint(
        settings, pair, inputs, mapping_settings, solvation_seed
    )
    prepared_manifest = workdir / "prepared" / "manifest.yaml"
    if prepared_manifest.is_file():
        protein, reference, preparation_manifest = _load_prepared_pair_bundle(
            workdir, preparation_fingerprint
        )
        mapping_payload = preparation_manifest["mapping"]
        parameterization = preparation_manifest["parameterization"]
        LOGGER.info(
            "Loaded persistent covalent preparation %s", preparation_fingerprint
        )
    else:
        if _runtime_artifacts_exist(workdir):
            raise CovalentResumeError(
                "resume_incompatible: existing covalent states or work lack the "
                "persistent prepared-system bundle; use a new workdir"
            )
        required, attachment_pairs, atom_map, mapping_provenance = (
            _prepare_covalent_atom_map(inputs, mapping_settings)
        )
        cache = workroot / "forcefield_cache"
        parameters_a = parameterize_capped_product(
            inputs["ligand_a"]["product"], cache_dir=cache
        )
        parameters_b = parameterize_capped_product(
            inputs["ligand_b"]["product"], cache_dir=cache
        )
        meta_a = inputs["ligand_a"]["metadata"]
        meta_b = inputs["ligand_b"]["metadata"]
        receptor = settings["dataset_root"] / settings["dataset"]["receptor"]
        residue_id = int(settings["dataset"]["covalent_residue"]["id"])
        backbone_charges = ff19sb_backbone_charges(receptor, residue_id)
        parameters_a = apply_modified_residue_charges(
            parameters_a, meta_a, backbone_charges
        )
        parameters_b = apply_modified_residue_charges(
            parameters_b, meta_b, backbone_charges
        )
        _validate_softcore_endpoint_charge(config, parameters_a, parameters_b)
        hybrid = build_covalent_hybrid_molecule(
            parameters_a,
            parameters_b,
            required_pairs=required,
            atom_map=atom_map,
            attachment_pairs=attachment_pairs,
            dummy_bonded_scales=DummyBondedScales(**config["dummy_bonded_scales"]),
        )
        physical_reference_a = create_solvated_capped_reference(
            parameters_a,
            padding_a=padding,
            ionic_strength_molar=ionic_strength,
            template_cache=cache / "templates.json",
            solvation_seed=solvation_seed + 1,
        )
        reference = solvate_capped_reference_hybrid(hybrid, physical_reference_a)
        protein = prepare_protein_covalent_hybrid(
            receptor,
            parameters_a,
            parameters_b,
            hybrid,
            meta_a,
            meta_b,
            residue_id=residue_id,
            padding_a=padding,
            ionic_strength_molar=ionic_strength,
            solvation_seed=solvation_seed,
        )
        _validate_prepared_endpoint_charges(protein, "protein")
        _validate_prepared_endpoint_charges(reference, "reference")
        mapping_payload = {
            "schema_version": 1,
            **mapping_provenance,
            "map_a_to_b_0based": {
                int(atom_a): int(atom_b) for atom_a, atom_b in sorted(hybrid.map_a_to_b.items())
            },
            "anchor_pairs_0based": [list(pair) for pair in hybrid.anchor_pairs],
            "attachment_pairs_0based": [list(pair) for pair in attachment_pairs],
            "unique_a_0based": list(hybrid.unique_a),
            "unique_b_0based": list(hybrid.unique_b),
            "dummy_bonded_scales": config["dummy_bonded_scales"],
            "dummy_nonbonded": "full_unique_branch_vacuum",
        }
        parameterization = {
            "ligand_a": parameters_a.provenance,
            "ligand_b": parameters_b.provenance,
            "protein": protein.provenance,
            "reference": reference.provenance,
        }
        preparation_manifest = _write_prepared_pair_bundle(
            workdir,
            protein,
            reference,
            fingerprint=preparation_fingerprint,
            fingerprint_inputs=fingerprint_inputs,
            mapping_payload=mapping_payload,
            parameterization=parameterization,
        )
        LOGGER.info(
            "Persisted covalent preparation %s", preparation_fingerprint
        )
    _validate_prepared_endpoint_charges(protein, "protein")
    _validate_prepared_endpoint_charges(reference, "reference")
    _write_yaml_atomic(workdir / "covalent_mapping.yaml", mapping_payload)
    platform, properties = _platform(workflow)
    running = yaml.safe_load((workdir / "result.yaml").read_text()) or {}
    running["progress"] = {"stage": "production", "environment": "protein"}
    _write_yaml_atomic(workdir / "result.yaml", running)
    protein_forward, protein_reverse, protein_rest2 = _run_environment(
        "protein", protein, config, workdir, platform, properties, config["random_seed"]
    )
    running["progress"] = {"stage": "production", "environment": "reference"}
    _write_yaml_atomic(workdir / "result.yaml", running)
    reference_forward, reference_reverse, reference_rest2 = _run_environment(
        "reference", reference, config, workdir, platform, properties, config["random_seed"] + 100000
    )
    work = {
        "leg_a_forward": protein_forward,
        "leg_a_reverse": protein_reverse,
        "leg_b_forward": reference_forward,
        "leg_b_reverse": reference_reverse,
    }
    analysis = analyze_two_leg_work(
        work, config["temperature_k"], config["bootstrap_samples"], config["random_seed"]
    )
    failed_switches = sum(
        int(np.count_nonzero(~np.isfinite(np.asarray(values, dtype=float))))
        for values in work.values()
    )
    result = {
        "schema_version": 1,
        "tool": "atom_openmm_rbfe",
        "jobname": jobname,
        "status": "completed" if analysis is not None else "partial",
        "method": "neqti",
        "system_mode": "covalent",
        "ligand_a": pair["ligand_a"],
        "ligand_b": pair["ligand_b"],
        "workdir": str(workdir.resolve()),
        "convention": {
            "ddg_definition": "G(ligand_b)-G(ligand_a)",
            "positive_value_meaning": "ligand_b binds weaker than ligand_a",
        },
        "experimental": {
            key: pair[key]
            for key in ("serotype", "experimental_ddg_kj_per_mol")
            if key in pair
        } or None,
        "result": None if analysis is None else {
            "ddg_kcal_per_mol": analysis["bar_dg_kcal_per_mol"],
            "ddg_error_kcal_per_mol": analysis["bar_bootstrap_std_kcal_per_mol"],
            "ddg_kj_per_mol": analysis["bar_dg_kj_per_mol"],
            "ddg_error_kj_per_mol": analysis["bar_bootstrap_std_kj_per_mol"],
            "estimator": "BAR",
            "components": _semantic_components(analysis),
            "samples": {
                "protein_forward": len(protein_forward),
                "protein_reverse": len(protein_reverse),
                "reference_forward": len(reference_forward),
                "reference_reverse": len(reference_reverse),
            },
        },
        "quality": {
            "convergence_status": "usable" if analysis and analysis["overlap_score"] >= 0.01 else "partial",
            "overlap_score": None if analysis is None else analysis["overlap_score"],
            "warnings": [],
            "rest2": {"protein": protein_rest2, "reference": reference_rest2},
        },
        "inputs": {
            "dataset": str(settings["dataset_path"]),
            "receptor": str((settings["dataset_root"] / settings["dataset"]["receptor"]).resolve()),
            "ligand_a_file": str(inputs["ligand_a"]["product"].resolve()),
            "ligand_b_file": str(inputs["ligand_b"]["product"].resolve()),
            "workflow_yaml": str(settings["config_path"]),
        },
        "parameterization": {
            **parameterization,
            "preparation_fingerprint": preparation_fingerprint,
        },
        "switching_protocol": switch_protocol,
        "performance": {
            "long_range_correction": config["softcore"]["long_range_correction"],
            "switching": _switch_timing_summary(workdir / "switch_timing.csv"),
        },
        "artifacts": {
            "covalent_mapping": "covalent_mapping.yaml",
            "switch_protocol": "switch_protocol.yaml",
            "switch_timing_csv": "switch_timing.csv",
            "prepared_manifest": "prepared/manifest.yaml",
            "protein_topology": "prepared/protein_topology.pdb",
            "protein_endpoint_a_system": "prepared/protein_endpoint_a.xml",
            "protein_endpoint_b_system": "prepared/protein_endpoint_b.xml",
            "protein_endpoint_a_state": "protein_endpoint_a_state.xml",
            "protein_endpoint_b_state": "protein_endpoint_b_state.xml",
            "protein_endpoint_a_equilibrated": "protein_endpoint_a_equilibrated.pdb",
            "protein_endpoint_b_equilibrated": "protein_endpoint_b_equilibrated.pdb",
            "reference_topology": "prepared/reference_topology.pdb",
            "reference_endpoint_a_system": "prepared/reference_endpoint_a.xml",
            "reference_endpoint_b_system": "prepared/reference_endpoint_b.xml",
            "reference_endpoint_a_state": "reference_endpoint_a_state.xml",
            "reference_endpoint_b_state": "reference_endpoint_b_state.xml",
            "reference_endpoint_a_equilibrated": "reference_endpoint_a_equilibrated.pdb",
            "reference_endpoint_b_equilibrated": "reference_endpoint_b_equilibrated.pdb",
            "protein_forward_work_csv": "protein_forward.csv",
            "protein_reverse_work_csv": "protein_reverse.csv",
            "reference_forward_work_csv": "reference_forward.csv",
            "reference_reverse_work_csv": "reference_reverse.csv",
        },
    }
    if config["softcore"]["long_range_correction"] == "endpoint_correction":
        result["artifacts"]["switch_lrc_diagnostics_csv"] = (
            "switch_lrc_diagnostics.csv"
        )
    if failed_switches:
        result["quality"]["warnings"].append(
            f"{failed_switches} numerical switches were retained as +inf work observations"
        )
    _write_yaml_atomic(workdir / "result.yaml", result)
    return {
        "jobname": jobname,
        "status": result["status"],
        "workdir": str(workdir),
        "ddg": None if analysis is None else analysis["bar_dg_kcal_per_mol"],
        "ddg_std": None if analysis is None else analysis["bar_bootstrap_std_kcal_per_mol"],
    }


def run_covalent_workflow(path):
    validate_covalent_workflow(path)
    _, settings = load_covalent_workflow(path)
    results = []
    workflow = settings["workflow"]
    workroot = _resolve(workflow.get("workdir", "run"), settings["config_path"].parent)
    for pair in settings["pairs"]:
        try:
            results.append(run_covalent_pair(settings, pair))
        except (Exception, KeyboardInterrupt) as exc:
            jobname = f"{pair['ligand_a']}--{pair['ligand_b']}"
            workdir = workroot / jobname
            workdir.mkdir(parents=True, exist_ok=True)
            result_path = workdir / "result.yaml"
            payload = (
                yaml.safe_load(result_path.read_text()) or {}
                if result_path.exists()
                else {}
            )
            payload.update(
                {
                    "schema_version": 1,
                    "tool": "atom_openmm_rbfe",
                    "jobname": jobname,
                    "status": "failed",
                    "method": "neqti",
                    "system_mode": "covalent",
                    "ligand_a": pair["ligand_a"],
                    "ligand_b": pair["ligand_b"],
                    "workdir": str(workdir.resolve()),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "stage": (payload.get("progress") or {}).get("stage", "production"),
                    },
                }
            )
            _write_yaml_atomic(result_path, payload)
            raise
    return results


def analyze_covalent_workflow(path):
    validate_covalent_workflow(path)
    _, settings = load_covalent_workflow(path)
    workflow = settings["workflow"]
    config = _normalized_settings(workflow)
    workroot = _resolve(workflow.get("workdir", "run"), settings["config_path"].parent)
    results = []
    for pair in settings["pairs"]:
        jobname = f"{pair['ligand_a']}--{pair['ligand_b']}"
        workdir = workroot / jobname
        work = {
            "leg_a_forward": _read_work(workdir / "protein_forward.csv"),
            "leg_a_reverse": _read_work(workdir / "protein_reverse.csv"),
            "leg_b_forward": _read_work(workdir / "reference_forward.csv"),
            "leg_b_reverse": _read_work(workdir / "reference_reverse.csv"),
        }
        analysis = analyze_two_leg_work(
            work, config["temperature_k"], config["bootstrap_samples"], config["random_seed"]
        )
        if analysis is None:
            raise CovalentWorkflowError(f"no finite covalent BAR estimate is available for {jobname}")
        result_path = workdir / "result.yaml"
        if result_path.exists():
            payload = yaml.safe_load(result_path.read_text()) or {}
            payload["status"] = "completed"
            payload["result"] = {
                "ddg_kcal_per_mol": analysis["bar_dg_kcal_per_mol"],
                "ddg_error_kcal_per_mol": analysis["bar_bootstrap_std_kcal_per_mol"],
                "ddg_kj_per_mol": analysis["bar_dg_kj_per_mol"],
                "ddg_error_kj_per_mol": analysis["bar_bootstrap_std_kj_per_mol"],
                "estimator": "BAR",
                "components": _semantic_components(analysis),
            }
            payload.setdefault("quality", {})["overlap_score"] = analysis["overlap_score"]
            temporary = workdir / "result.yaml.tmp"
            temporary.write_text(yaml.safe_dump(payload, sort_keys=False))
            os.replace(temporary, result_path)
        results.append({
            "jobname": jobname,
            "status": "completed",
            "workdir": str(workdir),
            "ddg": analysis["bar_dg_kcal_per_mol"],
            "ddg_std": analysis["bar_bootstrap_std_kcal_per_mol"],
        })
    return results
