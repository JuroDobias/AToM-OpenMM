"""Reference-only soft-bond path screen with a shared, held-out snapshot bank."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import logging
from pathlib import Path
import subprocess
import time

import numpy as np
import openmm as mm
from openmm import app, unit
import yaml

from atom_openmm.covalent_hybrid import (
    build_hybrid_molecule, HybridBondedScales, alchemical_bond_pair_metadata,
)
from atom_openmm.covalent_softcore import (
    EXPLICIT_ENDPOINT_A_CONTROLS, EXPLICIT_ENDPOINT_B_CONTROLS,
    create_softcore_hamiltonian, resolve_softcore_path,
)
from atom_openmm.covalent_systems import (
    write_prepared_hybrid_bundle, load_prepared_hybrid_bundle,
    solvate_capped_reference_hybrid,
)
from atom_openmm.covalent_workflow import (
    _normalized_settings, _equilibrate_endpoint, _sample_endpoint,
    _softcore_switch_context, _reset_softcore_context, _apply_state,
    _run_segmented_protocol, _is_numerical_switch_failure,
    _write_yaml_atomic, _load_state, _write_state, _validate_prepared_endpoint_charges,
)
from atom_openmm.hybrid_mapping import build_hybrid_atom_map
from atom_openmm.hybrid_workflow import _mapping_settings
from atom_openmm.hybrid_parameters import parameterize_ligand
from atom_openmm.hybrid_systems import create_physical_ligand_environment
from atom_openmm.hybrid_virtual_sites import add_alchemical_sigma_holes
from atom_openmm.neqti import (
    _allocate_segment_steps, _optimizer_cycle_scores, analyze_neqti_work,
    _bar_overlap_score,
)

LOG = logging.getLogger(__name__)
OPTIMIZER = dict(score_hysteresis_weight=0.7, score_absolute_weight=0.3,
                 score_power=1.5, score_ewma_alpha=0.3, min_segment_steps=250,
                 max_segment_steps=15000, min_update_factor=0.5, max_update_factor=2.0)


def path_variants(total_steps=50000):
    """Generate 24 paths; all corrections are exchanged with branch coupling off."""
    variants = []
    for angular, closure, bonded, mapped in itertools.product(
        ("together", "angular_first", "torsion_first"),
        ("full", "half"), ("crossfade", "overlap"), ("together", "after"),
    ):
        nodes = [dict(label="physical_a", controls=dict(EXPLICIT_ENDPOINT_A_CONTROLS))]

        def add(label, **updates):
            controls = {**nodes[-1]["controls"], **updates}
            if controls != nodes[-1]["controls"]:
                nodes.append(dict(label=label, controls=controls))

        add("decharge_a", charge_a=0)
        add("remove_a_sterics", sterics_a=0,
            soft_bond_a=0.5 if closure == "half" else 1)
        if angular == "torsion_first":
            add("release_a_torsions", soft_torsions_a=0)
        if angular != "together":
            add("release_a_angles", soft_angles_a=0, soft_torsions_a=0)
        add("open_a", soft_bond_a=0, soft_angles_a=0, soft_torsions_a=0)
        add("exchange_topology_pairs", bond_nonbonded_charge_a=1,
            bond_nonbonded_vdw_a=1, bond_one_four_charge_a=0,
            bond_one_four_vdw_a=0, bond_nonbonded_charge_b=0,
            bond_nonbonded_vdw_b=0, bond_one_four_charge_b=1,
            bond_one_four_vdw_b=1)
        if bonded == "overlap":
            add("introduce_b_regular_bonded", bonded_b=1)
        changes = dict(bonded_a=0, bonded_b=1)
        if mapped == "together":
            changes.update(mapped_charge=1, mapped_vdw=1)
        add("exchange_regular_bonded", **changes)
        if mapped == "after":
            add("exchange_mapped_nonbonded", mapped_charge=1, mapped_vdw=1)
        changes = dict(soft_bond_b=0.5 if closure == "half" else 1)
        if angular == "together":
            changes.update(soft_angles_b=1, soft_torsions_b=1)
        add("close_b", **changes)
        if angular != "together":
            add("form_b_angles", soft_angles_b=1,
                soft_torsions_b=0 if angular == "torsion_first" else 1)
        if angular == "torsion_first":
            add("form_b_torsions", soft_torsions_b=1)
        add("grow_b_sterics", sterics_b=1, soft_bond_b=1)
        add("charge_b", charge_b=1)
        for index, node in enumerate(nodes):
            node["at"] = index / (len(nodes) - 1)
        # The exclusion-only interval is energetically idle at zero branch coupling.
        # Keep it explicit and small so its settings are visible and auditable.
        weights = [0.1 if n["label"] == "exchange_topology_pairs" else 1.0
                   for n in nodes[1:]]
        cumulative = np.cumsum([0.0, *weights]) / sum(weights)
        for node, at in zip(nodes, cumulative):
            node["at"] = float(at)
        nodes[-1]["at"] = 1.0
        segments = [1 if n["label"] == "exchange_topology_pairs" else 3
                    for n in nodes[1:]]
        # At least one step is needed for the explicit exclusion-only interval.
        resolve_softcore_path(control_nodes=nodes, total_steps=total_steps,
                              segments_per_interval=segments)
        variants.append(dict(
            name=f"p{len(variants)+1:02d}_{angular}_{closure}_{bonded}_{mapped}",
            factors=dict(angular=angular, closure=closure, bonded=bonded, mapped=mapped),
            nodes=nodes, segments_per_interval=segments, total_steps=total_steps,
        ))
    return variants


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def platform():
    return mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"}


def prepare_bank(root):
    root = Path(root).resolve()
    cfg = yaml.safe_load((root / "screen.yaml").read_text())
    workflow = yaml.safe_load((root / "source_workflow.yaml").read_text())["workflow"]
    bank = root / "bank"
    bank.mkdir(exist_ok=True)
    if (bank / "complete.yaml").exists():
        LOG.info("Snapshot bank already complete")
        return
    setup = deepcopy(workflow["setup"])
    setup["protein_forcefield"] = ["amber14-all.xml"]
    setup.pop("metal_ions", None)
    settings = _normalized_settings(workflow)
    manifest_file = bank / "prepared.yaml"
    if manifest_file.exists():
        prepared = load_prepared_hybrid_bundle(bank, yaml.safe_load(manifest_file.read_text()))
    else:
        ligand_ids = workflow["pairs"][0]["ligands"]
        params = [parameterize_ligand(
            workflow["ligands"][name], ligand_forcefield=setup["ligand_forcefield"],
            ligand_charge_model=setup["ligand_charge_model"],
            ligand_parameter_cache=setup["ligand_parameter_cache"],
            ligand_parameter_protocol=setup["ligand_parameter_protocol"],
        ) for name in ligand_ids]
        atom_map, mapping = build_hybrid_atom_map(*params, _mapping_settings(workflow))
        kwargs = {}
        for end in ("a", "b"):
            for field in ("inactive_bonded_atoms", "inactive_z_matrix_root_atoms", "force_unique_atoms"):
                kwargs[f"{field}_{end}"] = set(mapping[f"{field}_{end}_0based"])
            kwargs[f"alchemical_bonds_{end}"] = {
                tuple(entry["atoms_0based"])
                for entry in mapping["alchemical_bonds"][f"ligand_{end}"]}
        hybrid = build_hybrid_molecule(
            *params, atom_map=atom_map,
            dummy_bonded_scales=HybridBondedScales(**settings["dummy_bonded_scales"]),
            dummy_core_nonbonded=settings["dummy_core_nonbonded"],
            transmuted_pairs={tuple(p) for p in mapping["transmuted_pairs_0based"]},
            inactive_bonded_geometry=mapping["inactive_bonded_geometry"], **kwargs)
        hybrid = add_alchemical_sigma_holes(hybrid, *params)
        physical = create_physical_ligand_environment(
            params[0], receptor=None, setup=setup, solvation_seed=cfg["seed"])
        prepared = solvate_capped_reference_hybrid(hybrid, physical)
        prepared.provenance["soft_bond_system_pairs"] = [
            [prepared.provenance[f"ligand_{end}_system_atom_indices"][i] for i in pair]
            for end in ("a", "b") for pair in getattr(hybrid, f"alchemical_bonds_{end}")]
        prepared.provenance["soft_bond_pair_changes"] = [
            {**entry, "system_atoms_0based": [
                prepared.provenance[f"ligand_{entry['endpoint']}_system_atom_indices"][i]
                for i in entry["atoms_0based"]]}
            for entry in alchemical_bond_pair_metadata(hybrid)]
        prepared.provenance["parameterization"] = [p.provenance for p in params]
        _validate_prepared_endpoint_charges(prepared, "solvent")
        _write_yaml_atomic(bank / "mapping.yaml", mapping)
        _write_yaml_atomic(manifest_file, write_prepared_hybrid_bundle(prepared, bank, "solvent"))
    plat, props = platform()
    eq = dict(minimization_tolerance_kj_mol_nm=10, minimization_max_iterations=2000,
              nvt_steps=100000, nvt_timestep_fs=1.0, npt_steps=500000,
              npt_timestep_fs=2.0)
    for ei, end in enumerate(("a", "b")):
        system = getattr(prepared, f"endpoint_{end}")
        initial = bank / f"equilibrated_{end}.xml"
        _equilibrate_endpoint(system, prepared.positions, initial, protocol=eq,
                              temperature_k=300, pressure_bar=1, platform=plat,
                              properties=props, seed=cfg["seed"]+ei, label=f"solvent {end}")
        files = bank / end
        files.mkdir(exist_ok=True)
        for index in range(30):
            target = files / f"snapshot_{index:03d}.xml"
            if target.exists():
                continue
            previous = initial if index == 0 else files / f"snapshot_{index-1:03d}.xml"
            # Write the next state independently; never modify a committed snapshot.
            _write_state(target.with_suffix(".input.xml"), _load_state(previous))
            state, _ = _sample_endpoint(
                system, prepared.topology, target.with_suffix(".input.xml"),
                prepared.hot_atom_indices, ensemble=end, steps=100000,
                rest2_config={"enabled": False}, output_dir=bank / "sampling",
                platform=plat, properties=props, temperature_k=300, timestep_fs=2,
                seed=cfg["seed"]+1000*ei+index+100)
            _write_state(target, state)
            with target.with_suffix(".pdb").open("w") as handle:
                app.PDBFile.writeFile(prepared.topology, state.getPositions(), handle)
            LOG.info("Bank endpoint %s: %d/30 snapshots saved", end, index+1)
    entries = {str(p.relative_to(bank)): digest(p) for p in bank.glob("*/snapshot_*.xml")
               if not p.name.endswith(".input.xml")}
    entries["prepared.yaml"] = digest(manifest_file)
    _write_yaml_atomic(bank / "complete.yaml", dict(schema_version=1, snapshots=entries,
                       optimizer_indices=list(range(10)), evaluation_indices=list(range(10, 30))))


def _audit_endpoint(prepared, hamiltonian, end, state, plat, props):
    """Compare switched physical endpoints to their prepared endpoint Hamiltonians."""
    energies = []
    forces = []
    for system in (getattr(prepared, f"endpoint_{end}"), hamiltonian.system):
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator, plat, props)
        _apply_state(context, state)
        if system is hamiltonian.system:
            for name, values in hamiltonian.parameter_values.items():
                context.setParameter(name, values[0 if end == "a" else -1])
        value = context.getState(energy=True, forces=True)
        energies.append(value.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        forces.append(value.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))
        del context, integrator
    error = abs(energies[1]-energies[0])
    force_error = float(np.max(np.abs(forces[1]-forces[0])))
    if not np.isfinite(error) or error > 0.1 or not np.isfinite(force_error) or force_error > 1.0:
        raise ValueError(f"Endpoint {end} identity failed: energy={error}, max force={force_error}")
    return dict(energy_error_kj_mol=error, max_force_error_kj_mol_nm=force_error)


def run_path(root, name):
    root = Path(root).resolve()
    cfg = yaml.safe_load((root / "screen.yaml").read_text())
    spec = next(v for v in cfg["variants"] if v["name"] == name)
    bank = root / "bank"
    manifest = yaml.safe_load((bank / "complete.yaml").read_text())
    for relative, checksum in manifest["snapshots"].items():
        if digest(bank / relative) != checksum:
            raise ValueError(f"Bank checksum mismatch: {relative}")
    prepared = load_prepared_hybrid_bundle(bank, yaml.safe_load((bank / "prepared.yaml").read_text()))
    out = root / "paths" / name
    out.mkdir(parents=True, exist_ok=True)
    record = dict(spec=spec, bank_sha256=digest(bank / "complete.yaml"), optimizer=OPTIMIZER)
    protocol = out / "protocol.yaml"
    if protocol.exists() and yaml.safe_load(protocol.read_text()) != record:
        raise ValueError("Screen protocol changed; use a new output directory")
    _write_yaml_atomic(protocol, record)
    hamiltonian = create_softcore_hamiltonian(
        prepared.endpoint_a, prepared.endpoint_b,
        prepared.provenance["unique_a_particle_indices"],
        prepared.provenance["unique_b_particle_indices"],
        function="beutler", coulomb_function="linear_pme", alpha=0.3,
        sigma_nm=0.25, power=1, use_long_range_correction=True,
        soft_bond_pairs=prepared.provenance["soft_bond_system_pairs"],
        soft_bond_pair_changes=prepared.provenance["soft_bond_pair_changes"],
        control_nodes=spec["nodes"], total_steps=spec["total_steps"],
        segments_per_interval=spec["segments_per_interval"])
    plat, props = platform()
    audit = {end: _audit_endpoint(prepared, hamiltonian, end,
             _load_state(bank / end / "snapshot_000.xml"), plat, props) for end in ("a", "b")}
    _write_yaml_atomic(out / "endpoint_audit.yaml", audit)
    state_file = out / "optimizer.yaml"
    state = yaml.safe_load(state_file.read_text()) if state_file.exists() else dict(
        completed=0, steps=list(hamiltonian.segment_steps), scores=None, history=[])

    def switch(index, end, phase, steps):
        target = out / f"{phase}_{index:03d}_{end}.yaml"
        if target.exists():
            result = yaml.safe_load(target.read_text())
            if result["forward_segment_steps"] != steps:
                raise ValueError("Saved switch allocation differs from resumed optimizer")
            return result
        context, integrator, parameters = _softcore_switch_context(
            hamiltonian, start=end, timestep_fs=2, temperature_k=300,
            platform=plat, properties=props, seed=cfg["seed"]+index+(1000 if end == "b" else 0))
        direction_steps = steps if end == "a" else list(reversed(steps))
        integrator.set_segment_steps(direction_steps)
        _reset_softcore_context(context, integrator, parameters)
        _apply_state(context, _load_state(bank / end / f"snapshot_{index:03d}.xml"))
        started = time.monotonic()
        segments = []
        completed_segments = []
        def callback(**progress):
            completed_segments.append(progress)
            _write_yaml_atomic(target.with_suffix(".progress.yaml"), progress)
            LOG.info("%s %s snapshot %d segment %d: %d/%d steps, work %.6g kJ/mol",
                     name, end, index, progress["segment"]+1,
                     progress["completed_steps"], progress["total_steps"],
                     progress["cumulative_work_kj_per_mol"])
        error = None
        try:
            work, segments = _run_segmented_protocol(integrator, direction_steps,
                                                     segment_callback=callback)
            if not np.isfinite(work):
                raise FloatingPointError("Nonfinite protocol work")
        except Exception as exc:
            if not _is_numerical_switch_failure(exc) and not isinstance(exc, FloatingPointError):
                raise
            work, error = float("inf"), str(exc)
        result = dict(work_kj_mol=float(work), segment_work=[float(v) for v in segments],
                      forward_segment_steps=steps, error=error,
                      completed_segments=completed_segments,
                      ns_day=sum(steps)*2e-6*86400/(time.monotonic()-started))
        _write_yaml_atomic(target, result)
        del context, integrator
        LOG.info("%s %s %s snapshot %d: work %s kJ/mol; %.2f ns/day",
                 name, phase, end, index, result["work_kj_mol"], result["ns_day"])
        return result

    for index in range(state["completed"], 10):
        results = [switch(index, end, "optimizer", state["steps"]) for end in ("a", "b")]
        previous = list(state["steps"])
        if all(r["error"] is None for r in results):
            scores = _optimizer_cycle_scores(
                np.array(results[0]["segment_work"])/4.184,
                np.array(results[1]["segment_work"])/4.184, OPTIMIZER)
            scores = scores if state["scores"] is None else (
                0.3*scores + 0.7*np.array(state["scores"]))
            state["steps"] = _allocate_segment_steps(scores, spec["total_steps"], previous, OPTIMIZER)
            state["scores"] = scores.tolist()
        state["history"].append(dict(cycle=index+1, previous_steps=previous,
                                     steps=state["steps"], errors=[r["error"] for r in results]))
        state["completed"] = index+1
        _write_yaml_atomic(state_file, state)
    _write_yaml_atomic(out / "frozen_schedule.yaml", dict(
        segment_steps=state["steps"], control_nodes=spec["nodes"],
        segments_per_interval=spec["segments_per_interval"], total_steps=sum(state["steps"])))
    work = {end: [] for end in ("a", "b")}
    for index in range(10, 30):
        for end in ("a", "b"):
            work[end].append(switch(index, end, "evaluation", state["steps"])["work_kj_mol"]/4.184)
        analysis = analyze_neqti_work(work["a"], work["b"], 300, 500, cfg["seed"])
        dg = None if analysis is None else analysis["bar_dg_kcal_per_mol"]
        _write_yaml_atomic(out / "result.yaml", dict(
            status="completed" if index == 29 else "running", evaluation_samples_per_direction=index-9,
            analysis=analysis, overlap=_bar_overlap_score(work["a"], work["b"], dg, 300),
            failed_samples={e: int(np.sum(~np.isfinite(v))) for e, v in work.items()},
            optimizer_failed_cycles=sum(any(h["errors"]) for h in state["history"])))


def generate(root, source):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError("Use a new screen directory")
    root.mkdir(parents=True)
    workflow = yaml.safe_load(Path(source).read_text())
    _write_yaml_atomic(root / "source_workflow.yaml", workflow)
    cfg = dict(seed=20260917, duration_ps=100, variants=path_variants())
    _write_yaml_atomic(root / "screen.yaml", cfg)
    repo = Path(__file__).resolve().parents[1]
    for name in ["bank", *[v["name"] for v in cfg["variants"]]]:
        command = f'python -m atom_openmm.soft_bond_screen {"bank" if name == "bank" else "run"} .'
        if name != "bank":
            command += f" --variant {name}"
        script = f'''#!/bin/bash
#SBATCH --job-name=sb-{name[:35]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-d
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myatom_openmm86
export PYTHONPATH="{repo}${{PYTHONPATH:+:$PYTHONPATH}}"
{command}
'''
        (root / f"{name}.slurm").write_text(script)
    (root / "README.md").write_text(
        "# cGAS ring-contraction schedule screen\n\n"
        "ms_491 -> ms_539; pyridine to thiazole, mapped coordinating nitrogen.\n"
        "Solvent only, cached RESP/sigma-hole GAFF2, TIP4P-Ew, no REST2.\n"
        "Minimization, 100 ps NVT (1 fs), 1 ns NPT (2 fs), then 30 snapshots\n"
        "per endpoint separated by 200 ps MD. Indices 0-9 optimize each path;\n"
        "indices 10-29 evaluate its frozen 100 ps schedule. 24 independent paths.\n"
        "No production samples are included in optimization. Numerical switch\n"
        "failures remain infinite work samples; failed optimizer cycles retain\n"
        "the preceding allocation. Endpoint bank failures stop preparation.\n"
        "Bank hashes and saved per-switch records protect resume and provenance.\n"
        "All jobs use gen-d with 12-hour limits; resubmit the same script to resume.\n")


def submit(root):
    root = Path(root).resolve()
    path = root / "submission.yaml"
    if path.exists():
        raise ValueError("Submission already recorded; inspect submission.yaml before resubmitting")
    config = yaml.safe_load((root / "screen.yaml").read_text())
    jobs = {}
    for name in ["bank", *[v["name"] for v in config["variants"]]]:
        command = ["sbatch", "--parsable"]
        if name != "bank":
            command.append(f"--dependency=afterok:{jobs['bank']}")
        command.append(f"{name}.slurm")
        job = subprocess.check_output(command, cwd=root, text=True).strip().split(";")[0]
        jobs[name] = job
        _write_yaml_atomic(path, jobs)
        LOG.info("Submitted %s as %s", name, job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "bank", "run", "submit"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--variant")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.action == "generate":
        generate(args.root, args.source)
    elif args.action == "bank":
        prepare_bank(args.root)
    elif args.action == "submit":
        submit(args.root)
    else:
        run_path(args.root, args.variant)


if __name__ == "__main__":
    main()
