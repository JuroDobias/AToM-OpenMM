# Relative Binding Free Energy

Relative binding free energy (RBFE) workflows estimate free energy differences between two related bound states. The `atom-rbfe` wrapper accepts one YAML file, prepares each requested ligand pair, and runs either alchemical Hamiltonian replica exchange or experimental nonequilibrium switching (NEQTI).

The most useful local examples to study are:

- `examples/RBFE/cdk2`: small-molecule ligand pairs with generated alignment atoms.
- `examples/RBFE/protein-peptide/tiam1`: protein-peptide variants where a mutation residue defines the common and variable atoms.

The CDK2 and protein-peptide workflows are the best templates for the current `setup-settings.sh` plus `defaults.yaml` pattern.

## When to Use RBFE

Use RBFE when you care about a difference between two states: ligand A versus ligand B, peptide variant A versus peptide variant B, or one molecular variant versus another. RBFE is usually the natural choice for congeneric ligand series, local chemical changes, or mutation studies where a perturbation network can be planned.

Use ABFE instead when each ligand should be treated independently or when there is no meaningful relative perturbation path.

## Small-Molecule Ligand Pairs

The CDK2 workflow is the current small-molecule RBFE template. It assumes:

- A prepared receptor PDB in `receptor/`.
- Ligand SDF files in `ligands/`.
- Docked or modeled ligand poses in the binding site.
- A reference ligand and three reference alignment atoms.
- A list of ligand pairs to process.

The key setup file is `scripts/setup-settings.sh`. For CDK2, it defines the receptor basename, the reference ligand, the reference ligand alignment atoms, and the ligand pairs:

```text
receptor=cdk2
ref_ligand=H1Q
ref_ligand_alignment_atoms="14,21,18"
ligands=( "H1Q H1R" "H1Q H1S" ... )
```

When `scripts/setup-atm.sh` runs, it first calls `scripts/find_alignment_atoms.py` to generate `ligands/alignments.yaml`. The generated alignment file lets `run-atm.py` assign `ALIGN_LIGAND1_REF_ATOMS` and `ALIGN_LIGAND2_REF_ATOMS` for each ligand pair. These atoms are used by the orientation and roll components of the ligand alignment restraint.

The CDK2 defaults use variable displacement: `ALIGN_KF_SEP` is zero, while `ALIGN_K_THETA` and `ALIGN_K_PSI` restrain relative orientation. During production, the workflow swaps the positions of the two ligands based on their current anchor-atom separation, confines the bound ligand to the binding-site region, and uses a short-range exclusion potential to keep the displaced ligand from interacting with the receptor.

## Protein-Peptide and Mutation Workflows

For peptide variants or mutation-like transformations, study `examples/RBFE/protein-peptide/tiam1`. This workflow uses the same overall launch structure as CDK2, but each entry in `scripts/setup-settings.sh` includes a mutation residue:

```text
ligands=( "sdc1E4Q sdc1wt 4"
          "sdc1wt  sdc1A8F 8" )
```

The two peptide or protein-partner PDB files are passed to `run-atm.py` as `LIG1` and `LIG2`. The `--mutationResid` value is written as `MUTATION_RESID` and used by the shared `make_pp_indexes()` helper to identify:

- The full atom lists for the two partners.
- The common and variable atoms around the mutation site.
- The attachment atoms.
- The alignment reference atoms derived from the mutation-residue backbone.

By default, the helper treats `N`, `CA`, `C`, `O`, and `H` as backbone atoms. Set `PP_BACKBONE` in `scripts/defaults.yaml` if the partner naming or chemistry requires a different backbone definition.

Use this protein-peptide pattern when the perturbation is better described by a residue or local molecular variant than by small-molecule SDF alignment. The important requirement is that the receptor and partner structures are prepared consistently and that the mutation residue can be identified in both partners.

## Running an RBFE Workflow

The single-YAML wrapper can run the small-molecule RBFE workflow from one input file:

```bash
cd $HOME/AToM-OpenMM/examples/RBFE/cdk2
atom-rbfe workflow.yaml
```

`workflow.yaml` contains the receptor, ligand directory, ligand pairs, alignment atom selection, and the original AToM options under `atom_options`. Relative input paths are resolved from the workflow YAML location. Each ligand pair is expanded into `complexes/<jobname>/`, where the wrapper writes the final per-pair `<jobname>.yaml` used by the existing `rbfe_structprep`, `rbfe_production`, and UWHAM analysis code. If `alignments_out` is set, relative output paths are written under the workflow `workdir`.

Set `prepare_only: true` under `workflow` to create the per-pair directories and final YAML files without starting production.

Preflight modes are available for workflow managers and batch systems:

```bash
atom-rbfe --validate workflow.yaml
atom-rbfe --plan-only workflow.yaml
atom-rbfe --analyze-only workflow.yaml
```

`--validate` resolves inputs and alignment settings without creating pair directories. `--plan-only` prints a YAML execution plan with resolved ligand files, expected pair workdirs, expected `result.yaml` paths, and external metadata. `--analyze-only` reuses existing pair outputs to refresh `result.yaml` without running setup or simulation.

Workflows may also use explicit ligand file mappings and opaque external metadata:

```yaml
workflow:
  ligands:
    H1Q: /abs/path/H1Q.sdf
    H1R: relative/path/H1R.sdf
  external_metadata:
    graph_id: 12
  pairs:
    - ligands: [H1Q, H1R]
      external_metadata:
        edge_id: 44
```

Pair-specific `external_metadata` is merged over workflow-level metadata and copied unchanged to the pair `result.yaml`.

Alignment atoms can be provided in three ways. Explicit `workflow.alignments` has highest priority and reads an existing alignment YAML file. Existing workflows can continue to use:

```yaml
workflow:
  reference_ligand: H1Q
  reference_alignment_atoms: [14, 21, 18]
```

For template workflows that should apply to a ligand series, use SMARTS alignment:

```yaml
workflow:
  alignment:
    method: smarts
    smarts: "[#6]-[#6]-[#6]"
    smarts_atom_ids: [1, 2, 3]
  alignments_out: alignments.yaml
```

`smarts_atom_ids` are 1-based positions inside the SMARTS match, not full ligand atom IDs. The SMARTS is matched to both ligands in each pair. If symmetry creates multiple matches, the wrapper evaluates all match combinations and picks the one with the smallest direct coordinate RMSD over the three selected atoms. No structural alignment is performed during this check; input ligand coordinates are expected to already be aligned.

Setup-time force fields can be selected in `workflow.setup`:

```yaml
workflow:
  setup:
    protein_forcefield:
      - amber14-all.xml
    solvent_forcefield:
      - amber14/tip3p.xml
    solvent_model: tip3p
    ligand_forcefield: espaloma-0.3.2
    ligand_charge_model: nn
```

`ligand_charge_model: nn` is supported for Espaloma ligand force fields. `ligand_charge_model: am1-bcc` is supported for Espaloma and is the expected GAFF setup behavior. OpenFF charge assignment is controlled by the selected OpenFF/SMIRNOFF force field and generator rather than a separate wrapper charge-model option.

Common combinations include:

```yaml
# Amber ff19SB protein, OPC water, Espaloma parameters and neural charges
setup:
  protein_forcefield: [amber19-all.xml]
  solvent_forcefield: [amber19/opc.xml]
  solvent_model: tip4pew
  ligand_forcefield: espaloma-0.3.2
  ligand_charge_model: nn
```

The accepted ligand force-field families are `gaff-*`, `openff-*`, and `espaloma-*`, subject to the versions available in the installed `openmmforcefields`. Protein and solvent values are OpenMM force-field XML files. Do not combine arbitrary protein and water XML files without checking that they are intended to be used together.

`solvent_model` is the OpenMM solvent packing model passed to `Modeller.addSolvent()`. If omitted, the wrapper infers it from `solvent_forcefield`. For example, `amber19/opc.xml` uses `solvent_model: tip4pew` for four-site water placement while parameterizing with OPC.

For phosphorylated proteins or other receptor chemistry that needs tleap-specific force fields, use AmberTools setup. Ligands can be supplied as pre-parameterized MOL2/FRCMOD files, or parameterized from SDF on the fly with antechamber:

```yaml
workflow:
  setup:
    mode: ambertools
    protein_forcefield: leaprc.protein.ff14SB
    additional_forcefields:
      - leaprc.phosaa14SB
    ligand_forcefield: leaprc.gaff2
    ligand_parameterization: antechamber
    ligand_charge_model: bcc
    ligand_net_charge: 0
    ligand_net_charges:
      charged_ligand: -1
    water_forcefield: leaprc.water.tip3p
    solvent_box: TIP3PBOX
    solvent_padding_a: 10.0
    neutralize: true
```

`ligand_net_charge` is the default integer charge passed to antechamber. `ligand_net_charges` can override it by ligand file stem.

The wrapper can also run the experimental NEQTI switching protocol instead of asynchronous replica exchange:

```yaml
workflow:
  production_method: neqti
  neqti:
    initial_equilibration_steps: 10000
    n_snapshots: 20
    decorrelation_steps: 10000
    switch_steps_per_segment: 100
    switch_integrator: custom
    validate_switch_integrator: false
    resume: true
```

NEQTI derives two half paths from the async-RE ATM schedule: A↔M+ on the positive direction and B↔M− on the negative direction. Independent M+ and M− snapshots are cross-evaluated to estimate the midpoint bridge. The final result is `DG(A→M+) + DG(M+→M−) - DG(B→M−)`.

For the example above, the execution order is:

1. Equilibrate M+ and M− independently, then A and B.
2. Collect decorrelated M+ and M− snapshots, cross-evaluate their midpoint energies, and switch them to A and B.
3. Collect decorrelated A and B snapshots and switch them to M+ and M−.
4. Estimate both half paths and the midpoint bridge with BAR, then combine them with joint bootstrap uncertainty.

| Setting | Meaning |
| --- | --- |
| `initial_equilibration_steps` | Additional MD run once for each of A, M+, B, and M−. |
| `n_snapshots` | Number of work samples in each of the four half-path directions. |
| `decorrelation_steps` | Endpoint MD between consecutive switching snapshots. |
| `switch_steps_per_segment` | Integration steps between neighboring nodes of each derived half path. |
| `switch_integrator` | `custom` (default) performs switching and work accumulation inside OpenMM; `python` retains the reference implementation. |
| `validate_switch_integrator` | Run one informational custom-versus-Python comparison in each direction from identical snapshots. |
| `resume` | Reuse completed work rows and completed custom endpoint states when available. |
| `bootstrap_samples` | Number of BAR bootstrap resamples; zero disables uncertainty estimation. |
| `random_seed` | Seed used for bootstrap resampling. |
| `platform` | Optional OpenMM platform override for NEQTI. |

Four half-path switches have approximately the same total integration length as two complete A↔B switches. Logs report effective ns/day for equilibration, decorrelation, and switching segments.

The custom switching path uses a dedicated BAOAB-style Langevin `CustomIntegrator` with the ATM parameter update centered in each timestep (`V R H O R V`). Endpoint equilibration and snapshot decorrelation continue to use the existing ATM MTS integrator. A `CompoundIntegrator` keeps both modes in one OpenMM context. The custom path executes one Python call per schedule segment rather than one call per integration step.

When `validate_switch_integrator: true`, the first pending snapshot on each half-path direction is switched with both custom and Python implementations after restoring the same state.

When `resume: true`, completed work rows and four sampling checkpoints are reused. `neqti_protocol.yaml` signs the ATM paths and segment length. Older full-path or linear artifacts are rejected.

The wrapper can replace the default equilibration stages with inline mdflow-style steps. Amber masks require `parmed`. For `async_re`, `pre_atm` replaces the physical minimization/thermalization/NPT/NVT stage and `async_re.midpoint` can replace the final lambda-0.5 equilibration. For `neqti`, `neqti.midpoint` is applied independently at M+ and M− and `neqti.endpoint` at A and B. When `neqti.midpoint` is omitted, it reuses the endpoint steps:

```yaml
workflow:
  production_method: neqti
  equilibration:
    pre_atm:
      steps:
        - id: min_heavy_restrained
          type: minimization
          tolerance_kj_mol_nm: 10.0
          max_iterations: 3000
          positional_restraints:
            mask: '!@H= & !:HOH,WAT,NA,CL,K,CA'
            k_kcal_mol_a2: 25.0
            tolerance_a: 0.01
    neqti:
      endpoint:
        steps:
          - id: endpoint_nvt
            type: md
            ensemble: NVT
            n_steps: 25000
            timestep_ps: 0.004
            thermostat:
              temperature_k: 310.0
              friction_per_ps: 1.0
            positional_restraints:
              mask: '!@H= & !:HOH,WAT,NA,CL,K,CA'
              k_kcal_mol_a2: 5.0
              tolerance_a: 0.25
```

Each custom step is either `minimization` or `md`. MD steps support `NVT` and `NPT`, `langevin_middle` or `verlet` integration, optional velocity reset, and state/XTC reporters. Positional restraints use Amber mask syntax. The two ligands are named `L1` and `L2` in the prepared system; for example, `!:L1,L2` excludes both ligands from a restraint selection. A 4 fs time step generally requires appropriate hydrogen mass repartitioning through `atom_options.HMASS`; choosing `timestep_ps: 0.004` alone does not make a system stable.

Custom equilibration writes one directory per step under `equilibration/`, including `final_state.xml`, `final_state.pdb`, optional reporter files, and `manifest.json`. NEQTI endpoint equilibration is performed independently at A and B rather than alternating between endpoints.

For the CDK2 small-molecule workflow:

```bash
cd $HOME/AToM-OpenMM/examples/RBFE/cdk2
bash ./scripts/setup-atm.sh
cd complexes

# On a SLURM cluster
for i in cdk2-*; do ( cd "$i" && sbatch ./run.sh ); done

# Or without SLURM
# for i in cdk2-*; do ( cd "$i" && bash ./run.sh > "${i}.log" 2>&1 ); done
```

For the TIAM1 protein-peptide workflow:

```bash
cd $HOME/AToM-OpenMM/examples/RBFE/protein-peptide/tiam1
bash ./scripts/setup-atm.sh
cd complexes

# On a SLURM cluster
for i in tiam1-*; do ( cd "$i" && sbatch ./run.sh ); done

# Or without SLURM
# for i in tiam1-*; do ( cd "$i" && bash ./run.sh > "${i}.log" 2>&1 ); done
```

Each generated `run.sh` calls `scripts/run-atm.py`, which loads `scripts/defaults.yaml`, adds pair-specific atom selections and displacement information, writes `<jobname>.yaml`, runs `rbfe_structprep`, runs `rbfe_production`, and performs UWHAM analysis when production data are available.

## Outputs to Check

Inside each `complexes/<jobname>/` directory, the main outputs are:

| Output | Use |
| --- | --- |
| `<jobname>.yaml` | Final merged options for this pair. |
| `<jobname>.pdb`, `<jobname>_sys.xml` | Prepared dual-topology or paired system. |
| `<jobname>_0.xml`, `<jobname>_0.pdb` | Prepared state used to start production. |
| `r*/<jobname>.out` | Perturbation-energy samples from each replica. |
| `<jobname>.log` | Runtime log and final relative free-energy estimate. |
| `<jobname>.png` | UWHAM quality-control plot when requested. |
| `vmd.in` | VMD helper file generated from the template when available. |

The final log lines report `DG` in kcal/mol, its estimated uncertainty, the two leg free energies, and the number of samples used after discarding initial samples.

For NEQTI, also inspect:

| Output | Use |
| --- | --- |
| `neqti_endpoint_A.xml`, `neqti_endpoint_B.xml` | Restartable endpoint states. |
| `neqti_midpoint_plus.xml`, `neqti_midpoint_minus.xml` | Restartable directional midpoint states. |
| `neqti_leg_*_*.csv` | Four half-path protocol-work datasets. |
| `neqti_midpoint_bridge.csv` | Midpoint cross-energy work in both directions. |
| `neqti_summary.yaml` | Component BAR estimates, overlap, combined DDG, and uncertainty. |
| `neqti_protocol.yaml` | Two-leg protocol and resume compatibility signature. |
| `*_swapped.pdb` | Diagnostic coordinates after applying the ATM virtual coordinate swap. |

The ordinary PDB contains the physical coordinates held by the OpenMM context. ATM evaluates an additional swapped coordinate state internally. NEQTI writes both forms for forward and reverse start, post-equilibration, and switching diagnostics. The existing snapshot pair, for example `neqti_forward_snapshot_4.pdb` and `neqti_forward_snapshot_4_swapped.pdb`, contains the coordinates immediately before the switch. The corresponding `neqti_forward_snapshot_4_post_switch.pdb` and `neqti_forward_snapshot_4_post_switch_swapped.pdb` files contain the coordinates immediately after the complete switch and before the sampling checkpoint is restored. Use the swapped PDBs to verify that the ligand expected in the binding site overlaps the intended reference pose at each endpoint. These files are diagnostics, not independent simulation states.

## Machine-Readable Results

Each pair directory contains `result.yaml`. The file is written atomically, so an external process may poll it while the workflow is running without observing partially serialized YAML. Input and work-directory paths are absolute; artifact paths are relative to the pair directory and are null until the corresponding file exists.

```yaml
schema_version: 1
tool: atom_openmm_rbfe
jobname: cdk2-H1Q-H1R
status: completed
method: neqti
ligand_a: H1Q
ligand_b: H1R
workdir: /abs/path/to/cdk2-H1Q-H1R
convention:
  edge_direction: ligand_a_to_ligand_b
  ddg_definition: G(ligand_b) - G(ligand_a)
  positive_value_meaning: ligand_b binds weaker than ligand_a
external_metadata: {}
result:
  ddg_kcal_per_mol: -1.23
  ddg_error_kcal_per_mol: 0.31
  ddg_kj_per_mol: -5.14632
  ddg_error_kj_per_mol: 1.29704
  estimator: BAR
  samples_forward: 50
  samples_reverse: 50
  samples_per_replica: null
quality:
  convergence_status: usable
  overlap_score: null
  cycle_closure_error: null
  warnings: []
error: null
inputs:
  receptor: /abs/path/receptor.pdb
  ligand_a_file: /abs/path/H1Q.sdf
  ligand_b_file: /abs/path/H1R.sdf
  workflow_yaml: /abs/path/workflow.yaml
  final_pair_yaml: /abs/path/cdk2-H1Q-H1R.yaml
artifacts:
  prepared_complex: cdk2-H1Q-H1R.pdb
  equilibrated_complex: cdk2-H1Q-H1R_equil.pdb
  endpoint_a: neqti_endpoint_A.pdb
  endpoint_b: neqti_endpoint_B.pdb
  endpoint_a_swapped: neqti_endpoint_A_swapped.pdb
  endpoint_b_swapped: neqti_endpoint_B_swapped.pdb
  midpoint_plus: neqti_midpoint_plus.pdb
  midpoint_minus: neqti_midpoint_minus.pdb
  leg_a_forward_work_csv: neqti_leg_a_forward.csv
  leg_a_reverse_work_csv: neqti_leg_a_reverse.csv
  leg_b_forward_work_csv: neqti_leg_b_forward.csv
  leg_b_reverse_work_csv: neqti_leg_b_reverse.csv
  midpoint_bridge_csv: neqti_midpoint_bridge.csv
  neqti_summary: neqti_summary.yaml
  neqti_protocol: neqti_protocol.yaml
  neqti_switch_validation: null
  async_re_log: null
  async_re_replica_output_pattern: null
  plot: null
progress:
  stage: production
  current_pair_index: 1
  total_pairs: 1
  forward_samples: 50
  reverse_samples: 50
  target_forward_samples: 50
  target_reverse_samples: 50
  last_update: '2026-07-09T12:00:00Z'
```

For asynchronous replica exchange, `estimator` is `UWHAM`, `samples_per_replica` is populated, and the forward/reverse sample fields are null. For NEQTI, `estimator` is `BAR`, the forward/reverse fields are populated, and `samples_per_replica` is null.

The top-level status has the following meaning:

| Status | Meaning |
| --- | --- |
| `prepared` | System preparation completed but production was not requested. |
| `running` | Setup, equilibration, production, or analysis is active. |
| `partial` | A finite estimate may exist, but requested sampling is incomplete. |
| `completed` | Requested sampling completed. |
| `failed` | The workflow raised an exception; `error` records its type, message, and stage. |

`quality.convergence_status: usable` means only that the run completed its requested sample count and produced a finite estimate. It is not a scientific convergence guarantee. Quantitative overlap and network cycle-closure analysis are not implemented in schema version 1 and remain null.

## Current Limitations

- The single-YAML wrapper currently supports `workflow.mode: small_molecule`.
- NEQTI is experimental and has not replaced asynchronous replica exchange as the established production method.
- NEQTI currently uses the configured discrete ATM schedule as interpolation knots; it does not yet implement an arbitrary continuous OpenMMTools-style alchemical function.
- GPU, CUDA, OpenMM, Espaloma, and `openmmforcefields` compatibility is the responsibility of the environment.
- A numerically completed run is not sufficient validation. Inspect endpoint structures, swapped structures, work distributions, forward/reverse overlap, and sensitivity to equilibration and switching time.

## Planning and Adapting

For small-molecule RBFE, plan the perturbation network before setup. Choose ligand pairs that are scientifically meaningful and that have reliable bound poses. The CDK2 workflow can generate alignment atoms from a reference ligand, but you should still inspect the chosen atoms and the resulting structures. Poor alignment atoms can lead to unstable restraints or unhelpful ligand orientations.

For mutation or protein-peptide RBFE, make sure the partner structures use consistent residue numbering and atom naming. The mutation residue should identify the local change cleanly in both partners. If the default backbone definition does not match the input structures, set `PP_BACKBONE` explicitly.

For either RBFE type, tune these files first:

| File | What to adjust |
| --- | --- |
| `scripts/setup-settings.sh` | Receptor name, ligand pairs, peptide pairs, mutation residues, and reference alignment settings. |
| `scripts/defaults.yaml` | Alchemical schedule, force field, receptor chains, restraints, displacement behavior, and production length. |
| `scripts/run_template.sh` | Conda environment activation, SLURM resources, time limit, and command-line options passed to `run-atm.py`. |

Tutorial defaults are intentionally short. For production work, increase `MAX_SAMPLES`, increase `WALL_TIME`, keep scheduler limits consistent, and compare quality-control plots and log summaries across the perturbation network before interpreting final rankings.
