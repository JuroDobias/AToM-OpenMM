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

Every workflow selects independent chemistry, alchemy, cycle, and sampling axes:

```yaml
workflow:
  type: rbfe
  chemistry: noncovalent
  alchemy:
    model: atm
    cycle: transfer
  sampling:
    method: neqti
```

ATM transfer supports `async_re`, `neqti`, and `awh`. Noncovalent and covalent
`hybrid_topology` currently require the `complex_solvent` cycle and `neqti`.
Unsupported combinations are rejected before preparation.

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
    structures:
      ligand_a: alignment/ligand_a.sdf
      ligand_b: alignment/ligand_b.sdf
  alignments_out: alignments.yaml
```

`smarts_atom_ids` are 1-based positions inside the SMARTS match, not full ligand atom IDs. The SMARTS is matched to both ligands in each pair. If symmetry creates multiple matches, the wrapper evaluates all match combinations and picks the one with the smallest direct coordinate RMSD over the three selected atoms. No structural alignment is performed during this check; input ligand coordinates are expected to already be aligned.

The optional `structures` mapping selects coordinate files used only for SMARTS matching. This is useful when simulation inputs are parameterized MOL2 files that a chemistry toolkit cannot parse directly; force-field setup still uses the ligand files listed under `pairs`. Mapping keys are ligand names without their filename suffixes.

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

Hybrid-topology workflows can load XML resources distributed by
`openmmforcefields` without using environment-specific absolute paths. Prefix the
path below that package's `ffxml` directory with `openmmforcefields:`. For
example, the compatible ff14SB, phosphorylated-amino-acid, and TIP3P stack is:

```yaml
setup:
  protein_forcefield:
    - openmmforcefields:amber/ff14SB.xml
    - openmmforcefields:amber/phosaa14SB.xml
  solvent_forcefield:
    - openmmforcefields:amber/tip3p_standard.xml
```

Do not combine `phosaa14SB.xml` with OpenMM's namespaced
`amber14-all.xml`; their atom-type namespaces are incompatible. The hybrid
builder also restores bonds for exact supplemental residue templates, such as
TPO, when OpenMM's PDB reader does not recognize the modified residue.

`solvent_box_shape` controls packing for hybrid-topology environments. It can
be `cube` (the default), `dodecahedron`, `octahedron`, or `rectangular`. The
rectangular mode applies `solvent_padding_a` independently to each coordinate
extent and is closest to tleap's rectangular solvent-box behavior.

Hybrid workflows reject undefined ligand stereochemistry by default. For a
source dataset that intentionally leaves a stereocenter unspecified, set
`setup.allow_undefined_stereo: true`. This is an explicit opt-in and does not
assign a configuration; the supplied conformer is retained.

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
  chemistry: noncovalent
  alchemy:
    model: atm
    cycle: transfer
  sampling:
    method: neqti
  neqti:
    initial_equilibration_steps: 10000
    n_snapshots: 20
    decorrelation_steps: 10000
    switch_steps_per_segment: 100
    work_sample_intervals: [1, 5, 10, 25, 50]
    switch_integrator: custom
    validate_switch_integrator: false
    failed_switch_policy: count_as_infinite
    rest2:
      enabled: true
      solute: '#ligand:"*"'
      effective_temperatures_k: [300, 351, 411, 481, 563, 658, 770, 900]
      exchange_interval_steps: 500
      checkpoint_interval_cycles: 10
    resume: true
```

NEQTI derives two bidirectional half paths from the async-RE ATM schedule, A↔M and B↔M, and samples one shared midpoint coordinate ensemble. The final result is `DG(A->M) - DG(B->M)`.

For the example above, the execution order is:

1. Optionally anneal the physical pre-ATM state to the shared midpoint M, then equilibrate M.
2. Optionally anneal the equilibrated M state to A and B, then equilibrate A and B.
3. Collect decorrelated M snapshots and switch copies of those snapshots to A and B.
4. Collect decorrelated A and B snapshots and switch copies back to M.
5. Estimate `DG(A->M)` and `DG(B->M)` with BAR, then combine them as `DG(A->M) - DG(B->M)`.

| Setting | Meaning |
| --- | --- |
| `initial_equilibration_steps` | Additional MD run once for each sampling ensemble: M, A, and B. |
| `n_snapshots` | Number of work samples in each of the four half-path directions. |
| `decorrelation_steps` | Endpoint MD between consecutive switching snapshots. |
| `switch_steps_per_segment` | Integration steps between neighboring nodes of each derived half path. |
| `work_sample_intervals` | Diagnostic work quadrature intervals evaluated from the same trajectory. Exact per-step work remains canonical. |
| `preparation_annealing_steps_per_segment` | Optional switching steps per schedule segment for pre-ATM->M, M->A, and M->B preparation annealing. |
| `sampling_order` | `interleaved` runs four directions by snapshot cycle; `batched` completes midpoint directions before endpoint directions. |
| `endpoint_system` | `atm` (default) samples endpoints with ATM; experimental `native` samples physical dual-ligand A and B systems without `ATMForce`. |
| `switch_integrator` | `custom` (default) performs switching and work accumulation inside OpenMM; `python` retains the reference implementation. |
| `validate_switch_integrator` | Run one informational custom-versus-Python comparison in each direction from identical snapshots. |
| `failed_switch_policy` | `abort` stops on a failed switch, `retry` excludes it and draws a replacement snapshot, and `count_as_infinite` includes a recognized numerical failure as `+inf` work without replacement. |
| `tolerate_failed_switches` | Legacy compatibility option: `false` maps to `abort` and `true` maps to `retry` when `failed_switch_policy` is absent. |
| `max_switch_attempts_per_direction` | Maximum attempts used to obtain `n_snapshots` complete work samples per direction. |
| `resume` | Reuse completed work rows and completed custom endpoint states when available. |
| `bootstrap_samples` | Number of BAR bootstrap resamples; zero disables uncertainty estimation. |
| `random_seed` | Seed used for bootstrap resampling. |
| `platform` | Optional OpenMM platform override for NEQTI. |
| `schedule_optimization` | Optional excluded pilot that redistributes a fixed total switch time among ATM schedule segments independently for the A and B legs. |
| `convergence` | Optional per-cycle BAR overlap, bootstrap uncertainty, and estimate-stability stopping criteria. |

When `rest2.enabled: true`, the additional NEQTI sampling at A, M, and B uses a synchronous REST2 ladder instead of ordinary MD. `initial_equilibration_steps` and `decorrelation_steps` are steps per replica and must be divisible by `rest2.exchange_interval_steps`. All replicas use the physical thermostat temperature; the effective temperatures define REST2 Hamiltonian scales. `solute` accepts the selection syntax described below. Legacy `both_ligands` remains accepted and is equivalent to `'#ligand:"*"'`.

Native endpoint REST2 is enabled explicitly:

```yaml
workflow:
  neqti:
    endpoint_system: native
    sampling_order: interleaved
    preparation_annealing_steps_per_segment: 10000
    rest2:
      enabled: true
      ensembles: [a, b]
      solute: '#unbound:"*"'
      effective_temperatures_k: [300, 357, 424, 505, 600]
      exchange_interval_steps: 500
    schedule_optimization:
      enabled: true
      pilot_samples: 10
      min_segment_steps: 1000
      max_segment_steps: 15000
    convergence:
      enabled: true
      min_samples_per_direction: 30
      min_overlap_score_per_leg: 0.05
      max_ddg_error_kcal_per_mol: 0.5
      consecutive_checks: 3
      max_ddg_range_kcal_per_mol: 0.25
```

This mode retains the current convention: A has L1 bound and L2 unbound; B has L2 bound and L1 unbound. M uses ordinary ATM MD. After ATM M-to-A/B preparation annealing, endpoint coordinates are converted to role-aware physical systems. Site, orientation, alignment, and optional receptor-exclusion restraints follow the bound/unbound roles. Endpoint equilibration and REST2 run without `ATMForce`. Native positions are converted back to the canonical ATM representation only for A/B-to-M switching; box vectors and per-atom velocities are preserved.

Native endpoint mode requires REST2, `rest2.ensembles: [a, b]`, and positive preparation annealing. Interleaved mode keeps both endpoint ladders resident for the full run. With five REST2 replicas this uses ten endpoint contexts plus one ATM context, so it is intended for high-memory GPUs such as the L40S. Batched mode keeps only one endpoint ladder resident and remains the lower-memory alternative.

The physical `s=1` replica supplies positions, velocities, and box vectors for each NEQTI switch. Switching itself always uses `s=1`, so work values and BAR analysis retain the standard ATM Hamiltonian. Legacy ATM endpoint REST2 requires interleaved sampling. Native endpoint REST2 supports interleaved and batched sampling. Both require the common/variable-region ATM coordinate-swap setup.

Adaptive scheduling runs the configured number of complete four-direction pilot cycles before production. Pilot work is written under `neqti_schedule_optimization/` and is never included in production BAR estimates. A and B segment scores combine forward/reverse hysteresis and absolute segment work, are smoothed across pilot cycles, and redistribute the fixed total switch steps within configured bounds. The final schedules are frozen in `neqti_schedule_optimization.yaml` and restored exactly on resume.

With convergence stopping enabled, BAR is recomputed after each complete production cycle once every direction reaches `min_samples_per_direction`. A run stops as `completed` with `termination_reason: converged` only after all overlap and bootstrap-error thresholds pass for the requested number of consecutive checks and the DDG range is stable. Reaching `n_snapshots` first records `termination_reason: max_samples` and leaves the run `partial`. State is stored atomically in `neqti_convergence.yaml`.

Four half-path switches have approximately the same total integration length as two complete A↔B switches. Logs report effective ns/day for equilibration, decorrelation, and switching segments.

## Conventional hybrid-topology NEQTI

Set `alchemy.model: hybrid_topology` and `alchemy.cycle: complex_solvent` to
run conventional dual-topology FEP in separate complex and solvent boxes. The
same mapped hybrid molecule is used in both environments. Inactive unique
branches do not interact with the environment, but retain their complete
intramolecular vacuum interactions. The reported result is
`DG_complex(A->B) - DG_solvent(A->B)`.

Mapping uses either an automatic connected MCS or a SMARTS-constrained MCS:

```yaml
workflow:
  chemistry: noncovalent
  alchemy:
    model: hybrid_topology
    cycle: complex_solvent
    mapping:
      method: mcs_core_smarts
      smarts: "Nc(nc1O)nc2c1ncn2"
      # Optional hard validation; coordinates are never fitted automatically.
      max_mapped_rmsd_a: 1.0
  sampling:
    method: neqti
```

Input ligands must already share the intended pose. Common atoms use ligand A
coordinates and unique ligand B atoms retain their input coordinates. The
selected mapping and direct mapped-atom RMSD are written to
`hybrid_mapping.yaml`. This initial implementation requires equal ligand formal
charges, Espaloma parameters with NN charges, and `softcore_linear` switching.
See `examples/RBFE/cdk2/workflow.hybrid.yaml` for a complete input.

Hybrid NEQTI can select switching duration independently in the complex and
solvent environments before production:

```yaml
workflow:
  neqti:
    n_snapshots: 100
    adaptive_switching:
      enabled: true
      candidate_times_ps: [100, 300, 1000]
      pilot_samples_per_direction: 20
      min_overlap_score_per_leg: 0.08
      max_failed_fraction_per_direction: 0.05
      reuse_selected_pilot_samples: true
      on_exhausted: use_longest
    convergence:
      enabled: true
      min_samples_per_direction: 30
      min_overlap_score_per_leg: 0.05
      max_ddg_error_kcal_per_mol: 0.5
      consecutive_checks: 3
      max_ddg_range_kcal_per_mol: 0.25
```

All candidate durations replay the same bank of endpoint snapshots. Rejected
candidate work remains diagnostic, while work from the selected duration is
promoted to the production CSVs and counts toward `n_snapshots`. Longer
durations preserve the relative allocation of the base softcore schedule. If
schedule optimization is enabled, its frozen allocation is scaled instead.
Selection state and candidate statistics are stored in
`neqti_adaptive_switching.yaml`.

After both environments have selected durations, convergence uses matched
complex and solvent prefixes. The first check is made at the configured minimum
sample count. Reaching `n_snapshots` without satisfying all overlap,
uncertainty, and stability criteria records `max_samples` and a partial result.
Because selection and estimation reuse the same pilot work, `result.yaml`
records this provenance and emits a quality warning.

## ATM-AWH with endpoint REST2

Set `workflow.sampling.method: awh` with `alchemy.model: atm` to run one expanded-ensemble walker over
the complete ATM schedule. Endpoint REST2 states extend the linear graph:

```text
A_hot ... A_physical - ATM states ... M+ - M- ... B_physical ... B_hot
```

The thermostat remains at the physical temperature. Effective REST2
temperatures define Hamiltonian scales, and the desired result is the free
energy difference between `A_physical` and `B_physical`. REST2 endpoint
selectors use the existing role-aware syntax. The default `#unbound:"*"`
therefore selects ligand B on the A branch and ligand A on the B branch.

```yaml
workflow:
  chemistry: noncovalent
  alchemy:
    model: atm
    cycle: transfer
  sampling:
    method: awh
  awh:
    state_move_interval_steps: 500
    state_sampling:
      method: hybrid_global_gibbs
      validation_interval_moves: 1000
      validation_states_per_check: 4
      validation_tolerance_kj_per_mol: 0.05
      direct_overflow_probability_tolerance: 1.0e-12
    # Optional interpolation of the original 22-state schedule.
    # atm_state_count: 100
    start_state: a
    start_annealing_steps_per_state: 10000
    initial_error_kj_per_mol: 50.0
    diffusion_per_ps: 0.005
    target_distribution: uniform
    adaptive:
      min_steps: 1000000
      max_steps: 20000000
      min_round_trips: 10
      min_visits_per_state: 100
      covering_fraction: 0.8
      learning_rate_kbt: 0.1
      refinement:
        enabled: true
        occupancy_half_life_moves: 5000
        stages:
          - {learning_rate_kbt: 0.1, min_steps: 500000, min_round_trips: 2, min_visits_per_state: 10}
          - {learning_rate_kbt: 0.05, min_steps: 500000, min_round_trips: 2, min_visits_per_state: 20, min_recent_target_overlap: 0.60, rest2_hot_fraction_tolerance: 0.10}
          - {learning_rate_kbt: 0.025, min_steps: 500000, min_round_trips: 3, min_visits_per_state: 30, min_recent_target_overlap: 0.70, rest2_hot_fraction_tolerance: 0.08}
          - {learning_rate_kbt: 0.0125, min_steps: 500000, min_round_trips: 3, min_visits_per_state: 50, min_recent_target_overlap: 0.75, rest2_hot_fraction_tolerance: 0.06}
          - {learning_rate_kbt: 0.005, min_steps: 500000, min_round_trips: 4, min_visits_per_state: 75, min_recent_target_overlap: 0.80, rest2_hot_fraction_tolerance: 0.05, min_endpoint_target_fraction: 0.5}
          - {learning_rate_kbt: 0.002, min_steps: 500000, min_round_trips: 4, min_visits_per_state: 100, min_recent_target_overlap: 0.80, rest2_hot_fraction_tolerance: 0.04, min_endpoint_target_fraction: 0.5}
      frozen_validation:
        min_steps: 2000000
        max_steps: 5000000
        burn_in_steps: 500000
        min_round_trips: 4
        min_visits_per_state: 100
        min_target_occupancy_overlap: 0.8
        rest2_hot_fraction_tolerance: 0.04
        min_endpoint_target_fraction: 0.5
      metric_target:
        enabled: false
        min_round_trips: 2
        update_interval_moves: 1000
        smoothing: 0.2
        max_relative_weight: 5.0
    production:
      steps: 5000000
      reduced_energy_interval_moves: 10
      bootstrap_samples: 500
    rest2:
      enabled: true
      effective_temperatures_k: [300, 344.6, 395.9, 454.7, 522.3, 600]
      endpoint_a_solute: '#unbound:"*"'
      endpoint_b_solute: '#unbound:"*"'
    analysis:
      trajectory:
        enabled: true
        interval_moves: 10
        atom_selection: '!:HOH,WAT,NA,CL,K,CA'
      friction:
        enabled: true
        max_correlation_lag_moves: 50
        min_effective_samples: 200
      thresholds:
        min_adjacent_overlap: 0.03
        min_endpoint_effective_samples: 50
        min_rest2_hot_returns: 5
        min_uniform_occupancy_overlap: 0.8
    resume: true
```

With `adaptive.refinement.enabled: true`, bias learning proceeds through the
strictly decreasing learning rates in `refinement.stages`. Each stage has
independent criteria, allowing aggressive early discovery and progressively
stricter recent-occupancy, REST2-group, and physical-endpoint checks. The
legacy `learning_rates_kbt` form with shared criteria remains supported, but it
cannot be mixed with `stages`.

After the final refinement stage, the bias is frozen for validation.
Production begins only after the fixed-bias trajectory passes its own
round-trip, per-state visit, and uniform-occupancy-overlap criteria. If
`frozen_validation.max_steps` is reached first, the workflow returns to the
smallest learning rate and tries validation again. Reaching
`adaptive.max_steps` before validation passes produces a partial result.
Stage transitions and their metrics are stored in
`awh_summary.yaml:adaptation.history`.

Frozen validation records a separate reduced-energy matrix after
`burn_in_steps`. A failed validation attempt discards that matrix. When
validation passes, its post-burn-in rows are combined with production for the
primary UWHAM estimate. The summary also reports validation-only and
production-only estimates so drift between the two fixed-bias phases remains
visible.

When refinement is disabled, the legacy adaptive stage uses the fixed
dimensionless update size set by `adaptive.learning_rate_kbt` and the original
top-level convergence criteria. The bias is then frozen directly for
production. In both modes, the fixed-bias production stage records sparse
complete reduced-energy matrices and reports UWHAM as the primary estimator.
The adaptive AWH bias estimate remains available under
`result.estimator_variants`.

`initial_error_kj_per_mol` and `diffusion_per_ps` remain accepted for
checkpoint and input compatibility, but they do not affect this fixed-rate
update.

`atm_state_count` can interpolate the two original ATM half paths while
preserving separate M+ and M- nodes. The default
`state_sampling.method: legacy_local_gibbs` evaluates the current and
neighboring Hamiltonians and preserves existing checkpoints. A nearest-neighbor
random walk requires approximately the square of the number of nodes to
traverse the graph.

The experimental `hybrid_global_gibbs` method instead samples once from all
graph states after every MD block. It obtains one physical ATM decomposition,
reconstructs all physical ATM energies analytically, and scans only the REST2
hot endpoint states with a device-side OpenMM integrator. The resulting full
probability vector updates the adaptive bias and is reused directly for
production UWHAM/MBAR rows. This avoids one host-driven force evaluation per
ATM state. An initial exhaustive comparison and periodic random direct checks
must remain within `validation_tolerance_kj_per_mol`; a mismatch stops the run
rather than silently using approximate energies.

Hybrid global-Gibbs diagnostics in `awh_summary.yaml` include selected jump
distances, conditional expected jump distance, probability mass captured by
local radii 1, 2, 4, 8, and 16, energy-scan and MD timing, and the largest
validation error. These data show whether a cheaper truncated Gibbs scan would
be justified on a later implementation. For approximately 100 ATM states,
consider a shorter interval such as `state_move_interval_steps: 100` and retain
enough adaptive steps for the configured round trips.

For configurations that are grossly incompatible with a remote ATM state, the
direct plugin expression can overflow while the stable analytical softplus
still returns a finite energy. Validation accepts this only when the
reconstructed global-Gibbs probability is no greater than
`direct_overflow_probability_tolerance`. Such a state cannot be selected at
that configuration. A nonfinite direct energy for a state with appreciable
probability remains fatal.

`awh_protocol.yaml` fingerprints the ATM schedule, REST2 selections, ladder,
and AWH settings. Resume restores coordinates, velocities, RNG state, current
node, free-energy estimate, reference histogram, visits, transitions, and
round trips. Incompatible artifacts are rejected.

For an independent B-start pilot, set `start_state: b`. Unless
`initial_state_file` points to an already equilibrated B state, the workflow
prepares B by traversing the configured ATM schedule with
`start_annealing_steps_per_state` MD steps at each node. The resulting state is
saved as `awh_start_B.xml`. This preparation is outside AWH statistics.

Important artifacts are `awh_state_trace.csv`, `awh_bias_history.csv`,
`awh_validation_reduced_energies.csv`, `awh_reduced_energies.csv`,
`awh_checkpoint.yaml`, and `awh_summary.yaml`.
Version 1 is a single walker. Shared-bias multiple walkers and overlapping
endpoint REST2 regions are not implemented. Independent A-start and B-start
calculations should agree before using the method for a benchmark series.

Friction analysis estimates the generalized force along the ordered state
graph using centered finite differences of neighboring reduced energies. It
accumulates probability-weighted force variance and autocorrelation statistics
for the complete endpoint-REST2 and ATM graph. `awh_friction_samples.csv`
contains the raw observations and `awh_friction.yaml` reports effective sample
counts, integrated autocorrelation times, friction, and a suggested target
proportional to the square root of friction. The units refer to graph-index
spacing rather than a Cartesian coordinate or the original ATM lambda.

Friction collection is analysis-only and can be enabled on a compatible
checkpoint. Setting `adaptive.metric_target.enabled: true` changes the dynamics
signature and requires a fresh run. Once every state reaches
`friction.min_effective_samples` and the configured round trips have completed,
the target is periodically moved toward the friction suggestion. Smoothing and
`max_relative_weight` limit noisy early changes. Keep this disabled until a
system's friction profile is reproducible.

By default, every expanded AWH state has equal target probability. This means
that reducing `atm_state_count` increases the aggregate probability assigned to
the fixed-size endpoint REST2 ladders. Use a grouped target to keep the total
hot-REST2 probability independent of ATM path resolution:

```yaml
awh:
  target_distribution: grouped
  target:
    rest2_hot_fraction: 0.1666666667
```

The configured fraction is divided equally over all auxiliary hot REST2 states;
the remainder is divided over the physical ATM path, including both physical
endpoints. Frozen validation then compares observed occupancy with this
configured target. `frozen_validation.min_target_occupancy_overlap` is the
preferred spelling for its threshold; the legacy
`min_uniform_occupancy_overlap` spelling remains accepted. Grouped targets
cannot currently be combined with friction-driven metric target adaptation.

`awh_diagnostics.yaml` reports adaptive and fixed-bias occupancy, observed and
expected neighboring transition probabilities, complete REST2
physical-hottest-physical returns, UWHAM overlap, and effective sample counts.
`awh_diagnostics.png` visualizes state visits, transition traffic, neighboring
overlap, effective samples, generalized-force friction, and active and
suggested targets. Failure of a configured quality threshold marks the result
`partial` and adds a specific warning to `result.yaml`.

When trajectory output is enabled, `awh_trajectory.xtc` contains the selected
atoms and `awh_trajectory_topology.pdb` is its matching topology.
`awh_trajectory_frames.csv` identifies the AWH stage, ATM state, REST2 region,
scale, and effective temperature for every frame. A frame is tagged with the
Hamiltonian that propagated it, before the following state draw. Adaptive
frames are useful for diagnosing state-space traversal but must not be used as
fixed-bias equilibrium populations. With the default 500-step state interval
and 2 fs timestep, `interval_moves: 10` saves one frame every 10 ps.

`atom-rbfe --analyze-only workflow.yaml` rebuilds the AWH summary and
diagnostics from the trace, reduced-energy matrix, bias history, and
checkpoint. Legacy runs without the extended trace or trajectory remain
analyzable, but unavailable stage-specific or structural diagnostics are
reported rather than reconstructed.

The custom switching path uses a dedicated BAOAB-style Langevin `CustomIntegrator` with the ATM parameter update centered in each timestep (`V R H O R V`). Endpoint equilibration and snapshot decorrelation continue to use the existing ATM MTS integrator. A `CompoundIntegrator` keeps both modes in one OpenMM context. The custom path executes one Python call per schedule segment rather than one call per integration step.

`work_sample_intervals` adds diagnostic left-point quadrature estimates without changing the switching trajectory or removing the exact per-step energy evaluations. Interval 1 must match exact work. Each configured interval gets its own two-leg BAR estimate in `neqti_summary.yaml` and `result.result.estimator_variants`, including the DDG difference from exact and a paired-bootstrap uncertainty for that difference. These diagnostics do not improve switching speed.

When `validate_switch_integrator: true`, the first pending snapshot on each half-path direction is switched with both custom and Python implementations after restoring the same state.

When `resume: true`, completed work rows and three sampling checkpoints are reused. REST2 runs instead maintain complete replica checkpoint banks under `neqti_rest2/{a,m,b}`. Adaptive pilot cycles and convergence checks are also resumed atomically. `neqti_protocol.yaml` signs the ATM paths, segment lengths, sampling order, REST2 ladder settings, optimizer, and convergence configuration. Incompatible resume settings are rejected.

`count_as_infinite` is intended for production protocols where a physical numerical instability is itself a zero-overlap outcome that must not be replaced selectively. It recognizes non-finite work, OpenMM NaN/non-finite state errors, and constraint convergence failures. Environment, CUDA/PTX, parameter-name, random-seed, I/O, and programming failures remain fatal. Counted rows use `status: counted_infinite` and `work_kcal_per_mol: inf`; ordinary failed rows from older runs remain excluded on resume. Summaries and `result.yaml` report total analyzed, finite, counted-infinite, and retryable failed counts separately. BAR returns no finite estimate when a required direction has no finite connecting sample.

The wrapper can replace the default equilibration stages with inline
mdflow-style steps. Amber masks require `parmed`. For `async_re`, `pre_atm`
replaces the physical minimization/thermalization/NPT/NVT stage and
`async_re.midpoint` can replace the final lambda-0.5 equilibration. AWH also
uses `pre_atm`, then starts its state-space adaptation from the resulting
physical endpoint. For `neqti`, `neqti.midpoint` is applied at the shared M
ensemble and `neqti.endpoint` at A and B. When `neqti.midpoint` is omitted, it
reuses the endpoint steps:

```yaml
workflow:
  chemistry: noncovalent
  alchemy:
    model: atm
    cycle: transfer
  sampling:
    method: neqti
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

Each custom step is either `minimization` or `md`. MD steps support `NVT` and
`NPT`, `langevin_middle` or `verlet` integration, optional velocity reset, and
state/XTC reporters. For Langevin heating, set
`thermostat.initial_temperature_k`, the final `temperature_k`, and optionally
`temperature_update_interval_steps`; the thermostat temperature is changed
linearly over the step. Positional restraints use Amber mask syntax. The two
ligands are named `L1` and `L2` in the prepared system; for example,
`!:L1,L2` excludes both ligands from a restraint selection. A 4 fs time step
generally requires appropriate hydrogen mass repartitioning through
`atom_options.HMASS`; choosing `timestep_ps: 0.004` alone does not make a
system stable.

Amber expressions can contain quoted, role-aware SMARTS leaves. Quote the complete YAML value with single quotes so `#` is not parsed as a YAML comment:

```yaml
positional_restraints:
  mask: '(!:HOH,WAT & #bound:"c1ncnc2ncnc12") | @CA'
rest2:
  solute: '#unbound:"*"'
```

`ligand_a` and `ligand_b` mean the canonical first and second input ligands. `bound` and `unbound` resolve by physical endpoint: A binds ligand A and B binds ligand B. `ligand` is the union of both ligand identities, so `'#ligand:"*"'` selects both complete copies. SMARTS is matched independently to the relevant canonical ligand graph, all symmetry matches are unioned, and the result is converted to explicit 1-based Amber atom IDs before ParmEd evaluates the complete mask. Pre-ATM equilibration uses endpoint A roles. Endpoint A/B equilibration and native endpoint REST2 use their respective roles. `bound` and `unbound` are invalid at the shared midpoint and in shared-ATM REST2 because no single role assignment exists there.

The prepared pair stores normalized ligand graph files and canonical system-atom mappings. An older prepared pair can continue using plain Amber masks or legacy `both_ligands`; using SMARTS roles requires re-preparation. Partial-ligand SMARTS hot regions are experimental: bonded terms crossing the hot/cold boundary use square-root REST2 scaling, so validate the chosen region and replica ladder for the system.

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
| `neqti_midpoint.xml` | Restartable shared midpoint state. |
| `neqti_leg_*_*.csv` | Four half-path protocol-work datasets, including configured diagnostic work columns. |
| `neqti_summary.yaml` | Exact and diagnostic component BAR estimates, overlap, combined DDG, and uncertainty. |
| `neqti_protocol.yaml` | Single-midpoint protocol and resume compatibility signature. |
| `neqti_rest2/{a,m,b}/` | REST2 walker checkpoints, assignments, exchange history, acceptance, and round-trip state. |
| `*_swapped.pdb` | Diagnostic coordinates after applying the ATM virtual coordinate swap. |

The ordinary PDB contains the physical coordinates held by the OpenMM context. ATM evaluates an additional swapped coordinate state internally. NEQTI writes both forms for start, post-equilibration, and switching diagnostics. The existing snapshot pair, for example `neqti_m_snapshot_4.pdb` and `neqti_m_snapshot_4_swapped.pdb`, contains the coordinates immediately before both midpoint switches from that sampled M geometry. The corresponding post-switch files include the leg name, for example `neqti_m_leg_a_reverse_snapshot_4_post_switch.pdb` and `neqti_m_leg_b_reverse_snapshot_4_post_switch.pdb`. Endpoint sampling uses the same pattern with `neqti_a_leg_a_forward_snapshot_*_post_switch.pdb` and `neqti_b_leg_b_forward_snapshot_*_post_switch.pdb`. Use the swapped PDBs to verify that the ligand expected in the binding site overlaps the intended reference pose at each endpoint. These files are diagnostics, not independent simulation states.

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
  midpoint: neqti_midpoint.pdb
  midpoint_swapped: neqti_midpoint_swapped.pdb
  leg_a_forward_work_csv: neqti_leg_a_forward.csv
  leg_a_reverse_work_csv: neqti_leg_a_reverse.csv
  leg_b_forward_work_csv: neqti_leg_b_forward.csv
  leg_b_reverse_work_csv: neqti_leg_b_reverse.csv
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

Without `workflow.neqti.convergence`, `quality.convergence_status: usable` means only that the run completed its requested sample count and produced a finite estimate. It is not a scientific convergence guarantee. With automatic convergence enabled, inspect `termination_reason`, `quality.convergence`, and the per-leg overlap values. Network cycle-closure analysis is not implemented in schema version 1 and remains null.

## Current Limitations

- The first unified release supports the documented ATM combinations and two-leg hybrid-topology NEQTI. Hybrid-topology ASYNC_RE/AWH and charge-balanced single-box hybrid transfer are not implemented.
- NEQTI is experimental and has not replaced asynchronous replica exchange as the established production method.
- NEQTI currently uses the configured discrete ATM schedule as interpolation knots; it does not yet implement an arbitrary continuous OpenMMTools-style alchemical function.
- NEQTI REST2 supports shared-ATM interleaved sampling and native A/B interleaved or batched sampling. SMARTS-defined partial hot regions and adaptive switching schedules are experimental; automatic REST2 ladder tuning is not implemented.
- GPU, CUDA, OpenMM, Espaloma, and `openmmforcefields` compatibility is the responsibility of the environment.
- A numerically completed run is not sufficient validation. Inspect endpoint structures, swapped structures, work distributions, forward/reverse overlap, and sensitivity to equilibration and switching time.

### Hybrid endpoint equilibration

Hybrid-topology NEQTI uses the fixed minimization, NVT, and NPT settings under
`workflow.neqti.endpoint_equilibration` unless a custom endpoint protocol is
provided. `workflow.equilibration.neqti.endpoint` is a shared fallback;
`complex_endpoint` and `solvent_endpoint` override it independently. An explicit
`mode: default` selects the fixed protocol for that environment.

```yaml
workflow:
  equilibration:
    neqti:
      complex_endpoint:
        steps:
        - id: restrained_minimization
          type: minimization
          tolerance_kj_mol_nm: 10.0
          max_iterations: 1000
          positional_restraints:
            mask: '#active:"Nc(nc1O)nc2c1ncn2"'
            k_kcal_mol_a2: 5.0
            tolerance_a: 0.25
        - id: restrained_npt
          type: md
          ensemble: NPT
          n_steps: 250000
          timestep_ps: 0.002
          thermostat:
            temperature_k: 300.0
            friction_per_ps: 1.0
          positional_restraints:
            mask: '#active:"Nc(nc1O)nc2c1ncn2"'
            k_kcal_mol_a2: 1.0
            tolerance_a: 0.5
          reporters:
            state:
              interval: 5000
      solvent_endpoint:
        mode: default
```

The same step sequence runs at endpoint A and endpoint B. In noncovalent hybrid
systems, `#active:"SMARTS"` resolves against ligand A at endpoint A and ligand B
at endpoint B, so one restraint definition follows the physical ligand.
Amber-mask operators and role-aware SMARTS leaves can be combined in one
expression. Each step has an independent checkpoint under
`equilibration/{complex|solvent}/endpoint_{a|b}`. The resolved protocol is signed
in `equilibration_protocol.yaml`; changing it requires a new workdir rather than
silently reusing endpoint states or protocol work.

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

## Covalent RBFE prototype

`chemistry: covalent`, `alchemy.model: hybrid_topology`, and
`alchemy.cycle: complex_solvent` select the experimental direct-endpoint NEQTI workflow for
congeneric cysteine-aldehyde inhibitors. It uses a chemically complete
ACE-Cys-product-NME thiohemiacetal in the aqueous reference leg and grafts the same
local product parameters onto the target cysteine in the protein leg. The endpoint
Hamiltonians are exact at lambda 0 and 1. The original smooth log-sum `envelope`
path and a native `softcore_linear` dual-topology path are available. The native
path supports both the original staged schedule and general node-based coupling
profiles.

Prepare a source dataset containing `protein_annealed.pdb`, `data.csv`,
`manifest.csv`, and the manifest-referenced ligand SDF files:

```bash
atom-rbfe-prepare-covalent-dataset SOURCE_DIR NORMALIZED_DIR
```

The preparation command keeps waters whose oxygen is within 5 A of a protein heavy
atom, converts the neutral aldehyde pose to the covalent thiohemiacetal, transfers
the Cys thiol proton to oxygen, records atom maps and stereochemistry, and writes
assay data without changing its units. The runner then uses ff19SB, OPC,
OpenFF 2.2.1 valence/Lennard-Jones parameters, and Espaloma NN charges from the
`espaloma-0.3.2` model. Charge inference has a graph-keyed cache and does not fall
back to AM1-BCC.

```yaml
workflow:
  type: rbfe
  chemistry: covalent
  alchemy:
    model: hybrid_topology
    cycle: complex_solvent
  sampling:
    method: neqti
  dataset: normalized/dataset.yaml
  workdir: run
  pairs:
  - ligand_a: I79DJ_543
    ligand_b: LIBA225
  setup:
    protein_forcefield: amber19/protein.ff19SB.xml
    water_forcefield: amber19/opc.xml
    ligand_forcefield: openff-2.2.1.offxml
    ligand_charge_model: espaloma_nn
    espaloma_model: espaloma-0.3.2
    solvent_padding_a: 10.0
    ionic_strength_molar: 0.15
    dummy_bonded_scales:
      bond: 1.0
      angle: 1.0
      proper_torsion: 1.0
      junction_angle: 1.0
      junction_proper_torsion: 0.0
  neqti:
    initial_equilibration_steps: 250000
    decorrelation_steps: 100000
    timestep_fs: 2.0
    n_snapshots: 10
    bootstrap_samples: 500
    interpolation: softcore_linear
    softcore:
      alpha: 0.3
      sigma_nm: 0.25
      power: 1
      charge_steps_per_stage: 10000
      sterics_steps: 30000
      long_range_correction: dynamic
    failed_switch_policy: count_as_infinite
    rest2:
      enabled: true
      effective_temperatures_k: [300.0, 344.6, 395.9, 454.7, 522.3, 600.0]
      exchange_interval_steps: 500
      execution: serial
```

Endpoint REST2 trajectories remain independent of switching trajectories: every
switch starts from a physical REST2 snapshot, and its final coordinates are not fed
back into decorrelation. Work is saved after each direction, so rerunning an
interrupted workflow resumes the missing direction without replacing completed
samples. Each pair directory contains `result.yaml`, four work CSV files, REST2
checkpoints, prepared endpoint PDBs, and serialized OpenMM systems. The reported
DDG convention is `G(ligand_b)-G(ligand_a)`.

REST2 `execution` defaults to `serial`, where resident replica Contexts are stepped
one after another. `execution: process` assigns each replica to a persistent worker
process and propagates all replicas concurrently before the synchronous exchange
barrier. `device_indices` may contain one device reused by every replica or one
device index per replica. CUDA MPS can improve same-GPU process concurrency when
it is available; checkpoint and exchange files are compatible between execution
modes.

Covalent atom mapping defaults to `method: dataset_core`, which uses the prepared
dataset scaffold to anchor a connected heavy-atom MCS through the
Cys-SG--warhead bond. `method: mcs_core_smarts` instead calculates the MCS among
ligand A, ligand B, and the supplied core pattern. This caps the common region so
that chemically shared atoms outside the selected core remain alchemical. All
matching combinations are evaluated without coordinate alignment and the pair
with the smallest direct RMSD is selected. A workflow-level `alchemy.mapping` applies to
the series; a pair-level `mapping` shallow-overrides it, and
`method: dataset_core` opts an individual pair back into the default behavior.

```yaml
workflow:
  alchemy:
    model: hybrid_topology
    cycle: complex_solvent
    mapping:
      method: mcs_core_smarts
      smarts: "O=CNc1cccn(C2(C(=O)N[C@H](C=O)C[C@@H]3CCNC3=O)Cc3ccccc3C2)c1=O"
```

The constrained core must include the common electrophile carbon and produce a
single protein-connected product subgraph. Explicit hydrogens are attached after
the heavy-atom match. Warhead mutations are rejected. Unique A and B branches are
noninteracting with the environment when inactive but retain full unique-unique
vacuum electrostatics, Lennard-Jones, exclusions, and 1-4 interactions. The input
core, generated MCS, selected match/RMSD, resolved map, and atom roles are recorded
in `covalent_mapping.yaml`.

For charge consistency, ff19SB charges are copied for Cys N, H, CA, HA, C, and O.
The remaining Cys sidechain, transferred hydrogen, and ligand charges are corrected
together to a neutral modified residue and copied unchanged into both protein and
capped-reference systems.

For `softcore_linear`, the forward protocol removes A-branch electrostatics,
softcore-transforms sterics and bonded parameters, then introduces B-branch
electrostatics. Mapped charges move through their endpoint midpoint during the two
charge stages. Reverse work uses the exactly reversed protocol. The total number
of steps is
`2 * charge_steps_per_stage + sterics_steps`; do not also set `switch_steps`.
The defaults (`alpha: 0.3`, `sigma_nm: 0.25`, `power: 1`) match the established
GROMACS softcore settings used for the RHINO calculations. Endpoint total charges
must currently be equal.

The Lennard-Jones softcore function is selectable. Existing workflows default to
the Beutler form. The Gapsys form linearly continues the short-range LJ force
below a coupling-dependent radius and can retain a stronger restoring force when
an appearing dummy branch overlaps its environment:

```yaml
softcore:
  function: gapsys
  gapsys_scale_linpoint_lj: 0.85
  gapsys_sigma_nm: 0.30
  long_range_correction: endpoint_correction
```

These are the standard GROMACS Gapsys LJ defaults. Gapsys is applied only to
unique-branch/environment LJ interactions and their exceptions. Electrostatics
continue to follow the configured staged PME charge path. `function: beutler`
retains the `alpha`, `sigma_nm`, and `power` settings.

A general path can replace the staged step settings:

```yaml
softcore:
  alpha: 0.3
  sigma_nm: 0.25
  power: 1
  total_steps: 150000
  long_range_correction: endpoint_correction
  path:
    nodes: []
    vdw_a: [1.0, 0.0]
    charge_a: [1.0, 0.0]
schedule_optimization:
  enabled: true
  pilot_samples: 10
  segments_per_interval: [30]
```

`nodes` contains the internal path coordinates. The A arrays contain one value
for each node plus both endpoints, begin at one, and end at zero. B values are
generated by reversing the A arrays. Mapped parameters use
`(1 - scale_a + scale_b) / 2`. Values interpolate linearly between nodes, and the
initial `total_steps` allocation follows interval width. The optimizer divides
each interval according to `segments_per_interval` and redistributes steps without
changing the coupling path or total budget.

To keep both unique branches fully coupled through van der Waals interactions at
the midpoint while transforming charges linearly:

```yaml
softcore:
  total_steps: 150000
  path:
    nodes: [0.5]
    vdw_a: [1.0, 1.0, 0.0]
    charge_a: [1.0, 0.5, 0.0]
schedule_optimization:
  enabled: true
  segments_per_interval: [15, 15]
```

Unique A/B branches remain mutually noninteracting. Changing bonded terms follow
the van der Waals path, while mapped bonded and nonbonded parameters retain a
normalized endpoint interpolation. General path fields are mutually exclusive
with `charge_steps_per_stage`, `sterics_steps`, and
`subdivisions_per_stage`. Electrostatics use PME charge interpolation; softcore
Coulomb is not currently implemented.

`softcore.long_range_correction` accepts `dynamic` or `endpoint_correction`.
`dynamic` is the backward-compatible default. `endpoint_correction` disables
long-range correction on the lambda-dependent custom Lennard-Jones forces during
integration, evaluates that correction only at the two fixed-volume endpoints,
and adds the final-minus-initial difference to protocol work. Switching systems
with a barostat are rejected in this mode. Corrected work remains in the standard
forward/reverse CSV files; `switch_lrc_diagnostics.csv` additionally records raw
work, endpoint corrections, their difference, and box volume.

Windowed switching diagnostics can localize where protocol work accumulates
without adding potential-energy evaluations:

```yaml
switch_work_profile:
  enabled: true
  interval_steps: 100
  phases: [optimizer]
```

`phases` accepts `optimizer`, `production`, or both. The resulting
`switch_work_profile.csv` records exact work increments, cumulative work, the
normalized A-to-B path coordinate, and all coupling scales. The
`delta_work_over_delta_lambda_kj_per_mol` column is a finite-window protocol-work
derivative, not an instantaneous analytical `dU/dlambda`. Sampling introduces a
host synchronization at each interval but does not perform an additional energy
calculation.

The softcore system evaluates common PME, protein, and solvent terms once. Unique
A/B cross interactions remain excluded, while the existing unique-branch vacuum
interactions remain active. Work is still accumulated exactly on-device from the
energy change at every lambda update. `switch_protocol.yaml` prevents incompatible
resume, and `switch_timing.csv` records elapsed time and ns/day for every switch.
Use `interpolation: envelope` with `switch_steps` to reproduce the original
two-endpoint log-envelope implementation.
