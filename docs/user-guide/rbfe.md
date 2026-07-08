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

`workflow.yaml` contains the receptor, ligand directory, ligand pairs, reference alignment atoms, and the original AToM options under `atom_options`. Relative input paths are resolved from the workflow YAML location. Each ligand pair is expanded into `complexes/<jobname>/`, where the wrapper writes the final per-pair `<jobname>.yaml` used by the existing `rbfe_structprep`, `rbfe_production`, and UWHAM analysis code. If `alignments_out` is set, relative output paths are written under the workflow `workdir`.

Set `prepare_only: true` under `workflow` to create the per-pair directories and final YAML files without starting production.

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

The wrapper can also run the experimental NEQTI switching protocol instead of asynchronous replica exchange:

```yaml
workflow:
  production_method: neqti
  neqti:
    initial_equilibration_steps: 10000
    n_snapshots: 20
    decorrelation_steps: 10000
    switch_steps_per_segment: 100
    resume: true
```

NEQTI reuses the existing ATM schedule as switching knots. By default it switches from the first state through every listed schedule state to the final state, and then repeats the reverse path. It writes `neqti_forward.csv`, `neqti_reverse.csv`, `neqti_summary.yaml`, and pmx-compatible integrated work files `integA.dat` and `integB.dat` in each pair directory. The integrated work files use kJ/mol so they can be read by `analyze_dhdl.py -iA integA.dat -iB integB.dat`.

For the example above, the execution order is:

1. Prepare the ATM system. If `equilibration.neqti.endpoint` is configured, run it independently at A and B and save both endpoint states.
2. Load endpoint A and run `initial_equilibration_steps` once.
3. Run `decorrelation_steps`, save a snapshot, switch A to B, and record one forward work value. Repeat until `n_snapshots` forward values exist.
4. Load endpoint B and run `initial_equilibration_steps` once.
5. Run `decorrelation_steps`, save a snapshot, switch B to A, and record one reverse work value. Repeat until `n_snapshots` reverse values exist.
6. Estimate the free-energy difference from forward and reverse work with BAR. If `bootstrap_samples` is greater than zero, estimate uncertainty by bootstrap resampling.

| Setting | Meaning |
| --- | --- |
| `initial_equilibration_steps` | Additional MD run once at A and once at B before collecting snapshots. |
| `n_snapshots` | Number of forward trajectories and number of reverse trajectories. |
| `decorrelation_steps` | Endpoint MD between consecutive switching snapshots. |
| `switch_steps_per_segment` | Integration steps used to interpolate between each neighboring pair of ATM schedule nodes. |
| `state_path` | Optional ordered list of ATM schedule indices; defaults to all configured states. |
| `resume` | Reuse completed work rows and completed custom endpoint states when available. |
| `bootstrap_samples` | Number of BAR bootstrap resamples; zero disables uncertainty estimation. |
| `random_seed` | Seed used for bootstrap resampling. |
| `platform` | Optional OpenMM platform override for NEQTI. |

`switch_steps_per_segment` applies to every neighboring pair of nodes in `state_path`. With the default 22-node schedule, a value of 1000 produces 21,000 integration steps per forward or reverse switch. A progress message is written after each complete node-to-node segment.

When `resume: true`, completed work rows and compatible endpoint states are reused. An interrupted switching trajectory is rerun because work is only marked complete after the whole trajectory finishes. Review the YAML and output files before restarting: changing the schedule or protocol while retaining old work CSV files can mix incompatible results.

The wrapper can replace the default equilibration stages with inline mdflow-style steps. Amber masks require `parmed`. For `async_re`, `pre_atm` replaces the physical minimization/thermalization/NPT/NVT stage and `async_re.midpoint` can replace the final lambda-0.5 equilibration. For `neqti`, preparation stops after the physical equilibrated state and `neqti.endpoint` is run separately at endpoint A and endpoint B before switching:

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
| `neqti_endpoint_A.xml`, `neqti_endpoint_B.xml` | Restartable endpoint states after custom endpoint equilibration. |
| `neqti_forward.csv`, `neqti_reverse.csv` | Per-trajectory protocol work and completion status. |
| `integA.dat`, `integB.dat` | Integrated work in kJ/mol for external analysis tools. |
| `neqti_summary.yaml` | BAR estimate and optional bootstrap uncertainty. |
| `*_swapped.pdb` | Diagnostic coordinates after applying the ATM virtual coordinate swap. |

The ordinary PDB contains the physical coordinates held by the OpenMM context. ATM evaluates an additional swapped coordinate state internally. Use the swapped PDBs to verify that the ligand expected in the binding site overlaps the intended reference pose at each endpoint. These files are diagnostics, not independent simulation states.

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
