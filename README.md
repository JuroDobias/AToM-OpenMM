AToM-OpenMM v8.5
====================

![AToM logo](AToM-logo.png)

The Alchemical Transfer Method for OpenMM (AToM-OpenMM) is an extensible Python package for estimating absolute and relative binding free energies of molecular complexes. It implements the [Alchemical Transfer Method (ATM)](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01129) with [OpenMM](https://github.com/openmm) and can run on GPU workstations or cluster nodes.

This fork adds a single-YAML RBFE workflow on top of the original AToM-OpenMM implementation. The wrapper separates chemistry, alchemical representation, thermodynamic cycle, and sampling. ATM transfer supports asynchronous replica exchange, experimental nonequilibrium switching (NEQTI), and experimental AWH. Conventional complex-plus-solvent hybrid topology supports NEQTI for noncovalent ligands and congeneric covalent inhibitors.

This version of AToM-OpenMM has been tested with OpenMM 8.5 and 8.4; it uses [ATMForce](https://github.com/openmm/openmm/pull/4110) in the 8.4.0 or later versions of [OpenMM](https://github.com/openmm/openmm).

Credits
-------

This software is developed and maintained by the [Emilio Gallicchio's lab](http://www.compmolbiophysbc.org) with support from current and past grants from the National Science Foundation (ACI 1440665 and CHE 1750511) and the National Institutes of Health (R15 GM151708).

Maintainer/Author:

- Emilio Gallicchio <egallicchio@brooklyn.cuny.edu>

Contributors:

- Elian Tiudic
- Sylvester Sakyi
- Stefan Doerr
- Sheenam Khuttan 
- Joe Z Wu
- Solmaz Azimi
- Baofeng Zhang 
- Rajat Pal

The asynchronous replica exchange method was first implemented in the [AsyncRE](https://github.com/ComputationalBiophysicsCollaborative/AsyncRE) package for the IMPACT program.

Bibliography
------------

Please [cite us](http://www.compmolbiophysbc.org/publications) if you use this software in your research:

- [Relative Binding Free Energy Calculations for Ligands with Diverse Scaffolds with the Alchemical Transfer Method](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01129)

- [Relative Binding Free Energy Estimation of Congeneric Ligands and Macromolecular Mutants with the Alchemical Transfer with Coordinate Swapping Method](https://arxiv.org/abs/2412.19971)

- [Alchemical Transfer Approach to Absolute Binding Free Energy Estimation](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00266)

- [Asynchronous Replica Exchange Software for Grid and Heterogeneous Computing](http://www.compmolbiophysbc.org/publications#asyncre_software_2015)

Installation & Usage
--------------------

It is recommended that the installation is performed in a personal Python environment (`miniforge`, `miniconda`, `conda`, or similar). AToM-OpenMM requires the `openmm` and other Python modules. 

This version of AToM-OpenMM requires OpenMM 8.4.0 or later. This conda command installs the necessary requirements:
```
mamba create -n atm8.5.0 -c conda-forge 'openmm>=8.4' ambertools openmmforcefields configobj setproctitle r-base espaloma
mamba activate atm8.5.0
```
`setproctitle` above is optional but useful to track the names of the processes started by AToM-OpenMM. The `ambertools`, `openmmforcefields`, and `espaloma` packages are not actual dependencies; they are used to prepare the molecular systems. `openmmforcefields`, in particular, is used to assign force field parameters using OpenFF, GAFF, or `espaloma`. [`espaloma`](https://github.com/choderalab/espaloma) is a machine-learning system by the Chodera lab to assign force field parameters.  The `r-base` dependency with the `UWHAM R package` (see below) is required for free energy estimation in legacy workflows and will be removed in later versions. See [examples](examples/) for examples and tutorials.

Finally, install AToM-OpenMM:

- From the latest release:
```
pip install atom-openmm
```

- From this fork in editable mode (recommended for development):
```
git clone https://github.com/JuroDobias/AToM-OpenMM.git
cd AToM-OpenMM
python -m pip install -e .
```

Verify that the YAML wrapper was installed into the active environment:

```bash
atom-rbfe --help
```

And this will install the UWHAM R package:
```
Rscript -e 'install.packages("UWHAM", repos = "http://cran.us.r-project.org")' 
```

While we strive to develop and distribute high-quality and bug-free software, keep in mind that this is research software under heavy development. AToM-OpenMM is provided without any guarantees of correctness. Please report fork-specific issues [here](https://github.com/JuroDobias/AToM-OpenMM/issues). We welcome contributions and pull requests.

Single-YAML RBFE quick start
----------------------------

Start from [`examples/RBFE/cdk2/workflow.yaml`](examples/RBFE/cdk2/workflow.yaml). It defines the receptor, ligand SDF directory, ligand pairs, alignment atom selection, force fields, and ATM schedule in one file:

```bash
cd examples/RBFE/cdk2
atom-rbfe workflow.yaml
```

Relative paths are resolved from the workflow file. Results for each pair are written below `workflow.workdir` (the example uses `complexes/`). Use `prepare_only: true` to build the pair directories without running production.

Alignment atoms can be supplied explicitly with `workflow.alignments`, generated from the legacy `reference_ligand` plus `reference_alignment_atoms`, or selected from a SMARTS scaffold:

```yaml
workflow:
  alignment:
    method: smarts
    smarts: "[#6]-[#6]-[#6]"
    smarts_atom_ids: [1, 2, 3]
  alignments_out: alignments.yaml
```

For SMARTS alignment, `smarts_atom_ids` are 1-based positions inside the SMARTS match. If the SMARTS matches symmetrically, all match combinations are evaluated for each ligand pair and the one with the smallest direct coordinate RMSD is used. No fitting, rotation, or translation is performed; input ligands should already be aligned.

The workflow axes are explicit. For ATM NEQTI use:

```yaml
workflow:
  chemistry: noncovalent
  alchemy:
    model: atm
    cycle: transfer
  sampling:
    method: neqti
  neqti:
    initial_equilibration_steps: 25000
    n_snapshots: 10
    decorrelation_steps: 5000
    switch_steps_per_segment: 1000
    work_sample_intervals: [1, 5, 10, 25, 50]
    failed_switch_policy: count_as_infinite
    rest2:
      enabled: true
      sampler_backend: custom  # or openmm_native with OpenMM 8.6+
      solute: '#ligand:"*"'
      effective_temperatures_k: [300, 351, 411, 481, 563, 658, 770, 900]
      exchange_interval_steps: 500
      checkpoint_interval_cycles: 10
    resume: true
    bootstrap_samples: 0
```

NEQTI reuses the async-RE ATM soft-core schedule as two bidirectional half paths that meet at one shared midpoint ensemble, A<->M and B<->M. It estimates both legs with BAR and combines them as `DG(A->M) - DG(B->M)`.

Experimental ATM-AWH instead uses one expanded-ensemble walker over the complete
ATM schedule. Optional role-aware REST2 branches extend the physical A and B
endpoints without heating the ATM interior. AWH first adapts a uniform-target
bias, requires physical A-B-A round trips and state coverage, then freezes the
bias for production. Fixed-bias UWHAM is the primary reported estimator. See
[`examples/RBFE/cdk2/workflow.awh.yaml`](examples/RBFE/cdk2/workflow.awh.yaml)
and the [RBFE guide](docs/user-guide/rbfe.md#atm-awh-with-endpoint-rest2).
Automatic diagnostics report neighboring UWHAM overlap, endpoint effective
samples, REST2 hot-state returns, and state occupancy. A state-tagged,
solute-only trajectory supports structural analysis of every AWH and REST2
node.
The adaptive learning rate remains fixed until freezing, and
`workflow.awh.atm_state_count` can densify both ATM half paths without changing
the user-supplied async-RE schedule. Optional generalized-force friction
diagnostics identify slow graph regions and suggest a square-root-friction
target. Applying that target remains disabled by default and requires a fresh
dynamics checkpoint.
Experimental `workflow.awh.state_sampling.method: hybrid_global_gibbs`
reconstructs every physical ATM energy from one ATM decomposition, scans the
REST2 endpoint states on-device, and draws from the complete graph. Direct
energy validation is mandatory, and the full energy vector is reused for
fixed-bias UWHAM/MBAR output.

For production sampling, `failed_switch_policy: count_as_infinite` preserves recognized numerical switching failures as `+inf` protocol-work observations instead of selectively replacing them. CUDA/environment and programming failures still stop the run. Use `retry` for the previous replacement behavior or `abort` to stop at the first failed switch.

By default, ATM parameter switching and protocol-work accumulation run inside a dedicated OpenMM `CustomIntegrator`; `openmmtools` is used as a design reference but is not a runtime dependency. Set `workflow.neqti.switch_integrator: python` to use the slower reference path or `validate_switch_integrator: true` for a one-shot comparison.

Optional NEQTI REST2 sampling replaces ordinary endpoint decorrelation with synchronous solute-tempering exchange. Its hot region can use role-aware SMARTS selectors such as `'#unbound:"*"'`; physical snapshots are taken only from the `s=1` replica, and all nonequilibrium switches remain at `s=1`. The existing equilibration and decorrelation counts are interpreted as steps per REST2 replica.

`rest2.sampler_backend` selects the exchange engine independently of the endpoint
Hamiltonian. `custom` remains the default and supports serial contexts or
process workers. `openmm_native` uses OpenMM 8.6's global replica exchange,
including non-neighbor state swaps, through one serial OpenMM context. Native
sampler banks are deliberately incompatible with custom-sampler banks; start a
new work directory when changing backend.

Role-aware SMARTS leaves can also be embedded in Amber masks used by custom equilibration, for example `'!:HOH,WAT & #bound:"c1ncnc2ncnc12"'`. The RBFE guide defines the canonical ligand and endpoint role semantics. Diagnostic work intervals produce additional BAR estimates from the same switching trajectories; exact per-step work remains the primary result and the diagnostics do not reduce energy-evaluation cost.

Experimental native endpoint sampling removes `ATMForce` from A/B equilibration and REST2 while retaining ATM for M and all switches. Select `endpoint_system: native` and `rest2.ensembles: [a, b]`. With `sampling_order: interleaved`, both endpoint REST2 ladders remain resident and adaptive pilot scheduling plus automatic convergence stopping are available. Use `sampling_order: batched` when GPU memory permits only one resident ladder. Endpoint ligand roles and restraints are exchanged consistently in B. Existing workflows continue to use ATM endpoints by default.

See the [RBFE user guide](docs/user-guide/rbfe.md) for the complete YAML schema, force-field examples, custom equilibration, restart behavior, outputs, and swapped-coordinate diagnostics.

For a conventional noncovalent dual-topology comparison, use [`examples/RBFE/cdk2/workflow.hybrid.yaml`](examples/RBFE/cdk2/workflow.hybrid.yaml). It maps the ligands by MCS or SMARTS-constrained MCS, preserves inactive-branch intramolecular interactions, and combines complex and solvent BAR estimates. Hybrid NEQTI can independently select switching durations and stop production for the complex and solvent environments using per-environment overlap, uncertainty, and DG-stability criteria. The covalent hybrid-topology workflow is documented in [`examples/RBFE/covalent-rhino`](examples/RBFE/covalent-rhino).

Mapped transmutations can mark a complete endpoint-unique branch with
`inactive_bonded_atoms_*_0based` or paired-SMARTS labels. `bond_only` preserves
the branch's internal bonded geometry while removing mixed branch/core angular
terms; `terminal_z_matrix` additionally retains one deterministic junction angle
and torsion frame. This is intended for local internal valence changes where
duplicating a large downstream ligand region would be inefficient.
Unambiguous one-anchor unique branches now receive this Z-matrix framing
automatically, with a recorded `bond_only` fallback when the mapped core cannot
define a heavy-atom frame. The optional `staged_bonded` softcore path independently
stages charge, bonded, steric, and mapped-atom changes within the original total
switching-time budget.

Every ligand-pair directory also contains an atomically updated `result.yaml` for integration with workflow managers and databases. It uses the same schema for asynchronous replica exchange and NEQTI, reports DDG in kcal/mol and kJ/mol, records input provenance and artifacts, and exposes `prepared`, `running`, `partial`, `completed`, or `failed` status.

Documentation
-------------

[AToM-OpenMM User Guide](http://gallicchio-lab.github.io/AToM-OpenMM)

[AToM-OpenMM Theory Introduction](https://www.compmolbiophysbc.org/atom-openmm)

See [examples](examples/) for examples, workflows, and tutorials.

See [example-notebooks](example-notebooks/) for example Notebooks.

Licensing
---------

 This software is licensed under the terms of the [GNU Lesser General Public License](https://opensource.org/license/lgpl-3-0). See [LICENSE](LICENSE). The AToM logo &copy; 2023 Solmaz Azimi. 
