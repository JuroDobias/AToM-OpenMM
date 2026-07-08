AToM-OpenMM v8.5
====================

![AToM logo](AToM-logo.png)

The Alchemical Transfer Method for OpenMM (AToM-OpenMM) is an extensible Python package for estimating absolute and relative binding free energies of molecular complexes. It implements the [Alchemical Transfer Method (ATM)](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01129) with [OpenMM](https://github.com/openmm) and can run on GPU workstations or cluster nodes.

This fork adds a single-YAML small-molecule RBFE workflow on top of the original AToM-OpenMM implementation. The wrapper can prepare and run complete ligand-pair calculations, select setup force fields and ligand charges, define custom equilibration protocols with Amber masks, and use either asynchronous replica exchange or an experimental nonequilibrium switching (NEQTI) protocol.

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

Start from [`examples/RBFE/cdk2/workflow.yaml`](examples/RBFE/cdk2/workflow.yaml). It defines the receptor, ligand SDF directory, ligand pairs, reference alignment atoms, force fields, and ATM schedule in one file:

```bash
cd examples/RBFE/cdk2
atom-rbfe workflow.yaml
```

Relative paths are resolved from the workflow file. Results for each pair are written below `workflow.workdir` (the example uses `complexes/`). Use `prepare_only: true` to build the pair directories without running production.

The default production method is the original asynchronous replica exchange implementation. To select experimental NEQTI switching:

```yaml
workflow:
  production_method: neqti
  neqti:
    initial_equilibration_steps: 25000
    n_snapshots: 10
    decorrelation_steps: 5000
    switch_steps_per_segment: 1000
    resume: true
    bootstrap_samples: 0
```

NEQTI equilibrates endpoints A and B, collects decorrelated endpoint snapshots, performs forward A-to-B and reverse B-to-A switches through the configured ATM schedule, records protocol work, and estimates the free-energy difference with BAR. This implementation is experimental and should be validated against established calculations before production use.

By default, ATM parameter switching and protocol-work accumulation run inside a dedicated OpenMM `CustomIntegrator`; `openmmtools` is used as a design reference but is not a runtime dependency. Set `workflow.neqti.switch_integrator: python` to use the slower reference path or `validate_switch_integrator: true` for a one-shot comparison.

See the [RBFE user guide](docs/user-guide/rbfe.md) for the complete YAML schema, force-field examples, custom equilibration, restart behavior, outputs, and swapped-coordinate diagnostics.

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
