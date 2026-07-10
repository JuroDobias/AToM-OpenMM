# ATM GAFF2 Benchmark Workflows

This directory contains utilities to generate and collect one YAML RBFE workflow per
ATM benchmark pair. The generated workflows are intended for independent Slurm
jobs and use the experimental NEQTI production path with AmberTools/tleap setup.

Generate jobs:

```bash
python examples/RBFE/benchmarks/generate_atm_benchmark_workflows.py \
  --benchmark-csv /path/to/DDG_ATM_GAFF2.csv \
  --systems-root /path/to/ATM_benchmark \
  --outdir /path/to/atm_neqti_jobs \
  --reference-alignment-atoms 14,21,18
```

For the official ATM validation CSV, first add the per-pair alignment atoms from
the `*_asyncre.cntl` files:

```bash
python examples/RBFE/benchmarks/enrich_atm_benchmark_csv.py \
  --input-csv /path/to/ATM_benchmark/ATM_Validation/DDG_ATM_GAFF2.csv \
  --validation-root /path/to/ATM_benchmark/ATM_Validation \
  --output-csv /path/to/ATM_benchmark/ATM_Validation/DDG_ATM_GAFF2_enriched_CDK2.csv \
  --protein CDK2
```

Submit one pair:

```bash
cd /path/to/atm_neqti_jobs/<system>/<ligand_a>--<ligand_b>
sbatch run.sh
```

The generated Slurm scripts default to:

- GPU resource: `gpu:nvidia_L40S:1`
- CPUs: `16`
- memory: `100G`
- wall time: `12:00:00`
- conda setup: `source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate myatom`
- input materialization: copy receptor, ligand, and frcmod files into each job directory

You can override the conda setup at submission time:

```bash
export ATOM_CONDA_SH=/path/to/conda.sh
export ATOM_CONDA_ENV=atomopenmm_legacy
```

Or override it while generating scripts with `--conda-sh`, `--conda-env`,
`--link-mode symlink`, `--slurm-gres`, `--slurm-cpus-per-task`, `--slurm-mem`,
`--slurm-time`, `--neqti-n-snapshots`, `--neqti-switch-steps-per-segment`,
`--neqti-max-switch-attempts-per-direction`, and
`--production-restart-attempts`.

Example 4-edge CDK2 smoother-switch test set:

```bash
python examples/RBFE/benchmarks/generate_atm_benchmark_workflows.py \
  --benchmark-csv /path/to/ATM_benchmark/ATM_Validation/DDG_ATM_GAFF2_enriched_CDK2.csv \
  --systems-root /path/to/ATM_benchmark \
  --outdir /path/to/neqti_cdk2_4_300ps \
  --pairs-filter CDK2:17:22 CDK2:1oiu:26 CDK2:30:31 CDK2:1oiy:32 \
  --neqti-n-snapshots 40 \
  --neqti-switch-steps-per-segment 15000 \
  --neqti-max-switch-attempts-per-direction 80 \
  --production-restart-attempts 6
```

Collect results:

```bash
python examples/RBFE/benchmarks/collect_results.py \
  --index-csv /path/to/atm_neqti_jobs/index.csv \
  --output-csv /path/to/atm_neqti_jobs/benchmark_results.csv
```

The first benchmark protocol is intentionally fixed across pairs:

- system preparation: AmberTools/tleap
- protein force field: `leaprc.protein.ff14SB`
- additional protein parameters: `leaprc.phosaa14SB` for CDK2 `TPO`
- ligand force field: `leaprc.gaff2`
- water force field: `leaprc.water.tip3p`
- ligand inputs: prepared `*-p.mol2` files plus matching `*-p.frcmod`
- NEQTI snapshots: 40
- shared midpoint and endpoint sampling equilibration: 200,000 steps
- endpoint equilibration: weakly restrained NVT reheat, short weakly restrained NPT, then unrestrained NPT
- decorrelation: 100,000 steps
- preparation annealing: 10,000 steps per ATM schedule segment
- switching: 5,000 steps per ATM schedule segment
- custom NEQTI switch integrator enabled
- failed production switches are recorded and skipped, with up to 80 attempts per direction

The prepared `*-p.mol2` charges and `*-p.frcmod` parameters are used through
tleap. This avoids OpenEye-dependent MOL2 parsing and is closer to the original
ATM benchmark preparation route.

The generator does not guess reference alignment atoms. It uses CSV alignment
columns if present, otherwise the fallback supplied with
`--reference-alignment-atoms`.
