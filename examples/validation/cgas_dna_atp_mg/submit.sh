#!/bin/bash
set -euo pipefail

PREP_JOB=$(sbatch --parsable prepare.slurm)
PREP_JOB=${PREP_JOB%%;*}
sbatch --dependency=afterok:${PREP_JOB} --array=0-11 run_array.slurm
