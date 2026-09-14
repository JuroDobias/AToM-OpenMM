#!/bin/bash
#SBATCH --job-name=cgas-mg-matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-d
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myatom_openmm86
export PYTHONPATH="$(cd ../../.. && pwd):${PYTHONPATH:-}"

TASK=${SLURM_ARRAY_TASK_ID}
TASK_DIR=$(find run/tasks -maxdepth 1 -type d -name "$(printf '%02d' "${TASK}")_*" -print -quit 2>/dev/null || true)
RESULT=${TASK_DIR}/result.yaml
CHILD=""
WALLTIME=0

on_term() {
  WALLTIME=1
  if [[ -n "${CHILD}" ]]; then
    kill -TERM "${CHILD}" 2>/dev/null || true
  fi
}
trap on_term TERM

python -m atom_openmm.md_validation workflow.yaml --task "${TASK}" &
CHILD=$!
if wait "${CHILD}"; then
  exit 0
fi

TASK_DIR=$(find run/tasks -maxdepth 1 -type d -name "$(printf '%02d' "${TASK}")_*" -print -quit 2>/dev/null || true)
RESULT=${TASK_DIR}/result.yaml
if [[ "${WALLTIME}" -eq 1 && ! -f "${RESULT}" ]]; then
  sbatch --dependency=afterany:"${SLURM_JOB_ID}" --array="${TASK}" run_array.slurm
  exit 0
fi
echo "Task ${TASK} failed; inspect its Slurm log before resubmission." >&2
exit 1
