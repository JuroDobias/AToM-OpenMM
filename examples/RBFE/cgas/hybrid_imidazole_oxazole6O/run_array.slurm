#!/bin/bash
#SBATCH --job-name=cgas-hyb-mg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint="[gen-b|gen-d]"
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myatom_openmm86
export PYTHONPATH="$HOME/myAToM/AToM-OpenMM-cgas-validation${PYTHONPATH:+:$PYTHONPATH}"

WORKFLOWS=(workflow.standard.yaml workflow.panteva.yaml)
WORKFLOW=${WORKFLOWS[${SLURM_ARRAY_TASK_ID}]}
CHILD=""
WALLTIME=0

on_term() {
  WALLTIME=1
  if [[ -n "${CHILD}" ]]; then
    kill -TERM "${CHILD}" 2>/dev/null || true
  fi
}
trap on_term TERM

atom-rbfe "${WORKFLOW}" &
CHILD=$!
if wait "${CHILD}"; then
  exit 0
fi
if [[ "${WALLTIME}" -eq 1 ]]; then
  sbatch --dependency=afterany:"${SLURM_JOB_ID}" --array="${SLURM_ARRAY_TASK_ID}" run_array.slurm
  exit 0
fi
echo "${WORKFLOW} failed; inspect its Slurm log before resubmission." >&2
exit 1

