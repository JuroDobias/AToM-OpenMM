#!/bin/bash
#SBATCH -J atm-resp-cgas-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-d
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myatom_openmm86
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCRIPT_DIR"

CHILD=""
WALLTIME=0
on_term() {
  WALLTIME=1
  if [[ -n "$CHILD" ]]; then
    kill -TERM "$CHILD" 2>/dev/null || true
  fi
}
trap on_term TERM

atom-rbfe workflow.yaml &
CHILD=$!
if wait "$CHILD"; then
  exit 0
fi
if [[ "$WALLTIME" -eq 1 ]]; then
  sbatch --dependency=afterany:"${SLURM_JOB_ID}" run.slurm
  exit 0
fi
echo "Production workflow failed; inspect the Slurm log before resubmission." >&2
exit 1
