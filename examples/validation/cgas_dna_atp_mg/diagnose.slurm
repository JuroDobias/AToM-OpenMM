#!/bin/bash
#SBATCH --job-name=cgas-startup
#SBATCH --array=0,3,6,9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-d
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=diagnose-%A_%a.out

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myatom_openmm86
export PYTHONPATH="$HOME/myAToM/AToM-OpenMM-cgas-validation:${PYTHONPATH:-}"
python diagnose.py workflow.yaml \
  --state run/tasks/00_legacy_atp_12_6_rep1/minimized_state.xml \
  --variant "$SLURM_ARRAY_TASK_ID" --steps 10000
