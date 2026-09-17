#!/bin/bash
#SBATCH -J atm-resp-cgas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-d
#SBATCH --mem=100G
#SBATCH --time=12:00:00

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myatom_openmm86
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCRIPT_DIR"

atom-rbfe workflow.yaml
