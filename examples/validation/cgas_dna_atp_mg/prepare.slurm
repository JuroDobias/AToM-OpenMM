#!/bin/bash
#SBATCH --job-name=cgas-mg-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myatom_openmm86
export PYTHONPATH="$(cd ../../.. && pwd):${PYTHONPATH:-}"
python -m atom_openmm.md_validation workflow.yaml --prepare-only
