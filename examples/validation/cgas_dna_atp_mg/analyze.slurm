#!/bin/bash
#SBATCH --job-name=cgas-mg-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myatom_openmm86
export PYTHONPATH="$(cd ../../.. && pwd):${PYTHONPATH:-}"
python -m atom_openmm.md_validation workflow.yaml --analyze
python -m atom_openmm.md_validation_dcd \
  run_gaff2_tip4pew/prepared/canonical_topology.cif run_gaff2_tip4pew/tasks/*
