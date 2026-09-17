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
export PYTHONPATH="$HOME/myAToM/AToM-OpenMM${PYTHONPATH:+:$PYTHONPATH}"

atom-rbfe workflow.yaml
