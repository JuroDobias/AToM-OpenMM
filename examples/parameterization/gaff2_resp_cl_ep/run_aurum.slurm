#!/bin/bash -l
#SBATCH -J atom-resp
#SBATCH --constraint=gen-a
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --mem=87G
#SBATCH --exclusive

set -euo pipefail

submission_dir=$(pwd)
scratch_root="/scratch/slurm_jobs/$USER/$SLURM_JOB_ID"
if [[ ! -d "$scratch_root" ]]; then
    scratch_root="/dev/shm/$USER/$SLURM_JOB_ID"
    mkdir -p "$scratch_root"
fi
export ATOM_PARAMETERIZATION_WORKDIR="$scratch_root/atom-parameterization"
export GAUSS_SCRDIR="$scratch_root/gaussian"
mkdir -p "$ATOM_PARAMETERIZATION_WORKDIR" "$GAUSS_SCRDIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myatom_openmm86
export PYTHONPATH="$HOME/myAToM/AToM-OpenMM-covalent-repeat${PYTHONPATH:+:$PYTHONPATH}"

ulimit -s hard
export g16root=/uochb/soft/generic/gaussian/gaussian16_a03/arch/amd64-pgi_12.10-acml
export LD_LIBRARY_PATH="$g16root/g16${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GAUSS_EXEDIR="$g16root/g16"

cd "$submission_dir"
python -m atom_openmm.ligand_parameterization parameterization.yaml

rm -rf "$scratch_root"
