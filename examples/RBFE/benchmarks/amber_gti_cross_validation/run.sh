#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gen-d
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 00:30:00
#SBATCH --job-name=gti-xval
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}
SOURCE_DIR="$HOME/myAToM/AToM-OpenMM-ssc2"
METHANE_MOL2=${METHANE_MOL2:-methane.mol2}
AMBERHOME=/uochb/soft/generic/amber/ambertools26
PMEMDHOME=/uochb/soft/generic/amber/pmemd26

cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /uochb/soft/generic/conda_env/amber
set +u
source "$AMBERHOME/amber.sh"
source "$PMEMDHOME/amber.sh"
set -u

sed "s|@MOL2@|$METHANE_MOL2|g" tleap.in > tleap.runtime.in
tleap -f tleap.runtime.in > tleap.out
for value in 0.00 0.10 0.25 0.50 0.75 0.90 1.00; do
    sed "s/@LAMBDA@/$value/g" amber_mdin.template > "amber_${value}.in"
    pmemd.cuda_DPFP -O \
        -p system.parm7 -c system.rst7 -i "amber_${value}.in" \
        -o "amber_${value}.out" -r "amber_${value}.rst7"
done

conda activate myatom
export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/myAToM/openmm-build-env/lib:$HOME/myAToM/openmm-endpoint-gates-install/lib:${LD_LIBRARY_PATH:-}"
export OPENMM_PLUGIN_DIR="$HOME/myAToM/openmm-endpoint-gates-install/lib/plugins"
"$HOME/myAToM/atm-gates-venv/bin/python" compare_openmm.py "$ROOT" --platform CUDA
