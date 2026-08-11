#!/usr/bin/env bash
#SBATCH --job-name=gapsys-xval
#SBATCH --constraint=gen-d
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}
SOURCE_DIR=${SOURCE_DIR:-$HOME/myAToM/AToM-OpenMM-separated-topology}
AMBERHOME=/uochb/soft/generic/amber/ambertools26
SPACK_SETUP=/uochb/soft/d/spack/spack-1.1.1/share/spack/setup-env.sh
LAMBDA_VALUES=(0.00 0.10 0.25 0.50 0.75 0.90 1.00)
GMX_BIN=${GMX_BIN:-}
OPENMM_PLATFORM=${OPENMM_PLATFORM:-CUDA}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /uochb/soft/generic/conda_env/amber
set +u
source "$AMBERHOME/amber.sh"
set -u

prepare_system() {
    local name=$1
    local mol2=$2
    local target="$ROOT/$name"
    mkdir -p "$target"
    sed "s|@MOL2@|$mol2|g" \
        "$SOURCE_DIR/examples/RBFE/benchmarks/amber_gti_cross_validation/tleap.in" \
        > "$target/tleap.in"
    (cd "$target" && tleap -f tleap.in > tleap.out)
    (
        cd "$target"
        python - <<'PY'
import parmed as pmd
system = pmd.load_file("system.parm7", "system.rst7")
system.save("system.top", overwrite=True)
system.save("system.gro", overwrite=True)
PY
    )
}

prepare_system charged \
    "$SOURCE_DIR/examples/RBFE/benchmarks/amber_gti_cross_validation/methane.mol2"
prepare_system lj_only \
    "$SOURCE_DIR/examples/RBFE/benchmarks/amber_gti_cross_validation/methane_lj_only.mol2"

conda deactivate || true
if [[ -z "$GMX_BIN" ]]; then
    set +u
    source "$SPACK_SETUP"
    spack load gromacs@2026.2+cuda~mpi
    set -u
    GMX_BIN=gmx
fi
"$GMX_BIN" --version

run_gromacs() {
    local name=$1
    local target="$ROOT/$name"
    for value in "${LAMBDA_VALUES[@]}"; do
        local run="$target/gmx_$value"
        mkdir -p "$run"
        sed "s/@LAMBDA@/$value/g" "$ROOT/gromacs_mdp.template" > "$run/md.mdp"
        "$GMX_BIN" grompp -f "$run/md.mdp" -p "$target/system.top" \
            -c "$target/system.gro" -o "$run/topol.tpr" -maxwarn 1 \
            > "$run/grompp.out" 2>&1
        (
            cd "$run"
            "$GMX_BIN" mdrun -s topol.tpr -deffnm md -nb cpu -pme cpu \
                -dhdl dhdl.xvg -ntomp "${SLURM_CPUS_PER_TASK:-48}" \
                > mdrun.out 2>&1
            test -s dhdl.xvg
            printf 'Potential\n0\n' | "$GMX_BIN" energy -f md.edr -o potential.xvg \
                > energy.out 2>&1
        )
    done
}

run_gromacs charged
run_gromacs lj_only

conda activate myatom
export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/compare_openmm.py" "$ROOT/charged" --platform "$OPENMM_PLATFORM"
mv "$ROOT/charged/comparison.csv" "$ROOT/comparison_charged.csv"
mv "$ROOT/charged/result.yaml" "$ROOT/result_charged.yaml"
python "$ROOT/compare_openmm.py" "$ROOT/lj_only" --platform "$OPENMM_PLATFORM"
mv "$ROOT/lj_only/comparison.csv" "$ROOT/comparison_lj_only.csv"
mv "$ROOT/lj_only/result.yaml" "$ROOT/result_lj_only.yaml"
