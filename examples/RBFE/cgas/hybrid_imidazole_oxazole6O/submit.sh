#!/bin/bash
set -euo pipefail
sbatch --array=0-1 run_array.slurm
