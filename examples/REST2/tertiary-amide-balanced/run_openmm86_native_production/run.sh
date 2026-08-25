#!/bin/bash
set -euo pipefail

ENV_DIR=/home/juro/Software/pymol/envs/atomopenmm_86
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
CUSTOM_RUN="$RUN_DIR/../run_openmm86_production"

cd "$RUN_DIR"

"$ENV_DIR/bin/python" - <<'PY'
import openmm
from openmm import app
print(f"OpenMM {openmm.__version__} ({openmm.version.git_revision})")
print("Platforms:", ", ".join(
    openmm.Platform.getPlatform(i).getName()
    for i in range(openmm.Platform.getNumPlatforms())
))
if not hasattr(app, "ReplicaExchangeSampler"):
    raise RuntimeError("OpenMM ReplicaExchangeSampler is unavailable")
PY

# Use the identical prepared system and equilibrated coordinates as the custom run.
if [[ ! -f prepared/equilibrated_state.xml ]]; then
    if [[ ! -f "$CUSTOM_RUN/prepared/equilibrated_state.xml" ]]; then
        echo "The custom OpenMM 8.6 preparation is missing: $CUSTOM_RUN/prepared" >&2
        exit 1
    fi
    cp -a "$CUSTOM_RUN/prepared" .
fi

"$ENV_DIR/bin/python" -m atom_openmm.rest2_validation \
  workflow.yaml --stage rest2 --resume 2>&1 | tee -a rest2.log

"$ENV_DIR/bin/python" -m atom_openmm.rest2_validation \
  workflow.yaml --stage analyze --resume 2>&1 | tee -a analysis.log

"$ENV_DIR/bin/python" compare.py | tee comparison.log
