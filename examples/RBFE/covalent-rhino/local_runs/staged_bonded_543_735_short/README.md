# Staged-bonded DJ543 to DJ735 smoke test

This local smoke test exercises automatic endpoint-junction Z-matrices, the
soft annulation closure bond, CCW-style pair input, and the symmetric
five-stage bonded switching path. It runs one forward and reverse switch in
the protein and capped-reference environments.

The 2 ps switch contains 1000 steps distributed as:

```text
100 discharge A
200 promote B bonded topology
400 exchange sterics and mapped parameters
200 demote A bonded topology
100 charge B
```

Run it with the OpenMM 8.6 environment:

```bash
conda activate atomopenmm_86
cd examples/RBFE/covalent-rhino/local_runs/staged_bonded_543_735_short
atom-rbfe --validate workflow.yaml
atom-rbfe --plan-only workflow.yaml
atom-rbfe workflow.yaml
```

The `run/` directory is ignored by the repository and can be removed before
repeating the smoke test from a clean state.
