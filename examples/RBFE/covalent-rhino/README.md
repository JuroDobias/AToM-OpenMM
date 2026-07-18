# Covalent RHINO pilot

This example exercises the experimental direct-endpoint covalent NEQTI workflow.
It requires a normalized dataset produced with:

```bash
atom-rbfe-prepare-covalent-dataset /path/to/rhino_benchmarks \
  /path/to/rhino_benchmarks/normalized
```

Adjust `workflow.dataset` in `workflow.yaml`, then check and run it:

```bash
atom-rbfe --validate workflow.yaml
atom-rbfe --plan-only workflow.yaml
atom-rbfe workflow.yaml
```

The first setup can be CPU-heavy because Espaloma is loaded and both solvated
endpoint systems are built. Production switching and five-replica REST2 should be
run on a CUDA-capable GPU. This prototype is restricted to the prepared
cysteine-aldehyde thiohemiacetal series and the force fields shown in the workflow.
