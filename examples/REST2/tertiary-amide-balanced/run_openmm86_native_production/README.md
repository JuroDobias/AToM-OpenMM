# OpenMM 8.6 native replica-exchange comparison

This run repeats the balanced tertiary-amide production validation with
OpenMM 8.6 `ReplicaExchangeSampler`. It uses the same prepared system,
temperature ladder, seeds, starting rotamers, and 20 ns per replica as
`run_openmm86_production`, which uses the custom nearest-neighbor sampler.

The native sampler intentionally retains OpenMM's global random-pair exchange
kernel. Its output therefore reports observed state transitions instead of
nearest-neighbor proposal acceptance rates.

```bash
cd examples/REST2/tertiary-amide-balanced/run_openmm86_native_production
bash run.sh
```

The command is resumable. Final custom-versus-native statistics are written to
`comparison_custom_vs_native.yaml`.
