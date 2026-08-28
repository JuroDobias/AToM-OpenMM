# MCL-1 ring-closure repeat graph

This cohort tests soft alchemical ring closure in two independent
thermodynamic cycles:

- `27 -> 45 -> 46 -> 27`: phenyl, tetralin, and indane.
- `27 -> 43 -> 47 -> 27`: phenyl, naphthalene, and quinoline.

Every primary edge belongs to a thermodynamic cycle. Radial edges from ligand
27 preserve its complete heavy-atom graph and use one soft alchemical closure
bond. The `45 -> 46` and `43 -> 47` cross-edges close the thermodynamic cycles
with conventional MCS mappings.

The cohort also runs `27 -> 43` with the complete terminal phenyl and
naphthyl groups left endpoint-specific. This whole-ring-unmapped calculation
is a mapping-method control. It is analyzed separately and is not included as
a second observation in the primary graph fit.

Each edge is run in triplicate. Therefore a fresh cohort contains 18 primary
jobs and three control jobs. An existing first `27 -> 46` pilot, including an
active resumable pilot, can provide that edge's first replicate through a
read-only run-directory symlink. This reduces the new production calculations
to 20.

Generate the cohort with:

```bash
python examples/RBFE/benchmarks/generate_hybrid_mcl1_ring_cycles.py \
  --benchmark-root ../ATM_benchmark \
  --pilot-workflow examples/RBFE/mcl1/local_runs/soft_bond_annulation_27_46/workflow.aurum.yaml \
  --output /path/to/hybrid_mcl1_ring_cycles \
  --repeats 3 \
  --existing-27-46 /path/to/completed/27-46/run
```

After completion, run `primary_cycles/analyze_repeats.sh` for cycle closure,
random-effects edge estimates, graph-fitted node energies, and comparisons to
experiment and the published ATM/GAFF2 results. Analyze the mapping control
with `whole_ring_control/analyze_repeats.sh`.
