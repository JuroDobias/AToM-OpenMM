# Balanced tertiary-amide REST2 production test

This production validation uses `rest2_test2.sdf` and monitors the 1-based
amide torsion `26(O)-25(C=O)-27(N)-28(C)`. Atom numbering differs from the
related tertiary-amide pilot and must not be copied from that system.

Two independent REST2 ensembles share one prepared solvated system:

- `a_started`: every walker starts near 0 degrees.
- `b_started`: every walker starts near 180 degrees.

Each ensemble runs eight replicas for 20 ns per replica. The first 2 ns are
excluded from population, transition, and round-trip analysis. Outputs remain
separate under `rest2/a_started/` and `rest2/b_started/`; `result.yaml` reports
both estimates and their combined population.

```bash
cd examples/REST2/tertiary-amide-balanced
python -m atom_openmm.rest2_validation production.yaml --stage all --resume
```

Stages may be scheduled separately. With `--resume`, existing preparation and
completed checkpoints are preserved.
