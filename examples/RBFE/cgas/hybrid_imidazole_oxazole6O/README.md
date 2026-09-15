# cGAS imidazole to oxazole6O hybrid NEQTI pilot

This pilot compares standard 12-6 Mg and Panteva modified 12-6-4 Mg for the
same neutral `imidazole -> oxazole6O` edge. The transformation preserves the
fused ring topology and maps all 24 heavy atoms. Atom 19 is an explicit N-to-O
transmutation; the imidazole N-H hydrogen is endpoint-A unique.

The input heavy-atom coordinates are identical and the mapping has one match.
The old AToM star calculations relative to 6OMe imply a direct reference of
approximately `+1.67 +/- 0.80 kcal/mol` in the A-to-B convention.

Both workflows use legacy ATP, GAFF2/AM1-BCC ligands, TIP4P-Ew, the robust
cGAS equilibration protocol, no REST2, and identical random seeds. The active
ligand ether oxygen is restrained to the nearest Mg only during early
equilibration, with the restraint released before production sampling.

Run locally with either:

```bash
atom-rbfe workflow.standard.yaml
atom-rbfe workflow.panteva.yaml
```

On Aurum, `submit.sh` submits both variants as separate resumable array tasks.

