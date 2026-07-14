# Tertiary-amide REST2 stress test

This solvated-ligand benchmark tests whether REST2 accelerates a high-barrier
tertiary-amide rotation. The monitored 1-based SDF torsion is
`25(O)-24(C=O)-26(N)-27(C)`. The input starts near 0 degrees, and the two amide
rotamers are separated at +/-90 degrees for population and transition counting.

The neutral SDF is parameterized during preparation with AmberTools using
GAFF2 and AM1-BCC, solvated in TIP3P, and sampled with an eight-state REST2
ladder spanning effective temperatures from 300 to 900 K.

```bash
cd examples/REST2/tertiary-amide
python -m atom_openmm.rest2_validation pilot.yaml --stage all --resume
```

For scheduler use, run `prepare`, `md`, `rest2`, `umbrella`, and `analyze` as
separate stages. `rest2/state_torsions.csv` records the torsion before and after
every propagation block for every walker, together with its assigned scale and
effective temperature. The per-temperature transition rates are summarized in
`result.yaml` under `rest2.propagation_by_state`.
