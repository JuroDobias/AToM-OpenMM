# Amber GTI SSC2 cross-engine validation

This fixed-coordinate regression compares AToM-OpenMM's `amber_ssc2`
Hamiltonian with Amber 26 `pmemd.cuda_DPFP`. Both engines evaluate the same
neutral methane annihilation in periodic TIP3P water at seven lambda values.
The PME cutoff, Ewald coefficient, grid, SSC exponents, coefficients, and
8-10 A softcore cutoff smoothing are matched explicitly.

The comparison uses relative potential energies to remove lambda-independent
engine differences. Amber `DV/DL` is compared with an analytic OpenMM
chain-rule derivative assembled from force parameter derivatives and the two
weighted endpoint electrostatic Hamiltonians. It passes when the maximum
errors are at most 0.02 kcal/mol for relative energy and 0.1 kcal/mol for
`dU/dlambda`.

Run on an Aurum `gen-d` node:

```bash
sbatch run.sh
```

The charge-free control isolates SSC2 Lennard-Jones:

```bash
sbatch --export=ALL,METHANE_MOL2=methane_lj_only.mol2 run.sh
```

The validated Aurum runs used AmberTools/pmemd 26, `pmemd.cuda_DPFP`, and
OpenMM CUDA double precision. The charged test reached maximum energy and
derivative errors of 0.00337 and 0.00344 kcal/mol; the LJ-only control reached
0.00257 and 0.00250 kcal/mol.

Results are written to `comparison.csv` and `result.yaml`.
