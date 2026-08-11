# GROMACS Gapsys cross-engine validation

This fixed-coordinate regression reuses the charged and charge-free methane in
periodic TIP3P water from the Amber GTI SSC2 validation. The same Amber topology
and coordinates are converted to GROMACS with ParmEd. Both GROMACS and OpenMM
then annihilate `MTH` with the Gapsys LJ and Coulomb softcore functions at seven
lambda values.

The charged case validates the combined Gapsys electrostatic and LJ path. The
charge-free control isolates Gapsys LJ. Relative potential energies and
`dU/dlambda` pass at maximum errors of 0.02 and 0.10 kcal/mol, respectively.

Run on an Aurum gen-d node:

```bash
sbatch run.sh
```

GROMACS 2026 has no Gapsys free-energy GPU kernel. Its reference calculations
therefore use `-nb cpu -pme cpu` on 48 CPU cores. The allocated L40S runs the
OpenMM CUDA double-precision evaluations. Results are written to
`comparison_charged.csv`, `comparison_lj_only.csv`, `result_charged.yaml`, and
`result_lj_only.yaml`.
