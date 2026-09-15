# cGAS DNA/ATP/Mg validation report

## Scope

The corrected validation used one canonical 90,782-particle
cGAS/6OMe/ATP/2 Mg/Zn/DNA system with 20,866 TIP4P-Ew waters. The ligand used
GAFF 2.2.20 with AM1-BCC charges. Four ATP/Mg combinations were simulated in
three independent replicas using the robust restrained-to-unrestrained
equilibration protocol followed by 20 ns NPT production. All 12 replicas
completed with 2,000 production frames each.

Values below are final-10-ns means followed by the between-replica standard
deviation. DNA values come from the corrected component-aware imaging analysis.

## Structural stability

| ATP model | Mg model | Protein RMSD (nm) | ATP RMSD (nm) | 6OMe RMSD (nm) | DNA RMSD (nm) | Active-site RMSD (nm) |
|---|---|---:|---:|---:|---:|---:|
| Legacy | 12-6 | 0.170 +/- 0.005 | 0.055 +/- 0.002 | 0.060 +/- 0.003 | 0.419 +/- 0.085 | 0.060 +/- 0.007 |
| Legacy | Panteva m12-6-4 | 0.158 +/- 0.003 | 0.062 +/- 0.004 | 0.066 +/- 0.006 | 0.438 +/- 0.044 | 0.062 +/- 0.002 |
| Hu 2024 B3 | 12-6 | 0.168 +/- 0.011 | 0.063 +/- 0.016 | 0.096 +/- 0.036 | 0.478 +/- 0.088 | 0.071 +/- 0.012 |
| Hu 2024 B3 | Panteva m12-6-4 | 0.164 +/- 0.006 | 0.064 +/- 0.009 | 0.114 +/- 0.014 | 0.397 +/- 0.021 | 0.072 +/- 0.016 |

All four models preserve the global complex, ATP pose, DNA-bound assembly,
active site, and Zn coordination over 20 ns. The legacy ATP variants are more
reproducible in ligand pose and local Mg donor identity.

## Magnesium coordination

Mg1 is ligand-associated and Mg2 is ATP-side. Coordination is the number of O
or N atoms within 0.25 nm under periodic boundary conditions.

| ATP model | Mg model | Mg1 coordination | Mg1 ATP donors | Mg1 ligand donors | Mg1 water donors | Mg2 coordination | Mg-Mg (nm) |
|---|---|---:|---:|---:|---:|---:|---:|
| Legacy | 12-6 | 5.907 +/- 0.017 | 0.998 | 1.909 | 0.000 | 5.983 +/- 0.002 | 0.3528 +/- 0.0004 |
| Legacy | Panteva m12-6-4 | 5.999 +/- 0.001 | 1.000 | 1.999 | 0.000 | 6.005 +/- 0.010 | 0.3560 +/- 0.0006 |
| Hu 2024 B3 | 12-6 | 5.992 +/- 0.008 | 0.661 | 1.331 | 1.000 | 6.042 +/- 0.073 | 0.3870 +/- 0.0419 |
| Hu 2024 B3 | Panteva m12-6-4 | 6.000 +/- 0.001 | 0.667 | 0.895 | 1.438 | 6.600 +/- 0.429 | 0.3694 +/- 0.0511 |

Legacy ATP with standard 12-6 Mg retains almost two ligand donors and keeps
both Mg shells close to six-coordinate. Legacy ATP with Panteva m12-6-4 is
even more reproducible: both ligand donors remain coordinated and Mg2 stays
near six in every replica. This corrects the earlier result obtained before
GAFF2/TIP4P-Ew matching and the robust equilibration protocol.

The Hu B3 variants show replica-dependent Mg1 donor replacement. In the
standard model, the three replicas respectively sample approximately
ATP/ligand/water donor counts of `1/2/0`, `1/1/1`, and `0/1/2`. Hu/Panteva is
also heterogeneous and overcoordinates Mg2 on average. Combining Hu B3 ATP
with Panteva Mg is therefore not supported by this validation.

## Technical corrections

1. Ligand and water models are now matched to the Panteva parameter ecosystem:
   GAFF2/AM1-BCC and TIP4P-Ew.
2. Equilibration starts with ligand relaxation while restraining the
   environment, then releases solute and Mg-ligand restraints gradually.
3. The first NVT phase regenerates velocities after minimization.
4. Mg CustomNonbondedForce objects copy all base nonbonded exclusions.
5. DNA RMSD is recomputed after reconstructing separately wrapped solute
   components around the protein.

## Conclusion

Legacy ATP is the preferred ATP model for the first cGAS RBFE tests. Both Mg
models are stable enough for a controlled comparison. Panteva m12-6-4 is the
primary candidate because it most consistently preserves the intended Mg1
chelation without overcoordinating Mg2; standard 12-6 remains the sensitivity
control.

This experiment establishes structural stability, not RBFE accuracy. The next
test is the neutral `imidazole -> oxazole6O` hybrid edge using both Mg models,
identical equilibration, and no REST2.
