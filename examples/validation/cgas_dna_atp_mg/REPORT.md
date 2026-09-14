# cGAS DNA/ATP/Mg validation report

## Scope

The validation used one canonical 69,999-particle cGAS/6OMe/ATP/2 Mg/Zn/DNA
system with 20,893 TIP3P waters. Four parameter combinations were simulated in
three independent replicas. Each replica used restrained NVT, restrained NPT,
unrestrained NPT, and 20 ns production. All 12 replicas completed with 2,000
production frames each. Wall-clock time was 1:25-1:29 per replica on gen-d GPU
nodes.

The values below are means over the final 10 ns of each replica, followed by
the between-replica standard deviation where available. Structural RMSDs were
recomputed from the DCD trajectories after reconstructing independently wrapped
solute components around the largest protein chain.

## Structural stability

| ATP model | Mg model | Protein RMSD (nm) | ATP RMSD (nm) | 6OMe RMSD (nm) | DNA RMSD (nm) | Active-site RMSD (nm) |
|---|---|---:|---:|---:|---:|---:|
| Legacy | 12-6 | 0.175 +/- 0.006 | 0.061 +/- 0.005 | 0.082 +/- 0.026 | 0.382 +/- 0.035 | 0.068 +/- 0.007 |
| Legacy | Panteva m12-6-4 | 0.177 +/- 0.006 | 0.056 +/- 0.000 | 0.112 +/- 0.017 | 0.435 +/- 0.043 | 0.066 +/- 0.002 |
| Hu 2024 B3 | 12-6 | 0.172 +/- 0.005 | 0.061 +/- 0.006 | 0.094 +/- 0.023 | 0.431 +/- 0.014 | 0.069 +/- 0.006 |
| Hu 2024 B3 | Panteva m12-6-4 | 0.171 +/- 0.007 | 0.065 +/- 0.010 | 0.143 +/- 0.041 | 0.392 +/- 0.029 | 0.075 +/- 0.017 |

All models preserve the global complex, ATP pose, and active site over 20 ns.
The Hu/Panteva combination has larger ligand and active-site variability because
one replica undergoes an Mg1 coordination-shell rearrangement.

## Magnesium coordination

| ATP model | Mg model | Mg1 coordination | Mg2 coordination | Mg-Mg (nm) | Mg1 initial-donor mean (nm) | Mg2 initial-donor mean (nm) |
|---|---|---:|---:|---:|---:|---:|
| Legacy | 12-6 | 5.88 +/- 0.09 | 5.99 +/- 0.03 | 0.352 +/- 0.001 | 0.219 +/- 0.020 | 0.200 +/- 0.000 |
| Legacy | Panteva m12-6-4 | 6.00 +/- 0.00 | 6.60 +/- 0.43 | 0.343 +/- 0.009 | 0.245 +/- 0.004 | 0.208 +/- 0.002 |
| Hu 2024 B3 | 12-6 | 5.96 +/- 0.02 | 6.09 +/- 0.11 | 0.362 +/- 0.003 | 0.234 +/- 0.020 | 0.208 +/- 0.001 |
| Hu 2024 B3 | Panteva m12-6-4 | 6.00 +/- 0.01 | 6.64 +/- 0.28 | 0.371 +/- 0.037 | 0.284 +/- 0.042 | 0.210 +/- 0.000 |

Standard 12-6 Mg keeps Mg2 close to six-coordinate. Panteva m12-6-4 makes Mg1
six-coordinate very reproducibly but drives Mg2 toward seven-coordinate: the
last 10 ns are seven-coordinate in 88% and 81% of legacy/Panteva replicas 2 and
3, and 76% and 84% of Hu/Panteva replicas 2 and 3.

Mg1 ligand/water identity is heterogeneous with standard 12-6. Legacy/Panteva
converges all three replicas to one ATP donor, one ligand donor, and one water
donor inside 0.25 nm. Hu/Panteva does not:

- Replicate 1 has no Mg1 ATP donor throughout production. It later loses its
  ligand donor and replaces the local shell with three waters.
- Replicate 2 intermittently loses the ATP and ligand donors during the final
  10 ns.
- Replicate 3 retains one ATP, one ligand, and one water donor.

This behavior means that the independently developed Hu B3 ATP and Panteva Mg
models should not be treated as a validated combined model from this experiment.
The result may reflect incompatibility of their cross interactions rather than
a defect in either model used with its intended parameter ecosystem.

## ATP conformation

The legacy models sample mean O3B-PB-O3A-PA torsions near -108 to -116 degrees
and PB-O3A-PA-O5* near -92 to -99 degrees. Hu B3 shifts these distributions to
approximately -83 to -104 and -100 to -123 degrees, respectively. This is a
clear parameter-model effect, but the present stability experiment cannot say
which distribution is more accurate. That requires comparison with the Hu
reference simulations/QM targets or experimental observables.

## Technical issues found

1. Added CustomNonbondedForce objects initially lacked the base
   NonbondedForce exception list. OpenMM therefore rejected the system with
   `All Forces must have identical exclusions`. Both Mg-related correction
   forces now copy every base exception.
2. Carrying velocities from minimization into NVT produced a reproducible NaN
   at step 251, at both 1 and 2 fs. Regenerating 300 K velocities from the same
   minimized coordinates was stable. The first restrained NVT phase now resets
   velocities, while checkpoint resumes preserve their saved velocities.
3. Live protein-aligned RMSDs were corrupted when separate protein/DNA chains
   crossed periodic boundaries. The raw trajectories were valid. The
   component-aware postprocessor now reconstructs solute images before fitting.
   Use `dna_analysis_corrected.yaml` and each task's
   `dna_rmsd_corrected.csv`, not the raw DNA RMSD in `analysis.yaml`.

## Interpretation

For a conservative first RBFE setup, legacy ATP with standard 12-6 Mg is the
least surprising baseline: it retains the crystallographic site without Mg2
overcoordination and has low replicate variability. Legacy ATP with Panteva
m12-6-4 is also globally stable and gives a very reproducible Mg1 shell, but its
strong Mg2 seven-coordination signal needs validation against the intended
Panteva coordination geometry and reference system.

Hu B3 with standard 12-6 is stable and reproducible, while changing the ATP
polyphosphate torsional ensemble as designed. Hu B3 plus Panteva m12-6-4 is not
recommended yet because its three replicas do not agree on Mg1 donor identity.

This is a structural-stability comparison, not a binding-free-energy
validation. It does not establish which model gives more accurate cGAS RBFE.
The next useful test is a small ligand-edge comparison using the two strongest
candidates, legacy/12-6 and Hu-B3/12-6, while preserving DNA and ATP.
