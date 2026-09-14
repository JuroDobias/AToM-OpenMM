# cGAS DNA/ATP/Mg validation matrix

The archived initial 3 x 20 ns comparison and interpretation are in
[`REPORT.md`](REPORT.md).

The current experiment compares four parameter combinations using one canonical
GAFF 2.2.20/TIP4P-Ew solvated cGAS system and three independent 20 ns replicas
per combination:

1. Legacy ATP with standard 12-6 Mg.
2. Legacy ATP with Panteva modified 12-6-4 Mg.
3. Hu 2024 B3 ATP with standard 12-6 Mg.
4. Hu 2024 B3 ATP with Panteva modified 12-6-4 Mg.

The fourth combination is experimental: the ATP and Mg models were developed
independently. All variants retain the protein, DNA, ATP, two Mg ions, Zn, and
crystallographic waters. Legacy water extra particles are removed before all
retained and newly added waters are parameterized consistently as TIP4P-Ew.
Standard Mg uses the TIP4P-Ew 12-6 parameters; the Panteva variants use the
matching 180.5 kcal Angstrom^4/mol Mg-water C4 coefficient and published ATP
pair overrides.

Create `inputs/` with `protein.pdb`, `6OMe.sdf`, and the legacy `ATP.xml`. The
receptor is the DNA-containing `/home/jurajdobias/gromacs/myatm/cGAS/copy/protein/protein.pdb`;
the ligand and legacy ATP XML are from the earlier DNA-free cGAS example. Put the
official Hu supporting-information files `ATP-B3.prepi`, `ATP-B3.frcmod`, and
`mod.py` in `parameters/`. The Hu files come from the supporting information for
[Hu et al. (2024)](https://doi.org/10.1021/acs.jctc.4c01142), article 27886062
on ACS Figshare, file `ct4c01142_si_002.zip`. They are not redistributed here.
Also place AmberTools' `dat/leap/parm/lj_1264_pol.dat` in `parameters/`. The
ligand uses GAFF 2.2.20 with AM1-BCC charges, and the same GAFF atom classes
provide the generic Mg C4 polarizabilities.
The m12-6-4 overrides follow
[Panteva et al. (2015)](https://doi.org/10.1021/acs.jpcb.5b10423).

Run `bash submit.sh` on Slurm from a checkout available in the
`myatom_openmm86` environment on gen-d GPU nodes. Preparation asserts two Mg,
one Zn, one ATP, 36 DNA residues, and 137 retained four-site waters rebuilt as
TIP4P-Ew. Metrics are written every 10 ps and a portable restart state every
100 ps. Resubmitted tasks trim metrics to the last completed checkpoint;
`trajectory_segments.yaml` records the valid range of each DCD segment.

Preparation follows the restraint-release structure of the older GROMACS
workflow. It starts with a ligand-relaxation minimization while the receptor,
DNA, ATP, and metals are restrained, followed by a restrained whole-solute
minimization, 300 ps NVT, 1 ns strong-restraint NPT, 1 ns soft-restraint NPT,
500 ps receptor/DNA-restraint NPT, and 1 ns unrestrained NPT. A flat-bottom
Mg1-O2x restraint permits 1.8-3.0 Angstrom and is gradually weakened, then
removed for final unrestrained equilibration and production.

After completion, run:

```bash
python -m atom_openmm.md_validation workflow.yaml --analyze
python -m atom_openmm.md_validation_dcd \
  run_gaff2_tip4pew/prepared/canonical_topology.cif run_gaff2_tip4pew/tasks/*
```

The second command also writes `run_gaff2_tip4pew/dna_analysis_corrected.yaml` with
replicate means and between-replicate standard deviations. On Slurm,
`analyze.slurm` runs both commands after the simulation array completes.

The task metrics track Mg coordination, initial Mg-donor distances, Mg-Mg
distance, and heavy-atom RMSDs for ATP, 6OMe, and DNA.
The live CSV's structural RMSDs can jump when separate solute components cross
periodic-image boundaries. Use the corrected RMSD columns in
`dna_rmsd_corrected.csv`; the postprocessor reconstructs protein, DNA, ATP, and
ligand components before fitting and leaves the live CSV unchanged.
