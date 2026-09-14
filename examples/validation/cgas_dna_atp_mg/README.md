# cGAS DNA/ATP/Mg validation matrix

This experiment compares four parameter combinations using one canonical solvated
cGAS system and three independent 20 ns replicas per combination:

1. Legacy ATP with standard 12-6 Mg.
2. Legacy ATP with Panteva modified 12-6-4 Mg.
3. Hu 2024 B3 ATP with standard 12-6 Mg.
4. Hu 2024 B3 ATP with Panteva modified 12-6-4 Mg.

The fourth combination is experimental: the ATP and Mg models were developed
independently. All variants retain the protein, DNA, ATP, two Mg ions, Zn, and
crystallographic waters. Legacy four-site water particles are removed before the
system is parameterized as TIP3P.

Create `inputs/` with `protein.pdb`, `6OMe.sdf`, and the legacy `ATP.xml`. The
receptor is the DNA-containing `/home/jurajdobias/gromacs/myatm/cGAS/copy/protein/protein.pdb`;
the ligand and legacy ATP XML are from the earlier DNA-free cGAS example. Put the
official Hu supporting-information files `ATP-B3.prepi`, `ATP-B3.frcmod`, and
`mod.py` in `parameters/`. The Hu files come from the supporting information for
[Hu et al. (2024)](https://doi.org/10.1021/acs.jctc.4c01142), article 27886062
on ACS Figshare, file `ct4c01142_si_002.zip`. They are not redistributed here.
Also place AmberTools' `dat/leap/parm/lj_1264_pol.dat` in `parameters/`;
GAFF2 typing by `antechamber` is used only for the Mg polarization lookup,
not for the Espaloma ligand force field.
The m12-6-4 overrides follow
[Panteva et al. (2015)](https://doi.org/10.1021/acs.jpcb.5b10423).

Run `bash submit.sh` on Slurm from a checkout available in the
`myatom_openmm86` environment on gen-d GPU nodes. Preparation asserts two Mg,
one Zn, one ATP, 36 DNA residues, and 137 retained four-site waters converted to
TIP3P. Metrics are written every 10 ps and a portable restart state every
100 ps. Resubmitted tasks trim metrics to the last completed checkpoint;
`trajectory_segments.yaml` records the valid range of each DCD segment.
After completion, run:

```bash
python -m atom_openmm.md_validation workflow.yaml --analyze
python -m atom_openmm.md_validation_dcd \
  run_v2/prepared/canonical_topology.cif run_v2/tasks/*
```

The second command also writes `run_v2/dna_analysis_corrected.yaml` with
replicate means and between-replicate standard deviations. On Slurm,
`analyze.slurm` runs both commands after the simulation array completes.

The task metrics track Mg coordination, initial Mg-donor distances, Mg-Mg
distance, and heavy-atom RMSDs for ATP, 6OMe, and DNA.
The live CSV's DNA RMSD can jump when DNA atoms cross the periodic-image
boundary. Use `dna_rmsd_corrected.csv` from the second command for DNA RMSD;
it computes protein-aligned, minimum-image displacements directly from each
committed DCD frame and leaves the live CSV unchanged.
