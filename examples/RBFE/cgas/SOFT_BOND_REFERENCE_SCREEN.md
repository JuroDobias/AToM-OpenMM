# Pyridine-to-thiazole solvent path screen

Purpose: compare soft-bond stage ordering for `ms_491 -> ms_539`, retaining
the mapped coordinating nitrogen. This is ring contraction, not inversion of
the polycyclic stereocenters. The original input is task 10 of CCW job
`eda51b8bd36e49e78eef04f8f5789d3f` (mapping revision 59).

The experiment uses fresh solvent endpoints, cached GAFF2 RESP/sigma-hole
parameters, TIP4P-Ew, 14 A padding, and the existing 0.9 nm cutoff. There is no
protein, Mg, ATP, or 12-6-4 correction in this solvent leg. REST2 remains disabled
as in the source workflow. The screen uses its own checkout and work directory.

Each endpoint undergoes minimization, 100 ps NVT at 1 fs, and 1 ns NPT at 2 fs.
Thirty subsequent endpoint snapshots are saved 200 ps apart at fixed volume.
Snapshots 0-9 are for optimization; snapshots 10-29 are held out for evaluation.
All 24 paths share the same bank, with per-file checksums.

The 24 paths are the Cartesian product of:

| Factor | Options |
|---|---|
| Closure angular terms | Together with stretching; angular terms released first; torsions, then angles, then stretching |
| Closure strength before B vdW | Full; half, completing during vdW growth |
| Regular changing bonded terms | Crossfade A/B; introduce B before removing A |
| Mapped charge/vdW parameters | Together with regular bonded exchange; immediately afterwards |

A unique-branch charges are removed before A vdW. Both branches are decoupled
when their topology-derived nonbonded corrections are exchanged. B charges are
introduced last. Reverse switching follows the exact reverse of the same path.
Identical bonded terms and the endpoint dummy interaction policy remain intact.

Every switch is 100 ps at 2 fs. The existing optimizer runs 10 cycles per path,
using forward/reverse segment hysteresis and absolute work with an EWMA. Its
allocation is frozen before the 20 held-out switches in each direction.
The optimizer may redistribute steps, but cannot change control-node order.

Numerical switch failures are saved as infinite work; they are not replaced.
An optimizer cycle with a failed direction retains the previous allocation and
is explicitly recorded. Failed endpoint propagation stops bank preparation.
Each path audits endpoint energy and force identity before starting switches.
Completed switches and optimizer cycles resume without being repeated.

On Aurum, from the dedicated checkout:

```bash
python -m atom_openmm.soft_bond_screen generate /path/to/new-screen --source /path/to/workflow.yaml
python -m atom_openmm.soft_bond_screen submit /path/to/new-screen
```

Submission creates one bank job and 24 independent jobs depending on successful
bank completion. Each uses one gen-d GPU, no MPS, four CPU cores, and a 4-hour
limit. Resubmit an individual Slurm script from the screen directory to resume.

Inspect `bank/mapping.yaml`, the endpoint PDB snapshots, and `bank/complete.yaml`.
Each `paths/<name>/` contains `optimizer.yaml`, `frozen_schedule.yaml`,
`endpoint_audit.yaml`, per-switch records with segment work and failures, and
`result.yaml` with held-out BAR overlap, sampling uncertainty, and failure counts.
The reported free energy is a solvent transformation, not a binding ddG.
Pilot work is never pooled into held-out BAR. Because all paths share a small
bank, comparisons are paired and remain exploratory rather than independent
replicate validation.
