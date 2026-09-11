# Covalent RHINO pilot

This example exercises the experimental direct-endpoint covalent NEQTI workflow.
It requires a normalized dataset produced with:

```bash
atom-rbfe-prepare-covalent-dataset /path/to/rhino_benchmarks \
  /path/to/rhino_benchmarks/normalized
```

Adjust `workflow.dataset` in `workflow.yaml`, then check and run it:

```bash
atom-rbfe --validate workflow.yaml
atom-rbfe --plan-only workflow.yaml
atom-rbfe workflow.yaml
```

The first setup can be CPU-heavy because Espaloma is loaded and both solvated
endpoint systems are built. Production switching and five-replica REST2 should be
run on a CUDA-capable GPU. This prototype is restricted to the prepared
cysteine-aldehyde thiohemiacetal series and the force fields shown in the workflow.

The A/B products are mapped with a connected heavy-atom MCS that must include the
Cys-SG--warhead attachment bond. Unique branches are dummies against the common
core and environment, but retain their complete unique-unique vacuum nonbonded
energy, including source-force-field 1-4 terms. `dummy_bonded_scales` reproduces
the earlier PMX defaults: full bonds, angles, and internal proper torsions, with
junction proper torsions disabled while the branch is inactive. The resolved map,
unique atom roles, and scaling values are written to `covalent_mapping.yaml`.
For edges where the unrestricted MCS is chemically undesirable, set
`workflow.alchemy.mapping.method: mcs_core_smarts` and provide a molecule-like SMARTS.
The code calculates the MCS of both aldehyde ligands and that core, selects
ambiguous matches by the smallest direct coordinate RMSD without alignment, and
transfers the result into the generated covalent products. A pair-level `mapping`
can override the workflow default.

The modified residue uses the receptor ff19SB charges for N, H, CA, HA, C, and O.
CB, HB2, HB3, SG, the transferred hydrogen, and ligand atoms are adjusted together
to make the modified Cys-ligand residue neutral. These exact charges are reused in
the protein and capped-reference systems.

Each physical endpoint is equilibrated with separately checkpointed minimization,
NVT, and NPT stages. A resumed workflow skips every completed stage, including
completed protein samples, before continuing the capped-reference calculation.

Every nonequilibrium switch writes a full hybrid structure named
`{protein|reference}_{forward|reverse}_sample_NNN_{pre|post}_switch.pdb`.
Inactive dummy atoms remain present with PDB occupancy `0.00`; active and common
atoms have occupancy `1.00`. `REMARK 901` lists the one-based dummy particle
indices for the represented endpoint, making the dummy branch directly selectable
in VMD or PyMOL.

The example uses the native `softcore_linear` path. A forward switch first removes
endpoint-A unique-branch charges, then transforms bonded and softcore Lennard-Jones
terms, and finally introduces endpoint-B charges. The reverse path executes the
same stages in reverse. `charge_steps_per_stage` applies independently to each
charge stage, so the configured 10000 + 30000 + 10000 steps produce a 100 ps
switch at 2 fs. When the mapping selects an `alchemical_bonds` ring opening, this
central stage also transforms the affected exclusions, ordinary nonbonded pairs,
and 1-4 terms. The implementation keeps PME and common protein/water forces in a
single system rather than evaluating two complete endpoint Hamiltonians.

Set `softcore.path.mode: staged_bonded` together with `softcore.total_steps` to
separate the bonded transformation from sterics. Its symmetric five-stage path
uses `10/20/40/20/10` percent of the requested total time: decharge A, promote B
bonded terms, exchange sterics and mapped parameters, demote A bonded terms, and
charge B. This is useful when a local valence change must reorganize before the
new branch acquires steric interactions. It does not increase a configured 100,
300, or 1000 ps switch.

For an edge with one selected `alchemical_bonds` closure, use
`softcore.path.mode: scheme1_soft_bond` to apply the finer two-half soft-bond
schedule. It independently controls closure stretch, angles, torsions, changing
ordinary nonbonded pairs, and 1-4 terms without changing the total switch time.

Unique branches with one mapped-core attachment use automatic
`terminal_z_matrix` framing when the mapped core supplies a connected heavy-atom
reference chain. Explicit `junction_bonds` entries override this choice. A branch
without a valid frame falls back to `bond_only` with a warning in the log and
mapping metadata.

For experimental paths, replace the two staged step settings with
`softcore.total_steps` and `softcore.path`. An empty `nodes` list with
`vdw_a: [1, 0]` and `charge_a: [1, 0]` transforms charge and van der Waals terms
simultaneously. A midpoint path with `nodes: [0.5]`,
`vdw_a: [1, 1, 0]`, and `charge_a: [1, 0.5, 0]` keeps both unique branches fully
van der Waals coupled at the midpoint. When schedule optimization is enabled,
`segments_per_interval` supplies one segment count for each path interval.

`softcore.long_range_correction` defaults to `dynamic`, which evaluates the
custom softcore Lennard-Jones long-range correction throughout the switch. The
alternative `endpoint_correction` runs the fixed-volume switch without that
expensive dynamic correction and adds its exact final-minus-initial energy
difference to the recorded work. Raw work and both endpoint corrections are
retained in `switch_lrc_diagnostics.csv`; the normal work CSVs contain corrected
values and remain the inputs to BAR.

Each pair directory records the resolved settings and compatibility fingerprint in
`switch_protocol.yaml`. Existing work is resumed only when that protocol matches.
`switch_timing.csv` and `result.yaml` report measured switching throughput. The
older `interpolation: envelope` path remains available and continues to use the
single `switch_steps` setting.

Set `neqti.rest2.execution: process` to propagate each resident REST2 replica in
its own persistent process. Exchanges remain synchronous. Optional
`device_indices` accepts either one GPU index shared by all replicas or one index
per replica; `serial` remains the default.
