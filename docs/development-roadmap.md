# Development Roadmap

This document records the intended development order for improving hybrid RBFE
sampling and alchemical pathways. It is a planning document, not a description
of currently supported behavior.

The immediate goal is to validate the existing hybrid NEQTI implementation on
CDK2 before adding more complex methods. Later work is informed by Amber's
smoothstep softcore (SSC), optimized phase-space overlap (Opt-PSO), and
alchemical enhanced sampling (ACES) methods.

## Development Principles

- Change one scientific component at a time and compare from identical prepared
  systems and snapshot banks.
- Compare methods at matched GPU time as well as matched sample count.
- Validate energies, forces, endpoint behavior, and restart behavior before
  using a new path for free-energy estimates.
- Keep experimental options explicit in YAML and out of default production
  settings until benchmarked.
- Implement published equations independently. Do not copy implementation code
  from Amber or other restrictively licensed sources.
- Preserve machine-readable diagnostics needed to explain failures: work
  distributions, overlap, effective sample counts, path derivatives, and timing.

## Ordered Milestones

### 1. Establish the Hybrid NEQTI Baseline

Finish and analyze the current hybrid NEQTI CDK2 cohort using the existing
staged decharge, sterics, and recharge pathway.

Record:

- binding free energies and uncertainties;
- protein and solvent-leg overlap;
- selected switching duration and failed-switch rate;
- GPU time, switching throughput, and equilibrium-MD throughput;
- comparison with experiment, published ATM results, and prior local ATM runs.

**Done when:** the cohort has a reproducible analysis table and the difficult
edges selected for subsequent paired pathway tests are documented.

### 2. Add Smoothstep Stage Interpolation

Add the second-order smoothstep function

\[
s(x) = 6x^5 - 15x^4 + 10x^3
\]

as an optional interpolation within each existing alchemical stage. Keep the
current staged path and softcore model otherwise unchanged.

Validate endpoint values, zero endpoint slopes, monotonicity, restart
serialization, and agreement between Python and integrator parameter updates.

**Done when:** unit tests pass and paired switches from identical snapshots show
that any work-distribution change comes only from interpolation.

### 3. Implement Pairwise SSC(2) Lennard-Jones Softcore

Add an experimental `amber_ssc2` sterics option based on the published SSC(2)
equations:

- pair-specific contact radii derived from the interacting atoms;
- softcore exponent 2;
- configurable LJ softcore coefficient, initially 0.5;
- smoothstep interpolation for the coupling coordinate;
- continuous energy and force behavior at endpoints and cutoffs.

Initially retain staged electrostatic decoupling. Exact SSC electrostatics under
PME require separate treatment of direct, reciprocal, self, common-core, and
softcore terms and must not be approximated by a simple custom `q/r_soft`
interaction.

**Done when:** analytical and finite-difference forces agree, endpoint energies
match the physical systems, no force discontinuities are observed, and restart
tests reproduce uninterrupted switching.

### 4. Run Controlled Softcore Comparisons

Compare Beutler, Gapsys, and SSC(2) LJ paths on selected easy and difficult CDK2
edges. Reuse identical endpoint snapshots and matched switching durations.

Measure:

- BAR overlap and uncertainty;
- dissipated work and work tails;
- failed-switch frequency;
- per-segment work and `dH/dlambda` profiles;
- GPU throughput.

Do not promote SSC(2) to a default based on stability alone. It must improve
overlap or robustness without introducing endpoint or force artifacts.

**Done when:** a benchmark report identifies which path should be used for the
next sampling-method comparison.

### 5. Implement Opt-PSO State Placement

Implement a dense-grid pilot that estimates neighboring phase-space overlap and
selects a smaller production state grid with approximately uniform overlap.
This optimizes equilibrium lambda placement and is distinct from the current
NEQTI optimizer, which redistributes switching steps along a fixed path.

Suggested first implementation:

1. Run a 31-51 state pilot grid.
2. Build the neighboring overlap matrix from reduced potentials.
3. Select the smallest state set satisfying a configurable overlap target.
4. Freeze and report the selected grid before production.

**Done when:** synthetic tests recover known difficult regions and an RBFE pilot
shows improved round trips or reduced state count at unchanged accuracy.

### 6. Compare Hybrid AWH and NEQTI Without Enhanced Sampling

Use the selected hybrid path and optimized state grid to compare AWH and NEQTI
on the same CDK2 edges. Disable REST2 and ACES for this comparison so the effect
of the free-energy sampling method is isolated.

Compare at matched total GPU time, including equilibration, adaptation, and
discarded pilot work. Report convergence versus wall time rather than only the
final uncertainty.

**Done when:** the comparison establishes which method reaches stable estimates
more efficiently and identifies edge classes where either method fails.

### 7. Add REST2 Endpoint Branches

Add optional REST2 sampling at physical endpoints, using the existing SMARTS and
ligand-role selection machinery to define the hot region. Keep the physical
replica available for unbiased endpoint snapshots and diagnostics.

Validate replica acceptance, round trips, torsion populations, snapshot
provenance, and reweighting before coupling REST2 snapshots to production
switches or state-space methods.

**Done when:** REST2 demonstrably improves slow endpoint conformational sampling
without changing the physical endpoint distribution beyond uncertainty.

### 8. Add ACES-Style Targeted Endpoint Sampling

Implement an experimental, reusable force-scaling layer that can selectively
reduce internal electrostatics, torsion barriers, and associated 1-4 terms for a
SMARTS-selected region while preserving bonds, angles, and appropriate packing
interactions.

Treat ACES and REST2 as complementary options:

- REST2 broadly tempers interactions of a hot region.
- ACES targets selected internal barriers more directly and may require fewer
  replicas for a known problematic torsion.

Test ACES first on an isolated torsional validation system, then at physical
RBFE endpoints. Do not combine it with REST2 until each method is independently
validated.

**Done when:** the physical replica reproduces the reference distribution and
the enhanced replicas increase transitions without force or exchange errors.

### 9. Add Lambda-Dependent Structural Restraints

Investigate floating-reference RMSD and Boresch-style distance, angle, and
torsion restraints for scaffold hopping, cyclization, and absolute binding free
energy calculations. Restraint free-energy corrections and symmetry treatment
must be explicit and tested.

This is lower priority for standard CDK2 common-core RBFE, where unnecessary
restraints can bias conformational populations.

**Done when:** restraint activation is continuous, analytical corrections are
validated against a controlled calculation, and YAML output fully records the
restraint definition and correction.

### 10. Validate Experimental Concerted SSC Electrostatics

An opt-in concerted Coulomb-plus-LJ SSC(2) path is now implemented from the
Amber 26 GTI CUDA equations. It keeps the PME reciprocal/self contribution,
reconstructs real-space Coulomb and exception interactions with the source-
matched SSC effective distances, and leaves the staged PME charge path as the
default. Focused analytical, endpoint energy/force, finite-difference force,
restart, and legacy-checkpoint tests cover the decomposition. The earlier
effective-distance approximation remains available as
`effective_distance_ssc2` and is deliberately not labelled Amber SSC(2).

**Done when:** an executable Amber/OpenMM cross-engine comparison and GPU
benchmarks on representative easy and poor-overlap edges show whether the extra
custom forces improve overlap enough to justify their runtime cost. Source-level
equation tests alone are not cross-engine validation. Do not promote this path
to the default before those comparisons.

## Required Benchmark Matrix

Each pathway or sampling change should include at least:

| Case | Purpose |
| --- | --- |
| Easy CDK2 edge | Detect regressions and estimator bias |
| Poor-overlap CDK2 edge | Measure practical overlap improvement |
| Large substituent change | Stress steric insertion and work tails |
| Torsion-limited edge | Evaluate REST2 or ACES sampling |
| Restarted calculation | Verify deterministic artifact and checkpoint handling |

For comparisons, keep prepared systems, initial snapshots, random seeds where
appropriate, timestep, pressure treatment, and estimator settings fixed.

## References

- Lee et al., [A Smoothstep Softcore Potential Function for Alchemical Free
  Energy Calculations](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00237).
- He et al., [A Generalized Framework for Smoothstep Softcore
  Potentials](https://pmc.ncbi.nlm.nih.gov/articles/PMC10329732/).
- Li et al., [Optimized Phase-Space Overlap for Alchemical Free-Energy
  Calculations](https://pmc.ncbi.nlm.nih.gov/articles/PMC11157682/).
- Lee et al., [Alchemical Enhanced Sampling for Protein-Ligand Binding Free
  Energy Calculations](https://pmc.ncbi.nlm.nih.gov/articles/PMC10333454/).
