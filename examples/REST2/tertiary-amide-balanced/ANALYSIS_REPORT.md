# Balanced Tertiary-Amide REST2 Validation Report

Analysis date: 2026-07-15

## Objective

This experiment tests whether the REST2 implementation reproduces the equilibrium
population of two tertiary-amide torsional basins. It compares ordinary molecular
dynamics, two REST2 simulations initialized in opposite basins, and an independent
umbrella-sampling PMF.

The monitored O=C-N-C torsion uses 1-based SDF atom IDs `26, 25, 27, 28`.
Basin A is defined as `-90 <= torsion <= 90` degrees; all remaining angles belong
to basin B.

## Simulation Protocol

- Physical temperature: 300 K
- Timestep: 2 fs
- Ordinary MD: four independent 20 ns runs
- REST2: two independently initialized ensembles
- REST2 sampling: 20 ns per replica and ensemble
- REST2 burn-in: 2 ns per ensemble
- REST2 effective temperatures: 300, 351, 411, 481, 563, 658, 770, and 900 K
- Exchange attempts: every 500 steps, or 1 ps
- Umbrella sampling: 24 windows spaced by 15 degrees
- Umbrella production: 2 ns per window

## Equilibrium Populations

| Method | Basin A population | 95% confidence interval |
| --- | ---: | ---: |
| REST2, combined | 0.5359 | 0.5060-0.5662 |
| REST2, A-started | 0.5699 | 0.5237-0.6081 |
| REST2, B-started | 0.5019 | 0.4612-0.5344 |
| Umbrella PMF | 0.5322 | 0.4534-0.6054 |
| Ordinary MD, combined | 0.6453 | 0.5942-0.7034 |

The absolute difference between the combined REST2 and umbrella populations is
`0.0037`. Their confidence intervals overlap.

At 300 K, the populations correspond to:

| Method | DG(B-A) |
| --- | ---: |
| REST2 | 0.086 kcal/mol |
| Umbrella PMF | 0.077 kcal/mol |

The difference between the two estimates is approximately `0.009 kcal/mol`.

## Distribution Agreement

Comparison of the post-burn-in physical REST2 histogram with the umbrella-derived
probability distribution gives:

- Histogram correlation: 0.993
- Jensen-Shannon divergence: 0.0015
- Total variation distance: 0.046

The agreement extends beyond the integrated basin populations to the shape of the
torsional distribution.

## REST2 Exchange Quality

Neighbor acceptance rates were uniformly high:

```text
A-started: 0.407 0.407 0.415 0.416 0.429 0.428 0.448
B-started: 0.398 0.411 0.415 0.416 0.431 0.440 0.423
```

Both ensembles completed 20,000 exchange cycles. The analysis recorded about 765
complete ladder round trips, with no evident exchange bottleneck.

Direct basin changes observed during propagation increased with effective
temperature:

| Effective temperature | Approximate basin changes/ns |
| ---: | ---: |
| 300 K | 0 |
| 351 K | 0-0.33 |
| 411 K | 0.39-0.67 |
| 481 K | 1.06-1.39 |
| 563 K | 3.17-3.72 |
| 658 K | 6.72-7.83 |
| 770 K | 11.0-11.9 |
| 900 K | 18.6-20.0 |

The barrier begins to be crossed on a useful timescale around 400-500 K and is
sampled rapidly above approximately 560 K.

The thousands of changes in the physical-replica torsion trace are not 300 K
kinetic transitions. Replica exchange changes which walker occupies the physical
Hamiltonian, transporting conformations generated in heated states into the
physical ensemble.

## Ordinary-MD Behavior

Ordinary MD remained strongly dependent on the initialized rotamer:

| Run | Initial basin | Basin A fraction | Sustained transitions |
| --- | --- | ---: | ---: |
| 1 | A | 1.0000 | 0 |
| 2 | B | 0.5810 | 1, at approximately 8.38 ns |
| 3 | A | 1.0000 | 0 |
| 4 | B | 0.00005 | 0 |

Run 4 contained two adjacent saved-frame basin changes at approximately 15.34 ns,
but no sustained transition. The combined ordinary-MD population is therefore
controlled by starting conditions and one rare transition rather than converged
equilibrium sampling.

## Umbrella PMF

- Basin A minimum: approximately -2.5 degrees
- Basin B minimum: approximately -147.5 degrees
- Basin B minimum relative to A: 2.62 kJ/mol
- Barrier near -90 degrees: 22.0 kJ/mol
- Barrier near +90 degrees: 24.7 kJ/mol
- Minimum adjacent-window histogram overlap: 0.0895

The 22-25 kJ/mol barriers explain the lack of reliable transitions in ordinary
300 K MD.

## Convergence Assessment

The two REST2 ensembles differ in basin A population by `0.068`. Their confidence
intervals overlap only narrowly. Post-burn-in 2 ns block populations fluctuate
from approximately 0.33 to 0.68.

Consequently, the combined REST2 result is strongly consistent with umbrella
sampling, but each 18 ns physical ensemble is not independently converged to high
precision. Some of the exceptional combined agreement may arise from cancellation
between residual biases from the opposing starting conformations.

For quantitative endpoint populations, use longer physical sampling or multiple
independent REST2 ensembles initialized in both basins. Retaining opposite starting
conformations is valuable because it exposes residual convergence error that a
single trajectory could hide.

## Conclusion

This experiment validates the REST2 implementation for the tested solvated
tertiary amide:

1. The temperature ladder has healthy and uniform exchange acceptance.
2. Conformations make many complete trips through the ladder.
3. Heated replicas cross the amide barrier at the expected increasing rate.
4. The physical REST2 ensemble reproduces the independent umbrella distribution.
5. Ordinary MD is not sufficient for this torsion on a 20 ns timescale.

The eight-state 300-900 K ladder is suitable for initial ATM endpoint integration.
Longer or replicated sampling remains necessary when precise conformational
populations are needed for free-energy population corrections.
