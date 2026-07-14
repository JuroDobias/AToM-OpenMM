# REST2 validation

`atom_openmm.rest2` transforms a standard fixed-charge OpenMM `System` into a
REST2-capable system. The solute atoms are explicit: solute-solute interactions
scale by `s`, mixed interactions by `sqrt(s)`, and environment interactions are
unchanged. `s=1` reproduces the physical Hamiltonian.

The initial implementation supports `HarmonicBondForce`, `HarmonicAngleForce`,
`PeriodicTorsionForce`, and one `NonbondedForce`. Unsupported force layouts fail
before simulation. `CustomExternalForce` restraints, barostats, and center-of-mass
motion removal are retained without scaling.

The standalone `atom-rest2-validate` command intentionally does not use ATM. It
runs conventional MD, synchronous neighboring REST2 exchange, and periodic
torsion umbrellas. Exchanges use exact cross-evaluation of both configurations at
both proposed REST scales. Only samples from the walker currently assigned to
`s=1` enter the physical REST2 population.

See `examples/REST2/1oiy-solvated` for pilot and production configurations.
The `examples/REST2/tertiary-amide` stress test instead parameterizes an SDF
with GAFF2/AM1-BCC and extends the REST ladder to 900 K for a high-barrier amide
rotation.

For each REST propagation block, `state_torsions.csv` stores the torsion before
and after dynamics, the walker, assigned REST state, scale, effective
temperature, and whether the configured basin changed. Measuring changes within
a propagation block is important: comparing coordinates before and after a
replica exchange would incorrectly attribute an exchanged configuration to a
torsional transition at that temperature.
