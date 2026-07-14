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
