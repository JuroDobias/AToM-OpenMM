# cGAS ATM cached RESP sigma-hole pilot

This is a short ATM/NEQTI implementation pilot for the `ms_491 -> ms_492`
stereoinversion. It deliberately uses standard 12-6 Mg and disables REST2 so
the first run isolates cached GAFF2/RESP/sigma-hole installation and ATM
switching.

The two fitted chlorine extra points are included in their corresponding ATM
ligand groups. The receptor is normalized during setup, including legacy
`HD1`, `CY3`, and `ZN3` residue names and regeneration of retained TIP4P-Ew
extra particles.

The absolute `ligand_parameter_cache` path in `workflow.yaml` is the Aurum
shared cache. Adjust it when running on another machine.

```bash
atom-rbfe --validate workflow.yaml
atom-rbfe --plan-only workflow.yaml
atom-rbfe workflow.yaml
```

This protocol is a smoke test, not a converged free-energy calculation.
Panteva modified 12-6-4 is intentionally rejected for ATM until its
`CustomNonbondedForce` is transformed inside `ATMForce` and endpoint energies
are validated against direct coordinate-swapped systems.
