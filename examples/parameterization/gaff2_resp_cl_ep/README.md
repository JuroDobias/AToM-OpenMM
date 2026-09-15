# GAFF2 multi-conformer RESP with chlorine sigma holes

This example prepares a content-addressed ligand parameter bundle before an
RBFE GPU job. Copy or link an explicit-hydrogen 3D ligand as `ligand.sdf`, then
submit the CPU-only Gaussian job:

```bash
sbatch run_aurum.slurm
```

The script invokes `python -m atom_openmm.ligand_parameterization`, so a source
checkout on `PYTHONPATH` is sufficient even before reinstalling the console
entry point.

Gaussian runs only in node-local scratch. The completed bundle is published
under `parameter_cache/artifacts/`, with an identity/protocol index below
`parameter_cache/index/`. Repeating the command returns the completed cache
entry without rerunning QM.

Use the artifact from a noncovalent hybrid workflow with:

```yaml
workflow:
  setup:
    ligand_forcefield: gaff-2.2.20
    ligand_charge_model: resp-sigma-hole
    ligand_parameter_cache: /absolute/shared/path/parameter_cache
    ligand_parameter_protocol: gaff2-resp-cl-ep-v1
```

The first implementation requires every sigma-hole C-Cl group and its frame
atom to be mapped between both endpoint ligands. Halogen creation or deletion
is rejected. `ligand.mol2` and `ligand.frcmod` are interoperability outputs;
the manifest and `system.xml` are authoritative because MOL2 cannot represent
the off-center virtual site.
