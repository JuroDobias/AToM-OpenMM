# Solvated 1oiy REST2 validation

This benchmark validates REST2 independently of ATM using the CDK2 ligand `1oiy`.
It compares the `1-22-23-13` Ar-NH-Ar torsion population from conventional MD,
the physical REST2 replica, and a periodic umbrella PMF.

The GAFF2/BCC MOL2 and frcmod inputs were copied from
`compsciencelab/ATM_benchmark`, `ATM_Validation/CDK2/ligands`, on 2026-07-14.
The rounded MOL2 charges sum to approximately zero and the intended net charge is
zero. The files retain the benchmark residue name `UNL`.

Install the checkout so the console entry point is current, then run the pilot:

```bash
python -m pip install -e .
cd examples/REST2/1oiy-solvated
atom-rest2-validate pilot.yaml --stage all --resume
```

Each stage can be scheduled separately:

```bash
atom-rest2-validate pilot.yaml --stage prepare
atom-rest2-validate pilot.yaml --stage md --resume
atom-rest2-validate pilot.yaml --stage rest2 --resume
atom-rest2-validate pilot.yaml --stage umbrella --resume
atom-rest2-validate pilot.yaml --stage analyze
```

Use `production.yaml` only after the pilot acceptance rates and umbrella overlap
have been inspected. Generated files are written below `run_pilot/` or
`run_production/` and are ignored by Git. The final machine-readable report is
`result.yaml` in the selected run directory.
