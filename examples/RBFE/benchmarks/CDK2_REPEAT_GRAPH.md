# CDK2 repeat-aware graph proposal

The ten representative published CDK2 transformations are prediction targets.
They are not all treated as mandatory direct simulations. The proposal in
`cdk2_repeat_graph.yaml` uses the 16 measured benchmark ligands as nodes and 25
same-charge, single-attachment-site hybrid transformations as simulated edges.

The edge set contains shortest feasible paths for every published target, links
all components to `1h1q`, and adds enough redundant edges to produce ten
independent fundamental cycles. Graph fitting therefore estimates all ten target
DDGs even when a target is represented by an indirect path. The experimental and
published ATM edge tables are fitted independently to the same reference before
node-level correlation analysis.

The graph was proposed from RDKit MCS mappings with these default filters:

- identical endpoint formal charge;
- exactly one mapped attachment site carrying all changed heavy atoms;
- at least half of the larger ligand's heavy atoms mapped;
- no automatic generation of intermediate chemistry.

The proposal still requires visual mapping review before submission. Five edges
already have three completed repeats in the previous seven-edge cohort. Importing
those results leaves 20 new edges and therefore 60 new production jobs. Repeats use unique seeds and
independent equilibration, REST2 endpoint sampling, schedule optimization, and
switching. Existing completed edge directories are retained when the graph is
expanded.

Generate a clean template and its repeats with:

```bash
python examples/RBFE/benchmarks/generate_hybrid_cdk2_repeat_graph.py \
  --graph examples/RBFE/benchmarks/cdk2_repeat_graph.yaml \
  --source-cohort /path/to/aligned/CDK2/cohort \
  --benchmark-root /path/to/ATM_benchmark \
  --output /path/to/template

atom-rbfe-generate-repeats /path/to/template /path/to/repeats \
  --repeats 3 --seed-base 20260817 \
  --reuse-repeats-from /path/to/completed/three-repeat/cohort
```

Do not run `submit_all.sh` until the mappings and cycle design have been reviewed.
