# RHINO covalent repeat cohort

This cohort validates mature covalent NEQTI controls and conservative mapped-atom
transmutation on ten B14 edges. Three repeats produce 30 independent production
jobs. The graph contains the `543-644-675` and `543-733-734` cycles, plus hard
P4 branches to `736` and the weak inhibitor `793`.

`733 -> 734` maps Br directly to Cl. `820 -> 866` maps the two aromatic H atoms
to F. Local additions use maximal MCS, while larger P4 replacements use a
restricted common scaffold and endpoint-unique branches.

The production protocol uses Gapsys sterics, staged charge removal/addition,
schedule optimization, adaptive 100/300/1000 ps switching, independent
protein/reference convergence, REST2 endpoint sampling, and stationarity checks.
Old covalent results are comparison data only because their mapping and stopping
rules differ.

Generate the template and repeats with:

```bash
python examples/RBFE/benchmarks/generate_covalent_rhino_repeat_graph.py \
  --graph examples/RBFE/benchmarks/rhino_covalent_repeat_graph.yaml \
  --dataset ../rhino_benchmarks/normalized/dataset.yaml \
  --runtime-dataset /home/jurajdobias/myAToM/rhino_benchmarks/normalized/dataset.yaml \
  --output /path/to/template

atom-rbfe-generate-repeats /path/to/template /path/to/repeats \
  --repeats 3 --seed-base 20260823
```

Review every generated `covalent_mapping.yaml` after preparation, especially the
two paired-SMARTS mappings, before treating production results as valid.
