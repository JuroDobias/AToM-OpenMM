# Explicit covalent ring-closure mapping

This example defines the `I79DJ_543 -> I79DJ_735` transformation using raw,
zero-based atom indices from the input aldehyde SDF files. It is suitable as a
template for external workflow generators that provide a curated atom map.

The complete common scaffold, including the isoxazole ring, is mapped. The
DJ543 methyl group and its three hydrogens are ligand-A-specific. The hydrogen
on the adjacent ring carbon is also ligand-A-specific. Both disappearing
junctions use terminal Z-matrix geometry.

DJ735 atoms 26-29 form the four-carbon benzo branch. Bond 25-26 is its terminal
Z-matrix anchor, while bond 29-30 is introduced as a soft alchemical closure
bond. The branch resolver treats the declared soft bond as open when deriving
the Z-matrix branch. This avoids mapping the DJ543 methyl carbon onto an
aromatic carbon in DJ735.

Validate without starting preparation or simulation:

```bash
python -m atom_openmm.rbfe_workflow --validate workflow.yaml
```
