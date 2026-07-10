def _test_restraint_resolution_is_indexed_not_step_id(monkeypatch):
    from atom_openmm import equilibration

    class FakeResolver:
        def __init__(self, topology, positions):
            pass

        def resolve(self, mask, label):
            return [7]

    monkeypatch.setattr(equilibration, "AmberMaskResolver", FakeResolver)
    steps = [
        {
            "id": "repeat",
            "type": "md",
            "n_steps": 1,
            "positional_restraints": {"mask": "@CA", "k_kcal_mol_a2": 1.0},
        },
        {"id": "repeat", "type": "md", "n_steps": 1},
    ]

    resolved = equilibration._resolve_step_restraints(steps, topology=None, positions=None)

    assert resolved == [{"positional_atom_indices": [7]}, {}]
