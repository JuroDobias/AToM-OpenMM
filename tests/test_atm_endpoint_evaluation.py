import logging

from atom_openmm.awh import GlobalGibbsDiagnostics
from atom_openmm.ommsystem import OMMSystem


class _GateForce:
    def setState0EvaluationExpression(self, expression):
        self.state0 = expression

    def setState1EvaluationExpression(self, expression):
        self.state1 = expression


def _configure(multisoftplus=False, force=None):
    system = object.__new__(OMMSystem)
    system.atmforce = force if force is not None else _GateForce()
    system.multisoftplus = multisoftplus
    system.logger = logging.getLogger("test-atm-endpoint-evaluation")
    enabled = system.configure_atm_endpoint_evaluation()
    return system, enabled


def _test_endpoint_evaluation_gates_standard_softplus():
    system, enabled = _configure()
    assert enabled
    assert system.atmforce.state0 == (
        "select(Lambda1^2+Lambda2^2, 1, step(Direction))"
    )
    assert system.atmforce.state1 == (
        "select(Lambda1^2+Lambda2^2, 1, step(-Direction))"
    )


def _test_endpoint_evaluation_gates_multisoftplus_include_lambda3():
    system, enabled = _configure(multisoftplus=True)
    assert enabled
    assert "Lambda3^2" in system.atmforce.state0
    assert "Lambda3^2" in system.atmforce.state1


def _test_endpoint_evaluation_falls_back_on_older_openmm():
    _, enabled = _configure(force=object())
    assert not enabled


def _test_global_gibbs_saturation_diagnostics_round_trip():
    diagnostics = GlobalGibbsDiagnostics()
    diagnostics.record_saturated_inner_energies(["u1"])
    diagnostics.record_saturated_inner_energies(["u0", "u1"])
    restored = GlobalGibbsDiagnostics.from_dict(diagnostics.to_dict())
    assert restored.saturated_inner_energy_scans == 2
    assert restored.saturated_inner_energy_counts == {"u0": 1, "u1": 2}
