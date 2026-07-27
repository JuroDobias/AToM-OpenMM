"""Analytical reconstruction of ATM state energies from one decomposition."""

from __future__ import annotations

import math

import numpy as np
from openmm import unit
from scipy.special import logsumexp


def _value(value, target_unit=None):
    if target_unit is not None and hasattr(value, "value_in_unit"):
        return float(value.value_in_unit(target_unit))
    return float(getattr(value, "_value", value))


def softcore_perturbation_energy(
    energy,
    maximum_energy,
    core_energy,
    core_exponent,
):
    """Return the ATM soft-core perturbation energy in kJ/mol."""
    energy = float(energy)
    maximum_energy = float(maximum_energy)
    core_energy = float(core_energy)
    core_exponent = float(core_exponent)
    if core_exponent == 0.0 or energy <= core_energy:
        return energy
    reduced = (energy - core_energy) / (
        core_exponent * (maximum_energy - core_energy)
    )
    zeta = 1.0 + 2.0 * reduced * (reduced + 1.0)
    powered = zeta**core_exponent
    return (
        (maximum_energy - core_energy) * (powered - 1.0) / (powered + 1.0)
        + core_energy
    )


def softplus_energy(lambda1, lambda2, alpha, threshold, offset, energy):
    """Evaluate the single-softplus ATM bias in kJ/mol."""
    lambda1 = float(lambda1)
    lambda2 = float(lambda2)
    alpha = float(alpha)
    energy = float(energy)
    result = lambda2 * energy + float(offset)
    if alpha > 0.0:
        result += (
            (lambda2 - lambda1)
            / alpha
            * np.logaddexp(0.0, -alpha * (energy - float(threshold)))
        )
    return float(result)


def multi_softplus_energy(
    lambda1,
    lambda2,
    lambda3,
    alpha,
    threshold0,
    threshold1,
    offset,
    energy,
):
    """Evaluate the three-branch ATM bias in kJ/mol."""
    lambda1 = float(lambda1)
    lambda2 = float(lambda2)
    lambda3 = float(lambda3)
    alpha = float(alpha)
    energy = float(energy)
    offset = float(offset)
    if alpha <= 0.0:
        return lambda3 * energy + offset
    f3 = lambda3 * energy + offset
    f2 = lambda2 * energy + offset + (lambda3 - lambda2) * float(threshold1)
    f1 = (
        lambda1 * energy
        + offset
        + (lambda3 - lambda2) * float(threshold1)
        + (lambda2 - lambda1) * float(threshold0)
    )
    return float(logsumexp(alpha * np.asarray([f1, f2, f3])) / alpha - math.log(3.0) / alpha)


def atm_state_energy(
    state,
    *,
    environment_energy,
    u0,
    u1,
    multisoftplus=False,
):
    """Reconstruct one ATM state's total potential energy in kJ/mol."""
    kj = unit.kilojoule_per_mole
    direction = float(state["atmdirection"])
    uoffset = _value(state["uoffset"], kj)
    raw = u1 - (u0 + uoffset)
    reference = u0
    if direction < 0.0:
        raw = -raw
        reference = u1
    perturbation = softcore_perturbation_energy(
        raw,
        _value(state["Umax"], kj),
        _value(state["Ubcore"], kj),
        float(state["Acore"]),
    )
    alpha = _value(state["alpha"], kj**-1)
    if multisoftplus:
        bias = multi_softplus_energy(
            state["lambda1"],
            state["lambda2"],
            state["lambda3"],
            alpha,
            _value(state["uh"], kj),
            _value(state["uh1"], kj),
            _value(state["w0"], kj),
            perturbation,
        )
    else:
        bias = softplus_energy(
            state["lambda1"],
            state["lambda2"],
            alpha,
            _value(state["uh"], kj),
            _value(state["w0"], kj),
            perturbation,
        )
    return float(environment_energy + reference + bias)


def reconstruct_atm_energies(
    states,
    *,
    reference_total_energy=None,
    reference_atm_energy=None,
    environment_energy=None,
    u0,
    u1,
    multisoftplus=False,
):
    """Reconstruct all physical ATM energies from one physical-state query."""
    if environment_energy is None:
        if reference_total_energy is None or reference_atm_energy is None:
            raise ValueError(
                "environment_energy or both reference energies are required"
            )
        environment = float(reference_total_energy) - float(
            reference_atm_energy
        )
    else:
        environment = float(environment_energy)
    return np.asarray(
        [
            atm_state_energy(
                state,
                environment_energy=environment,
                u0=float(u0),
                u1=float(u1),
                multisoftplus=multisoftplus,
            )
            for state in states
        ],
        dtype=float,
    )
