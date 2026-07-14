import numpy as np
import pytest


def _test_torsion_angle_degrees_distinguishes_rotamers():
    from atom_openmm.rest2_validation import torsion_angle_degrees

    first = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]])
    second = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, -1, 0]])
    assert abs(torsion_angle_degrees(first, (0, 1, 2, 3))) < 1e-10
    assert abs(abs(torsion_angle_degrees(second, (0, 1, 2, 3))) - 180.0) < 1e-10


def _test_exchange_log_acceptance_uses_cross_energies():
    from atom_openmm.rest2_validation import exchange_log_acceptance

    assert exchange_log_acceptance(0.5, 10.0, 14.0, 20.0, 12.0) == pytest.approx(2.0)


def _test_periodic_delta_wraps_at_boundary():
    from atom_openmm.rest2_validation import periodic_delta_degrees

    assert periodic_delta_degrees([179.0, -179.0], [-179.0, 179.0]).tolist() == pytest.approx([-2.0, 2.0])


def _test_periodic_wham_recovers_uniform_distribution_from_unbiased_windows():
    from atom_openmm.rest2_validation import periodic_wham

    samples = np.linspace(-179.5, 179.5, 360)
    centers, probability, pmf, histograms, offsets = periodic_wham(
        [samples, samples], [-90.0, 90.0], 0.0, 300.0, bin_width_deg=30.0
    )
    assert len(centers) == 12
    assert probability == pytest.approx(np.full(12, 1.0 / 12.0))
    assert pmf == pytest.approx(np.zeros(12), abs=1e-10)


def _test_periodic_wham_rejects_empty_window():
    from atom_openmm.rest2_validation import REST2ValidationError, periodic_wham

    with pytest.raises(REST2ValidationError, match="every umbrella window"):
        periodic_wham([[0.0], []], [0.0, 90.0], 50.0, 300.0)
