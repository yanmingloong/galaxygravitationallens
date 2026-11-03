"""Tests for mock_pipeline.lensing (arc ray-tracing + dataset merge)."""

import numpy as np
import pytest

from galaxylens.mock_pipeline.lensing import (
    build_lens_galaxy, einstein_radius, make_arc, scale_arc, merge_arc,
)


def test_scale_arc_peak():
    arc = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
    scaled, s = scale_arc(arc, arc_snr=10.0, noise_rms=0.5)
    assert scaled.max() == pytest.approx(10.0 * 0.5)
    assert s == pytest.approx(5.0 / 2.0)


def test_merge_arc_centered():
    data = np.ones((9, 9))
    arc = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    merged = merge_arc(data, arc)
    # the arc lands centred: centre pixel of the 9x9 = index (4, 4)
    assert merged[4, 4] == pytest.approx(1.0)  # arc centre is 0 -> unchanged
    assert merged[3, 4] == pytest.approx(2.0)  # arc's top edge = 1 added


def test_arc_smoke():
    """Synthetic true-mass lens -> positive theta_E and a non-empty arc."""
    surf_mass_kpc2 = np.array([1.0e9])
    sig_as = np.array([0.5])
    q_obs = np.array([0.7])
    lens, mstar = build_lens_galaxy(
        surf_mass_kpc2, sig_as, q_obs, v_vir=200.0, c_minus2=5.0, gamma=1.5,
        q_h=0.7, zlens=0.138, zsrc=0.7127, inclination=78.5,
    )
    assert mstar > 0
    theta_e, kbar = einstein_radius(lens)
    assert theta_e > 0
    arc = make_arc(lens, beta_as=0.5 * theta_e, zsrc=0.7127)
    assert arc.shape == (121, 121)
    assert arc.sum() > 0
