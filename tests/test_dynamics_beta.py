"""Tests for the jampy beta_z-direct parameterization."""

import numpy as np
import pytest

from galaxylens.lensfit.dynamics import (
    beta_cyl, beta_z_to_ratio, ratio_to_beta_z, beta_cyl_beta_z,
)


def test_beta_z_ratio_roundtrip():
    for bz in (-0.3, 0.0, 0.4, 0.7):
        r = beta_z_to_ratio(bz)
        assert ratio_to_beta_z(r) == pytest.approx(bz)


def test_beta_z_to_ratio_monotonic():
    # more radial beta_z -> smaller ratio
    assert beta_z_to_ratio(0.5) < beta_z_to_ratio(-0.5)


def test_beta_cyl_beta_z_matches_ratio():
    # beta_cyl_beta_z(bz) == beta_cyl(sqrt(1-bz)) pointwise
    bz = 0.3
    f_bz = beta_cyl_beta_z(bz)
    f_ratio = beta_cyl(beta_z_to_ratio(bz))
    for r in (1.0, 5.0, 10.0):
        for th in (0.0, 0.5, 1.0, 1.5):
            b1, d1 = f_bz(r, th)
            b2, d2 = f_ratio(r, th)
            assert b1 == pytest.approx(b2)
            assert d1 == pytest.approx(d2)


def test_beta_cyl_beta_z_at_th_zero():
    # at theta=0 (major axis), the cylindrical-alignment beta reduces to
    # beta = 1 - ratio^-2 = 1 - 1/(1-beta_z) = -beta_z/(1-beta_z)
    bz = 0.4
    f = beta_cyl_beta_z(bz)
    beta_at_0, _ = f(1.0, 0.0)
    assert beta_at_0 == pytest.approx(-bz / (1 - bz))
