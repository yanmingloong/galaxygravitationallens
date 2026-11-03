"""Tests for validation prior sensitivity + corner helpers."""

import numpy as np
import pytest

from galaxylens.lensfit.validation import ml_prior_reweight, ess


def test_ess_uniform():
    # equal weights over N samples -> ESS = N
    w = np.ones(1000)
    assert ess(w) == pytest.approx(1000.0, rel=1e-3)


def test_ml_prior_reweight_centers_mass():
    rng = np.random.default_rng(0)
    lg_ml = rng.normal(9.0, 0.3, 100000)
    w = np.ones(100000)
    # a prior centred above the sample mean shifts the reweighted mean up
    w_hi = ml_prior_reweight(lg_ml, w, lg_ml_center=9.5)
    mean_hi = np.average(lg_ml, weights=w_hi)
    w_lo = ml_prior_reweight(lg_ml, w, lg_ml_center=8.5)
    mean_lo = np.average(lg_ml, weights=w_lo)
    assert mean_hi > 9.0 > mean_lo
