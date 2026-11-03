"""Tests for validation diagnostics (Fourier bar test) + closure smoke."""

import numpy as np
import pytest

from galaxylens.lensfit.structures import (
    Geometry, StellarTemplate, Calibration,
)
from galaxylens.lensfit.validation import fourier_modes, sample_f_dm


def test_fourier_modes_bar():
    # a two-fold (bar) distribution -> m=2 amplitude significant
    rng = np.random.default_rng(0)
    n = 20000
    R = rng.uniform(1.0, 10.0, n)
    # bar: angles clustered along +-x axis (phi = 0 and pi)
    phi = rng.choice([0.0, np.pi], size=n) + rng.normal(0, 0.3, n)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    rbins = np.linspace(0, 12, 13)
    Rc, amps = fourier_modes(x, y, np.ones(n), rbins)
    # m=2 amplitude near 1 in the well-populated bins
    assert np.nanmax(amps[2]) > 0.7


def test_fourier_modes_axisymmetric():
    # a round disk -> m=2 amplitude near 0
    rng = np.random.default_rng(1)
    n = 20000
    R = rng.uniform(1.0, 10.0, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    rbins = np.linspace(0, 12, 13)
    Rc, amps = fourier_modes(x, y, np.ones(n), rbins)
    assert np.nanmax(amps[2]) < 0.1


def test_sample_f_dm_bounds():
    geom = Geometry(zlens=0.138, zsrc=0.7127, inclination_deg=78.5,
                    da_mpc=500.0, pc_per_arcsec=1000.0)
    template = StellarTemplate(
        mge=(np.array([1.0]), np.array([0.5]), np.array([1.0])),
        l2d_rel=1.0, calibration=Calibration(calib=1.0),
    )
    fd = sample_f_dm(9.0, 200.0, 1.5, 5.0, 0.7, template, geom,
                     {"2.2Rd": 1.0})
    for key in ("f_dm_spherical", "f_dm_cyl_edgeon"):
        assert 0.0 < fd["2.2Rd"][key] < 1.0
