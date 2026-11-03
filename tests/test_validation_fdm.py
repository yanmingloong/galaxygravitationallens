"""Regression tests for lensfit.validation (mock-closure f_dm apertures)."""

import numpy as np
import pytest

from galaxylens.lensfit.structures import Geometry
from galaxylens.lensfit.validation import (
    deproject_q, cylindrical_mass, spherical_f_dm, cylindrical_f_dm,
    bias, f_dm_bias_table,
)


def _geom(inc=90.0, pc_per_arcsec=1.0):
    return Geometry(zlens=0.138, zsrc=0.7127, inclination_deg=inc,
                    da_mpc=500.0, pc_per_arcsec=pc_per_arcsec)


def test_deproject_q_edge_on():
    # edge-on: intrinsic axis ratio == observed
    assert deproject_q(0.5, 90.0) == pytest.approx(0.5)


def test_deproject_q_face_on_nan():
    # face-on: intrinsic flattening unconstrained -> NaN
    assert np.isnan(deproject_q(0.5, 0.0))


def test_cylindrical_mass_round_gaussian():
    # single round Gaussian at edge-on -> analytic 2 pi sigma^2 S0 (1 - e^-1/2)
    geom = _geom(inc=90.0, pc_per_arcsec=1.0)
    m = cylindrical_mass([1.0], [1.0], [1.0], 1.0, geom)
    expected = 2.0 * np.pi * 1.0 ** 2 * 1.0 * (1.0 - np.exp(-0.5))
    assert m == pytest.approx(expected, rel=1e-9)


def test_spherical_f_dm_bounds():
    # f_dm must stay in [0, 1]: no DM -> 0, DM-dominated -> close to 1
    geom = _geom(inc=90.0, pc_per_arcsec=1000.0)
    surf_lum = np.array([1.0])
    sig_s = np.array([0.5])
    q_s = np.array([1.0])
    f_no_dm = spherical_f_dm(surf_lum, sig_s, q_s, np.array([1e-12]),
                             np.array([2.0]), 1.0, 1.0, geom)
    assert f_no_dm == pytest.approx(0.0, abs=1e-6)
    f_dm_dom = spherical_f_dm(np.array([1e-12]), sig_s, q_s, np.array([1.0]),
                              np.array([2.0]), 1.0, 1.0, geom)
    assert f_dm_dom == pytest.approx(1.0, abs=1e-6)


def test_bias_and_table():
    assert bias(0.8, 1.0) == pytest.approx(-20.0)
    fit = {"2.2Rd": {"f_dm_spherical": 0.9, "f_dm_cyl_edgeon": 0.8}}
    truth = {"2.2Rd": {"f_dm_spherical": 0.6, "f_dm_cyl_edgeon": 0.5}}
    t = f_dm_bias_table(fit, truth)
    assert t["2.2Rd"]["spherical_pct"] == pytest.approx(50.0)
    assert t["2.2Rd"]["cyl_edgeon_pct"] == pytest.approx(60.0)
