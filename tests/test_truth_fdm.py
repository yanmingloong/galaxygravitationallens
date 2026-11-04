"""Regression tests for truth particle loading, f_dm, and anisotropy.

Pure-numerical assertions validate the conventions; the f_dm / beta_z
regression anchors against the known J2141 analog 348489 (f_dm spherical
2.2Rd = 0.695, cylindrical = 0.797, beta_z 2.2Rd = +0.40) when the particle
HDF5 cutouts are available.
"""

import os

import numpy as np
import pytest

from galaxylens.mock_pipeline.truth import (
    inertia_rotation, surface_density, fit_exp_rd, f_dm_apertures, beta_z,
    load_particles, com_iterative, solve_c_from_delta, fit_halo,
)


# ---------------------------------------------------------------------------
# pure numerical
# ---------------------------------------------------------------------------
def test_inertia_rotation_aligns_long_axis():
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.normal(0, 10, n)   # long axis
    y = rng.normal(0, 10, n)
    z = rng.normal(0, 1, n)    # thin direction
    pos = np.column_stack([x, y, z])
    R, evals = inertia_rotation(pos, np.ones(n))
    assert evals[0] < evals[2]                     # ascending [min, mid, max]
    pos_al = pos @ R.T
    # after rotation the thin (normal) axis lands on z, the long axis on x
    assert pos_al[:, 2].std() < pos_al[:, 0].std()


def test_fit_exp_rd_recovers():
    rng = np.random.default_rng(1)
    rd = 5.0
    R = rng.uniform(0.5, 20.0, 500)
    sigma = np.exp(-R / rd) * (1 + rng.normal(0, 0.05, 500))
    assert fit_exp_rd(R, sigma, 5.0) == pytest.approx(rd, abs=0.5)


def test_f_dm_apertures_dm_dominated():
    rng = np.random.default_rng(2)
    pos_s = rng.normal(0, 1.0, (10000, 3))      # stars compact
    m_cur = np.full(10000, 1e7)
    pos_dm = rng.normal(0, 10.0, (20000, 3))    # DM extended
    m_dm = 1e9
    out = f_dm_apertures(pos_s, m_cur, pos_dm, m_dm, {"10kpc": 10.0})
    assert out["10kpc"]["f_dm_spherical"] > 0.5


def test_beta_z_isotropic():
    rng = np.random.default_rng(3)
    n = 20000
    R = rng.uniform(0.1, 5.0, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    pos = np.column_stack([R * np.cos(theta), R * np.sin(theta),
                           rng.normal(0, 0.1, n)])
    vel = rng.normal(0, 30.0, (n, 3))           # isotropic velocity dispersion
    b, _ = beta_z(pos, vel, np.ones(n), 10.0)
    assert b == pytest.approx(0.0, abs=0.05)


def test_beta_z_tangential():
    """Tangential orbits -> sigma_R < sigma_z -> beta_z < 0."""
    rng = np.random.default_rng(4)
    n = 20000
    R = rng.uniform(1.0, 5.0, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    pos = np.column_stack([x, y, np.zeros(n)])
    vphi = 100.0
    vx = -vphi * np.sin(theta)
    vy = vphi * np.cos(theta)
    vz = rng.normal(0, 40.0, n)                 # larger vertical dispersion
    vel = np.column_stack([vx, vy, vz])
    b, _ = beta_z(pos, vel, np.ones(n), 10.0)
    assert b < -0.1


# ---------------------------------------------------------------------------
# data regression (skip if the particle HDF5 is unavailable)
# ---------------------------------------------------------------------------
RAW = os.environ.get("GALAXYLENS_TEST_RAW_DATA", "")


def _aligned(sim, snap, sid):
    star = f"{RAW}/raw_stars/{sim}_snap{snap}_id{sid}_stars.hdf5"
    dm = f"{RAW}/raw_dm/{sim}_snap{snap}_id{sid}_dm.hdf5"
    if not (os.path.exists(star) and os.path.exists(dm)):
        pytest.skip(f"particle HDF5 not available for {sim} snap{snap} id{sid}")
    pos_s, vel_s, m_cur, m_init, pos_dm, m_dm, H, A, box, z = \
        load_particles(star, dm)
    center = com_iterative(pos_s, m_init)
    R, _evals = inertia_rotation(pos_s - center, m_init)
    pos_s_al = (pos_s - center) @ R.T
    pos_dm_al = (pos_dm - center) @ R.T
    vel_s_al = vel_s @ R.T
    v_com = np.average(vel_s_al, axis=0, weights=m_cur)
    vel_s_rel = vel_s_al - v_com
    return pos_s_al, vel_s_rel, m_cur, pos_dm_al, m_dm


def test_348489_fdm_beta_regression():
    pos_s_al, vel_s_rel, m_cur, pos_dm_al, m_dm = _aligned("TNG100-1", 88, 348489)
    R_xy = np.hypot(pos_s_al[:, 0], pos_s_al[:, 1])
    order = np.argsort(R_xy)
    m_cum = np.cumsum(m_cur[order])
    r50 = float(R_xy[order][np.searchsorted(m_cum, 0.5 * m_cum[-1])])
    rbins = np.logspace(np.log10(max(R_xy.min(), 0.05)),
                        np.log10(R_xy.max()), 30)
    Rb, Sigma = surface_density(R_xy, m_cur, rbins)
    rd = fit_exp_rd(Rb, Sigma, r50)
    out = f_dm_apertures(pos_s_al, m_cur, pos_dm_al, m_dm, {"2.2Rd": 2.2 * rd})
    assert out["2.2Rd"]["f_dm_spherical"] == pytest.approx(0.695, abs=0.002)
    assert out["2.2Rd"]["f_dm_cyl_edgeon"] == pytest.approx(0.797, abs=0.002)
    b, n = beta_z(pos_s_al, vel_s_rel, m_cur, 2.2 * rd)
    assert n > 1000
    assert b == pytest.approx(0.40, abs=0.03)


# ---------------------------------------------------------------------------
# halo fit
# ---------------------------------------------------------------------------
def test_solve_c_from_delta_roundtrip():
    from galaxylens.lensfit.models.dm.gnfw_virial import zeta
    for c_true in (5.0, 10.0, 20.0):
        gamma = 1.5
        delta = (200.0 / 3.0) * c_true ** 3 / zeta(c_true, gamma, 1.0)
        assert solve_c_from_delta(delta, gamma) == pytest.approx(c_true, rel=1e-6)


def test_348489_halo_regression():
    star = f"{RAW}/raw_stars/TNG100-1_snap88_id348489_stars.hdf5"
    dm = f"{RAW}/raw_dm/TNG100-1_snap88_id348489_dm.hdf5"
    if not (os.path.exists(star) and os.path.exists(dm)):
        pytest.skip("348489 particle HDF5 not available")
    pos_s, _vel_s, _m_cur, m_init, pos_dm, m_dm, _H, _A, _b, _z = \
        load_particles(star, dm)
    center = com_iterative(pos_s, m_init)
    R, _ = inertia_rotation(pos_s - center, m_init)
    pos_dm_al = (pos_dm - center) @ R.T
    halo = fit_halo(pos_dm_al, m_dm, zlens=0.138, zsrc=0.7127)
    assert halo["v_vir_km_s"] == pytest.approx(218.0, abs=2.0)
    assert halo["c_minus2"] == pytest.approx(5.55, abs=0.05)
    assert halo["gamma"] == pytest.approx(1.53, abs=0.05)
    assert halo["q_h"] == pytest.approx(0.74, abs=0.03)
