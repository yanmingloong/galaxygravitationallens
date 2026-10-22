"""Regression tests for mock_pipeline.truth (ground-truth computation).

Pure-numerical assertions validate the cosmology / measures conventions;
the disk-kinematics test regresses against the known J2141 analog 348489
(f_disk=0.643, B/D=0.55) when its particle HDF5 is available.
"""

import os

import numpy as np
import pytest

from galaxylens.mock_pipeline.truth import (
    H, BOX_BY_SIM, KAPPA_THRESH,
    kpc_per_arcsec, theta_E_SIS, cosmic_time, Dls_over_Ds,
    tng_mass_to_msun, tng_length_to_physical_kpc, snap_to_a,
    half_mass_radius, projected_ellipticity, intrinsic_axis_ratios,
    extinction, major_axis_angle,
    unwrap, com_iterative, angular_momentum, circularity, disk_axis,
    disk_fraction, disk_kinematics, counter_rotation_angle,
)


# ---------------------------------------------------------------------------
# cosmology
# ---------------------------------------------------------------------------
def test_kpc_per_arcsec_j2141():
    # J2141 z=0.138 -> 2.517 kpc/arcsec (project-validated value)
    assert kpc_per_arcsec(0.138) == pytest.approx(2.517, abs=0.01)


def test_kpc_per_arcsec_others():
    # Re unit-conversion anchors (Planck 2015)
    assert kpc_per_arcsec(0.131) == pytest.approx(2.408, abs=0.01)
    assert kpc_per_arcsec(0.104) == pytest.approx(1.972, abs=0.01)
    assert kpc_per_arcsec(0.158) == pytest.approx(2.818, abs=0.01)


def test_theta_e_sis_selfcheck():
    # SWELLS III J2141 sigma_SIE=197 -> theta_E ~ 0.867 arcsec
    assert theta_E_SIS(197, 0.138, 0.7127) == pytest.approx(0.867, abs=0.005)
    # canonical project mapping: sigma=200 -> theta_E ~ 0.89 arcsec
    assert theta_E_SIS(200, 0.138, 0.7127) == pytest.approx(0.89, abs=0.005)


def test_cosmic_time():
    # age of the universe today ~ 13.8 Gyr
    assert cosmic_time(1.0) == pytest.approx(13.8, abs=0.3)


def test_snap_to_a_unknown_raises():
    """Unknown snapshot must raise, not silently return a=1.0."""
    import pytest as _pytest
    from galaxylens.mock_pipeline.truth import snap_to_a
    with _pytest.raises(KeyError):
        snap_to_a(93)  # not in SNAP_Z


def test_tng_units():
    assert tng_mass_to_msun(42.48) == pytest.approx(6.27e11, rel=0.01)
    assert snap_to_a(88) == pytest.approx(1.0 / 1.1418, abs=1e-6)
    # 100 comoving kpc/h at snap88 -> 100 / 0.6774 * (1/1.1418) physical kpc
    assert tng_length_to_physical_kpc(100, 88) == pytest.approx(
        100 / 0.6774 / 1.1418, abs=1e-6)


# ---------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------
def test_half_mass_radius():
    R = np.array([1.0, 2.0, 3.0, 4.0])
    m = np.ones(4)
    # half the mass = 2 particles -> radius of the 2nd sorted particle
    assert half_mass_radius(R, m) == pytest.approx(2.0)


def test_projected_ellipticity_round():
    yy, xx = np.mgrid[0:64, 0:64]
    r2 = (xx - 32) ** 2 + (yy - 32) ** 2
    img = np.exp(-r2 / (2 * 10.0 ** 2))
    assert projected_ellipticity(img) < 0.01


def test_projected_ellipticity_elongated():
    yy, xx = np.mgrid[0:64, 0:64]
    img = np.exp(-((xx - 32) ** 2 / (2 * 20.0 ** 2)
                   + (yy - 32) ** 2 / (2 * 5.0 ** 2)))
    e = projected_ellipticity(img)
    assert 0.5 < e < 0.9


def test_intrinsic_axis_ratios():
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.normal(0, 10, n)
    y = rng.normal(0, 10, n)
    z = rng.normal(0, 1, n)
    m = np.ones(n)
    syx, szx, szy = intrinsic_axis_ratios(x, y, z, m)
    assert syx == pytest.approx(1.0, abs=0.05)
    assert szx == pytest.approx(0.1, abs=0.02)  # thin disk


def test_extinction():
    assert extinction(1.0, 1.0) == pytest.approx(0.0)
    assert extinction(0.5, 1.0) == pytest.approx(-2.5 * np.log10(0.5))


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------
def test_circularity_basic():
    """Particles on self-consistent circular orbits -> kappa ~ 1."""
    from galaxylens.mock_pipeline.truth.cosmology import GRAV
    rng = np.random.default_rng(1)
    n = 5000
    R = rng.uniform(0.5, 10.0, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    m = 1e8
    mass = np.full(n, m)
    pos = np.column_stack([R * np.cos(theta), R * np.sin(theta),
                           rng.normal(0, 0.05, n)])
    # enclosed mass within each radius (sorted cumsum), matching circularity()
    order = np.argsort(R)
    Rs = R[order]
    Menc = np.cumsum(mass[order])
    vc = np.sqrt(GRAV * np.interp(R, Rs, Menc) / R)
    vel = np.column_stack([-vc * np.sin(theta), vc * np.cos(theta), np.zeros(n)])
    center = np.zeros(3)
    kappa, _ = circularity(pos, vel, mass, center, np.array([0, 0, 1.0]))
    assert np.median(kappa) == pytest.approx(1.0, abs=0.15)


def test_disk_fraction():
    kappa = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
    mass = np.ones(5)
    f_disk, bd = disk_fraction(kappa, mass, kappa_thresh=0.7)
    assert f_disk == pytest.approx(0.6)
    assert bd == pytest.approx(0.4 / 0.6)


def test_disk_fraction_no_disk():
    """No disk particle -> f_disk=0 and B/D=inf (mathematically)."""
    kappa = np.array([0.1, 0.2, 0.3])
    mass = np.ones(3)
    f_disk, bd = disk_fraction(kappa, mass, kappa_thresh=0.7)
    assert f_disk == pytest.approx(0.0)
    assert not np.isfinite(bd)  # inf


def test_disk_kinematics_random_system():
    """Random (weak-disk) system -> no crash, finite output."""
    rng = np.random.default_rng(2)
    n = 5000
    pos = rng.normal(0, 3.0, (n, 3))
    vel = rng.normal(0, 50.0, (n, 3))  # random, no coherent rotation
    mass = np.full(n, 1e7)
    center = com_iterative(pos, mass)
    res = disk_kinematics(pos, vel, mass, center)
    assert res["f_disk"] < 0.3
    assert res["B_over_D"] >= 0  # finite (no round(inf) crash)


def test_disk_kinematics_348489_regression():
    """Regression against the known J2141 analog 348489 (f_disk=0.643, B/D=0.55)."""
    path = ("/mnt/d/HermesAgent/ResearchProjects/cosmology_simulations/"
            "IllustrisTNG/disk_lens_analogs/01_data/analogs/J2141/stars_l3/"
            "TNG100-1_snap88_id348489_stars.hdf5")
    if not os.path.exists(path):
        pytest.skip("348489 particle HDF5 not available")
    import h5py
    with h5py.File(path, "r") as f:
        pt = f["PartType4"]
        pos_raw = pt["Coordinates"][:].astype(np.float64)
        vel_raw = pt["Velocities"][:].astype(np.float64)
        mass_raw = pt["Masses"][:].astype(np.float64)
    a = snap_to_a(88)
    pos = unwrap(pos_raw.copy(), BOX_BY_SIM["TNG100-1"]) * a / H
    vel = vel_raw * np.sqrt(a)
    mass = mass_raw / H * 1e10
    center = com_iterative(pos, mass)
    res = disk_kinematics(pos, vel, mass, center)
    assert res["f_disk"] == pytest.approx(0.643, abs=0.01)
    assert res["B_over_D"] == pytest.approx(0.55, abs=0.05)


def test_major_axis_angle():
    """Elongated Gaussian at +30 deg -> recovered major axis ~ 30 deg."""
    yy, xx = np.mgrid[0:128, 0:128]
    cx = cy = 63.5
    dx = xx - cx
    dy = yy - cy
    ang = np.radians(30.0)
    u = dx * np.cos(ang) + dy * np.sin(ang)
    v = -dx * np.sin(ang) + dy * np.cos(ang)
    img = np.exp(-(u ** 2 / (2 * 20.0 ** 2) + v ** 2 / (2 * 4.0 ** 2)))
    assert major_axis_angle(img) == pytest.approx(30.0, abs=1.0)


def test_major_axis_angle_round():
    """A round Gaussian is axis-degenerate but must return a finite angle."""
    yy, xx = np.mgrid[0:64, 0:64]
    r2 = (xx - 31.5) ** 2 + (yy - 31.5) ** 2
    img = np.exp(-r2 / (2 * 8.0 ** 2))
    assert np.isfinite(major_axis_angle(img))


def _rotating(R, n, sign, rng):
    """Particles on tangential orbits at radius R about the z-axis."""
    theta = rng.uniform(0, 2 * np.pi, n)
    pos = np.column_stack([R * np.cos(theta), R * np.sin(theta),
                           rng.normal(0, 0.1 * R, n)])
    vphi = 100.0
    vel = np.column_stack([sign * -vphi * np.sin(theta),
                           sign * vphi * np.cos(theta), np.zeros(n)])
    return pos, vel, np.ones(n)


def test_counter_rotation_angle_kdc():
    """Core counter-rotating vs outer disk -> angle ~ 180 deg (KDC)."""
    rng = np.random.default_rng(3)
    po, vo, mo = _rotating(12.0, 6000, +1, rng)
    pc, vc, mc = _rotating(1.5, 1500, -1, rng)
    ang = counter_rotation_angle(
        np.vstack([po, pc]), np.vstack([vo, vc]),
        np.concatenate([mo, mc]), np.zeros(3))
    assert ang > 150.0


def test_counter_rotation_angle_co_rotating():
    """All co-rotating -> angle < 90 deg (no KDC)."""
    rng = np.random.default_rng(4)
    po, vo, mo = _rotating(12.0, 6000, +1, rng)
    pc, vc, mc = _rotating(1.5, 1500, +1, rng)
    ang = counter_rotation_angle(
        np.vstack([po, pc]), np.vstack([vo, vc]),
        np.concatenate([mo, mc]), np.zeros(3))
    assert ang < 90.0


def test_counter_rotation_angle_342537_regression():
    """Regression: 342537 weak counter-rotation bias -> angle ~ 115.7 deg."""
    path = ("/mnt/d/HermesAgent/ResearchProjects/cosmology_simulations/"
            "IllustrisTNG/disk_lens_analogs/01_data/analogs/J1331/stars_l3/"
            "TNG100-1_snap90_id342537_stars.hdf5")
    if not os.path.exists(path):
        pytest.skip("342537 particle HDF5 not available")
    import h5py
    with h5py.File(path, "r") as f:
        pt = f["PartType4"]
        pos_raw = pt["Coordinates"][:].astype(np.float64)
        vel_raw = pt["Velocities"][:].astype(np.float64)
        mass_raw = pt["Masses"][:].astype(np.float64)
    a = snap_to_a(90)
    pos = unwrap(pos_raw.copy(), BOX_BY_SIM["TNG100-1"]) * a / H
    vel = vel_raw * np.sqrt(a)
    mass = mass_raw / H * 1e10
    center = com_iterative(pos, mass)
    ang = counter_rotation_angle(pos, vel, mass, center)
    assert ang == pytest.approx(115.7, abs=2.0)


def test_major_axis_angle_342537_regression():
    """Regression: 342537 F814W roll-0 image -> major axis ~ -21.32 deg."""
    path = ("/mnt/d/HermesAgent/ResearchProjects/cosmology_simulations/"
            "IllustrisTNG/mock_observations/03_outputs/static_mock/"
            "342537_1e6/342537_F814W_F814W_total.fits")
    if not os.path.exists(path):
        pytest.skip("342537 F814W mock FITS not available")
    from astropy.io import fits
    img = fits.getdata(path)
    if img.ndim == 3:
        img = img[0]
    assert major_axis_angle(img) == pytest.approx(-21.32, abs=1.0)
