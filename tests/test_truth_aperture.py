"""Unit tests for ``truth.aperture.sigma_ap`` (elliptical-mock sigma_ap)."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from galaxylens.mock_pipeline.truth.aperture import sigma_ap  # noqa: E402


def test_sigma_ap_isotropic_inclination_independent():
    rng = np.random.default_rng(0)
    n = 20000
    pos = rng.normal(0.0, 2.0, (n, 3))   # kpc, isotropic
    vel = rng.normal(0.0, 200.0, (n, 3))  # km/s, isotropic
    mass = np.ones(n)

    sig0, nn = sigma_ap(pos, vel, mass, incl_deg=0.0, R_ap_kpc=6.0)
    sig90, _ = sigma_ap(pos, vel, mass, incl_deg=90.0, R_ap_kpc=6.0)

    # isotropic velocity -> sigma_ap recovers the input dispersion for any
    # inclination (projection of an isotropic tensor is isotropic).
    # pos/vel generated independently, so the R_sky truncation does not bias the
    # dispersion; only require the bulk of particles enclosed (Gaussian tail).
    assert nn > 0.9 * n, nn
    assert abs(sig0 - 200.0) < 5.0, sig0
    assert abs(sig90 - 200.0) < 5.0, sig90


def test_dispersion_removes_streaming_not_rms():
    rng = np.random.default_rng(1)
    n = 20000
    pos = rng.normal(0.0, 2.0, (n, 3))
    vel = rng.normal(0.0, 100.0, (n, 3))
    vel[:, 2] += 500.0  # bulk streaming along the disk normal (LOS at face-on)
    mass = np.ones(n)

    sig, _ = sigma_ap(pos, vel, mass, incl_deg=0.0, R_ap_kpc=6.0)

    # dispersion (mean removed) -> ~100; v_rms would be ~sqrt(100^2 + 500^2) ~ 510
    assert abs(sig - 100.0) < 5.0, sig
