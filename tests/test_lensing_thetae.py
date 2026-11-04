"""End-to-end integration regression: arc synthesis against the MF truth.

348489: the true-mass model (stellar MGE + halo truth) must reproduce the
arc meta's Einstein radius (theta_E = 0.609") and the stellar mass self-check.
Data-dependent, skipped when the MF MGE / truth files are absent.
"""

import json
import os

import numpy as np
import pytest

from galaxylens.mock_pipeline.lensing import build_lens_galaxy, einstein_radius
from galaxylens.lensfit.models.dm.gnfw_virial import halo_cosmo

MF = os.environ.get("GALAXYLENS_MF_ROOT", "")


def test_348489_theta_e_regression():
    if not MF:
        pytest.skip("GALAXYLENS_MF_ROOT not set (MF-series test data)")
    mge_path = f"{MF}/outputs/J2141/mge/348489_mge.npz"
    halo_path = f"{MF}/data/J2141/truth_halo_348489.json"
    truth_path = f"{MF}/data/J2141/truth_348489.json"
    if not all(os.path.exists(p) for p in (mge_path, halo_path, truth_path)):
        pytest.skip("348489 MGE / halo truth not available")

    d = np.load(mge_path)
    total_counts = d["total_counts"]          # integrated L_k per Gaussian
    sigma_kpc = d["sigma_kpc"]
    q_obs = d["q_obs"]
    halo = json.load(open(halo_path))
    truth = json.load(open(truth_path))

    c = halo_cosmo(0.138, 0.7127)             # H0=70 (Barnabe), matches MF
    sig_as = sigma_kpc / c.kpc_per_arcsec
    sigma_pc = sigma_kpc * 1000.0
    surf = total_counts / (2 * np.pi * sigma_pc ** 2 * q_obs)   # peak, flux/pc^2
    total_lum = float(total_counts.sum())
    ml = truth["m_star_total_msun"] / total_lum
    surf_mass_kpc2 = surf * ml * 1e6          # Msun/kpc^2

    lens, mstar_model = build_lens_galaxy(
        surf_mass_kpc2, sig_as, q_obs,
        v_vir=halo["v_vir_km_s"], c_minus2=halo["c_minus2"],
        gamma=halo["gamma"], q_h=halo["q_h"],
        zlens=0.138, zsrc=0.7127, inclination=78.5,
    )
    theta_e, kbar = einstein_radius(lens)

    # stellar-mass self-check (MGE + ml should recover the truth M*)
    assert mstar_model == pytest.approx(truth["m_star_total_msun"], rel=0.05)
    # Einstein radius anchor (arc meta: 0.609")
    assert theta_e == pytest.approx(0.609, abs=0.03)
