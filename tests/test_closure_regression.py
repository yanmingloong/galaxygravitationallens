"""End-to-end closure regression: fit samples -> f_dm -> truth bias.

343259 MF-2 (G2, no prior): the closure must recover f_dm spherical 2.2Rd
= 0.518 against the particle truth 0.548 (-5.5%). This wires the whole chain
together: samples.csv (ml_img -> lg_ml via the Sigma_crit bridge), the stellar
MGE template, gnfw_halo_mge, and the validation apertures. Data-dependent;
skipped when the MF samples / MGE / truth files are absent.
"""

import json
import os

import numpy as np
import pytest

from galaxylens.lensfit.structures import (
    Geometry, StellarTemplate, Calibration,
)
from galaxylens.lensfit.validation import closure_f_dm
from galaxylens.lensfit.models.dm.gnfw_virial import halo_cosmo

MF = ("/mnt/d/HermesAgent/ResearchProjects/GalaxyGravLens/DiskLenses/"
      "AutoLensFit/MF_series")
G2_343259 = "eff3fedb2825cd0f190966bd62277db4"


def test_343259_closure_regression():
    samples_csv = (f"{MF}/outputs/J2141/fit/mf2_343259/{G2_343259}/files/"
                   "samples.csv")
    mge_path = f"{MF}/outputs/J2141/mge/343259_mge.npz"
    truth_path = f"{MF}/data/J2141/truth_343259.json"
    data_path = f"{MF}/outputs/J2141/lensed/343259_data.fits"
    if not all(os.path.exists(p) for p in
               (samples_csv, mge_path, truth_path, data_path)):
        pytest.skip("343259 MF-2 samples / MGE / truth not available")

    import pandas as pd
    from astropy.io import fits

    df = pd.read_csv(samples_csv)
    df.columns = df.columns.str.strip()
    ml_cols = [c for c in df.columns
               if c.startswith("galaxies.lens.mge_")
               and c.endswith("mass_to_light_ratio")]
    ml_img = df[ml_cols[0]].values
    v_vir = df["galaxies.lens.dark.v_vir"].values
    gamma = df["galaxies.lens.dark.inner_slope"].values
    c_2 = df["galaxies.lens.dark.c_2"].values
    q_h = df["galaxies.lens.dark.q_h"].values
    beta_z = df["galaxies.lens.anisotropy.mass_to_light_ratio"].values
    w = df["weight"].values.astype(float)

    d = np.load(mge_path)
    total_counts = d["total_counts"]
    sigma_kpc = d["sigma_kpc"]
    q_obs = d["q_obs"]

    cosmo = halo_cosmo(0.138, 0.7127)            # H0=70, matches MF
    sig_as = sigma_kpc / cosmo.kpc_per_arcsec
    sigma_pc = sigma_kpc * 1000.0
    surf_s = total_counts / (2 * np.pi * sigma_pc ** 2 * q_obs)  # flux/pc^2
    total_lum = float(total_counts.sum())

    # ml_img -> lg_ml bridge (the Sigma_crit bridge of the MF-2 closure)
    data = fits.getdata(data_path)
    calib = data.sum() / total_lum
    ml_to_dyn = calib * cosmo.pc_per_arcsec ** 2 * cosmo.sigma_crit / 1e6
    lg_ml = np.log10(ml_img * ml_to_dyn)
    theta = np.column_stack([lg_ml, v_vir, gamma, c_2, q_h, beta_z])

    template = StellarTemplate(
        mge=(surf_s, sig_as, q_obs), l2d_rel=total_lum,
        calibration=Calibration(calib=1.0),
    )
    geom = Geometry(zlens=0.138, zsrc=0.7127, inclination_deg=78.5,
                    da_mpc=cosmo.da_mpc, pc_per_arcsec=cosmo.pc_per_arcsec)

    truth = json.load(open(truth_path))
    rd = truth["rd_kpc"]
    apertures = {
        "2.2Rd": 2.2 * rd / cosmo.kpc_per_arcsec,
        "5kpc": 5.0 / cosmo.kpc_per_arcsec,
        "10kpc": 10.0 / cosmo.kpc_per_arcsec,
    }
    out = closure_f_dm(theta, w, template, geom, truth["f_dm"], apertures)

    assert out["2.2Rd"]["spherical"][1] == pytest.approx(0.518, abs=0.01)
    assert out["2.2Rd"]["spherical_bias_pct"] == pytest.approx(-5.5, abs=1.0)
