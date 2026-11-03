#!/usr/bin/env python3
"""Elliptical-lens aperture dispersion (sigma_ap) regression.

Verifies the JampyEngine computes a single luminosity-weighted aperture velocity
dispersion sigma_ap (the elliptical-lens kinematic observable) inside a circular
aperture, via the jam.axi.proj ``pixsize`` + ``sigmapsf`` convolution. Uses the
J2141 template + B7 parameters and checks the 3"-aperture sigma_ap lands near
the central v_rms (~200 km/s), and that the chi2 carries the aperture term.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from galaxylens.lensfit.structures import (
    Geometry,
    KinData,
    PipelineConfig,
    PriorSet,
    PriorSpec,
    SourcePriors,
)
from galaxylens.lensfit.models.stellar.mge_const_ml import MgeConstMl, stellar_mge_peak
from galaxylens.lensfit.models.dm.gnfw_virial import GNFWVirial
from galaxylens.lensfit.dynamics.jampy import JampyEngine

# ---- J2141 geometry / B7 parameters ----
ZL, ZSRC = 0.138, 0.7127
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
da = cosmo.angular_diameter_distance(ZL).value
geometry = Geometry(
    zlens=ZL, zsrc=ZSRC, inclination_deg=78.5, da_mpc=da,
    pc_per_arcsec=da * np.pi / 0.648, re_arcsec=0.88, rd_arcsec=1.51,
)
B7 = dict(
    ml_img=13.575937273609815, v_vir=212.26679152167054, c_minus2=5.095233792763365,
    gamma=0.77151875639625, q_h=0.69384859965147, ratio=0.5907773483998032,
)

template = stellar_mge_peak(pc_per_arcsec=geometry.pc_per_arcsec)
priors = PriorSet(
    ml_img=(2.0, 100.0), v_vir=(150.0, 350.0), gamma=(0.3, 1.5), q_h=(0.4, 1.0),
    ratio=(0.5, 1.0), c_2=PriorSpec(kind="gaussian", mean=5.9, sigma=1.49, lower=3.0, upper=20.0),
)
cfg = PipelineConfig(
    geometry=geometry, kinematics=KinData(
        xbin=np.array([]), ybin=np.array([]), vrms=np.array([]), vrms_err=np.array([])
    ),
    dark="gnfw_virial", stellar="mge_const_ml", source="single_sersic", dynamics="jampy",
    priors=priors, stellar_template=template, source_priors=SourcePriors(),
)

stellar_comp = MgeConstMl(template)
dm_comp = GNFWVirial.from_config(cfg)
engine = JampyEngine()

inst = SimpleNamespace(
    galaxies=SimpleNamespace(
        lens=SimpleNamespace(
            mge_0=SimpleNamespace(mass_to_light_ratio=B7["ml_img"]),
            dark=SimpleNamespace(
                v_vir=B7["v_vir"], inner_slope=B7["gamma"],
                c_2=B7["c_minus2"], q_h=B7["q_h"],
            ),
            anisotropy=SimpleNamespace(mass_to_light_ratio=B7["ratio"]),
        )
    )
)

stellar_mge = stellar_comp.dyn_mge(inst, geometry, template.calibration)
dm_mge = dm_comp.dyn_mge(inst, geometry, template.calibration)
model_lum = stellar_mge
model_pot = (
    np.concatenate([stellar_mge[0], dm_mge[0]]),
    np.concatenate([stellar_mge[1], dm_mge[1]]),
    np.concatenate([stellar_mge[2], dm_mge[2]]),
)

# ---- single aperture dispersion (3" circular aperture, 0.1" PSF) ----
kin = KinData(
    xbin=np.array([]), ybin=np.array([]), vrms=np.array([]), vrms_err=np.array([]),
    aperture_radius=1.5, aperture_vrms=200.0, aperture_vrms_err=20.0,
    kin_psf_sigma=0.1,
)
assert kin.n_points == 1

chi2, info = engine.chi2(model_lum, model_pot, B7["ratio"], geometry, kin)
sigma_ap = info["sigma_ap"]
print(f"[ellip] sigma_ap (3\" aperture) = {sigma_ap:.1f} km/s")
print(f"[ellip] chi2_ap = {info['chi2_ap']:.3f}  (obs 200 +/- 20)")
# The J2141 central v_rms spans ~158-236 km/s; a 3" light-weighted aperture
# dispersion must land in a sane band around it.
assert 140.0 < sigma_ap < 280.0, sigma_ap
# chi2_ap = ((sigma_ap - 200)/20)^2 must be finite and non-negative
assert np.isfinite(info["chi2_ap"]) and info["chi2_ap"] >= 0.0

print("\nElliptical-lens aperture-dispersion regression passed.")
