#!/usr/bin/env python3
"""Regression: D-series components (nfw + mge_radial_ml + jampy_kdc + Mein) vs D4/D5.

Verifies the componentized D-series layer reproduces the D5 joint-fit anchors:
  1. dynamics chi2 + v_rms at the radial-M/L MAP (v200=253, c200=10.3, beta_z=-0.42)
  2. the FIXED (corrected) Einstein-mass constraint: M_star(<Rein) = 3.676e10
     (the D5 script omitted PC_PER_ARCSEC^2; this test anchors the CORRECTED form).

Run with autolens_jax (jampy). Does NOT require the D-series scripts: the anchors
are hardcoded from a verbatim run of d5_map_radial.py / d4_jam_fit.py.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]  # repo root (portable)
sys.path.insert(0, str(REPO / "src"))

from galaxylens.lensfit.structures import Calibration, Geometry, KinData  # noqa: E402
from galaxylens.lensfit.models.dm.nfw import NFW  # noqa: E402
from galaxylens.lensfit.models.stellar.mge_radial_ml import MgeRadialMl  # noqa: E402
from galaxylens.lensfit.dynamics.jampy_kdc import JampyKdcEngine  # noqa: E402
from galaxylens.lensfit.fit.constraints import MeinConstraint  # noqa: E402

# ---- J1331 constants (D4/D5) ----
SIGMA = np.array([0.051, 0.178, 0.503, 1.180, 3.891])  # arcsec
QOBS = np.array([1.00, 0.76, 0.58, 0.56, 0.57])
SURF_LUM = np.array([20768.0, 3161.2, 1588.2, 502.25, 136.51])  # Lsun/pc^2 (Table 3 I0,i)
ML_J = np.array([4.7, 2.2, 2.2, 4.7, 4.7])  # bulge 4.7 + inner disk 2.2 + outer disk 4.7

D_A = 414.0
INC = 70.0
PC_PER_ARCSEC = D_A * 1e6 / 206265.0
R9 = np.array([0.032, 0.601, 0.663, 1.230, 1.294, 1.863, 1.924, 2.808, 2.870])
DATA_VRMS = np.array([185.4, 199.2, 197.4, 213.6, 214.7, 224.9, 232.8, 226.8, 230.4])
DATA_ERR = np.full_like(DATA_VRMS, 5.0)

REIN_ARCSEC = 0.907
MEIN = 7.72e10
MEIN_FRAC_ERR = 0.10

geometry = Geometry(
    zlens=0.113, zsrc=0.254, inclination_deg=INC, da_mpc=D_A,
    pc_per_arcsec=PC_PER_ARCSEC, re_arcsec=REIN_ARCSEC,
)
kinematics = KinData(xbin=R9, ybin=np.zeros(9), vrms=DATA_VRMS, vrms_err=DATA_ERR)
calib = Calibration(1.0)

# ---- components ----
stellar = MgeRadialMl((SURF_LUM, SIGMA, QOBS), ML_J)
dm = NFW()
engine = JampyKdcEngine()

def _instance(v200, c200):
    return SimpleNamespace(galaxies=SimpleNamespace(lens=SimpleNamespace(
        dark=SimpleNamespace(v200=v200, c200=c200),
    )))

def model_mges(v200, c200):
    inst = _instance(v200, c200)
    lum = stellar.lum_mge(inst, geometry, calib)
    mass = stellar.dyn_mge(inst, geometry, calib)
    dm_mge = dm.dyn_mge(inst, geometry, calib)
    model_lum = lum
    model_pot = (
        np.concatenate([mass[0], dm_mge[0]]),
        np.concatenate([mass[1], dm_mge[1]]),
        np.concatenate([mass[2], dm_mge[2]]),
    )
    return model_lum, model_pot

# ---- check 1: dynamics chi2 + v_rms at (253.0, 10.3, -0.42) ----
# (exact input point; the D5 joint MAP's chi2_dyn=28.6 sits at a slightly
# different fitted point, not these rounded values)
V200, C200, BETAZ = 253.0, 10.3, -0.42
model_lum, model_pot = model_mges(V200, C200)
chi2_dyn, info = engine.chi2(model_lum, model_pot, BETAZ, geometry, kinematics)
vrms = info["vrms_model"]
print(f"[1] chi2_dyn = {chi2_dyn:.6f}  (expect 30.2633)")
print(f"    v_rms    = {np.round(vrms, 3)}")
assert abs(chi2_dyn - 30.2633) < 1e-3, chi2_dyn
# exact anchor: the D5 compute_vrms_ml at (253.0, 10.3, -0.42)
VRMS_EXPECT = np.array([
    190.178846, 212.925290, 215.251232, 219.698495, 219.436537,
    220.528031, 220.877749, 228.627548, 228.826606,
])
assert np.allclose(vrms, VRMS_EXPECT, atol=1e-3), (vrms, VRMS_EXPECT)

# ---- check 2: FIXED (corrected) Einstein-mass constraint ----
mein_constraint = MeinConstraint(REIN_ARCSEC, MEIN, MEIN_FRAC_ERR)
inst = _instance(V200, C200)
chi2_lens = mein_constraint.chi2(inst, [stellar, dm], geometry, calib)

# M_star(<Rein) with the PC_PER_ARCSEC^2 factor (corrected D5 unit bug)
from galaxylens.lensfit.fit.constraints import m2d_within  # noqa: E402

M_star = m2d_within(stellar.dyn_mge(inst, geometry, calib), REIN_ARCSEC, PC_PER_ARCSEC)
M_nfw = m2d_within(dm.dyn_mge(inst, geometry, calib), REIN_ARCSEC, PC_PER_ARCSEC)
print(f"[2] M_star(<Rein) = {M_star:.4e}  (expect 3.676e10)")
print(f"    M_nfw(<Rein)  = {M_nfw:.4e}")
print(f"    chi2_lens     = {chi2_lens:.4f}  (fixed-Mein value)")
assert abs(M_star - 3.676e10) / 3.676e10 < 1e-3, M_star
assert abs(chi2_lens - 10.5) < 0.5, chi2_lens  # ((3.68e10 + 1.53e10 - 7.72e10)/7.72e9)^2

print("\nALL D-series component regression checks passed.")
