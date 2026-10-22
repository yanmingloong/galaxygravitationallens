#!/usr/bin/env python3
"""Regression: facade point-level entries (fit_positions / fit_mein) vs D1/D5.

Exercises the two point-level facade paths added in stage 5.5:
  1. gl.fit_positions reproduces the D1 scale-free Rein/Mein (Table 5).
  2. gl.fit_mein reproduces the corrected-Mein D5 3-param MAP (v200/c200/beta_z).
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]  # repo root (portable)
sys.path.insert(0, str(REPO / "src"))

import galaxylens as gl  # noqa: E402
from galaxylens.lensfit.structures import Geometry, KinData  # noqa: E402

# ---- check 1: gl.fit_positions (D1) ----
PIXSCALE = 0.05
IMG_PIX = {"A": (12.1, 16.6), "B": (-8.5, -10.4), "C": (21.7, -0.5), "D": (-3.3, 19.2)}
G_PIX = (0.5, 0.5)
theta_obs = np.array([(np.array(v) - np.array(G_PIX)) * PIXSCALE for v in IMG_PIX.values()])
err = 1.0 * PIXSCALE

sf = gl.fit_positions(theta_obs, err, zlens=0.113, zsrc=0.254, kpc_per_arcsec=2.006)
print(f"[1] fit_positions: Rein={sf.rein_arcsec:.4f}\"  Mein={sf.mein_1e10:.3f}e10  chi2={sf.chi2:.3f}")
assert abs(sf.rein_arcsec - 0.9068) < 1e-3, sf.rein_arcsec
assert abs(sf.mein_1e10 - 7.718) < 1e-2, sf.mein_1e10
assert abs(sf.chi2 - 1.945) < 1e-2, sf.chi2

# ---- check 2: gl.fit_mein (D5, corrected Mein) ----
D_A = 414.0
PC_PER_ARCSEC = D_A * 1e6 / 206265.0
geometry = Geometry(zlens=0.113, zsrc=0.254, inclination_deg=70.0, da_mpc=D_A, pc_per_arcsec=PC_PER_ARCSEC)

R9 = np.array([0.032, 0.601, 0.663, 1.230, 1.294, 1.863, 1.924, 2.808, 2.870])
DATA_VRMS = np.array([185.4, 199.2, 197.4, 213.6, 214.7, 224.9, 232.8, 226.8, 230.4])
DATA_ERR = np.full_like(DATA_VRMS, 5.0)
kinematics = KinData(xbin=R9, ybin=np.zeros(9), vrms=DATA_VRMS, vrms_err=DATA_ERR)

SIGMA = np.array([0.051, 0.178, 0.503, 1.180, 3.891])
QOBS = np.array([1.00, 0.76, 0.58, 0.56, 0.57])
SURF_LUM = np.array([20768.0, 3161.2, 1588.2, 502.25, 136.51])
ML_J = np.array([4.7, 2.2, 2.2, 4.7, 4.7])

res = gl.fit_mein(
    geometry, kinematics, (SURF_LUM, SIGMA, QOBS), ML_J,
    rein_arcsec=0.907, mein_msun=7.72e10,
)
print(f"[2] fit_mein: v200={res['v200']:.0f}  c200={res['c200']:.1f}  beta_z={res['beta_z']:.2f}")
print(f"    chi2_dyn={res['chi2_dyn']:.1f}  chi2_lens={res['chi2_lens']:.1f}")

# Anchors. chi2_dyn/chi2_lens are robust (the dynamics uses the same MGE in both
# implementations). v200 lands ~242 here vs 252 in d5_map_radial_fixmein.py because
# the MeinConstraint sums m2d_within over the component MGEs (M_nfw via 10
# Gaussians ~ 1.55e10 vs the D5 script's exact NFW projected mass 1.538e10), and
# the neg_log_prob is flat along the beta_z x c200 degeneracy. The ~4% v200 shift
# is within the 10% Mein error and the degeneracy.
assert abs(res["chi2_dyn"] - 27.3) < 1.5, res["chi2_dyn"]
assert abs(res["chi2_lens"] - 11.2) < 1.0, res["chi2_lens"]
assert 235 <= res["v200"] <= 255, res["v200"]
assert 9.0 <= res["c200"] <= 12.0, res["c200"]
assert abs(res["beta_z"] - (-0.42)) < 0.06, res["beta_z"]

print("\nALL facade point-level regression checks passed.")
