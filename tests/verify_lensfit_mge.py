#!/usr/bin/env python3
"""Regression check: lensfit.mge + lensfit.lens.rescale against B-series products.

Run with jampy_env (has mgefit + numpy). Verifies the extracted lensfit
modules reproduce the B2/B3/B4 numbers exactly.

Checks:
  1. noise_rescale_factor reproduces f=5.124 (B4 job 127231)
  2. load_mge reads J2141 mge_params.csv; bulge sigma_min == 0.0637" (post-rbounds)
  3. fit_mge_1d with rbounds reproduces the B2 bulge (sigma_min == PSF sigma)
  4. deproject/project_back round-trips exactly
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root (portable)
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from galaxylens.lensfit.mge import deproject, fit_mge_1d, load_mge, project_back
from galaxylens.lensfit.lens.rescale import noise_rescale_factor

MGE_CSV = Path(os.environ.get("GALAXYLENS_TEST_MGE_CSV", ""))

# --- 1. rescale factor ---
f = noise_rescale_factor(824219.3, 31389)
print(f"[1] noise_rescale_factor(824219.3, 31389) = {f:.4f}  (expect 5.124)")
assert abs(f - 5.124) < 0.01

# --- 2. load_mge (bulge/disk labels preserved) ---
if not MGE_CSV.is_file():
    print("SKIP: GALAXYLENS_TEST_MGE_CSV not set")
    sys.exit(0)
mge = load_mge(MGE_CSV)
n = len(mge)
bulge = [g for g in mge if g.component == "bulge"]
disk = [g for g in mge if g.component == "disk"]
sigma_min_bulge = min(g.sigma_arcsec for g in bulge)
print(
    f"[2] load_mge: {n} Gaussians ({len(bulge)} bulge, {len(disk)} disk), "
    f"bulge sigma_min = {sigma_min_bulge:.4f} arcsec (expect 0.0637)"
)
assert len(bulge) == 6 and len(disk) == 10
assert abs(sigma_min_bulge - 0.0637) < 0.001

# --- 3. fit_mge_1d reproduces B2 bulge with rbounds ---
PIXEL_SCALE = 0.039686
PSF_SIGMA_PX = (0.15 / 2.355) / PIXEL_SCALE
BULGE_RE_ARCSEC = 0.26
BULGE_N = 1.21
re_px = BULGE_RE_ARCSEC / PIXEL_SCALE
r = np.geomspace(max(re_px * 0.02, 0.05), re_px * 6, 200)
b_n = 2 * BULGE_N - 1 / 3 + 4 / (405 * BULGE_N)
prof = np.exp(-b_n * ((r / re_px) ** (1 / BULGE_N) - 1))

total, sigma = fit_mge_1d(r, prof, ngauss=6, psf_sigma=PSF_SIGMA_PX)
sigma_min = sigma.min() * PIXEL_SCALE
print(f"[3] fit_mge_1d: sigma_min = {sigma_min:.4f} arcsec (expect 0.0637)")
assert abs(sigma_min - 0.0637) < 0.001

# --- 4. deproject / project_back round-trip ---
q_obs = np.array([0.93, 0.93, 0.31, 0.31])
q_intr = deproject(q_obs, 78.0)
q_back = project_back(q_intr, 78.0)
max_err = np.max(np.abs(q_back - q_obs))
print(f"[4] deproject round-trip max|q_back - q_obs| = {max_err:.2e} (expect < 1e-12)")
assert max_err < 1e-12

print("\nALL mge/rescale regression checks passed.")
