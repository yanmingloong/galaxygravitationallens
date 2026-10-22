#!/usr/bin/env python3
"""Regression: scale-free quad lensing (D1) reproduces Trick+2016 Table 5.

Ports d1_scale_free_fit.py into lensfit.lens.scale_free and asserts the fit to
the J1331 four image positions reproduces the paper Table 5 peak values.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]  # repo root (portable)
sys.path.insert(0, str(REPO / "src"))

from galaxylens.lensfit.lens.scale_free import (  # noqa: E402
    fit_image_positions,
    rein_mein,
    sigma_crit,
)

# ---- J1331 constants (Trick+2016 / D1) ----
PIXSCALE = 0.05  # arcsec / px (drizzled WFPC2)
ERR_IMG_PIX = 1.0
IMG_PIX = {"A": (12.1, 16.6), "B": (-8.5, -10.4), "C": (21.7, -0.5), "D": (-3.3, 19.2)}
G_PIX = (0.5, 0.5)
ZL, ZS = 0.113, 0.254
KPC_PER_ARCSEC = 2.006  # Table 1 (Dd=414 Mpc)

TABLE5_PEAK = {
    "a0": 1.814, "a2": 0.012, "b2": -0.057, "xs": 0.095, "ys": 0.107,
    "Rein": 0.907, "Mein": 7.72, "Mcrit": 7.87,
}

theta_obs = np.array([(np.array(v) - np.array(G_PIX)) * PIXSCALE for v in IMG_PIX.values()])
err = ERR_IMG_PIX * PIXSCALE

sc = sigma_crit(ZL, ZS)
print(f"[d1] Sigma_crit = {sc:.4e} Msun/kpc^2")

params, chi2 = fit_image_positions(theta_obs, err)
a0, a2, b2, xs, ys = params
rein_a, rein_kpc, mein, mcrit = rein_mein(a0, a2, b2, KPC_PER_ARCSEC, sc)

print(f"[d1] chi2 = {chi2:.3f}  (expect 1.945)")
print(f"     a0={a0:.4f} a2={a2:.4f} b2={b2:.4f} xs={xs:.4f} ys={ys:.4f}")
print(f"     Rein={rein_a:.4f}\"  Mein={mein:.3f}e10  Mcrit={mcrit:.3f}e10")

# peak values vs Table 5 col 1 (exact, D1 reproduced these to ~1e-4)
assert abs(a0 - TABLE5_PEAK["a0"]) < 1e-3, a0
assert abs(a2 - TABLE5_PEAK["a2"]) < 1e-3, a2
assert abs(b2 - TABLE5_PEAK["b2"]) < 1e-3, b2
assert abs(xs - TABLE5_PEAK["xs"]) < 1e-3, xs
assert abs(ys - TABLE5_PEAK["ys"]) < 1e-3, ys
assert abs(rein_a - TABLE5_PEAK["Rein"]) < 1e-3, rein_a
assert abs(mein - TABLE5_PEAK["Mein"]) < 1e-2, mein
assert abs(mcrit - TABLE5_PEAK["Mcrit"]) < 1e-2, mcrit
assert abs(chi2 - 1.945) < 1e-2, chi2

print("\nALL scale-free (D1) regression checks passed.")
