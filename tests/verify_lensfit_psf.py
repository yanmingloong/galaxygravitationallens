#!/usr/bin/env python3
"""Smoke check: lensfit.psf.find_field_stars runs on the J2141 K'-band field.

Run with astro_kai (has photutils + astropy). The badmask FITS is not present
in the combo dir, so this smoke uses an all-good mask (the point of the check is
that the detection/classification code path runs end-to-end, not to reproduce
B1's exact catalog).
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root (portable)
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from astropy.io import fits

from galaxylens.lensfit.psf import find_field_stars

SCI = Path(os.environ.get("GALAXYLENS_TEST_NIRC2_SCI", ""))
if not SCI.is_file():
    print("SKIP: GALAXYLENS_TEST_NIRC2_SCI not set")
    sys.exit(0)

sci = fits.getdata(SCI).astype(float)
goodmask = np.ones(sci.shape, dtype=bool)

res = find_field_stars(
    sci, goodmask,
    pixel_scale=0.039686,
    psf_fwhm_px=3.6,
)

print(f"sources detected = {len(res['sources'])}, point candidates = {res['n_point']}")
print(f"galaxy centre = {res['galaxy_center']}")
print(f"background median = {res['background']['median']:.2f}, std = {res['background']['std']:.2f}")

assert res["n_point"] >= 0
assert len(res["sources"]) > 0
print("PSF field-star detection smoke passed.")
