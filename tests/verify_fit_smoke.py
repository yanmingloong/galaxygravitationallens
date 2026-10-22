#!/usr/bin/env python3
"""Smoke test: gl.fit plumbing with a tiny synthetic dataset (no cluster data).

Builds a 40x40 synthetic FITS triple (data/noise/psf), loads it through
``api.dataset`` (mask + oversample, no galaxy-subtract to keep it fast),
assembles the model, and evaluates a single log-likelihood -- catching wiring
bugs in the full chain without the hours-long Nautilus run or the cluster F814W
data. The real end-to-end ``gl.fit`` on J2141 (stage 3) runs on the cluster.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

REPO = Path(__file__).resolve().parents[1]  # repo root (portable)
sys.path.insert(0, str(REPO / "src"))

from galaxylens.api.config_loader import DataSpec, load_config
from galaxylens.api.dataset import load_dataset as load_imaging

# ---- synthetic FITS triple ----
tmp = Path(tempfile.mkdtemp())
shape = (40, 40)
rng = np.random.RandomState(1)
data = rng.normal(0.0, 0.5, shape)
noise = np.full(shape, 0.5)
xx, yy = np.meshgrid(np.arange(5) - 2, np.arange(5) - 2)
psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * 0.8 ** 2))
psf /= psf.sum()

paths = {}
for name, arr in [("sci", data), ("noise", noise), ("psf", psf)]:
    p = tmp / f"syn_{name}.fits"
    hdu = fits.PrimaryHDU(arr.astype(np.float32))
    hdu.header["CDELT1"] = 0.05
    hdu.header["CDELT2"] = 0.05
    hdu.writeto(p, overwrite=True)
    paths[name] = str(p)

# ---- load through api.dataset (no galaxy-subtract, no rescale) ----
data_spec = DataSpec(
    data_path=paths["sci"],
    noise_map_path=paths["noise"],
    psf_path=paths["psf"],
    pixel_scale=0.05,
    mask_radius=1.0,
    oversample=1,
)
dataset, n_unmasked = load_imaging(
    data_spec, zlens=0.138, subtract_stellar=False, noise_rescale=False
)
print(f"[smoke] dataset loaded: unmasked={n_unmasked}")

# ---- assemble + single likelihood ----
cfg = load_config(REPO / "configs/fit/j2141_barnabe.yaml")
from galaxylens.lensfit.pipeline.assemble import assemble

collection, analysis = assemble(cfg.pipeline, dataset, n_unmasked)

instance = collection.random_instance()
ll = analysis.log_likelihood_function(instance)
print(f"[smoke] single log_likelihood = {ll:.4f}  (finite={np.isfinite(ll)})")
assert np.isfinite(ll)

print("\nSMOKE passed: dataset -> assemble -> likelihood chain works.")
