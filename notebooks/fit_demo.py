# %% [markdown]
# # Fitting + Post-fit Analysis Tutorial
#
# The core "recipe -> fit -> analyse" chain through the `gl` facade:
# `gl.load_config` (YAML recipe) -> `gl.fit` (assemble + Nautilus sampling) ->
# `gl.analyze` (posterior summary + derived + prior-edge check + corner).
#
# Two parts:
# - **Offline part** (§1 config / §2 synthetic-FITS smoke): purely local,
#   seconds-level, runs out of the box. It exercises the full
#   load_dataset -> assemble -> likelihood wiring with a synthetic 40x40 FITS
#   triple, catching wiring bugs without the hours-long sampling.
# - **Heavy part** (§3 fit / §4 analyse): needs the real HST FITS (under
#   `GALAXYLENS_DATA_ROOT`) and a multi-hour Nautilus run (or the cluster); this
#   tutorial marks the prerequisites and shows the exact calls.

# %%
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # repo root/src

import galaxylens as gl

# %% [markdown]
# ## 1. Load a recipe (config-driven)
#
# One YAML = one reproducible fit: geometry, kinematics, component names,
# priors, stellar template, data paths, search settings. The disk recipe
# `j2141_barnabe.yaml` is the Barnabe-style bulge+disk + gNFW + Sersic source +
# jampy-dynamics joint fit.

# %%
cfg = gl.load_config("configs/fit/j2141_barnabe.yaml")
print(f"geometry: zlens={cfg.pipeline.geometry.zlens}  zsrc={cfg.pipeline.geometry.zsrc}")
print(f"          inclination={cfg.pipeline.geometry.inclination_deg} deg  "
      f"re={cfg.pipeline.geometry.re_arcsec}\"  rd={cfg.pipeline.geometry.rd_arcsec}\"")
print(f"components: dark={cfg.pipeline.dark}  stellar={cfg.pipeline.stellar}  "
      f"source={cfg.pipeline.source}  dynamics={cfg.pipeline.dynamics}")
print(f"search: backend={cfg.search.backend}  name={cfg.search.name}  n_live={cfg.search.n_live}")

# %% [markdown]
# ## 2. Synthetic-FITS smoke (offline wiring check)
#
# Build a tiny 40x40 data/noise/psf triple, load it through `api.dataset`
# (mask + oversample, no galaxy-subtract), assemble the model, and evaluate one
# log-likelihood. This is the same plumbing `gl.fit` uses, minus the hours of
# Nautilus sampling and the real data.

# %%
from galaxylens.api.config_loader import DataSpec
from galaxylens.api.dataset import load_dataset as load_imaging
from galaxylens.lensfit.pipeline.assemble import assemble

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
print(f"dataset loaded: unmasked={n_unmasked} pixels")

collection, analysis = assemble(cfg.pipeline, dataset, n_unmasked)
instance = collection.random_instance()
ll = analysis.log_likelihood_function(instance)
print(f"single log_likelihood = {ll:.4f}  (finite={np.isfinite(ll)})")
assert np.isfinite(ll), "likelihood is not finite — wiring bug"

# %% [markdown]
# ## 3. Fit (heavy — real data + Nautilus, hours)
#
# `gl.fit` runs the recipe end-to-end (galaxy-subtract, noise rescale,
# gNFW + MGE x M/L + Sersic source + jampy dynamics) and samples with Nautilus.
# Prerequisites: the real FITS under `GALAXYLENS_DATA_ROOT` (see the recipe's
# `data:` block) and several hours (or submit via `examples/*.slurm` on the
# cluster). Commented out here — run it once the data is present.

# %%
# result = gl.fit(cfg)               # assemble -> Nautilus -> Result
#
# print(result.beta_z)               # weighted beta_z = 1 - ratio^2
# print(result.c_minus2, result.gamma, result.v_vir, result.q_h)
# print(result.c200)                 # c_200 = (2 - gamma) * c_-2
# print(result.m_star)               # M* = ml_img x CALIB x L_2D_REL
# print(result.f_dm(2.2 * cfg.pipeline.geometry.rd_arcsec))
print("[skip] gl.fit commented out — needs real FITS + hours of Nautilus; see examples/*.slurm")

# %% [markdown]
# ## 4. Post-fit analysis (heavy — needs a samples.csv)
#
# `gl.analyze` summarises a finished fit: the six mass parameters, derived
# quantities (beta_z / M* / f_dm at median / theta_E / gamma_tot), a prior-edge
# check (2.5/97.5% within 1% of a bound), and the corner plot. Point it at the
# samples.csv of a completed run.

# %%
# gl.analyze(cfg)                    # reads the fit's samples.csv under the search output root
print("[skip] gl.analyze commented out — needs a completed fit's samples.csv")
