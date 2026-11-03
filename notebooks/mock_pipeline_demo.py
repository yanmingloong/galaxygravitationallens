# %% [markdown]
# # mock_pipeline End-to-End Tutorial
#
# From IllustrisTNG selection of J2141 disk-lens analogs → generating SKIRT mock images → synthesizing fake HST → measurement.
# This is an end-to-end demonstration of three subpackages: `mock_pipeline.selection` + `mock_pipeline.skirt` + `mock_pipeline.truth`.
#
# Divided into two parts:
# - **Offline part** (§1 config / §4 .ski / §6 fake HST / §7 measurement): purely local, seconds-level, runs out of the box.
# - **Heavy part** (§2 selection / §3 download & convert / §5 SKIRT): requires a TNG key / local particle data / the SKIRT binary; this tutorial marks the prerequisites and shows real products.
#
# Prerequisite environment variables (if any is missing, the corresponding part is skipped with a note):
# - `TNG_API_KEY`: TNG API key (used by §2 selection, §3 particle download)
# - `GALAXYLENS_DATA_ROOT`: particle data root (`catalog/` + `stars_l3/` + `raw_stars/` + `raw_gas/`)
# - `GALAXYLENS_OUTPUT_ROOT`: mock output root (.ski / .dat / SKIRT products)
# - `SKIRT_BIN`: SKIRT 9 binary (used by §5, defaults to `skirt` on PATH)

# %%
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # repo root/src

import numpy as np
from astropy.io import fits

from galaxylens.mock_pipeline.selection import run_selection
from galaxylens.mock_pipeline.selection.config import J2141
from galaxylens.mock_pipeline.skirt import (
    BandConfig, MockObsConfig, J2141_BANDS, J2141_OBS,
    tng_to_skirt, write_ski, run_skirt, make_fake_hst, extinction_map,
)
from galaxylens.mock_pipeline.truth import (
    projected_ellipticity, theta_E_SIS, kpc_per_arcsec,
)

DATA_ROOT = os.environ.get("GALAXYLENS_DATA_ROOT", ".")
OUTPUT_ROOT = os.environ.get("GALAXYLENS_OUTPUT_ROOT", os.path.join(DATA_ROOT, "mock_out"))
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"DATA_ROOT   = {DATA_ROOT}")
print(f"OUTPUT_ROOT = {OUTPUT_ROOT}")

# %% [markdown]
# ## 1. Target + observation config (fully data-driven)
#
# Changing the target = changing a `TargetConfig`; changing the band/instrument = changing a `BandConfig`. Zero code changes in the pipeline.

# %%
print("Built-in target J2141:")
print(J2141)
print()
print("Default observation (J2141_OBS) bands:")
for b in J2141_OBS.bands:
    print(f"  {b.label:6s} -> {b.skirt_name:24s} px={b.pixel_scale:.4f} arcsec/px")

# %% [markdown]
# Changing-instrument example: J1331 uses WFPC2 F814W (single band, no F160W), inclination 70°.
# Just create a new `BandConfig`; the rest of the code is unchanged.

# %%
wfpc2_f814w = BandConfig("F814W", "HST WFPC2 F814W", 0.10)  # 0.10"/px is the WF3 chip
j1331_obs = MockObsConfig(bands=[wfpc2_f814w])
print("J1331 observation:", [(b.label, b.skirt_name) for b in j1331_obs.bands])

# %% [markdown]
# ## 2. Selection: scan → L1 → L3 → catalog (requires TNG_API_KEY)
#
# A real census runs `python -m galaxylens.mock_pipeline.selection.run --target j2141`.
# In the notebook, use `max_detail=10` as a smoke test (only 10 details per snapshot).
# Results are not committed to the repo; they land at `OUTPUT_ROOT/catalog.csv`.

# %%
_key = os.environ.get("TNG_API_KEY", "")
if not _key:
    print("⚠ TNG_API_KEY not set, skipping selection (§2).")
    print("  Get one: register at tng-project.org → API docs; export TNG_API_KEY=... to run this.")
    rows = []
else:
    rows = run_selection(
        "j2141", DATA_ROOT, OUTPUT_ROOT,
        workers=1, l3_workers=4, max_detail=10,  # smoke test
    )
    print(f"\n{len(rows)} disk-dominated candidates -> {os.path.join(OUTPUT_ROOT, 'catalog.csv')}")
    for r in rows[:3]:
        print({k: r[k] for k in ("subhalo_id", "f_disk") if k in r})

# %% [markdown]
# ## 3. Download full-field particles + convert to SKIRT .dat (requires local particle data)
#
# Here we directly demonstrate the known J2141 analog **348489** (the disk-dominated candidate with the highest f_disk).
# If §2 has been run, you can switch to candidates in `rows`. When particles are missing, first run `download_stars` / `download_gas`.

# %%
label, sim, snap = "348489", "TNG100-1", 88  # demo target (real runs can use §2 candidates)
stars_h5 = os.path.join(DATA_ROOT, "raw_stars", f"{sim}_snap{snap}_id{label}_stars.hdf5")
gas_h5 = os.path.join(DATA_ROOT, "raw_gas", f"{sim}_snap{snap}_id{label}_gas.hdf5")

if os.path.exists(stars_h5) and os.path.exists(gas_h5):
    res = tng_to_skirt(stars_h5, gas_h5, label, sim, snap, OUTPUT_ROOT)
    print(f"{label}: {res['n_stars']} stellar particles, {res['n_dust']} dust particles, "
          f"flat={res['flat']:.3f}, M_star={res['M_star']:.2e} Msun")
else:
    print("⚠ Missing full-field particles, skipping §3.")
    print(f"  stars_h5 = {stars_h5}")
    print(f"  gas_h5   = {gas_h5}")
    print("  Download: from galaxylens.mock_pipeline.selection import download_stars, download_gas")

# %% [markdown]
# ## 4. Generate .ski (fully runnable, purely local)
#
# One dust / nodust dual version per band. Band name, instrument name, inclination, and geometry all come from config.

# %%
dust_ski, nodust_ski = write_ski(label, J2141_BANDS[0], OUTPUT_ROOT,
                                 obs=J2141_OBS, incl=J2141.incl, roll="0 deg")
print(f"dust   : {dust_ski}")
print(f"nodust : {nodust_ski}")
print("\n--- key lines in dust .ski ---")
for line in open(dust_ski, encoding="utf-8"):
    s = line.strip()
    if any(k in s for k in ("FrameInstrument", "BroadBand", "BruzualCharlot", "ThemisDustMix")):
        print(s)

# %% [markdown]
# ## 5. Run SKIRT (requires SKIRT binary + several hours)
#
# `run_skirt` invokes the SKIRT binary (`SKIRT_BIN` env var or `skirt` on PATH).
# 1e9 photons/galaxy, ~3–4 hours on 64 cores — real runs happen on a cluster; commented out in the notebook.

# %%
# run_skirt(dust_ski, outdir=OUTPUT_ROOT)
# run_skirt(nodust_ski, outdir=OUTPUT_ROOT)
print("SKIRT runs are commented out (slow + needs the binary). §4 already generated the .ski files; run on a cluster:")
print(f"  skirt {dust_ski}")
print(f"  skirt {nodust_ski}")

# %% [markdown]
# ## 6. fake HST (fully runnable)
#
# `make_fake_hst`: SKIRT noiseless total.fits → resample to HST pixel scale → convolve PSF →
# peak normalization → add Gaussian noise / cosmic rays / background sources. Requires SKIRT to have produced `total.fits`;
# here, when the real product is missing we fall back to a synthetic image so the API demo always runs.

# %%
skirt_dust_fits = os.path.join(OUTPUT_ROOT, f"{label}_1e9", "total.fits")

# PSF: real F814W uses the measured J2141 PSF (j21410001_f814w_psf.fits); here a 5x5 delta placeholder
psf = np.zeros((5, 5)); psf[2, 2] = 1.0

if os.path.exists(skirt_dust_fits):
    img_hst = make_fake_hst(skirt_dust_fits, px_mock=0.20, band_cfg=J2141_BANDS[0], psf=psf)
    print(f"Using real SKIRT product: {skirt_dust_fits}")
else:
    # Offline fallback: synthesize a 50x50 disk image to demo the API
    y, x = np.mgrid[0:50, 0:50]
    img = np.exp(-((x - 25) ** 2 + ((y - 25) / 0.4) ** 2) / 2 / 8 ** 2) * 4905.0
    tmp = os.path.join(OUTPUT_ROOT, "_demo_total.fits")
    fits.writeto(tmp, img[None], overwrite=True)
    img_hst = make_fake_hst(tmp, px_mock=0.20, band_cfg=J2141_BANDS[0], psf=psf)
    print(f"(Real SKIRT product not found, demoing with a synthetic image. Real path: {skirt_dust_fits})")

print("fake HST image shape:", img_hst.shape)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imshow(img_hst, origin="lower", cmap="gray_r")
    plt.title(f"fake HST {J2141_BANDS[0].skirt_name}")
    plt.colorbar(label="e-/s")
    out_png = os.path.join(OUTPUT_ROOT, "fake_hst.png")
    plt.savefig(out_png, dpi=100)
    plt.close()
    print(f"fake HST plot saved: {out_png}")
except ImportError:
    print("(matplotlib not installed, skipping plot)")

# %% [markdown]
# ## 7. Extinction map + morphology measurement (fully runnable)
#
# Extinction A = -2.5 log10(dust/nodust) uses the two SKIRT noiseless versions; morphology measurement goes through the `truth` package.

# %%
skirt_nodust_fits = os.path.join(OUTPUT_ROOT, f"{label}_1e9", "total_nodust.fits")
if os.path.exists(skirt_dust_fits) and os.path.exists(skirt_nodust_fits):
    A = extinction_map(skirt_dust_fits, skirt_nodust_fits, half_px=50)
    c = A.shape[0] // 2
    print("Extinction map A center 5x5 (mag):")
    print(np.round(A[c - 2:c + 3, c - 2:c + 3], 2))
    print(f"Central extinction A(center) = {A[c, c]:.2f} mag")
else:
    print("⚠ Missing SKIRT noiseless products, skipping extinction map.")

# %% [markdown]
# Morphology measurement (`truth` package, pure NumPy, works on any 2D image):

# %%
eps = projected_ellipticity(img_hst)
print(f"projected_ellipticity(fake HST) = {eps:.3f}  (1 - sqrt(l_min/l_max))")

# SIS Einstein radius + scale conversion (using J2141's σ_SIE and redshifts)
thE = theta_E_SIS(sigma_kms=197.0, z_lens=J2141.z_lens, z_src=J2141.z_source)
print(f"theta_E_SIS(σ=197, zl={J2141.z_lens}, zs={J2141.z_source}) = {thE:.3f} arcsec")
print(f"kpc_per_arcsec(z={J2141.z_lens}) = {kpc_per_arcsec(J2141.z_lens):.3f} kpc/arcsec")

# %% [markdown]
# Particle-level measurements (`half_mass_radius` / `disk_kinematics` / `intrinsic_axis_ratios`) require
# aligned particle coordinates + masses; after §3 you can call them with the same particles as `tng_to_skirt`, omitted here.
