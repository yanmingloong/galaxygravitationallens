#!/usr/bin/env python3
"""B9 extraction regression: sky RMS noise, AGN point-source subtraction, pixelized source.

Run with autolens_jax for check 3 (autolens/autofit); checks 1-2 need only
numpy/scipy/astropy. Verifies the three newly extracted B9 pieces reproduce
their research-project values:

  1. estimate_sky_noise      -- pure numerical: recovers a known synthetic sky std.
  2. subtract_stellar_light_mge_agn -- data-dependent (skip if missing): the B9
     K'-band MGE + AGN point-source subtraction lands at reduced chi2 ~ 1.11
     (the pure-MGE version was 2.58), i.e. the point source removes the compact
     nucleus.
  3. PixelizedSource         -- model structure: 7 explicit params (6 mass + 1
     regularization coefficient) and n_source = 225 for a 15x15 mesh (the
     imaging dof subtract the mesh pixels, not the explicit params).
"""
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]  # repo root (portable)
sys.path.insert(0, str(REPO / "src"))

from galaxylens.lensfit.lens.rescale import estimate_sky_noise
from galaxylens.lensfit.lens.subtract import subtract_stellar_light_mge_agn

# ----------------------------------------------------------------------------
# check 1: estimate_sky_noise recovers a synthetic sky std (pure numerical)
# ----------------------------------------------------------------------------
rng = np.random.RandomState(42)
sci = 100.0 + rng.normal(0.0, 3.5, (64, 64))  # positive sky background + noise
badmask = np.ones((64, 64), dtype=int)
yy, xx = np.mgrid[0:64, 0:64]
r = np.sqrt((xx - 32) ** 2 + (yy - 32) ** 2)
sci[r < 10] = 1000.0  # a bright central source, kept OUT of the sky annulus
# mask_radius 1.0 arcsec / 0.04 arcsec/px = 25 px > 10 px source radius
est = estimate_sky_noise(sci, badmask, mask_radius_arcsec=1.0, pixel_scale=0.04)
print(f"[1] estimate_sky_noise = {est:.3f}  (injected 3.5)")
assert abs(est - 3.5) < 0.5, est

# ----------------------------------------------------------------------------
# check 2: AGN point-source subtraction (data-dependent, skip if missing)
# ----------------------------------------------------------------------------
SCIP = Path(
    "/mnt/d/HermesAgent/ResearchProjects/GalaxyGravLens/ObservationalPipeline"
    "/NIRC2/J2141-0001/kai_work/combo/mag09aug13_kp.fits"
)
BADMASKP = Path(
    "/mnt/d/HermesAgent/ResearchProjects/GalaxyGravLens/ObservationalPipeline"
    "/NIRC2/J2141-0001/archive/selfsky_reduction_v1/mag09aug13_kp_badmask.fits"
)
MGEP = Path(
    "/mnt/d/HermesAgent/ResearchProjects/GalaxyGravLens/DiskLenses/AutoLensFit"
    "/B_series/outputs/b2_mge/mge_params.csv"
)

if SCIP.exists() and BADMASKP.exists() and MGEP.exists():
    from astropy.io import fits

    PIXEL_SCALE = 0.039686
    PSF_SIGMA_PX = (0.15 / 2.355) / PIXEL_SCALE
    MASK_RADIUS = 4.0
    CROP = 512
    ROW_C, COL_C = 1076.0, 1129.0

    Ls, sigs, qs = [], [], []
    with open(MGEP) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            _, L, _sp_px, sp_arc, q = row
            Ls.append(float(L))
            sigs.append(float(sp_arc))
            qs.append(float(q))
    Ls = np.array(Ls)
    sigs = np.array(sigs)
    qs = np.array(qs)

    sci = fits.getdata(SCIP).astype(float)
    badmask = fits.getdata(BADMASKP).astype(np.uint8)
    half = CROP // 2
    sci_c = sci[int(ROW_C) - half : int(ROW_C) + half,
                int(COL_C) - half : int(COL_C) + half]
    bad_c = badmask[int(ROW_C) - half : int(ROW_C) + half,
                    int(COL_C) - half : int(COL_C) + half]

    sky_noise = estimate_sky_noise(sci_c, bad_c, MASK_RADIUS, PIXEL_SCALE)
    noise_c = np.full_like(sci_c, sky_noise)

    subtracted, red = subtract_stellar_light_mge_agn(
        sci_c, noise_c, bad_c,
        mge_L=Ls, mge_sigma_arcsec=sigs, mge_q=qs,
        psf_sigma_px=PSF_SIGMA_PX,
        mask_radius_arcsec=MASK_RADIUS, pixel_scale=PIXEL_SCALE,
    )
    print(f"[2] sky RMS = {sky_noise:.2f}  (B9 printed ~10.05)")
    print(f"[2] MGE+AGN subtraction reduced chi2 = {red:.3f}  (B9 local ~1.11, pure MGE 2.58)")
    assert red < 1.5, red
else:
    print("[2] SKIP: B9 K'-band data / MGE params not present")

# ----------------------------------------------------------------------------
# check 3: pixelized source structure (needs autolens/autofit)
# ----------------------------------------------------------------------------
try:
    from astropy.cosmology import FlatLambdaCDM

    from galaxylens.lensfit.structures import (
        Geometry,
        KinData,
        PipelineConfig,
        PriorSet,
        PriorSpec,
        SourcePriors,
    )
    from galaxylens.lensfit.models.stellar.mge_const_ml import stellar_mge_peak
    from galaxylens.lensfit.pipeline.assemble import build_collection

    ZL, ZSRC = 0.138, 0.7127
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    da = cosmo.angular_diameter_distance(ZL).value
    geometry = Geometry(
        zlens=ZL, zsrc=ZSRC, inclination_deg=78.5, da_mpc=da,
        pc_per_arcsec=da * np.pi / 0.648, re_arcsec=0.88, rd_arcsec=1.51,
    )
    vrms = np.array([
        (0.4123, 236.2, 15.0), (0.9297, 231.4, 15.0), (1.4456, 211.4, 15.0),
        (1.9669, 164.2, 15.0), (2.4815, 168.9, 15.0), (3.0028, 178.8, 15.0),
        (3.5178, 171.0, 15.0), (4.0331, 158.8, 18.0), (4.5529, 180.3, 18.0),
        (5.0696, 209.0, 18.0), (5.5893, 227.5, 20.0),
    ])
    kinematics = KinData(
        xbin=vrms[:, 0], ybin=np.zeros(11), vrms=vrms[:, 1],
        vrms_err=vrms[:, 2], vcirc=289.0, vcirc_err=15.0,
    )
    template = stellar_mge_peak(pc_per_arcsec=geometry.pc_per_arcsec)
    priors = PriorSet(
        ml_img=(2.0, 100.0), v_vir=(150.0, 350.0), gamma=(0.3, 1.5),
        q_h=(0.4, 1.0), ratio=(0.5, 1.0),
        c_2=PriorSpec(kind="gaussian", mean=5.9, sigma=1.49, lower=3.0, upper=20.0),
    )
    base = dict(
        geometry=geometry, kinematics=kinematics, dark="gnfw_virial",
        stellar="mge_const_ml", dynamics="jampy",
        priors=priors, stellar_template=template, source_priors=SourcePriors(),
    )

    # parametric source: 6 mass + 7 Sersic = 13
    cfg_sersic = PipelineConfig(source="single_sersic", **base)
    col_sersic = build_collection(cfg_sersic)
    print(f"[3] single_sersic prior_count = {col_sersic.prior_count}  (expect 13)")
    assert col_sersic.prior_count == 13

    # pixelized source: 6 mass + 1 regularization = 7, but n_source = 225
    cfg_pix = PipelineConfig(source="pixelized", **base)
    col_pix = build_collection(cfg_pix)
    print(f"[3] pixelized    prior_count = {col_pix.prior_count}  (expect 7)")
    assert col_pix.prior_count == 7

    from galaxylens.lensfit.pipeline.assemble import _component
    src = _component("pixelized", cfg_pix)
    print(f"[3] pixelized n_free_params = {src.n_free_params} (expect 1), "
          f"n_source = {src.n_source} (expect 225)")
    assert src.n_free_params == 1
    assert src.n_source == 225
except ImportError as e:
    print(f"[3] SKIP: heavy deps unavailable ({e})")

print("\nALL B9 extraction regression checks passed (skips allowed).")
