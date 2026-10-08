"""PSF construction (split into sub-modules).

Sub-modules:
  - stars.py    Field-star detection and selection
  - metrics.py  EE50 / ellipticity metrics
  - epsf.py     photutils ePSF construction
  - analytic.py Moffat / Gaussian analytic PSF
  - tinytim.py  TinyTim (Massey grid + direct compilation)

This module provides the build_psf automatic selection entry point and
re-exports all public functions, keeping the
`from ...stages.psf import ...` call style unchanged.
"""
from __future__ import annotations
from typing import Optional, List, Tuple

import numpy as np

from .stars import detect_stars, filter_field_stars
from .metrics import psf_ee50, psf_ellipticity
from .epsf import build_epsf
from .analytic import (
    build_moffat_psf,
    build_gaussian_psf,
    build_elliptical_gaussian_psf,
)
from .tinytim import (
    build_tinytim_massey_psf,
    build_tinytim_direct_psf,
    MASSEY_AVAILABLE_BANDS,
)
from ...config import PSF_EE50_MIN_PX, PSF_EE50_MAX_PX, PSF_ELL_MAX


def build_psf(
    data: np.ndarray,
    field_stars: List[Tuple[float, float, float]],
    psf_fwhm_pix: float = 2.5,
    pixel_scale: float = 0.05,
    prefer_epsf: bool = True,
    psf_method: Optional[str] = None,
    band: Optional[str] = None,
    date_obs: Optional[str] = None,
    detector_pos: Optional[Tuple[int, int]] = None,
    instrument: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """Automatically select and build the PSF (comparative selection).

    Automatic selection (when psf_method=None):
      1. Generate candidates: ePSF (empirical, captures real aberrations)
         + TinyTim Massey (physical fallback)
      2. Compute metrics: EE50 + second-moment ellipticity
      3. Hard-threshold filtering: EE50 ∈ [0.8, 3.0]px and ellipticity < 0.15
      4. Selection: passing ePSF preferred -> passing TinyTim (fallback)
         -> Moffat as last resort
    Specifying psf_method skips automatic selection.

    Args:
        data: Background-subtracted science image
        field_stars: List of field stars [(x, y, flux), ...]
        psf_fwhm_pix: Estimated PSF FWHM (pixels)
        pixel_scale: Pixel scale (arcsec/pixel)
        prefer_epsf: Whether to prefer trying ePSF
        psf_method: Force a specific method ("epsf", "moffat", "gaussian",
                    "gaussian_ell", "tinytim_massey", "tinytim_direct")
        band: Band (required by TinyTim)
        date_obs: Observation date (required by TinyTim focus correction)
        detector_pos: Detector pixel (required by TinyTim position mapping)
        instrument: Instrument name (required by TinyTim direct)

    Returns:
        (psf_array, source_label)
    """
    # ── Force a specific method ──
    if psf_method == "epsf":
        if len(field_stars) > 0:
            result = build_epsf(data, field_stars)
            if result is not None:
                psf, label, n = result
                print(f"  PSF: ePSF ({n} stars)", flush=True)
                return psf, label

    elif psf_method == "moffat":
        fwhm_as = psf_fwhm_pix * pixel_scale
        psf, label = build_moffat_psf(fwhm_arcsec=fwhm_as, pixel_scale=pixel_scale)
        print(f"  PSF: Moffat (FWHM={fwhm_as:.2f}\", beta=3.0)", flush=True)
        return psf, label

    elif psf_method == "gaussian":
        fwhm_as = psf_fwhm_pix * pixel_scale
        psf, label = build_gaussian_psf(fwhm_arcsec=fwhm_as, pixel_scale=pixel_scale)
        print(f"  PSF: Gaussian (FWHM={fwhm_as:.2f}\")", flush=True)
        return psf, label

    elif psf_method == "gaussian_ell":
        fwhm_as = psf_fwhm_pix * pixel_scale
        psf, label = build_elliptical_gaussian_psf(fwhm_arcsec=fwhm_as, pixel_scale=pixel_scale)
        print(f"  PSF: Elliptical Gaussian (FWHM={fwhm_as:.2f}\", e=0.05)", flush=True)
        return psf, label

    elif psf_method == "tinytim_massey":
        if instrument and "ACS" not in str(instrument).upper():
            print(f"  PSF: TinyTim Massey skipped (instrument={instrument}, ACS only)", flush=True)
        elif band and band in MASSEY_AVAILABLE_BANDS:
            result = build_tinytim_massey_psf(
                band=band, date_obs=date_obs or "",
                detector_pos=detector_pos, pixel_scale=pixel_scale,
            )
            if result is not None:
                psf, label = result
                print(f"  PSF: TinyTim Massey ({label})", flush=True)
                return psf, label

    elif psf_method == "tinytim_direct":
        if band and instrument:
            result = build_tinytim_direct_psf(
                band=band, instrument=instrument,
                date_obs=date_obs or "",
                detector_pos=detector_pos or (400, 400),
                pixel_scale=pixel_scale,
            )
            if result is not None:
                psf, label = result
                print(f"  PSF: TinyTim Direct ({label})", flush=True)
                return psf, label

    # ── Automatic comparative selection (TinyTim fallback + ePSF candidate, EE50/ellipticity threshold) ──
    candidates = []  # [(name, psf, label, ee50, ell, valid)]

    # 1. ePSF (empirical candidate, captures real aberrations)
    if prefer_epsf and len(field_stars) > 0:
        result = build_epsf(data, field_stars)
        if result is not None:
            psf, label, n_used = result
            ee50, fwhm = psf_ee50(psf)
            ell = psf_ellipticity(psf)
            ok = (PSF_EE50_MIN_PX <= ee50 <= PSF_EE50_MAX_PX) and (ell < PSF_ELL_MAX)
            candidates.append(("epsf", psf, label, ee50, ell, ok))
            print(f"  PSF candidate ePSF ({n_used} stars): EE50={ee50:.2f}px, "
                  f"ell={ell:.3f} -> {'OK' if ok else 'REJECT'}", flush=True)

    # 2. TinyTim Massey (physical fallback, independent of field stars)
    if band and band in MASSEY_AVAILABLE_BANDS:
        result = build_tinytim_massey_psf(
            band=band, date_obs=date_obs or "",
            detector_pos=detector_pos, pixel_scale=pixel_scale,
        )
        if result is not None:
            psf, label = result
            ee50, fwhm = psf_ee50(psf)
            ell = psf_ellipticity(psf)
            ok = (PSF_EE50_MIN_PX <= ee50 <= PSF_EE50_MAX_PX) and (ell < PSF_ELL_MAX)
            candidates.append(("tinytim_massey", psf, label, ee50, ell, ok))
            print(f"  PSF candidate TinyTim Massey: EE50={ee50:.2f}px, "
                  f"ell={ell:.3f} -> {'OK' if ok else 'REJECT'}", flush=True)

    # 3. Selection: passing ePSF preferred (captures real aberrations) -> passing TinyTim (fallback)
    for pref in ("epsf", "tinytim_massey"):
        for name, psf, label, ee50, ell, ok in candidates:
            if name == pref and ok:
                print(f"  PSF selected: {name} ({label}, EE50={ee50:.2f}px, ell={ell:.3f})",
                      flush=True)
                return psf, label

    # 4. Moffat fallback (no passing candidate)
    fwhm_as = psf_fwhm_pix * pixel_scale
    psf, label = build_moffat_psf(fwhm_arcsec=fwhm_as, pixel_scale=pixel_scale)
    print(f"  PSF: Moffat fallback (FWHM={fwhm_as:.2f}\", beta=3.0)", flush=True)
    return psf, label

__all__ = [
    "detect_stars", "filter_field_stars",
    "psf_ee50", "psf_ellipticity",
    "build_epsf",
    "build_moffat_psf", "build_gaussian_psf", "build_elliptical_gaussian_psf",
    "build_tinytim_massey_psf", "build_tinytim_direct_psf",
    "build_psf",
]
