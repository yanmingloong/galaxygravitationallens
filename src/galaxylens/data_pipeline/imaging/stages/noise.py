"""
Noise map generation
==========
Compute the σ-map (noise standard deviation map) from HST weight maps (WHT) and/or science images.

Methods:
  1. noise_from_wht()             — CALACS standard: σ = 1/√(WHT)
  2. noise_from_wht_plus_poisson()— WHT + Poisson shot noise (DRC)
  3. noise_from_wht_poisson_calib()— + two-pass σ-clipping empirical calibration (disk galaxy final)
  4. noise_poisson_readnoise()    — Poisson + read noise (no-WHT fallback)
  5. noise_stacked()              — multi-exposure stacking: σ/√N

References:
  - CALACS ISR 2015-03: ACS WHT weighting scheme
  - MultiDrizzle Handbook: WHT for drizzled products
  - preprocess_swells.py v1.3: empirical noise calibration method

Processing flow:
    Input: sci_bgsub + wht (optional)
           │
           ├─ WHT present?
           │   ├─ YES → input type?
           │   │   ├─ FLT/FLC → noise_from_wht()
           │   │   │              └─ multiple exposures? → noise_stacked()
           │   │   └─ DRC/DRZ  → noise_from_wht_plus_poisson()
           │   │                  └─ calibrate? → noise_from_wht_poisson_calibrated()
           │   └─ NO  → noise_poisson_readnoise()
           │
           └─ Output: noise_map (σ-map, 2D float32)

Usage example:
  >>> from galaxylens.data_pipeline.imaging.stages.noise import noise_from_wht
  >>> noise = noise_from_wht(wht_data)
"""

from __future__ import annotations
from typing import Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# Method 1: WHT-only (CALACS standard)
# ══════════════════════════════════════════════════════════════════════

def noise_from_wht(
    wht_data: np.ndarray,
    n_exposures: int = 1,
) -> np.ndarray:
    """CALACS standard noise: compute the σ-map directly from the WHT weight map.

    The WHT extension of the CALACS pipeline contains the inverse variance of
    all noise components: read noise + dark current + flat-field error +
    Poisson shot noise. Hence noise = 1/√(WHT), with no additional Poisson term
    required.

    Args:
        wht_data: WHT weight map (2D array), in units of electrons⁻²·s²
        n_exposures: number of stacked exposures. When >1, noise /= √(n_exposures)

    Returns:
        Noise map σ-map (2D float32), in units of electrons/s

    Raises:
        ValueError: if wht_data contains no positive values

    Note:
        Only applicable to FLT/FLC products of ACS/WFC.
        The DRC WHT comes from drizzle and contains inter-pixel correlations,
        so this method underestimates the noise.
    """
    # Handle non-positive WHT values
    wht_positive = wht_data[wht_data > 0]
    if len(wht_positive) == 0:
        raise ValueError("WHT data contains no positive values")

    wht_median = float(np.median(wht_positive))
    wht_safe = np.where(wht_data > 0, wht_data, wht_median)

    noise = 1.0 / np.sqrt(np.maximum(wht_safe, 1e-10))

    # Handle infinite values
    noise = np.where(np.isfinite(noise), noise, np.median(noise[np.isfinite(noise)]))

    # Multi-exposure scaling
    if n_exposures > 1:
        noise = noise / np.sqrt(float(n_exposures))

    return noise.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Method 2: WHT + Poisson (DRC products)
# ══════════════════════════════════════════════════════════════════════

def noise_from_wht_plus_poisson(
    wht_data: np.ndarray,
    sci_bgsub: np.ndarray,
) -> np.ndarray:
    """WHT + Poisson shot noise (used for DRC products).

    The DRC WHT map comes from the drizzle process (1/σ²),
    so the source-photon Poisson noise must be considered separately.

    Warning: this method double-counts the sky background noise,
    because the MultiDrizzle WHT already includes the sky variance.
    For DRC, noise_from_wht_poisson_calibrated() is recommended.

    Args:
        wht_data: WHT weight map
        sci_bgsub: background-subtracted science image (electrons/s)

    Returns:
        Noise map σ-map (2D float32)
    """
    # WHT noise
    noise_wht = noise_from_wht(wht_data)

    # Poisson noise: σ_poisson = √(max(sci, 0))
    noise_poisson = np.sqrt(np.maximum(sci_bgsub, 0))

    # Sum in quadrature
    noise = np.sqrt(noise_wht**2 + noise_poisson**2)
    return noise.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Method 3: WHT + Poisson + empirical calibration (disk-galaxy DRC final)
# ══════════════════════════════════════════════════════════════════════

def noise_from_wht_poisson_calibrated(
    wht_data: np.ndarray,
    sci_bgsub: np.ndarray,
    min_bg_pixels: int = 20,
) -> np.ndarray:
    """WHT + Poisson + two-pass σ-clipping empirical global scaling calibration.

    Corrects the double-counting of sky background in the WHT+Poisson
    combination for DRC products.

    Algorithm:
      1. First compute noise_raw = √(σ²_wht + σ²_poisson)
      2. Identify background pixels with two-pass sigma-clipping
      3. Compare the std of SCI against the median of noise_raw in the background region
      4. scale = σ_sci_bg / median(noise_raw_bg)
      5. noise_final = noise_raw × scale

    Args:
        wht_data: WHT weight map
        sci_bgsub: background-subtracted science image
        min_bg_pixels: minimum background pixel count, below which calibration is skipped

    Returns:
        Empirically calibrated noise map σ-map (2D float32)

    References:
        preprocess_swells.py v1.3
        fix_noise_maps_disk_v2.py
    """
    # Step 1: raw WHT+Poisson
    noise_raw = noise_from_wht_plus_poisson(wht_data, sci_bgsub).astype(np.float64)

    # Step 2: first-pass sigma-clipping (MAD-based)
    sci_flat = sci_bgsub.ravel()
    med_sci = float(np.median(sci_flat))
    mad_sci = float(np.median(np.abs(sci_flat - med_sci)))
    sigma_est = mad_sci * 1.4826

    lo = med_sci - 3.0 * sigma_est
    hi = med_sci + 5.0 * sigma_est
    bg1 = (sci_bgsub > lo) & (sci_bgsub < hi)

    if bg1.sum() < min_bg_pixels:
        return noise_raw.astype(np.float32)

    # Step 3: second-pass sigma-clipping (within the background subset)
    bg1_sci = sci_bgsub[bg1]
    bg_med1 = float(np.median(bg1_sci))
    bg_std1 = float(np.std(bg1_sci))

    bg2 = np.abs(sci_bgsub - bg_med1) < 2.5 * bg_std1
    bg_final = bg1 & bg2

    if bg_final.sum() < min_bg_pixels:
        return noise_raw.astype(np.float32)

    # Step 4: scaling factor
    sci_std_bg = float(np.std(sci_bgsub[bg_final]))
    noise_med_bg = float(np.median(noise_raw[bg_final]))

    if noise_med_bg < 1e-15 or sci_std_bg < 1e-15:
        return noise_raw.astype(np.float32)

    scale = np.clip(sci_std_bg / noise_med_bg, 0.01, 100.0)
    noise = noise_raw * scale

    return noise.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Method 3.5: pure WHT + global rescaling (Barnabè-style, disk-galaxy FLC)
# ══════════════════════════════════════════════════════════════════════

def noise_from_wht_rescaled(
    wht_data: np.ndarray,
    sci_bgsub: np.ndarray,
    min_bg_pixels: int = 20,
) -> np.ndarray:
    """Pure WHT noise × single global rescaling factor (Barnabè+2012 style).

    Pure WHT noise (σ = 1/√WHT) is physically correct but too small for
    disk-galaxy Nautilus fits: model systematic errors (≳16 e⁻/s) dwarf the
    measurement noise (≈0.35 e⁻/s), blowing up χ²/px to ≈2167. Barnabè+2012
    rescales the noise by a single global factor so that reduced χ² ≈ 1. This
    preprocessing-stage approximation uses the background region to estimate
    that factor:

        scale = σ_sci_bg / median(noise_raw_bg)

    Unlike the signal-dependent bug noise (√(1/WHT + SCI)), this method scales
    the whole map uniformly and adds no double-counted Poisson term — matching
    Barnabè's single-factor assumption.

    Algorithm:
      1. noise_raw = 1/√WHT (pure WHT, no Poisson term)
      2. Two-pass sigma-clipping to identify background pixels
      3. scale = σ_sci_bg / median(noise_raw_bg)
      4. noise_final = noise_raw × scale (uniform upscale)

    Args:
        wht_data: WHT weight map.
        sci_bgsub: Background-subtracted science image.
        min_bg_pixels: Minimum background pixels; below this, skip calibration.

    Returns:
        Globally rescaled noise σ-map (2D float32).

    References:
        Barnabè+2012 (SWELLS IV, mnras0423-1073): global rescaling to reduced χ²~1
    """
    # Step 1: pure WHT noise (no Poisson term)
    noise_raw = noise_from_wht(wht_data).astype(np.float64)

    # Step 2: first-pass sigma-clipping (MAD-based)
    sci_flat = sci_bgsub.ravel()
    med_sci = float(np.median(sci_flat))
    mad_sci = float(np.median(np.abs(sci_flat - med_sci)))
    sigma_est = mad_sci * 1.4826

    lo = med_sci - 3.0 * sigma_est
    hi = med_sci + 5.0 * sigma_est
    bg1 = (sci_bgsub > lo) & (sci_bgsub < hi)

    if bg1.sum() < min_bg_pixels:
        return noise_raw.astype(np.float32)

    # Step 3: second-pass sigma-clipping
    bg1_sci = sci_bgsub[bg1]
    bg_med1 = float(np.median(bg1_sci))
    bg_std1 = float(np.std(bg1_sci))
    bg2 = np.abs(sci_bgsub - bg_med1) < 2.5 * bg_std1
    bg_final = bg1 & bg2

    if bg_final.sum() < min_bg_pixels:
        return noise_raw.astype(np.float32)

    # Step 4: global rescaling factor
    sci_std_bg = float(np.std(sci_bgsub[bg_final]))
    noise_med_bg = float(np.median(noise_raw[bg_final]))

    if noise_med_bg < 1e-15 or sci_std_bg < 1e-15:
        return noise_raw.astype(np.float32)

    scale = np.clip(sci_std_bg / noise_med_bg, 0.01, 100.0)
    noise = noise_raw * scale

    return noise.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Method 4: Poisson + read noise (no-WHT fallback)
# ══════════════════════════════════════════════════════════════════════

def noise_poisson_readnoise(
    sci_bgsub: np.ndarray,
    bg_rms: float,
    apply_median_filter: bool = True,
    median_filter_size: int = 3,
) -> np.ndarray:
    """Poisson shot noise + read noise estimate (fallback when no WHT is available).

    Used for products without a WHT extension, such as WFPC2.

    Args:
        sci_bgsub: background-subtracted science image
        bg_rms: background RMS (from estimate_background)
        apply_median_filter: whether to apply median-filter smoothing (removes drizzle striping)
        median_filter_size: median filter window size

    Returns:
        Noise map σ-map (2D float32)
    """
    noise = np.sqrt(np.maximum(sci_bgsub, 0) + bg_rms**2)

    if apply_median_filter:
        from scipy.ndimage import median_filter as mf
        noise = mf(noise, size=median_filter_size)

    # Rescale to match background RMS
    noise_pos = noise[noise > 0]
    if len(noise_pos) > 0:
        noise_med = float(np.median(noise_pos))
        if noise_med > 1e-15:
            noise = noise * bg_rms / noise_med

    return noise.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Method 5: multi-exposure stacking noise
# ══════════════════════════════════════════════════════════════════════

def noise_stacked(
    noise_single: np.ndarray,
    n_exposures: int,
) -> np.ndarray:
    """Noise scaling after multi-exposure stacking: σ_stacked = σ_single / √N.

    Args:
        noise_single: noise map of a single exposure
        n_exposures: number of stacked exposures

    Returns:
        Stacked noise map
    """
    if n_exposures <= 1:
        return noise_single
    return (noise_single / np.sqrt(float(n_exposures))).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# One-stop interface: automatically select the method based on product type
# ══════════════════════════════════════════════════════════════════════

def compute_noise(
    sci_bgsub: np.ndarray,
    wht_data: Optional[np.ndarray],
    product_type: str,
    bg_rms: float = 0.0,
    n_exposures: int = 1,
    calibrate: bool = True,
    use_bug_noise: bool = False,
    use_rescaled: bool = False,
) -> np.ndarray:
    """Automatically select the noise method based on product type.

    Args:
        sci_bgsub: background-subtracted science image
        wht_data: WHT weight map (may be None)
        product_type: "FLT", "FLC", "DRC", "DRZ"
        bg_rms: background RMS (for the no-WHT fallback)
        n_exposures: number of exposures (stacking scaling applied when >1)
        calibrate: whether to apply empirical calibration for DRC/DRZ
        use_bug_noise: FLT/FLC use the signal-dependent bug noise √(1/WHT+SCI) (absorbs model residuals)
        use_rescaled: FLT/FLC use pure WHT × global rescaling factor (Barnabè-style)

    Returns:
        Noise map σ-map (2D float32)
    """
    if wht_data is not None:
        if product_type in ("FLT", "FLC"):
            if use_bug_noise:
                # Signal-dependent bug noise: σ = √(1/WHT + SCI), absorbs model residuals.
                noise = noise_from_wht_plus_poisson(wht_data, sci_bgsub)
            elif use_rescaled:
                # Global rescaling: pure WHT × single factor (Barnabè-style).
                noise = noise_from_wht_rescaled(wht_data, sci_bgsub)
            else:
                noise = noise_from_wht(wht_data, n_exposures=n_exposures)
        elif product_type in ("DRC", "DRZ"):
            if calibrate:
                noise = noise_from_wht_poisson_calibrated(wht_data, sci_bgsub)
            else:
                noise = noise_from_wht_plus_poisson(wht_data, sci_bgsub)
        else:
            noise = noise_from_wht(wht_data, n_exposures=n_exposures)
    else:
        noise = noise_poisson_readnoise(sci_bgsub, bg_rms)

    return noise
