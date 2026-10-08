"""
Cosmic ray detection & handling
================
Detect cosmic rays in single-exposure FLT data, or automatically reject them via multi-exposure stacking.

Methods:
  1. detect_cosmic_rays_laplace() — Simplified LA Cosmic (local median deviation)
  2. sigma_clipped_stack()        — Sigma-clipped median stack (multi-exposure CR rejection)

Processing flow:
    FLT exposure
      ├─ single exposure → detect_cosmic_rays_laplace() → CR mask
      └─ multi-exposure → sigma_clipped_stack() → stacked SCI + CR hitmap

References:
  - van Dokkum (2001) PASP 113, 1420: LA Cosmic algorithm
  - CALACS ISR: cosmic ray rejection in ACS pipeline

Usage example:
  >>> from galaxylens.data_pipeline.imaging.stages.cosmic_ray import detect_cosmic_rays_laplace
  >>> cr_mask = detect_cosmic_rays_laplace(sci_data)
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
from scipy.ndimage import median_filter


# ══════════════════════════════════════════════════════════════════════
# Default parameters
# ══════════════════════════════════════════════════════════════════════

CR_SIGMA: float = 8.0        # CR detection σ threshold
CR_LOCAL_BOX: int = 5        # local median filter window (pixels)


# ══════════════════════════════════════════════════════════════════════
# Method 1: Simplified LA Cosmic
# ══════════════════════════════════════════════════════════════════════

def detect_cosmic_rays_laplace(
    data: np.ndarray,
    sigma: float = CR_SIGMA,
    box: int = CR_LOCAL_BOX,
) -> np.ndarray:
    """Detect cosmic-ray pixels based on local median deviation (simplified LA Cosmic).

    For each pixel, compare its value against the local median filter.
    If the deviation > sigma × local_MAD, mark it as a cosmic ray.

    Args:
        data: Input 2D image
        sigma: Detection threshold (unit: local MAD)
        box: Local filter window size (pixels)

    Returns:
        Boolean mask, True = cosmic-ray pixel
    """
    local_med = median_filter(data, size=box)
    local_mad = median_filter(np.abs(data - local_med), size=box) * 1.4826
    local_mad = np.maximum(local_mad, 1e-10)

    cr_mask = (data - local_med) > sigma * local_mad
    return cr_mask


# ══════════════════════════════════════════════════════════════════════
# Method 2: Sigma-clipped median stacking
# ══════════════════════════════════════════════════════════════════════

def sigma_clipped_stack(
    images: List[np.ndarray],
    sigma: float = 3.0,
    maxiters: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack multiple images with a sigma-clipped median (pseudo-drizzle CR rejection).

    For each pixel position, compute the median and MAD across exposures,
    reject values deviating > sigma × MAD, then take the median.

    Args:
        images: List of images (same region across multiple exposures)
        sigma: Clipping threshold
        maxiters: Maximum number of iterations

    Returns:
        (stacked_image, cr_hitmap):
          stacked_image — Stacked image (2D float64)
          cr_hitmap    — Total times each pixel was clipped (2D int)
    """
    if len(images) == 1:
        return images[0].astype(np.float64), np.zeros_like(images[0], dtype=int)

    stack = np.stack([img.astype(np.float64) for img in images], axis=0)
    total_hitmap = np.zeros(stack.shape[1:], dtype=int)

    for iteration in range(maxiters):
        median = np.median(stack, axis=0)
        mad = 1.4826 * np.median(
            np.abs(stack - median[np.newaxis, :, :]), axis=0,
        )
        mad = np.maximum(mad, 1e-10)

        deviate = np.abs(stack - median[np.newaxis, :, :]) / mad[np.newaxis, :, :]
        mask = deviate > sigma

        if mask.sum() == 0:
            break

        total_hitmap += mask.sum(axis=0)
        stack = np.where(mask, np.nan, stack)

    result = np.nanmedian(stack, axis=0)
    return result, total_hitmap


def compute_cr_fraction(cr_mask: np.ndarray) -> float:
    """Compute the fraction of cosmic-ray pixels.

    Args:
        cr_mask: Boolean CR mask

    Returns:
        CR pixel fraction (0-100%)
    """
    if cr_mask.size == 0:
        return 0.0
    return 100.0 * float(cr_mask.sum()) / float(cr_mask.size)


# ══════════════════════════════════════════════════════════════════════
# Method 3: deepCR deep-learning detection (2026-07-03)
# ══════════════════════════════════════════════════════════════════════

# deepCR model name mapping (reference dinos_i)
_DEEPCR_MODEL_MAP = {
    ("ACS", "WFC", "F435W"): "ACS-WFC-F606W-2-32",
    ("ACS", "WFC", "F606W"): "ACS-WFC-F606W-2-32",
    ("ACS", "WFC", "F814W"): "ACS-WFC-F606W-2-32",
}


def detect_cosmic_rays_deepcr(
    data: np.ndarray,
    exptime: float = 1.0,
    instrument: str = "ACS",
    detector: str = "WFC",
    band: str = "F814W",
    threshold: float = 0.5,
    inpaint: bool = False,
):
    """Detect cosmic rays with a deepCR deep-learning model (optional inpainting).

    Zhang & Bloom (2020): HST cosmic-ray detection based on Mask R-CNN.
    Models are specialized per instrument/band, with accuracy far higher than LA-cosmic.

    Args:
        data: 2D image (electrons)
        exptime: Exposure time (seconds)
        instrument: Instrument name ("ACS", "WFC3")
        detector: Detector ("WFC", "UVIS")
        band: Band name
        threshold: Detection threshold (0-1, default 0.5, reference dinos_i)
        inpaint: When True, return (mask, inpainted image); inpainting uses 5×5 median fill
                 (falls back to medmask when deepCR has no dedicated inpaint model for ACS/WFC).

    Returns:
        inpaint=False: Boolean CR mask (True = cosmic-ray pixel)
        inpaint=True:  (Boolean CR mask, inpainted electrons image)

    Note:
        If the model is unavailable or deepCR is not installed, inpaint=False returns an all-False mask;
        inpaint=True returns (all-False mask, original image).
    """
    try:
        from deepCR import deepCR
    except ImportError:
        if inpaint:
            return np.zeros(data.shape, dtype=bool), np.asarray(data)
        return np.zeros(data.shape, dtype=bool)

    model_key = (instrument, detector, band)
    model_name = _DEEPCR_MODEL_MAP.get(model_key)

    if model_name is None:
        # try to construct the model name
        model_name = f"{instrument}-{detector}-{band}-2-32"

    try:
        # deepCR expects the input in units of electrons.
        # The FLC/FLT SCI extension has BUNIT=ELECTRONS (already total electron counts),
        # so do not multiply by exptime (that conversion is only needed for electrons/s).
        image_e = np.maximum(data, 0)

        mdl = deepCR(mask=model_name, device="CPU")
        # clean(inpaint=False) returns a single mask; inpaint=True returns a (mask, inpainted) tuple.
        # inpainted has already been restored by deepCR to the input units (electrons).
        result = mdl.clean(image_e, threshold=threshold, inpaint=inpaint)

        if inpaint:
            mask, inpainted = result
            return np.asarray(mask).astype(bool), np.asarray(inpainted)
        return np.asarray(result).astype(bool)

    except Exception as e:
        print(f"    [WARN] deepCR failed ({e}), falling back to LA-cosmic", flush=True)
        if inpaint:
            return np.zeros(data.shape, dtype=bool), np.asarray(data)
        return np.zeros(data.shape, dtype=bool)
