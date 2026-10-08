"""
Stellar stray-light mask
=============
Generate stellar masks that flag the regions around field stars to prevent their stray light from contaminating the lens modeling.

Features:
  1. Generate a circular mask for each detected field star
  2. Exclude the lens center region (protecting the galaxy and arcs)
  3. Mask radius = MASK_RADIUS_FACTOR × PSF_FWHM

Processing flow:
    Field star list → exclude center region → generate circular mask per star → combine output

Example:
  >>> from galaxylens.data_pipeline.imaging.stages.starmask import make_starmask
  >>> mask = make_starmask(shape, field_stars, psf_fwhm_pix, center, pixel_scale)
"""

from __future__ import annotations
from typing import Optional, List, Tuple

import numpy as np

from ..config import MASK_RADIUS_FACTOR


# ══════════════════════════════════════════════════════════════════════
# Public interface
# ══════════════════════════════════════════════════════════════════════

def make_starmask(
    shape: Tuple[int, int],
    star_coords: List[Tuple[float, float, float]],
    psf_fwhm_pix: float,
    lens_center: Tuple[float, float],
    pixel_scale: float,
    lens_exclusion_arcsec: float = 5.0,
    arc_positions: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Generate the stellar stray-light mask.

    Create a circular mask for each detected field star,
    excluding the lens center region and known arc positions.

    Args:
        shape: Image shape (ny, nx)
        star_coords: Field star list [(x, y, flux), ...]
        psf_fwhm_pix: PSF FWHM (pixels)
        lens_center: Lens center (cx, cy)
        pixel_scale: Pixel scale (arcsec/pixel)
        lens_exclusion_arcsec: Center protection radius (arcsec)
        arc_positions: Additional protected arc coordinates [(x, y), ...]

    Returns:
        Boolean mask (True = pixel to be masked out)
    """
    mask = np.zeros(shape, dtype=bool)
    cx, cy = lens_center

    lens_excl_pix = lens_exclusion_arcsec / pixel_scale
    mask_radius = int(np.ceil(MASK_RADIUS_FACTOR * psf_fwhm_pix))

    # Pre-generate the coordinate grid
    yy, xx = np.ogrid[:shape[0], :shape[1]]

    for sx, sy, flux in star_coords:
        # Skip the lens center region
        if np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2) < lens_excl_pix:
            continue

        # Skip arc regions
        if arc_positions:
            skip = False
            for ax, ay in arc_positions:
                if np.sqrt((sx - ax) ** 2 + (sy - ay) ** 2) < mask_radius:
                    skip = True
                    break
            if skip:
                continue

        # Circular mask
        dist = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
        mask[dist < mask_radius] = True

    return mask


def combine_masks(
    masks: List[np.ndarray],
    target_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Combine multiple masks (logical OR), unified to the target shape.

    Used to combine starmasks from different bands/exposures.

    Args:
        masks: List of Boolean masks
        target_shape: Target shape; if None, use the smallest common size of the masks

    Returns:
        Combined Boolean mask
    """
    if not masks:
        raise ValueError("No masks to combine")

    if target_shape is None:
        shapes = [m.shape for m in masks]
        target_shape = (
            min(s[0] for s in shapes),
            min(s[1] for s in shapes),
        )

    combined = np.zeros(target_shape, dtype=bool)
    for m in masks:
        if m.shape == target_shape:
            combined = combined | m
        else:
            # center-crop to target_shape
            dy = (m.shape[0] - target_shape[0]) // 2
            dx = (m.shape[1] - target_shape[1]) // 2
            cropped = m[dy:dy + target_shape[0], dx:dx + target_shape[1]]
            combined = combined | cropped

    return combined


def get_mask_stats(mask: np.ndarray) -> dict:
    """Compute statistics of the mask.

    Args:
        mask: Boolean mask

    Returns:
        {"n_masked": int, "fraction_pct": float}
    """
    n = int(mask.sum())
    return {
        "n_masked": n,
        "fraction_pct": 100.0 * n / mask.size if mask.size > 0 else 0.0,
    }
