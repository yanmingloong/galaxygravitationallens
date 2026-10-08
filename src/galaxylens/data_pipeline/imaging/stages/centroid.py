"""
2D Gaussian centroid refinement
==============
Refine the lens center position via 2D Gaussian fitting, correcting the WCS pointing error.

Features:
  1. Perform a 2D Gaussian fit on the central region of the cropped image
  2. Return the refined sub-pixel centroid

Principle:
  The lens galaxy center is the brightest region of the image; fitting its brightness peak
  with a Gaussian yields an (x0, y0) that is more accurate than the WCS pointing.

Example:
  >>> from galaxylens.data_pipeline.imaging.stages.centroid import fit_centroid
  >>> cx_fit, cy_fit = fit_centroid(sci_cut, sci_cut.shape[1]//2, sci_cut.shape[0]//2)
  >>> print(f"Offset: ({cx_fit-cx:+.1f}, {cy_fit-cy:+.1f}) px")
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit


# ══════════════════════════════════════════════════════════════════════
# Internal: 2D Gaussian model
# ══════════════════════════════════════════════════════════════════════

def _gaussian_2d(
    xy: Tuple[np.ndarray, np.ndarray],
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float,
) -> np.ndarray:
    """Two-dimensional elliptical Gaussian function."""
    x, y = xy
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    return offset + amplitude * np.exp(
        -(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2)
    )


# ══════════════════════════════════════════════════════════════════════
# Public interface
# ══════════════════════════════════════════════════════════════════════

def fit_centroid(
    data: np.ndarray,
    cx: float,
    cy: float,
    box_size: int = 20,
) -> Tuple[float, float]:
    """Fit the central brightness peak with a 2D Gaussian and return the sub-pixel centroid.

    Within a box_size×box_size region around (cx, cy),
    use an elliptical Gaussian fit to precisely locate the brightness center.

    Args:
        data: 2D image (should be background-subtracted data)
        cx: Initial center x coordinate (pixels)
        cy: Initial center y coordinate (pixels)
        box_size: Size of the fitting region (pixels)

    Returns:
        (cx_fit, cy_fit): Refined center coordinates (pixels)

    Note:
        Returns the input (cx, cy) when the fit fails.
    """
    half = box_size // 2
    yc, xc = int(round(cy)), int(round(cx))

    # Ensure no out-of-bounds
    y1, y2 = max(0, yc - half), min(data.shape[0], yc + half)
    x1, x2 = max(0, xc - half), min(data.shape[1], xc + half)

    sub = data[y1:y2, x1:x2]
    if sub.size < 9:  # Too small to fit
        return cx, cy

    yy, xx = np.mgrid[y1:y2, x1:x2]

    # Initial guess
    guess = (
        float(sub.max()),
        x1 + half,
        y1 + half,
        2.0,
        2.0,
        0.0,
        float(np.median(sub)),
    )

    # Parameter bounds
    bounds = (
        [0, x1, y1, 0.5, 0.5, -np.pi / 2, -np.inf],
        [np.inf, x2, y2, box_size / 2, box_size / 2, np.pi / 2, np.inf],
    )

    try:
        popt, _ = curve_fit(
            _gaussian_2d,
            (xx.ravel(), yy.ravel()),
            sub.ravel(),
            p0=guess,
            bounds=bounds,
            maxfev=5000,
        )
        return float(popt[1]), float(popt[2])  # x0, y0
    except Exception:
        return cx, cy


def compute_offset(
    cx_fit: float,
    cy_fit: float,
    cx_initial: float,
    cy_initial: float,
    pixel_scale: float = 0.05,
) -> dict:
    """Compute the centroid offset (in pixels and arcsec).

    Args:
        cx_fit, cy_fit: Fitted centroid
        cx_initial, cy_initial: Initial center
        pixel_scale: Pixel scale (arcsec/pixel)

    Returns:
        {"dx_px": float, "dy_px": float, "dx_as": float, "dy_as": float}
    """
    dx_px = cx_fit - cx_initial
    dy_px = cy_fit - cy_initial
    return {
        "dx_px": float(dx_px),
        "dy_px": float(dy_px),
        "dx_as": float(dx_px * pixel_scale),
        "dy_as": float(dy_px * pixel_scale),
    }
