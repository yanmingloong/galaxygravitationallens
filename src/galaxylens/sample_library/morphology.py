"""Image morphology measurement for lens identification.

Measure morphological parameters directly from a FITS image and classify the
lens as disk or elliptical. This is the image-level fallback for the
identification layer, used when catalog morphology fields (``lens_type_label``
/ ``n_sersic``) are missing or unreliable. Pure numpy + astropy (lazy import),
no mgefit / autolens dependency.

Criteria (thresholds calibrated on real samples, 2026-11-03):
  C = 5 * log10(r80 / r20)  -- light concentration (primary, most robust)
      J2141 edge-on S0 disk C = 1.58; J1630 SLACS elliptical C = 4.21
      thresholds: C < 3.1 -> disk, C > 3.4 -> elliptical, between -> fall back
      to axis ratio
  q = b / a                  -- axis ratio (photometric 2nd moment, auxiliary)
      NOTE: the lens arc + PSF drag q round (J2141/J1630 both measure q~0.89,
      unreliable), so q is only used as a tie-breaker in the C gap, never as
      the primary criterion

Method:
  - background: outer-annulus median estimate + subtract
  - centre: light centroid (1st moment)
  - C: circular-aperture cumulative-flux 20% / 80% radii
  - q/PA: photometric 2nd moment (inertia tensor) eigenvalue ratio /
    eigenvector direction

Usage:
  morph = measure_morphology("xxx_f814w_sci.fits")
  lens_type, basis = classify_morphology(morph)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Light-concentration thresholds (calibrated on real samples; initial values
# from the typical C range of a Sersic index).
C_DISK_MAX = 3.1      # C below this -> disk
C_ELLIP_MIN = 3.4     # C above this -> elliptical; in between use axis ratio


def measure_morphology(
    fits_path,
    center=None,
    rmax_pix: Optional[float] = None,
) -> dict:
    """Measure morphology parameters from a FITS image, return a dict.

    Args:
      fits_path: science-image FITS path (sci frame).
      center:    lens centre (x, y); defaults to the light centroid.
      rmax_pix:  photometric radius cap (pixels); defaults to
                 ``min(shape)/2 * 0.8`` (avoid the arc / outer halo).
    """
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float64)

    if data.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {data.shape}")

    bg = _estimate_background(data)
    img = np.clip(data - bg, 0.0, None)

    if center is None:
        center = _centroid(img)
    center = (float(center[0]), float(center[1]))

    if rmax_pix is None:
        rmax_pix = float(min(data.shape)) / 2.0 * 0.8

    r20, r80, C = _concentration(img, center, rmax_pix)
    q, pa = _axis_ratio(img, center, rmax_pix)

    return {
        "C": float(C),
        "q": float(q),
        "pa_deg": float(pa),
        "r20_pix": float(r20),
        "r80_pix": float(r80),
        "center": center,
        "background": float(bg),
        "rmax_pix": float(rmax_pix),
    }


def classify_morphology(morph: dict) -> Tuple[str, str]:
    """C + q -> ``(lens_type, basis)`` (disk / elliptical)."""
    C = morph["C"]
    q = morph.get("q", 1.0)

    if C < C_DISK_MAX:
        return "disk", f"C={C:.2f}_disk_like"
    if C > C_ELLIP_MIN:
        return "elliptical", f"C={C:.2f}_bulge_like"

    # In the C gap: use axis ratio (edge-on disks are very flat).
    if q < 0.5:
        return "disk", f"C={C:.2f}_q={q:.2f}_edge_on_disk"
    return "elliptical", f"C={C:.2f}_q={q:.2f}_ambiguous_ellip"


def _estimate_background(data: np.ndarray) -> float:
    """Outer-annulus (outer 10%) median as the background."""
    r = _radial_grid(data)
    rmax = r.max()
    annulus = data[(r > 0.85 * rmax) & (r <= rmax)]
    return float(np.median(annulus)) if annulus.size else 0.0


def _radial_grid(data: np.ndarray) -> np.ndarray:
    ny, nx = data.shape
    yy, xx = np.indices((ny, nx))
    return np.sqrt((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2)


def _centroid(img: np.ndarray) -> Tuple[float, float]:
    """Light centroid (1st moment)."""
    ny, nx = img.shape
    yy, xx = np.indices((ny, nx))
    w = img.sum()
    if w <= 0:
        return (nx / 2.0, ny / 2.0)
    xc = (img * xx).sum() / w
    yc = (img * yy).sum() / w
    return (float(xc), float(yc))


def _concentration(img: np.ndarray, center, rmax_pix: float) -> Tuple[float, float, float]:
    """Circular-aperture cumulative flux -> (r20, r80, C)."""
    ny, nx = img.shape
    yy, xx = np.indices((ny, nx))
    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)

    mask = r <= rmax_pix
    total = img[mask].sum()
    if total <= 0:
        return (np.nan, np.nan, np.nan)

    radii = np.arange(1.0, rmax_pix, 0.5)
    cum = np.array([img[r <= rr].sum() for rr in radii])
    frac = cum / total

    r20 = radii[np.argmin(np.abs(frac - 0.20))]
    r80 = radii[np.argmin(np.abs(frac - 0.80))]
    C = 5.0 * np.log10(r80 / r20) if r20 > 0 else np.nan
    return (float(r20), float(r80), float(C))


def _axis_ratio(img: np.ndarray, center, rmax_pix: float) -> Tuple[float, float]:
    """Photometric 2nd moment (inertia tensor) -> (q=b/a, PA_deg)."""
    ny, nx = img.shape
    yy, xx = np.indices((ny, nx))
    dx = xx - center[0]
    dy = yy - center[1]
    r = np.sqrt(dx ** 2 + dy ** 2)

    w = img * (r <= rmax_pix)
    wsum = w.sum()
    if wsum <= 0:
        return (1.0, 0.0)

    Ixx = (w * dx * dx).sum() / wsum
    Iyy = (w * dy * dy).sum() / wsum
    Ixy = (w * dx * dy).sum() / wsum

    a = (Ixx + Iyy) / 2.0
    b = np.sqrt(((Ixx - Iyy) / 2.0) ** 2 + Ixy ** 2)
    lam1 = a + b
    lam2 = a - b
    q = float(np.sqrt(lam2 / lam1)) if lam1 > 0 else 1.0
    pa = float(0.5 * np.degrees(np.arctan2(2.0 * Ixy, Ixx - Iyy)))
    return (q, pa)
