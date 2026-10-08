"""PSF quality metrics: EE50 (half-light radius) and second-moment ellipticity."""
from __future__ import annotations
from typing import Tuple

import numpy as np


def psf_ee50(psf: np.ndarray) -> Tuple[float, float]:
    """EE50 (half-light radius, px) of a normalized PSF and its Gaussian-approximate FWHM.

    For a 2D Gaussian, EE50 = FWHM / 2, so FWHM_est = 2 * EE50.

    Returns:
        (ee50_px, fwhm_est_px)
    """
    d = np.asarray(psf, dtype=float)
    d = np.maximum(d, 0)
    total = d.sum()
    if total <= 0:
        return float("inf"), float("inf")
    d /= total
    yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    cy = (yy * d).sum()
    cx = (xx * d).sum()
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    order = np.argsort(r.ravel())
    cs = d.ravel()[order].cumsum()
    ee50 = r.ravel()[order][np.searchsorted(cs, 0.5)]
    return float(ee50), 2.0 * float(ee50)

def psf_ellipticity(psf: np.ndarray) -> float:
    """Second-moment ellipticity of a normalized PSF, e = sqrt(e1^2 + e2^2)."""
    d = np.asarray(psf, dtype=float)
    d = np.maximum(d, 0)
    total = d.sum()
    if total <= 0:
        return float("inf")
    yy, xx = np.indices(d.shape)
    cy = (yy * d).sum() / total
    cx = (xx * d).sum() / total
    x2 = (d * (xx - cx) ** 2).sum() / total
    y2 = (d * (yy - cy) ** 2).sum() / total
    xy = (d * (xx - cx) * (yy - cy)).sum() / total
    if x2 + y2 <= 0:
        return 0.0
    e1 = (x2 - y2) / (x2 + y2)
    e2 = 2.0 * xy / (x2 + y2)
    return float(np.sqrt(e1 ** 2 + e2 ** 2))

