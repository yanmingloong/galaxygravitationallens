"""Light MGE fit from a mock image (mgefit), the arc-building mass model's front end.

Fit a light MGE (surface brightness) from a mock image (the SKIRT noiseless
``total.fits`` or the fake-HST ``data.fits``), producing
``(surf, sig_as, q_obs, total_lum)`` for
:func:`~galaxylens.mock_pipeline.lensing.arc.build_lens_galaxy` to multiply by
M/L into a mass MGE.

This is the arc stage's front end (previously missing, backed by external MGE
artifacts); extracted from the MF series ``mf1a_mge.py``, logic frozen:
two-pass truncated 2nd moment for centre/eps/theta -> ``sectors_photometry``
elliptical-ring photometry -> ``fit_sectors`` MGE fit. ``mgefit`` is a lazy
import (importing this module stays light; only the actual fit needs it).

Key conventions (same as ``mf1a_mge.py``, do NOT change):
  - ``sigmapsf_px`` deconvolves with the PSF 2nd-moment sigma to avoid a double
    PSF when PyAutoLens convolves again;
  - an outer faint stellar halo would fit a spurious ~20 kpc-wide Gaussian, so
    truncate to R < 2.5 x R50;
  - ``qbounds`` lower limit = sqrt(cos^2 i + q_int_min^2 sin^2 i), keeping the
    deprojected q_int >= 0.05.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def measure_geometry(img, rmax_px):
    """Two-pass 2D 2nd moment -> (col_c, row_c, eps, theta_deg).

    First pass coarsely locates the centre over the whole image; the second
    pass refines eps/theta only inside the R<rmax_px truncation region,
    excluding the outer faint stellar halo (otherwise eps is dragged round, or
    over-thinned by an aggressive threshold).
    """
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    # First pass: whole-image coarse centre (gentle threshold)
    sel0 = img > 0.01 * img.max()
    m0 = img[sel0].astype(np.float64)
    x0 = xx[sel0].astype(np.float64)
    y0 = yy[sel0].astype(np.float64)
    xc0 = float((x0 * m0).sum() / m0.sum())
    yc0 = float((y0 * m0).sum() / m0.sum())
    # Second pass: truncation-region refine
    r2 = (xx - xc0) ** 2 + (yy - yc0) ** 2
    sel = (img > 0.05 * img.max()) & (r2 < rmax_px ** 2)
    m = img[sel].astype(np.float64)
    x = xx[sel].astype(np.float64)
    y = yy[sel].astype(np.float64)
    xc = float((x * m).sum() / m.sum())
    yc = float((y * m).sum() / m.sum())
    dx = x - xc
    dy = y - yc
    Sxx = (dx * dx * m).sum() / m.sum()
    Syy = (dy * dy * m).sum() / m.sum()
    Sxy = (dx * dy * m).sum() / m.sum()
    evals, evecs = np.linalg.eigh(np.array([[Sxx, Sxy], [Sxy, Syy]]))
    eps = float(1.0 - np.sqrt(evals[0] / evals[1]))          # ellipticity
    theta = float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])))
    return xc, yc, eps, theta


def read_pixel_scale_kpc(skirt_params_xml: str) -> float:
    """Read the pixel scale (kpc/px) = fieldOfViewX / numPixelsX from SKIRT's parameters.xml."""
    import re

    xml = Path(skirt_params_xml).read_text()
    fov_pc = float(re.search(r'fieldOfViewX="([\d.eE+-]+) pc"', xml).group(1))
    npx = int(re.search(r'numPixelsX="(\d+)"', xml).group(1))
    return fov_pc / 1000.0 / npx          # pc -> kpc


def read_psf_sigma_px(psf_path: str) -> float:
    """Circular Gaussian sigma (px) from a PSF FITS, for mgefit sigmapsf deconvolution."""
    from astropy.io import fits

    psf = fits.getdata(psf_path).astype(np.float64)
    ny, nx = psf.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    psf_n = psf / psf.sum()
    var = float((r2 * psf_n).sum())
    return float(np.sqrt(var / 2.0))       # 1D Gaussian sigma = sqrt(<r^2>/2)


def fit_light_mge(
    fits_path,
    pixel_scale_kpc: float,
    r50_kpc: float,
    qbounds,
    kpc_per_arcsec: float,
    ngauss: int = 8,
    sigmapsf_px: float = 0.0,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a light MGE from a mock image -> ``(surf_pc2, sig_as, q_obs, total_lum)``.

    ``surf_pc2`` is the peak surface density (flux/pc^2); ``sig_as`` the
    per-Gaussian width (arcsec); ``q_obs`` the projected axis ratio;
    ``total_lum`` the relative total luminosity (SKIRT flux units). The returned
    4-tuple matches the MF series ``load_mge`` output, ready for
    ``build_lens_galaxy``.
    """
    import mgefit as mge
    from astropy.io import fits

    d = fits.getdata(fits_path)
    img = d[0].astype(np.float64) if d.ndim == 3 else d.astype(np.float64)

    rmax_px = 2.5 * r50_kpc / pixel_scale_kpc
    col_c, row_c, eps, theta = measure_geometry(img, rmax_px)

    mask = img > 0
    s = mge.sectors_photometry(img, eps, theta, row_c, col_c,
                               minlevel=0.0, mask=mask, plot=False)
    sel = s.radius < rmax_px

    m = mge.fit_sectors(s.radius[sel], s.angle[sel], s.counts[sel], eps,
                        ngauss=ngauss, sigmapsf=sigmapsf_px,
                        qbounds=qbounds, scale=pixel_scale_kpc,
                        plot=False, linear=False, quiet=quiet)
    total_counts, sigma_px, q_obs = m.sol

    sigma_kpc = np.asarray(sigma_px, dtype=np.float64) * pixel_scale_kpc
    sigma_pc = sigma_kpc * 1000.0
    q_obs = np.asarray(q_obs, dtype=np.float64)
    total_counts = np.asarray(total_counts, dtype=np.float64)
    surf = total_counts / (2.0 * np.pi * sigma_pc ** 2 * q_obs)   # flux/pc^2
    sig_as = sigma_kpc / kpc_per_arcsec
    total_lum = float(total_counts.sum())
    return surf, sig_as, q_obs, total_lum
