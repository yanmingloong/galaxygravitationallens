"""Analytic PSF: Moffat / circular Gaussian / elliptical Gaussian."""
from __future__ import annotations
from typing import Tuple

import numpy as np

from ...config import PSF_STAMP_SIZE


def build_moffat_psf(
    fwhm_arcsec: float = 0.09,
    beta: float = 3.0,
    pixel_scale: float = 0.05,
    size: int = PSF_STAMP_SIZE,
) -> Tuple[np.ndarray, str]:
    """Build a normalized Moffat-profile PSF.

    I(r) propto [1 + (r/alpha)^2]^(-beta)
    alpha = FWHM / (2 * sqrt(2^(1/beta) - 1))

    Args:
        fwhm_arcsec: PSF FWHM (arcseconds)
        beta: Moffat beta parameter (2.5-4.5 are typical HST values)
        pixel_scale: pixel scale (arcsec/pixel)
        size: PSF stamp size (odd number of pixels)

    Returns:
        (psf_array, source_label)
    """
    fwhm_px = fwhm_arcsec / pixel_scale
    alpha = fwhm_px / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))

    cy = cx = size // 2
    yy, xx = np.indices((size, size))
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / alpha ** 2

    psf = (1.0 + r2) ** (-beta)
    psf /= psf.sum()

    label = f"moffat_fwhm{fwhm_arcsec:.2f}_beta{beta:.1f}"
    return psf, label

def build_gaussian_psf(
    fwhm_arcsec: float = 0.09,
    pixel_scale: float = 0.05,
    size: int = PSF_STAMP_SIZE,
) -> Tuple[np.ndarray, str]:
    """Build a normalized circular Gaussian PSF.

    Args:
        fwhm_arcsec: PSF FWHM (arcseconds)
        pixel_scale: pixel scale (arcsec/pixel)
        size: PSF stamp size

    Returns:
        (psf_array, source_label)
    """
    fwhm_px = fwhm_arcsec / pixel_scale
    sigma = fwhm_px / 2.355

    cy = cx = size // 2
    yy, xx = np.indices((size, size))
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2

    psf = np.exp(-r2 / (2.0 * sigma ** 2))
    psf /= psf.sum()

    label = f"gaussian_fwhm{fwhm_arcsec:.2f}"
    return psf, label

def build_elliptical_gaussian_psf(
    fwhm_arcsec: float = 0.09,
    pixel_scale: float = 0.05,
    size: int = PSF_STAMP_SIZE,
    ellipticity: float = 0.05,
) -> Tuple[np.ndarray, str]:
    """Build a normalized elliptical Gaussian PSF (with ~5% ellipticity).

    Args:
        fwhm_arcsec: PSF FWHM (arcseconds)
        pixel_scale: pixel scale (arcsec/pixel)
        size: PSF stamp size
        ellipticity: ellipticity e = (a-b)/(a+b), default 0.05

    Returns:
        (psf_array, source_label)
    """
    fwhm_px = fwhm_arcsec / pixel_scale
    sigma = fwhm_px / 2.355

    sigma_x = sigma * (1.0 + ellipticity)
    sigma_y = sigma * (1.0 - ellipticity)

    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]

    psf = np.exp(
        -(xx ** 2) / (2.0 * sigma_x ** 2)
        - (yy ** 2) / (2.0 * sigma_y ** 2)
    )
    psf /= psf.sum()

    label = f"gaussian_ell_e{ellipticity:.2f}_fwhm{fwhm_arcsec:.2f}"
    return psf, label

