"""Noise rescale for galaxy-subtracted lens fitting.

Barnabe's method: after subtracting the lens-galaxy light, the residual image
still carries systematic residuals (dust lane, imperfect bulge/disk subtraction)
that the raw measurement noise does not cover. The standard fix is to scale the
noise map by ``f = sqrt(reduced_chi2)`` before the arc fit, so that the fit's
chi2/px lands near 1.

Without this rescale the arc fit sees chi2/px ~ 30, the posterior collapses
into an over-narrow peak, and Nautilus can grind for days without converging
(observed: 2 days, no samples.csv written).

References
----------
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (SWELLS IV,
  galaxy-subtracted lens modelling with reduced-chi2 rescaling)
"""

from __future__ import annotations

import math

import numpy as np


def noise_rescale_factor(chi2: float, n_unmasked: int) -> float:
    """Compute the noise rescale factor ``f = sqrt(chi2 / n_unmasked)``.

    Args:
        chi2: the chi-squared of the step whose residuals are to be absorbed
            (e.g. the Step-1 galaxy-subtraction chi2).
        n_unmasked: number of unmasked image pixels used in that chi2.

    Returns:
        ``f = sqrt(reduced chi2)``. Multiply the noise map by ``f`` so the
        subsequent fit's chi2/px is ~1.

    Raises:
        ValueError: if ``n_unmasked <= 0``.
    """
    if n_unmasked <= 0:
        raise ValueError(f"n_unmasked must be positive, got {n_unmasked}")
    return math.sqrt(chi2 / n_unmasked)


def estimate_sky_noise(
    sci,
    badmask,
    mask_radius_arcsec: float,
    pixel_scale: float,
) -> float:
    """Estimate a single global sky-RMS noise from the sky annulus (B9 K'-band).

    Why this exists: the KAI reduction's ``_sig.fits`` is the drizzle OUTPUT
    WEIGHT map (``out_wht``, per-pixel exposure-frame coverage), NOT a pixel
    noise map. Using it as noise overestimates the noise by 3-7x and carries a
    spatial gradient opposite to the true noise (see kai-nirc2-reduction
    pitfalls 18/19). This function bypasses the weight map entirely and measures
    the noise directly from the sky region, returning one uniform RMS (the
    Barnabe uniform-noise assumption).

    The sky region is defined as: pixels that are unmasked (``badmask == 1``),
    lie OUTSIDE the fitting-mask radius (``radius > mask_radius_arcsec``, i.e.
    beyond the lens/arc light), and have positive flux (``sci > 0``). The
    standard deviation of those pixels is the global noise.

    Args:
        sci: science image (2D array), counts.
        badmask: bad-pixel mask (2D array, integer); ``1`` = good/unmasked,
            ``0`` = bad/masked.
        mask_radius_arcsec: fitting-mask radius in arcsec. The sky annulus is
            the good pixels at radius larger than this.
        pixel_scale: pixel scale in arcsec per pixel (converts the mask radius
            to pixels).

    Returns:
        Global sky RMS (float), uniform across the whole image.

    Note:
        Only ``numpy`` is required (no ``autolens``/``astropy``), so this stays
        importable without the heavy imaging dependencies.
    """
    sci = np.asarray(sci, dtype=float)
    badmask = np.asarray(badmask)
    if sci.shape != badmask.shape:
        raise ValueError(f"shape mismatch: sci {sci.shape} vs badmask {badmask.shape}")
    half = sci.shape[0] // 2
    yy, xx = np.mgrid[0 : sci.shape[0], 0 : sci.shape[1]]
    rpx = np.sqrt((xx - half) ** 2 + (yy - half) ** 2)
    sky = (badmask == 1) & (rpx > mask_radius_arcsec / pixel_scale) & (sci > 0)
    if not np.any(sky):
        raise ValueError(
            "no sky pixels selected; check badmask / mask_radius / pixel_scale"
        )
    return float(sci[sky].std())
