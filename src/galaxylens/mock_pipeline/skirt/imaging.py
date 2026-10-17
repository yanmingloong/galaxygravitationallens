"""Fake HST imaging from SKIRT raw output.

Turns the SKIRT noiseless ``total.fits`` into a fake HST image (PSF
convolution + Gaussian noise + cosmic rays + faint background sources) and
derives the extinction map from the raw dust/no-dust output. Requires
``scipy`` and ``astropy`` (core optional deps).

PSF / noise / peak / pixel scale are band-specific and come from a
:class:`~.config.BandConfig` (see the J2141 defaults in ``config.py``).
F814W uses the observed HST PSF; F160W uses a TinyTim WFC3/IR PSF
(generated 2026-10-17). The caller passes the band-appropriate ``psf``.
"""

import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom
from scipy.signal import fftconvolve

def make_fake_hst(fits_path, px_mock, band_cfg, psf, seed=42):
    """Make one fake HST image from a SKIRT noiseless total.fits.

    ``band_cfg`` is a :class:`~.config.BandConfig` (pixel_scale /
    noise_sigma / peak); ``psf`` is the PSF array for that band.
    ``px_mock`` is the SKIRT image pixel scale (arcsec/px); the image
    is resampled to ``band_cfg.pixel_scale``, convolved with ``psf``,
    peak-normalized to ``band_cfg.peak``, then Gaussian noise
    (``band_cfg.noise_sigma``), cosmic rays and 40 faint background
    sources are added. Returns the noisy image.
    """
    img = fits.getdata(fits_path)[0]
    factor = px_mock / band_cfg.pixel_scale
    img_rs = zoom(img, factor, order=1)
    img_conv = fftconvolve(img_rs, psf, mode="same")
    img_norm = img_conv / img_conv.max() * band_cfg.peak

    rng = np.random.default_rng(seed)
    img_noisy = img_norm + rng.normal(0, band_cfg.noise_sigma, img_norm.shape)

    # cosmic rays: 0.05% of pixels get a bright spike
    ncr = int(img_noisy.size * 0.0005)
    idx = rng.choice(img_noisy.size, ncr, replace=False)
    img_noisy.ravel()[idx] += rng.uniform(500, 5000, ncr)

    # faint background sources
    yy, xx = np.mgrid[0:img_noisy.shape[0], 0:img_noisy.shape[1]]
    for _ in range(40):
        bx = rng.uniform(0, img_noisy.shape[0])
        by = rng.uniform(0, img_noisy.shape[1])
        amp = rng.uniform(3, 60)
        sig = rng.uniform(1.0, 2.5)
        img_noisy += amp * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sig ** 2))
    return img_noisy


def extinction_map(dust_fits, nodust_fits, half_px):
    """Extinction A(mag) from SKIRT raw noiseless dust/no-dust output.

    A = -2.5 log10(dust/nodust) on the pixels above 1% of each peak, over a
    central ``half_px`` box. NaN where the mask fails. This must be computed
    on the raw output, not on the noisy fake-HST image.
    """
    d = fits.getdata(dust_fits)[0]
    n = fits.getdata(nodust_fits)[0]
    c = d.shape[0] // 2
    half_px = min(c, half_px)
    d = d[c - half_px:c + half_px, c - half_px:c + half_px]
    n = n[c - half_px:c + half_px, c - half_px:c + half_px]
    mask = (d > 0.01 * d.max()) & (n > 0.01 * n.max())
    A = np.full_like(d, np.nan)
    A[mask] = -2.5 * np.log10(np.clip(d[mask], 1e-9, None)
                              / np.clip(n[mask], 1e-9, None))
    return A
