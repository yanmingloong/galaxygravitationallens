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

import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom, binary_fill_holes
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


def _auto_crop(img, thr_frac=1e-3, pad=3):
    """Crop a mock image to the bright region (``peak * thr_frac`` mask).

    ``thr_frac=1e-3`` (not ``> 0``) is deliberate: a ``> 0`` mask includes
    the faint noise halo and defeats the auto-crop. ``binary_fill_holes``
    keeps the thresholded mask contiguous.
    """
    peak = img.max()
    if peak <= 0:
        return img
    mask = img > peak * thr_frac
    mask = binary_fill_holes(mask)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    half = max(y1 - y0, x1 - x0) // 2 + pad
    half = min(half, cy, img.shape[0] - 1 - cy, cx, img.shape[1] - 1 - cx)
    return img[cy - half:cy + half + 1, cx - half:cx + half + 1]


def single_scale_normalize(images, anchor, anchor_peak):
    """Single photometric scale across a set of convolved band images.

    ``images`` is a ``{band_name: 2D array}`` dict; ``anchor`` the band
    whose peak is normalised to ``anchor_peak``. Returns
    ``(scale, {band_name: scaled array})``.

    Applying ONE scale factor to every band preserves the SKIRT SED colour
    ratio, which multi-band joint fitting needs. Normalising each band to
    its own peak instead would destroy the colour information.
    """
    scale = anchor_peak / images[anchor].max()
    return scale, {k: v * scale for k, v in images.items()}


def make_mock_dataset(band_items, px_mock, anchor, crop_px, outdir, label,
                      thr_frac=1e-3):
    """Multi-band fitting-ready mock FITS (single-scale, colour-preserving).

    ``band_items`` is a list of ``(name, fits_path, band_cfg, psf)`` tuples;
    ``anchor`` the band name whose peak sets the single photometric scale
    (its ``band_cfg.peak``); ``px_mock`` the SKIRT image pixel scale
    (arcsec/px); ``crop_px`` the square cutout size; ``outdir`` the output
    directory; ``label`` the file tag.

    Each band is auto-cropped, resampled to ``band_cfg.pixel_scale``,
    convolved with its PSF, then all bands are scaled by the SAME factor
    (see :func:`single_scale_normalize`). The noise map is a CONSTANT
    ``band_cfg.noise_sigma`` generated AFTER crop/pad — zero-padding the
    noise map would leave a zero border that the fitting reads as
    "infinitely precise". Writes ``<label>_<name>_mock_{data,noise,psf}.fits``
    and returns the per-band peak dict.
    """
    os.makedirs(outdir, exist_ok=True)

    prep, psfs, cfgs = {}, {}, {}
    for name, fits_path, band_cfg, psf in band_items:
        img = fits.getdata(fits_path)
        if img.ndim == 3:
            img = img[0]
        img = np.clip(np.asarray(img, float), 0, None)
        img = _auto_crop(img, thr_frac)
        factor = px_mock / band_cfg.pixel_scale
        img_rs = zoom(img, factor, order=1)
        prep[name] = fftconvolve(img_rs, psf, mode="same")
        psfs[name] = psf
        cfgs[name] = band_cfg

    _, scaled = single_scale_normalize(prep, anchor, cfgs[anchor].peak)

    peaks = {}
    for name in scaled:
        data = scaled[name]
        cy, cx = data.shape[0] // 2, data.shape[1] // 2
        half = crop_px // 2
        half = min(half, cy, data.shape[0] - 1 - cy, cx, data.shape[1] - 1 - cx)
        data = data[cy - half:cy + half + 1, cx - half:cx + half + 1]
        if data.shape[0] < crop_px or data.shape[1] < crop_px:
            py = (crop_px - data.shape[0]) // 2
            px = (crop_px - data.shape[1]) // 2
            data = np.pad(data, [(py, crop_px - data.shape[0] - py),
                                 (px, crop_px - data.shape[1] - px)])
        noise = np.full_like(data, cfgs[name].noise_sigma)  # AFTER crop/pad
        fits.writeto(f"{outdir}/{label}_{name}_mock_data.fits", data,
                     overwrite=True)
        fits.writeto(f"{outdir}/{label}_{name}_mock_noise.fits", noise,
                     overwrite=True)
        fits.writeto(f"{outdir}/{label}_{name}_mock_psf.fits", psfs[name],
                     overwrite=True)
        peaks[name] = float(data.max())
    return peaks
