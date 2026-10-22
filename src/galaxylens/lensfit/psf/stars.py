"""Empirical PSF measurement: field-star detection and classification.

First step of building an empirical PSF: scan the field, detect point sources
with DAOStarFinder, and classify them (point-like vs extended, in/out of the
galaxy exclusion zone, near the frame edge, on bad pixels). The surviving
point-source candidates are the input to the subsequent Moffat/PSF fit.

Pitfalls codified here (learned on the J2141 K'-band field):
- The galaxy centre must be located as the brightest pixel AFTER masking bad
  pixels; a cosmic-ray spike can outshine the true bulge and fool a raw argmax.
- Do not Gaussian-smooth the image before finding the peak: smoothing drags the
  peak toward the surrounding signal centroid by ~1 px.

References
----------
- Stetson 1987, PASP 99, 191 (DAOStarFinder)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def find_field_stars(
    sci: np.ndarray,
    goodmask: np.ndarray,
    pixel_scale: float,
    psf_fwhm_px: float,
    detect_sigma: float = 8.0,
    galaxy_excl_arcsec: float = 3.0,
    edge_margin_px: int = 15,
    sharpness_point: float = 0.6,
) -> dict:
    """Detect and classify point sources in an imaging field.

    Args:
        sci: science image (2-D array).
        goodmask: boolean mask, ``True`` where the pixel is good.
        pixel_scale: pixel scale in arcsec/pix.
        psf_fwhm_px: initial PSF FWHM estimate in pixels (for DAOStarFinder).
        detect_sigma: detection threshold in units of the background std.
        galaxy_excl_arcsec: exclusion radius around the galaxy centre (arcsec).
        edge_margin_px: frame-edge exclusion margin in pixels.
        sharpness_point: sharpness lower bound for a point-source candidate
            (point sources ~1.0, extended sources < 1).

    Returns:
        A dict with keys ``sources`` (list of per-source dicts: x_px, y_px,
        dist_gal_arcsec, flux, peak, sharpness, roundness1, roundness2,
        point_candidate), ``n_point`` (count of point-source candidates), and
        ``galaxy_center`` (the (x, y) masked-brightest-pixel centre).

    Note:
        ``photutils`` and ``astropy`` are imported lazily.
    """
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    sci = np.asarray(sci, dtype=float)
    goodmask = np.asarray(goodmask, dtype=bool)
    badpix = ~goodmask
    ny, nx = sci.shape

    mean, med, std = sigma_clipped_stats(sci[goodmask], sigma=3.0)

    # Galaxy centre: brightest pixel AFTER masking bad pixels (cosmic-ray spike
    # guard -- a raw argmax can land on a spike, not the bulge).
    sci_good = np.where(goodmask, sci, -np.inf)
    gy, gx = np.unravel_index(np.argmax(sci_good), sci.shape)

    daofind = DAOStarFinder(
        fwhm=psf_fwhm_px,
        threshold=detect_sigma * std,
        sharplo=0.2, sharphi=2.0,
        roundlo=-1.0, roundhi=1.0,
    )
    sources = daofind(sci - med, mask=badpix)

    rows: List[dict] = []
    n_point = 0
    if sources is not None and len(sources) > 0:
        def _col(table, names):
            for n in names:
                if n in table.colnames:
                    return table[n]
            raise KeyError(f"none of {names} in {table.colnames}")

        x = np.asarray(_col(sources, ["xcentroid", "x_centroid"]))
        y = np.asarray(_col(sources, ["ycentroid", "y_centroid"]))
        flux = np.asarray(_col(sources, ["flux", "flux_bb"]))
        peak = np.asarray(_col(sources, ["peak"]))
        sharp = np.asarray(_col(sources, ["sharpness"]))
        r1 = np.asarray(_col(sources, ["roundness1"]))
        r2 = np.asarray(_col(sources, ["roundness2"]))

        dist_gal = np.hypot(x - gx, y - gy) * pixel_scale
        near_edge = (
            (x < edge_margin_px) | (x > nx - edge_margin_px)
            | (y < edge_margin_px) | (y > ny - edge_margin_px)
        )
        in_badmask = badpix[
            np.clip(y.astype(int), 0, ny - 1),
            np.clip(x.astype(int), 0, nx - 1),
        ]
        is_galaxy = dist_gal < galaxy_excl_arcsec
        is_point = (sharp > sharpness_point) & ~is_galaxy & ~near_edge & ~in_badmask

        for i in np.argsort(-flux):
            rows.append(
                {
                    "x_px": float(x[i]), "y_px": float(y[i]),
                    "dist_gal_arcsec": float(dist_gal[i]),
                    "flux": float(flux[i]), "peak": float(peak[i]),
                    "sharpness": float(sharp[i]),
                    "roundness1": float(r1[i]), "roundness2": float(r2[i]),
                    "point_candidate": bool(is_point[i]),
                }
            )
        n_point = int(is_point.sum())

    return {
        "sources": rows,
        "n_point": n_point,
        "galaxy_center": (float(gx), float(gy)),
        "background": {"median": float(med), "std": float(std)},
    }
