"""
WCS astrometric cutout
===========
Cut out the image to a specified field of view based on the lens's celestial coordinates.

Features:
  1. Precise WCS-coordinate cutout (preferred)
  2. Pixel-coordinate fallback (when WCS is missing or fails)
  3. Support coordinate offsets (disk-galaxy centroid fine-tuning)

Processing flow:
    Input: sci_data + wcs + ra/dec + cutout_size
           │
           ├─ WCS available?
           │   ├─ YES → Cutout2D(wcs=wcs) celestial-coordinate cutout
           │   └─ NO  → pixel-coordinate fallback
           │
           └─ Return: cutout_sci, cutout_wht (optional)

Usage example:
  >>> from galaxylens.data_pipeline.imaging.stages.cutout import wcs_cutout
  >>> sci_cut = wcs_cutout(sci, wcs, ra=247.6, dec=45.3, size_arcsec=15, pixel_scale=0.05)
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u


# ══════════════════════════════════════════════════════════════════════
# Public interface
# ══════════════════════════════════════════════════════════════════════

def wcs_cutout(
    data: np.ndarray,
    wcs: Optional[WCS],
    ra: float,
    dec: float,
    size_arcsec: float,
    pixel_scale: float,
    offset_arcsec: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Cut out the image to a specified celestial position and field of view based on WCS coordinates.

    Args:
        data: Input 2D image
        wcs: WCS object (None triggers pixel fallback)
        ra: Target right ascension (deg)
        dec: Target declination (deg)
        size_arcsec: Cutout field-of-view size (arcsec)
        pixel_scale: Pixel scale (arcsec/pixel)
        offset_arcsec: Coordinate offset (dRA, dDec) in arcsec, for centroid fine-tuning

    Returns:
        The cutout 2D array

    Note:
        Falls back to image-center cutout automatically when WCS cutout fails.
    """
    # Apply coordinate offset
    if offset_arcsec != (0.0, 0.0):
        import numpy as np
        ra_eff = ra + offset_arcsec[0] / 3600.0 / np.cos(np.deg2rad(dec))
        dec_eff = dec + offset_arcsec[1] / 3600.0
    else:
        ra_eff, dec_eff = ra, dec

    skycoord = SkyCoord(ra=ra_eff, dec=dec_eff, unit="deg", frame="icrs")
    cutout_size_pix = int(size_arcsec / pixel_scale)

    # Prefer WCS cutout
    if wcs is not None:
        try:
            cutout = Cutout2D(
                data, skycoord, cutout_size_pix,
                wcs=wcs, mode="trim",
            )
            return cutout.data
        except Exception:
            pass  # fallback below

    # Fallback: pixel-coordinate cutout (image center)
    cx, cy = data.shape[1] // 2, data.shape[0] // 2
    try:
        if wcs is not None:
            xpix, ypix = wcs.world_to_pixel(skycoord)
        else:
            xpix, ypix = cx, cy
        cutout = Cutout2D(
            data, (float(xpix), float(ypix)),
            cutout_size_pix, mode="trim",
        )
        return cutout.data
    except Exception:
        # Final fallback: direct center cutout
        half = cutout_size_pix // 2
        y1, y2 = max(0, cy - half), min(data.shape[0], cy + half)
        x1, x2 = max(0, cx - half), min(data.shape[1], cx + half)
        return data[y1:y2, x1:x2].copy()


def wcs_cutout_with_wht(
    sci: np.ndarray,
    wht: Optional[np.ndarray],
    wcs: Optional[WCS],
    ra: float,
    dec: float,
    size_arcsec: float,
    pixel_scale: float,
    offset_arcsec: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Perform WCS cutout on SCI and WHT simultaneously.

    Args:
        sci: Science image
        wht: Weight map (may be None)
        wcs: WCS object
        ra, dec: Celestial coordinates
        size_arcsec: Field-of-view size
        pixel_scale: Pixel scale
        offset_arcsec: Coordinate offset

    Returns:
        (sci_cutout, wht_cutout) — wht_cutout is None if the input is None
    """
    sci_cut = wcs_cutout(sci, wcs, ra, dec, size_arcsec, pixel_scale, offset_arcsec)

    if wht is None:
        return sci_cut, None

    wht_cut = wcs_cutout(wht, wcs, ra, dec, size_arcsec, pixel_scale, offset_arcsec)
    return sci_cut, wht_cut
