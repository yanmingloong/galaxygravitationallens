"""photutils EPSFBuilder oversampled ePSF construction (Anderson & King 2000)."""
from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import EPSFBuilder, extract_stars
from scipy.ndimage import zoom

from ...config import PSF_STAMP_SIZE


def build_epsf(
    data: np.ndarray,
    star_coords: List[Tuple[float, float, float]],
    stamp_size: int = PSF_STAMP_SIZE,
    oversampling: int = 4,
    maxiters: int = 10,
) -> Optional[Tuple[np.ndarray, str, int]]:
    """Build an oversampled effective PSF using photutils EPSFBuilder.

    Based on Anderson & King (2000), iteratively constructs the ePSF and
    refines the stellar centroids, then downsamples back to the detector
    pixel scale.

    Args:
        data: 2D image after background subtraction
        star_coords: list of field star coordinates [(x, y, flux), ...]
        stamp_size: output PSF size (odd number of pixels)
        oversampling: oversampling factor (default 4)
        maxiters: maximum number of iterations (default 10)

    Returns:
        (psf_normalized, source_label, n_stars_used) or None (on failure)
    """
    if len(star_coords) == 0:
        return None

    try:
        star_table = Table(
            names=["x", "y", "flux"],
            data=[
                [s[0] for s in star_coords],
                [s[1] for s in star_coords],
                [s[2] for s in star_coords],
            ],
        )

        nddata = NDData(data=data.astype(np.float64))
        extract_size = min(stamp_size + 4, min(data.shape) // 2)
        stars = extract_stars(nddata, star_table, size=extract_size)

        if len(stars) == 0:
            return None

        builder = EPSFBuilder(
            oversampling=oversampling,
            maxiters=maxiters,
            progress_bar=False,
        )
        epsf_result, fitted_stars = builder(stars)

        epsf_data = epsf_result.data
        ds_factor = 1.0 / oversampling
        psf_ds = zoom(epsf_data, ds_factor, order=1)

        if psf_ds.shape[0] >= stamp_size:
            start = (psf_ds.shape[0] - stamp_size) // 2
            psf_out = psf_ds[start:start + stamp_size, start:start + stamp_size]
        else:
            psf_out = np.zeros((stamp_size, stamp_size), dtype=np.float64)
            start = (stamp_size - psf_ds.shape[0]) // 2
            s = psf_ds.shape[0]
            psf_out[start:start + s, start:start + s] = psf_ds

        psf_out = np.maximum(psf_out, 0)
        psf_out /= psf_out.sum()

        n_used = len(fitted_stars)
        source_label = f"epsf_{oversampling}x_{n_used}stars"

        return psf_out, source_label, n_used

    except Exception:
        return None

