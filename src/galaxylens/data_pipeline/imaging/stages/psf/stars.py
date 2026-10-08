"""Field star detection & filtering (for PSF construction)."""
from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree

from ...config import MIN_STAR_SNR, STAR_SHARPLO


def detect_stars(
    data: np.ndarray,
    psf_fwhm_pix: float = 2.5,
    min_snr: float = MIN_STAR_SNR,
) -> List[Tuple[float, float, float]]:
    """Detect point sources (field stars) in the image with DAOStarFinder.

    Args:
        data: Background-subtracted 2D image
        psf_fwhm_pix: Estimated PSF FWHM (pixels), which affects detection sensitivity
        min_snr: Minimum SNR threshold

    Returns:
        [(x_centroid, y_centroid, flux), ...] sorted by descending flux

    Note:
        - Use a sigma-clipped std to avoid bright galaxies raising the detection threshold
        - photutils 3.0+ uses x_centroid/y_centroid; older versions use xcentroid/ycentroid
    """
    _, _, std_clipped = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    threshold = max(
        min_snr * std_clipped,
        min_snr * np.median(np.abs(data - np.median(data))) * 1.4826,
    )

    finder = DAOStarFinder(
        fwhm=psf_fwhm_pix,
        threshold=threshold,
        sharplo=STAR_SHARPLO, sharphi=3.0,
        roundlo=-0.8, roundhi=0.8,
    )

    sources = finder(data)
    if sources is None:
        return []

    sources.sort("flux", reverse=True)

    xcol = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"

    return [
        (float(s[xcol]), float(s[ycol]), float(s["flux"]))
        for s in sources
    ]

def filter_field_stars(
    stars: List[Tuple[float, float, float]],
    center: Tuple[float, float],
    exclusion_radius_pix: float = 60.0,
    max_stars: int = 20,
    data_shape: Optional[Tuple[int, int]] = None,
    min_dist: float = 5.0,
    edge_margin: float = 20.0,
) -> List[Tuple[float, float, float]]:
    """Exclude the lens center region and filter isolated field stars, for PSF construction.

    Fix (2026-08-17): previously only center exclusion + count truncation was done,
    with no isolation filtering; galaxy clumps / arc brightness knots / low-SNR diffuse
    blobs were fed to EPSFBuilder as field stars, and the ePSF was diluted and inflated
    (EE50 5-12px, normal ~1.1px).

    Filtering steps:
      1. Exclude the lens center region (galaxy itself + arc brightness knots)
      2. Exclude edge-touching sources (stamp truncation would contaminate the ePSF)
      3. Isolation: distance to the nearest-neighbor source >= min_dist px (KD-tree, O(n log n))
      4. Truncate to max_stars sorted by descending brightness

    Args:
        stars: Output of detect_stars() [(x, y, flux), ...]
        center: Lens center (cx, cy)
        exclusion_radius_pix: Center exclusion radius (pixels)
        max_stars: Maximum number of field stars to keep
        data_shape: Image (ny, nx), used for edge exclusion; skipped if None
        min_dist: Isolation threshold (pixels)
        edge_margin: Minimum distance from the edge (pixels)

    Returns:
        Field star list [(x, y, flux), ...]
    """
    cx, cy = center
    field = [
        (sx, sy, fl) for sx, sy, fl in stars
        if np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2) >= exclusion_radius_pix
    ]
    # Exclude edge-touching sources (stamp truncation would contaminate the ePSF)
    if data_shape is not None:
        ny, nx = data_shape
        field = [(sx, sy, fl) for sx, sy, fl in field
                 if edge_margin <= sx <= nx - edge_margin
                 and edge_margin <= sy <= ny - edge_margin]
    if len(field) < 1:
        return []
    xs = np.array([s[0] for s in field], dtype=float)
    ys = np.array([s[1] for s in field], dtype=float)
    # Isolation: use a KD-tree to get each source's distance to its nearest other source (O(n log n), fast even for thousands of stars across the full frame)
    if len(field) >= 2:
        coords = np.column_stack([xs, ys])
        dists, _ = cKDTree(coords).query(coords, k=2)
        nearest = dists[:, 1]  # 2nd-nearest = nearest other source (1st is itself, distance 0)
    else:
        nearest = np.array([np.inf])
    isolated = [(sx, sy, fl) for j, (sx, sy, fl) in enumerate(field)
                if nearest[j] >= min_dist]
    if len(isolated) < 1:
        return []
    isolated.sort(key=lambda s: s[2], reverse=True)  # Sort by descending flux
    return isolated[:max_stars]

