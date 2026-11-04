"""
Background estimation & subtraction
==============
Estimate the sky background from the edge regions of the image, then subtract it.

Features:
  1. Sigma-clipped edge-strip statistics to estimate the background level
  2. Background subtraction

References:
  - Bertin & Arnouts 1996, A&AS 117, 393 (SExtractor, used for background estimation)

Methods:
  - estimate_background(): edge sigma-clipped median and RMS
  - subtract_background(): data - bg_median

Processing flow:
    data → extract 5px edge strips → sigma_clipped_stats(3σ) → return (median, rms)
         → data - median → output

Usage example:
  >>> from galaxylens.data_pipeline.imaging.stages.background import estimate_background, subtract_background
  >>> bg_med, bg_rms = estimate_background(sci)
  >>> sci_bgsub = subtract_background(sci, bg_med)
"""

from __future__ import annotations
from typing import Optional, Tuple
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats


# ══════════════════════════════════════════════════════════════════════
# Public interface
# ══════════════════════════════════════════════════════════════════════

def estimate_background(
    data: np.ndarray,
    sigma: float = 3.0,
    edge_width: int = 5,
) -> Tuple[float, float]:
    """Estimate the median and RMS of the sky background from edge strips.

    Take edge_width-pixel-wide strips along each of the four image edges
    and compute sigma-clipped statistics, excluding the influence of
    galaxies and bright sources.

    Args:
        data: 2D image
        sigma: sigma-clipping threshold
        edge_width: Edge strip width (pixels)

    Returns:
        (bg_median, bg_rms): background median and RMS

    Note:
        Assumes the lens is at the image center, so the edge strips
        contain mostly sky background.
        If the lens is too far off-center, this method may underestimate
        the background.
    """
    edge_strip = np.concatenate([
        data[:edge_width, :].ravel(),
        data[-edge_width:, :].ravel(),
        data[:, :edge_width].ravel(),
        data[:, -edge_width:].ravel(),
    ])

    mean, median, std = sigma_clipped_stats(
        edge_strip, sigma=sigma, maxiters=5,
    )
    return float(median), float(std)


def subtract_background(
    data: np.ndarray,
    bg_value: float,
) -> np.ndarray:
    """Subtract the background level from the image.

    Args:
        data: Input 2D image
        bg_value: Background value (usually the median from estimate_background)

    Returns:
        Background-subtracted image (data - bg_value)
    """
    return data - bg_value


def process_background(
    data: np.ndarray,
    sigma: float = 3.0,
) -> Tuple[np.ndarray, float, float]:
    """One-stop background estimation + subtraction.

    Prefer SExtractor; fall back to edge sigma-clipping on failure.

    Args:
        data: Input 2D image
        sigma: sigma-clipping threshold (used on fallback)

    Returns:
        (bg_subtracted_data, bg_median, bg_rms)
    """
    bg_med, bg_rms = estimate_background_sextractor(data)
    if bg_med is not None:
        print(f"    BG (SExtractor): median={bg_med:.4f}, rms={bg_rms:.4f}", flush=True)
        return subtract_background(data, bg_med), bg_med, bg_rms

    # Fallback: edge sigma-clipping
    bg_med, bg_rms = estimate_background(data, sigma=sigma)
    return subtract_background(data, bg_med), bg_med, bg_rms


# ══════════════════════════════════════════════════════════════════════
# Method 2: SExtractor background estimation (E3, 2026-07-03)
# ══════════════════════════════════════════════════════════════════════

def estimate_background_sextractor(
    data: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate the background mean and sigma using SExtractor.

    Based on dinos_i 3_Pre-Processor.ipynb, which uses pysex.

    Args:
        data: 2D image

    Returns:
        (bg_mean, bg_sigma) or (None, None) if SExtractor is unavailable
    """
    try:
        # Write temporary FITS
        tmpdir = Path(tempfile.mkdtemp())
        fits_path = tmpdir / "tmp_sci.fits"
        fits.writeto(str(fits_path), data.astype(np.float32), overwrite=True)

        # Config
        bg_path = tmpdir / "tmp_bg.fits"
        config = "\n".join([
            "CATALOG_TYPE    NONE",
            "CHECKIMAGE_TYPE BACKGROUND",
            "CHECKIMAGE_NAME " + str(bg_path),
            "BACK_SIZE       32",
            "BACK_FILTERSIZE 3",
            "DETECT_THRESH   3.0",
            "ANALYSIS_THRESH 3.0",
            "WEIGHT_TYPE     BACKGROUND",
        ])
        config_path = tmpdir / "default.sex"
        config_path.write_text(config)

        # Invoke sex with absolute path
        sex_bin = os.environ.get("SEXTRACTOR_BIN") or shutil.which("sex")
        if sex_bin is None:
            return None, None
        result = subprocess.run(
            [sex_bin, str(fits_path), "-c", str(config_path)],
            capture_output=True, text=True, timeout=60,
            cwd=str(tmpdir),
        )
        if result.returncode != 0:
            print(f"    [WARN] SExtractor exit={result.returncode}: {result.stderr[:200]}", flush=True)

        # Read background image
        if bg_path.exists():
            bg_data = fits.getdata(str(bg_path))
            bg_mean = float(np.mean(bg_data))
            bg_std = float(np.std(bg_data))
            return bg_mean, bg_std

        return None, None

    except Exception as e:
        print(f"    [WARN] SExtractor failed: {e}", flush=True)
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
