"""
Standardized FITS output
===============
Write the processed SCI/Noise/PSF/Mask to standard FITS files.

Features:
  1. Write FITS files following the unified naming convention
  2. SCI → float32, Noise → float32, PSF → float32, Mask → uint8
  3. Automatically create the output directory

Naming convention:
  {short_name}_{band_lower}_{type}.fits
  e.g.: j2141_f814w_sci.fits, 1630+4520_f435w_noise.fits

Usage example:
  >>> from galaxylens.data_pipeline.imaging.output.fits_writer import write_processed_band
  >>> write_processed_band(output_dir, "J2141-0001", "F814W", sci=sci, noise=noise, psf=psf)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from astropy.io import fits


# ══════════════════════════════════════════════════════════════════════
# Naming utilities
# ══════════════════════════════════════════════════════════════════════

def make_short_name(lens_id: str) -> str:
    """Convert a lens ID into a short filename prefix.

    Args:
        lens_id: e.g. "J2141-0001", "J1630+4520"

    Returns:
        Short name: e.g. "j21410001", "1630+4520"
    """
    # Strip the SDSS/SDSSJ prefix
    name = lens_id
    for prefix in ["SDSSJ", "SDSS_", "SDSS"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.lower().replace("-", "")


def make_prefix(lens_id: str, band: str) -> str:
    """Generate the output filename prefix.

    Args:
        lens_id: Lens identifier
        band: Band name (e.g. "F814W")

    Returns:
        Prefix: e.g. "1630+4520_f814w"
    """
    short = lens_id.replace("SDSSJ", "").replace("SDSS_", "")
    return f"{short.lower()}_{band.lower()}"


# ══════════════════════════════════════════════════════════════════════
# Writing interface
# ══════════════════════════════════════════════════════════════════════

def write_processed_band(
    output_dir: Path,
    lens_id: str,
    band: str,
    sci: np.ndarray,
    noise: np.ndarray,
    psf: np.ndarray,
    starmask: np.ndarray,
    cr_mask: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write the processed results for one lens + band to FITS files.

    Args:
        output_dir: Output directory (e.g. processed/J2141-0001/v2_flc_whtonly/)
        lens_id: Lens identifier
        band: Band name
        sci: Background-subtracted science image
        noise: Noise map σ-map
        psf: Normalized PSF kernel
        starmask: Star mask (Boolean)
        cr_mask: Cosmic-ray mask (Boolean, optional)

    Returns:
        {type: filepath} dictionary

    Output files:
        {prefix}_sci.fits, {prefix}_noise.fits, {prefix}_psf.fits,
        {prefix}_starmask.fits, [{prefix}_crmask.fits]
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = make_prefix(lens_id, band)

    files: dict[str, Path] = {}

    # SCI
    sp = output_dir / f"{prefix}_sci.fits"
    fits.writeto(sp, sci.astype(np.float32), overwrite=True)
    files["sci"] = sp

    # Noise
    np_ = output_dir / f"{prefix}_noise.fits"
    fits.writeto(np_, noise.astype(np.float32), overwrite=True)
    files["noise"] = np_

    # PSF
    pp = output_dir / f"{prefix}_psf.fits"
    fits.writeto(pp, psf.astype(np.float32), overwrite=True)
    files["psf"] = pp

    # Star mask
    mp = output_dir / f"{prefix}_starmask.fits"
    fits.writeto(mp, starmask.astype(np.uint8), overwrite=True)
    files["starmask"] = mp

    # CR mask (optional)
    if cr_mask is not None:
        cp = output_dir / f"{prefix}_crmask.fits"
        fits.writeto(cp, cr_mask.astype(np.uint8), overwrite=True)
        files["crmask"] = cp

    return files


def write_combined_starmask(
    output_dir: Path,
    lens_id: str,
    starmask: np.ndarray,
) -> Path:
    """Write the combined star mask (OR over all bands).

    Args:
        output_dir: Output directory
        lens_id: Lens identifier
        starmask: Combined Boolean mask

    Returns:
        Output file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    short = make_short_name(lens_id)
    path = output_dir / f"{short}_starmask.fits"
    fits.writeto(path, starmask.astype(np.uint8), overwrite=True)
    return path
