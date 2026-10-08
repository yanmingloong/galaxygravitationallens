"""
HST source file discovery
==============
Locate the correct FITS files in the MAST_download directory based on lens ID and band.

Features:
  1. Auto-match files by product_type (DRC/DRZ/FLT/FLC)
  2. Priority: FLC > FLT > DRC > DRZ
  3. Exclude skycell products (prefer target-pointed products)
  4. FLT/FLC support returning multiple exposures

Processing flow:
    Input: lens_id + band + product_type
           │
           ├─ Scan raw/{lens}/MAST_download/
           │
           ├─ Filter by product_type:
           │   ├─ DRC → *_drc.fits
           │   ├─ DRZ → *_drz.fits
           │   ├─ FLT → filename contains band and not drc/drz
           │   └─ FLC → filename contains band and not drc/drz
           │
           └─ Return List[Path] (FLT/FLC may be multiple)

Usage example:
  >>> from galaxylens.data_pipeline.imaging.io.file_discovery import find_source_files
  >>> files = find_source_files("J1630+4520", "F814W", "FLT")
  >>> [f.name for f in files]
"""

from __future__ import annotations
import glob
from pathlib import Path
from typing import List, Optional, Literal

from astropy.io import fits

from galaxylens.data_pipeline.imaging.paths import get_raw_root

ProductType = Literal["DRC", "DRZ", "FLT", "FLC"]


# ══════════════════════════════════════════════════════════════════════
# Public interface
# ══════════════════════════════════════════════════════════════════════

def find_source_files(
    lens_id: str,
    band: str,
    product_type: ProductType = "FLC",
    min_size_mb: float = 50.0,
) -> List[Path]:
    """Find source files for the given lens + band in MAST_download.

    Args:
        lens_id: Lens identifier, e.g. "J2141-0001"
        band: Band name, e.g. "F814W"
        product_type: Data product type (DRC/DRZ/FLT/FLC)
        min_size_mb: Minimum file size threshold (MB), to exclude partial downloads

    Returns:
        List of matching FITS file paths (FLT/FLC may have multiple exposures)

    Raises:
        FileNotFoundError: If no matching files are found

    Note:
        - DRC/DRZ usually return 1 file
        - FLT/FLC may return multiple files (multiple exposures)
        - skycell products are automatically excluded (prefer target-pointed ones)
    """
    mast_dir = get_raw_root() / lens_id / "MAST_download"
    if not mast_dir.is_dir():
        raise FileNotFoundError(
            f"MAST_download not found for {lens_id} at {mast_dir}"
        )

    # Collect all candidate FITS files
    all_fits: List[Path] = sorted(
        p for p in mast_dir.rglob("*.fits")
        if p.stat().st_size > min_size_mb * 1024 * 1024
    )

    if not all_fits:
        raise FileNotFoundError(
            f"No FITS files > {min_size_mb} MB in {mast_dir}"
        )

    # Filter by product type
    band_lower = band.lower()
    matched: List[Path] = []

    if product_type in ("DRC", "DRZ"):
        suffix = f"_{product_type.lower()}.fits"
        matched = [f for f in all_fits
                   if f.name.lower().endswith(suffix)
                   and band_lower in f.name.lower()]
    else:
        # FLT/FLC: match by band name first, exclude DRC/DRZ/ASN
        matched = [
            f for f in all_fits
            if band_lower in f.name.lower()
            and "_drc" not in f.name.lower()
            and "_drz" not in f.name.lower()
            and "_asn" not in f.name.lower()
        ]
        # If none found, try matching via the FILTER keyword in the FITS header (FLC files may not contain the band name)
        if not matched:
            for f in all_fits:
                if "_drc" in f.name.lower() or "_drz" in f.name.lower():
                    continue
                if "_asn" in f.name.lower():
                    continue
                try:
                    with fits.open(f) as hdul:
                        for ext in [0, 1]:
                            if ext < len(hdul):
                                hdr = hdul[ext].header
                                # FLC/FLT files use FILTER1/FILTER2, not FILTER
                                f1 = hdr.get("FILTER1", "").strip()
                                f2 = hdr.get("FILTER2", "").strip()
                                filt = hdr.get("FILTER", "").strip()
                                if band.upper() in (f1.upper(), f2.upper(), filt.upper()):
                                    matched.append(f)
                                    break
                except Exception:
                    continue

    # Exclude skycell products (prefer target-pointed ones)
    nonsky = [f for f in matched if "skycell" not in f.name.lower()]
    if nonsky:
        matched = nonsky

    # Secondary validation: ensure files are readable
    valid: List[Path] = []
    for f in matched:
        try:
            with fits.open(f) as hdul:
                # Only files with at least 2 extensions are FLT/DRC
                if len(hdul) >= 2:
                    valid.append(f)
        except Exception:
            continue

    if not valid:
        raise FileNotFoundError(
            f"No valid {product_type} files found for {lens_id} {band} "
            f"in {mast_dir}"
        )

    return sorted(valid)


def find_available_bands(lens_id: str) -> List[str]:
    """Scan MAST_download and return the list of bands actually present.

    Bands are determined by reading the FILTER keyword of each FITS file.

    Args:
        lens_id: Lens identifier

    Returns:
        List of band names, e.g. ["F435W", "F814W"]
    """
    mast_dir = get_raw_root() / lens_id / "MAST_download"
    if not mast_dir.is_dir():
        return []

    bands: set[str] = set()
    for fpath in mast_dir.rglob("*.fits"):
        if fpath.stat().st_size < 50_000_000:
            continue
        try:
            with fits.open(fpath) as hdul:
                for ext in [0, 1]:
                    if ext < len(hdul):
                        filt = hdul[ext].header.get("FILTER", "").strip()
                        if filt and filt not in ("", "N/A", "CLEAR"):
                            bands.add(filt)
                            break
        except Exception:
            continue

    return sorted(bands)


def has_files_for_band(lens_id: str, band: str) -> bool:
    """Check whether the given lens + band has available source files.

    Args:
        lens_id: Lens identifier
        band: Band name

    Returns:
        Whether files are available
    """
    try:
        files = find_source_files(lens_id, band)
        return len(files) > 0
    except FileNotFoundError:
        return False
