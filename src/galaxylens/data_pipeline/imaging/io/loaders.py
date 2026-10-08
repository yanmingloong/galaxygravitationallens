"""
HST image loaders
==============
Unified loading of SCI + WHT + WCS for DRC/FLT/FLC/DRZ products.

Features:
  1. Auto-detect the extension structure (DRC: ext 0/1, FLT: ext 1/2)
  2. Extract the SCI, WHT, WCS triple
  3. Return a standardized RawExposure dataclass

Product types:
  - DRC: drizzle-combined product, SCI in ext 0 or 1, WHT in ext 2
  - DRZ: drizzle-combined (WFPC2), similar to DRC
  - FLT: single exposure, SCI=ext 1, WHT=ext 2, CTX=ext 3
  - FLC: CTE-corrected FLT, same structure as FLT

Usage examples:
  >>> from galaxylens.data_pipeline.imaging.io.loaders import load_exposure, RawExposure
  >>> exp = load_exposure(Path("j9c706010.fits"))
  >>> exp.sci.shape, exp.wht.shape
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


@dataclass
class RawExposure:
    """Standard data structure for a single HST exposure.

    Attributes:
        path: source file path
        sci: science data (2D float64, units electrons/s)
        wht: weight map (2D float64, units electrons⁻²·s²) or None
        header: FITS header (from the SCI extension)
        wcs: WCS object or None
        product_type: product type ("DRC", "DRZ", "FLT", "FLC")
        exptime: exposure time (seconds)
        filter_name: filter name
        detector: detector name
    """
    path: Path
    sci: np.ndarray
    wht: Optional[np.ndarray]
    header: fits.Header
    wcs: Optional[WCS]
    product_type: str
    exptime: float
    filter_name: str
    detector: str


# ══════════════════════════════════════════════════════════════════════
# Public interface
# ══════════════════════════════════════════════════════════════════════

def load_exposure(filepath: Path, sci_ext: Optional[int] = None) -> RawExposure:
    """Load a single HST FITS file, auto-detecting product type and extension structure.

    Auto-detection logic:
      - if ext 0 has data → DRC/DRZ (data in PRIMARY)
      - if ext 0 has no data and ext 1 is named 'SCI' → FLT/FLC
      - otherwise try ext 1

    Args:
        filepath: FITS file path
        sci_ext: explicitly specify the SCI extension index (for FLT/FLC dual-chip data,
                 the target may be in chip2 = hdu[4] rather than the default chip1 = hdu[1])

    Returns:
        RawExposure object

    Raises:
        ValueError: if no science-data extension can be found
    """
    with fits.open(filepath) as hdul:
        n_ext = len(hdul)

        # ── Detect product type ──
        # Explicitly specify the SCI extension (used to load the chip containing the target for dual-chip data)
        if sci_ext is not None:
            if hdul[sci_ext].data is None:
                raise ValueError(f"Extension {sci_ext} has no data in {filepath}")
            data = hdul[sci_ext].data.astype(np.float64)
            header = hdul[sci_ext].header
        # DRC/DRZ: ext 0 usually has data
        elif hdul[0].data is not None:
            data = hdul[0].data.astype(np.float64)
            header = hdul[0].header
            sci_ext = 0
        elif n_ext > 1 and hdul[1].data is not None:
            data = hdul[1].data.astype(np.float64)
            header = hdul[1].header
            sci_ext = 1
        else:
            raise ValueError(f"Cannot find science data in {filepath}")

        # ── Detect product type ──
        filetype = header.get("FILETYPE", "").strip()
        if "DRC" in filetype or "_drc" in filepath.name.lower():
            product_type = "DRC"
        elif "DRZ" in filetype or "_drz" in filepath.name.lower():
            product_type = "DRZ"
        elif "FLC" in filetype or "_flc" in filepath.name.lower() or "_flc" in str(filepath).lower():
            product_type = "FLC"
        else:
            product_type = "FLT"

        # ── Find WHT extension ──
        wht: Optional[np.ndarray] = None
        if product_type in ("FLT", "FLC"):
            # FLT/FLC: WHT is in ext 2
            if n_ext > 2 and hdul[2].data is not None:
                wht = hdul[2].data.astype(np.float64)
        else:
            # DRC/DRZ: WHT may be in ext 2 or in an extension named 'WHT'
            for ext_name in ["WHT", "ERR"]:
                try:
                    wht_ext = hdul[ext_name]
                    if wht_ext.data is not None:
                        wht = wht_ext.data.astype(np.float64)
                        break
                except (KeyError, IndexError):
                    continue

        # ── INHERIT expansion (FLT/FLC global metadata in PRIMARY) ──
        # FLT/FLC global keywords (EXPTIME/FILTER*/DETECTOR/INSTRUME/DATE-OBS, etc.)
        # live in the PRIMARY header; the SCI header references them via INHERIT=True,
        # but astropy does not expand them automatically.
        # Manually fill SCI-missing/empty keys from PRIMARY into the header to avoid reading empty values.
        primary_header = hdul[0].header
        for key in primary_header:
            if key in header and header.get(key) not in (None, ""):
                continue
            try:
                header[key] = primary_header[key]
            except Exception:
                continue

        # ── WCS ──
        # SIP distortion (A_ORDER/B_ORDER) requires the CPDis lookup-table extension;
        # an HDUList must be passed as fobj for correct resolution, otherwise WCS(header) raises
        # "an astropy.io.fits.HDUList is required for Lookup table distortion".
        try:
            wcs = WCS(header, fobj=hdul)
        except Exception:
            try:
                wcs = WCS(header)
            except Exception:
                wcs = None

        # ── Metadata ──
        exptime = float(header.get("EXPTIME", 0) or 0.0)
        # Prefer FILTER; with the ACS/WFC dual filter wheel FILTER is often CLEAR1L, so fall back to FILTER2
        filter_name = header.get("FILTER", "").strip()
        if not filter_name or filter_name.upper().startswith("CLEAR"):
            for fk in ("FILTER2", "FILTER1"):
                cand = header.get(fk, "").strip()
                if cand and not cand.upper().startswith("CLEAR"):
                    filter_name = cand
                    break
        detector = header.get("DETECTOR", "").strip()

    return RawExposure(
        path=filepath,
        sci=data,
        wht=wht,
        header=header,
        wcs=wcs,
        product_type=product_type,
        exptime=exptime,
        filter_name=filter_name,
        detector=detector,
    )


def load_exposure_for_target(filepath: Path, ra: float, dec: float) -> RawExposure:
    """Load the chip containing the target celestial coordinates (auto-selects the correct SCI extension for FLT/FLC dual-chip data).

    ACS/WFC FLT/FLC has two chips, top and bottom (2048×4096 each); by default hdu[1] is the north chip.
    If the target lies in the south chip (hdu[4]), the default load would push WCS coordinates
    out of bounds and the cutout would crop the wrong position.
    This function uses the target coordinates to detect the containing chip and loads the corresponding extension.

    Args:
        filepath: FITS file path
        ra, dec: target celestial coordinates (deg)

    Returns:
        the RawExposure containing the target coordinates (returns the default chip if not found)
    """
    exp = load_exposure(filepath)
    if exp.product_type not in ("FLT", "FLC") or exp.wcs is None:
        return exp
    coord = SkyCoord(ra, dec, unit="deg")
    x, y = exp.wcs.world_to_pixel(coord)
    if 0 <= x < exp.sci.shape[1] and 0 <= y < exp.sci.shape[0]:
        return exp
    # Target not in the default chip; iterate over other SCI extensions
    with fits.open(filepath) as hdul:
        for i in range(len(hdul)):
            if i == 1 or hdul[i].data is None:
                continue
            if hdul[i].name != "SCI":
                continue
            try:
                exp2 = load_exposure(filepath, sci_ext=i)
                x2, y2 = exp2.wcs.world_to_pixel(coord)
                if 0 <= x2 < exp2.sci.shape[1] and 0 <= y2 < exp2.sci.shape[0]:
                    return exp2
            except Exception:
                continue
    return exp


# ══════════════════════════════════════════════════════════════════════
# WFPC2 DRZ chip localization
# ══════════════════════════════════════════════════════════════════════

_WFPC2_CHIP_INSTRUMENT = {
    1: "WFPC2/PC",
    2: "WFPC2/WF2",
    3: "WFPC2/WF3",
    4: "WFPC2/WF4",
}


def wfpc2_chip_position(
    filepath: Path,
    ra: float,
    dec: float,
    drz_wcs: Optional[WCS] = None,
) -> tuple[Optional[str], Optional[tuple[int, int]]]:
    """WFPC2 DRZ: use the CTX image + HDRTAB to precisely locate the chip containing the target and its chip coordinates.

    A WFPC2 DRZ is a drizzle combination of the entire focal plane (DETECTOR=MULTIPLE); the target
    may fall on any of the PC/WF2/WF3/WF4 chips. The DRZ WCS returns focal-plane coordinates
    (about 4588×4573), far beyond the single-chip 0-799 range, so they cannot be used directly
    with TinyTim (TinyTim requires in-chip coordinates).

    Method (all empirically verified):
      1. use the DRZ WCS to obtain the target's focal-plane pixels (fx, fy)
      2. read CTX[fy, fx] and take the lowest set bit → chip number
         (bit0=PC, bit1=WF2, bit2=WF3, bit3=WF4, one input exposure per 4 bits)
      3. read the OCRPIX/OCRVAL/CD(+SIP) from the HDRTAB row with EXTVER=chip to build the chip WCS
      4. world_to_pixel back-projects the chip coordinates (0-799)

    J2141 verification: CTX=17476→bit2→chip3(WF3), chip coordinates (434.7, 422.7),
    (34.7, 22.7)px from center, WFPC2 distortion <0.1px and negligible.

    Args:
        filepath: DRZ FITS file path
        ra, dec: target celestial coordinates (deg)
        drz_wcs: DRZ focal-plane WCS (optional, used for step 1)

    Returns:
        (instrument_full, detector_pos) or (None, None)
    """
    coord = SkyCoord(ra, dec, unit="deg")

    # Step 1: focal-plane pixels
    if drz_wcs is None:
        try:
            drz_wcs = WCS(fits.getheader(filepath, ext=1))
        except Exception:
            return None, None
    try:
        fx, fy = drz_wcs.world_to_pixel(coord)
    except Exception:
        return None, None
    fx, fy = int(round(float(fx))), int(round(float(fy)))

    with fits.open(filepath, memmap=True) as hdul:
        # Step 2: CTX image → chip number
        try:
            ctx = hdul["CTX"].data
        except (KeyError, IndexError):
            return None, None
        if not (0 <= fy < ctx.shape[0] and 0 <= fx < ctx.shape[1]):
            return None, None
        ctx_val = int(ctx[fy, fx])
        if ctx_val == 0:
            return None, None
        lowest_bit = (ctx_val & -ctx_val).bit_length() - 1
        chip = (lowest_bit % 4) + 1  # 1=PC, 2=WF2, 3=WF3, 4=WF4

        # Steps 3-4: back-project chip coordinates from HDRTAB
        try:
            ht = hdul["HDRTAB"].data
        except (KeyError, IndexError):
            return None, None
        rows = ht[ht["EXTVER"] == chip]
        if len(rows) == 0:
            return None, None
        row = rows[0]

        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = int(row["ONAXIS1"])
        hdr["NAXIS2"] = int(row["ONAXIS2"])
        # OCTYPE looks like "RA---TAN"/"DEC--TAN"; append the -SIP suffix to correctly enable SIP distortion
        hdr["CTYPE1"] = str(row["OCTYPE1"]).strip() + "-SIP"
        hdr["CTYPE2"] = str(row["OCTYPE2"]).strip() + "-SIP"
        hdr["CRVAL1"] = float(row["OCRVAL1"])
        hdr["CRVAL2"] = float(row["OCRVAL2"])
        hdr["CRPIX1"] = float(row["OCRPIX1"])
        hdr["CRPIX2"] = float(row["OCRPIX2"])
        hdr["CD1_1"] = float(row["CD1_1"])
        hdr["CD1_2"] = float(row["CD1_2"])
        hdr["CD2_1"] = float(row["CD2_1"])
        hdr["CD2_2"] = float(row["CD2_2"])
        # SIP distortion coefficients (A_ORDER/B_ORDER + A_i_j/B_i_j, i+j from 2 to order)
        for prefix in ("A", "B"):
            order = int(row[f"{prefix}_ORDER"])
            hdr[f"{prefix}_ORDER"] = order
            for i in range(order + 1):
                for j in range(order + 1):
                    if 2 <= i + j <= order:
                        key = f"{prefix}_{i}_{j}"
                        if key in ht.columns.names:
                            hdr[key] = float(row[key])

        try:
            chip_wcs = WCS(hdr)
            cx, cy = chip_wcs.world_to_pixel(coord)
        except Exception:
            # Fallback: no distortion (pure TAN + CD). When the target is near center, distortion <0.1px and negligible.
            hdr.remove("A_ORDER", ignore_missing=True)
            hdr.remove("B_ORDER", ignore_missing=True)
            for prefix in ("A", "B"):
                for i in range(4):
                    for j in range(4):
                        hdr.remove(f"{prefix}_{i}_{j}", ignore_missing=True)
            hdr["CTYPE1"] = str(row["OCTYPE1"]).strip()
            hdr["CTYPE2"] = str(row["OCTYPE2"]).strip()
            chip_wcs = WCS(hdr)
            cx, cy = chip_wcs.world_to_pixel(coord)

        return _WFPC2_CHIP_INSTRUMENT[chip], (int(round(float(cx))), int(round(float(cy))))
