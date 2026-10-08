"""TinyTim PSF: Massey grid (ACS) + direct compilation (WFPC2 etc.).
TinyTim PSF: Massey grid (ACS) + direct compilation (WFPC2 etc.).

Reference: dinos_i pipeline: https://github.com/cytan1/dinos-i
"""
from __future__ import annotations
from typing import Optional, Tuple
from pathlib import Path
import os
import csv
import gzip
import shutil
import subprocess
import tempfile

import numpy as np
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import zoom

from ...config import PSF_STAMP_SIZE
from galaxylens.config.paths import (
    get_tinytim_grid_base,
    get_tinytim_bin,
    get_tinytim_breathing_csv,
)


# ══════════════════════════════════════════════════════════════════════
# TinyTim configuration
# ══════════════════════════════════════════════════════════════════════

# Massey grid configuration
MASSEY_GRID_TILE_SIZE = 255
MASSEY_GRID_SOURCE_SCALE = 0.03   # arcsec/pixel (TinyTim native)
MASSEY_GRID_N_TILES = 30           # 30x30 tiles
# Available bands
MASSEY_AVAILABLE_BANDS = {"F435W", "F606W", "F775W", "F814W", "F850W"}

# TinyTim instrument configuration
# Note: detector_range is the pixel range of the【entire focal plane】, not a single chip.
# ACS/WFC's two chips (each 2048×4096) are stitched top-to-bottom into a 4096×4096 focal plane,
# world_to_pixel returns focal-plane coordinates (y can reach 4095). See PSF_NEEDS.md §6.
# J2141 verification: pixel=(2083,3135) -> tile=(22,15) requires ymax=4095 (ymax=2047 would give tile_row=28, wrong)
TINYTIM_CAMERAS = {
    "ACS/WFC":  {"camera_id": 15, "chip_key": "ACS_WFC",  "detector_range": (0, 4095, 0, 4095)},
    "WFPC2/PC": {"camera_id": 6,  "chip_key": "WFPC2_PC", "detector_range": (0, 799, 0, 799)},
    # WFPC2's three wide-field chips: after selecting 5 (WFPC2 WFC) in the tiny1 interactive menu,
    # you must further enter the chip number (2/3/4).
    # camera_id stores the tiny1 interactive menu selection number (5); the chip number is stored separately as wfpc2_chip.
    # tiny1.c case 5: sprintf("WFPC_II_WFC%d", chip); chip = chip-2+WFPC2_WFC_2 → internal numbers 12/13/14.
    "WFPC2/WF2": {"camera_id": 5, "chip_key": "WFPC2_WFC", "detector_range": (0, 799, 0, 799), "wfpc2_chip": 2},
    "WFPC2/WF3": {"camera_id": 5, "chip_key": "WFPC2_WFC", "detector_range": (0, 799, 0, 799), "wfpc2_chip": 3},
    "WFPC2/WF4": {"camera_id": 5, "chip_key": "WFPC2_WFC", "detector_range": (0, 799, 0, 799), "wfpc2_chip": 4},
}

# Z4 focal-plane offset (microns), reference dinos_i template
TINYTIM_FOCUS_CENTER = {
    "ACS/WFC":  0.028,
    "WFPC2/PC": -0.117,
    # WFPC2 WFC's three chips share -0.117 (the WF3 value in the dinos_i psf_templates template)
    "WFPC2/WF2": -0.117,
    "WFPC2/WF3": -0.117,
    "WFPC2/WF4": -0.117,
}

# Z4 scaling: secondary mirror despace -> Z4 ratio
TINYTIM_Z4_SCALING = 0.011  # used in dinos_i psf_templates

def _get_hst_focus_mjd(mjd: float) -> float:
    """Query the focus offset (microns) for a given MJD from the HST breathing model.

    Reference: dinos_i Hubble_dspace.csv.

    Args:
        mjd: Modified Julian Date

    Returns:
        Focus offset (microns), 0 if the model is unavailable
    """
    csv_path = get_tinytim_breathing_csv()
    if csv_path is None or not csv_path.exists():
        return 0.0

    try:
        # binary search for the nearest MJD
        mjd_str = f"{mjd:.5f}"
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            best_mjd = None
            best_focus = 0.0
            for row in reader:
                try:
                    row_mjd = float(row[0])
                    row_focus = float(row[1])
                    if best_mjd is None or abs(row_mjd - mjd) < abs(best_mjd - mjd):
                        best_mjd = row_mjd
                        best_focus = row_focus
                except (ValueError, IndexError):
                    continue
        return best_focus
    except Exception:
        return 0.0


def _compute_focus_offset(
    date_obs: str,
    instrument: str,
) -> float:
    """Compute the Z4 focus correction.

    1. Query the HST breathing model to get dm (secondary mirror despace, microns)
    2. Z4_offset = dm * Z4_SCALING

    Args:
        date_obs: Observation date "YYYY-MM-DD"
        instrument: "ACS/WFC" or "WFPC2/PC"

    Returns:
        Z4 correction value (microns)
    """
    try:
        t = Time(date_obs, format='iso')
        mjd = t.mjd
        dm = _get_hst_focus_mjd(mjd)
        z4_offset = dm * TINYTIM_Z4_SCALING
        return z4_offset
    except Exception:
        return 0.0


def _detector_pos_to_tile(
    x_pix: int, y_pix: int,
    detector_range: Tuple[int, int, int, int],
    n_tiles: int = MASSEY_GRID_N_TILES,
) -> Tuple[int, int]:
    """Detector pixel coordinates -> Massey grid tile coordinates.

    Reference dinos_i: tile = 1 + (coord/det_max) * (n_tiles-2), clamp 1..(n_tiles-2)

    Args:
        x_pix, y_pix: Detector pixel coordinates
        detector_range: (xmin, xmax, ymin, ymax)
        n_tiles: Grid dimension

    Returns:
        (tile_row, tile_col), 0-indexed
    """
    xmin, xmax, ymin, ymax = detector_range
    # normalize to 0..(n_tiles-2), leaving 1 edge tile
    tile_col = int(1 + (x_pix / xmax) * (n_tiles - 2))
    tile_row = int(1 + (y_pix / ymax) * (n_tiles - 2))
    # clamp
    tile_col = max(1, min(n_tiles - 2, tile_col))
    tile_row = max(1, min(n_tiles - 2, tile_row))
    return tile_row, tile_col


def _find_grid_file(base: Path, band: str, focus_idx: int) -> Optional[Path]:
    """Recursively search for the focus slice file in the Massey grid directory.

    Supports multiple directory structures:
      - F814W/TinyTim_models_F814W/TinyTim_f0.fits
      - F435W/435/TinyTim_f0.fits.gz

    Args:
        base: Grid root directory
        band: Band name
        focus_idx: Focus index

    Returns:
        File path or None
    """
    band_dir = base / band
    if not band_dir.is_dir():
        return None
    for root, dirs, files in os.walk(str(band_dir)):
        for fn in files:
            if fn == f"TinyTim_f{focus_idx}.fits" or fn == f"TinyTim_f{focus_idx}.fits.gz":
                return Path(root) / fn
    return None


def _load_grid_tile(
    grid_path: Path,
    tile_row: int,
    tile_col: int,
    tile_size: int = MASSEY_GRID_TILE_SIZE,
) -> Optional[np.ndarray]:
    """Load the grid and extract the specified tile."""
    try:
        if grid_path.suffix == '.gz':
            with gzip.open(grid_path, 'rb') as gz:
                with fits.open(gz) as hdul:
                    grid_data = hdul[0].data.copy()
        else:
            grid_data = fits.getdata(str(grid_path))
    except Exception:
        return None

    r0 = tile_row * tile_size
    c0 = tile_col * tile_size
    r1 = min(r0 + tile_size, grid_data.shape[0])
    c1 = min(c0 + tile_size, grid_data.shape[1])
    return grid_data[r0:r1, c0:c1]


def _find_best_focus_slice(
    grid_base: Path,
    band: str,
    tile_row: int,
    tile_col: int,
    tile_size: int = MASSEY_GRID_TILE_SIZE,
    focus_range: range = range(-10, 6),
) -> Tuple[Optional[np.ndarray], int]:
    """Find the tile with the largest peak across multiple focus slices."""
    best_peak = -1.0
    best_tile = None
    best_f = 0

    for f_idx in focus_range:
        fpath = _find_grid_file(grid_base, band, f_idx)
        if fpath is None:
            continue
        tile = _load_grid_tile(fpath, tile_row, tile_col, tile_size)
        if tile is not None and tile.max() > best_peak:
            best_peak = tile.max()
            best_tile = tile.copy()
            best_f = f_idx

    return best_tile, best_f

def build_tinytim_massey_psf(
    band: str,
    date_obs: str,
    detector_pos: Optional[Tuple[int, int]] = None,
    pixel_scale: float = 0.05,
    stamp_size: int = PSF_STAMP_SIZE,
    focus_idx: Optional[int] = None,
    grid_base: Optional[Path] = None,
) -> Optional[Tuple[np.ndarray, str]]:
    """Extract an ACS PSF from the TinyTim Massey grid.

    Uses the TinyTim v6.3 grid model precomputed by Richard Massey (Durham).
    Supports detector position mapping, HST breathing focus correction, and multi-focus-slice search.

    Args:
        band: Band (e.g. "F814W", "F435W")
        date_obs: Observation date "YYYY-MM-DD"
        detector_pos: Detector pixel coordinates (x, y); use center if None
        pixel_scale: Target pixel scale (arcsec/pixel)
        stamp_size: Output PSF size
        focus_idx: Focus slice index (None=auto search f-10..f+5)
        grid_base: Grid root directory; use default if None

    Returns:
        (psf_normalized, source_label) or None
    """
    if band not in MASSEY_AVAILABLE_BANDS:
        return None

    base = grid_base or get_tinytim_grid_base()
    if base is None:
        return None

    # detector position -> tile
    if detector_pos is not None:
        det_range = TINYTIM_CAMERAS["ACS/WFC"]["detector_range"]
        tile_row, tile_col = _detector_pos_to_tile(
            detector_pos[0], detector_pos[1], det_range
        )
    else:
        # default center tile
        tile_row = tile_col = MASSEY_GRID_N_TILES // 2

    # focus search or specified
    if focus_idx is not None:
        fpath = _find_grid_file(base, band, focus_idx)
        if fpath is None:
            return None
        tile = _load_grid_tile(fpath, tile_row, tile_col, MASSEY_GRID_TILE_SIZE)
        if tile is None:
            return None
        best_tile = tile
        best_f = focus_idx
    else:
        # auto-search for best focus
        result = _find_best_focus_slice(base, band, tile_row, tile_col)
        if result[0] is None:
            return None
        best_tile, best_f = result
        # if a date is provided, compute the focus correction
        if date_obs:
            z4 = _compute_focus_offset(date_obs, "ACS/WFC")
            if abs(z4) > 0.001:
                # Z4 corresponds to roughly 1 focus-slice step per micron
                f_correction = int(round(z4))
                corrected_f = best_f + f_correction
                corrected_f = max(-10, min(5, corrected_f))
                if corrected_f != best_f:
                    result2 = _find_best_focus_slice(base, band, tile_row, tile_col,
                                                      focus_range=range(corrected_f, corrected_f+1))
                    if result2[0] is not None:
                        best_tile, best_f = result2

    # extract PSF from tile (rebin 0.03" -> target pixel scale, crop center)
    source_scale = MASSEY_GRID_SOURCE_SCALE
    target_scale = pixel_scale

    # find tile peak position
    y_peak, x_peak = np.unravel_index(best_tile.argmax(), best_tile.shape)
    margin = stamp_size * int(target_scale / source_scale) + 10
    y0, y1 = max(0, y_peak - margin), min(best_tile.shape[0], y_peak + margin + 1)
    x0, x1 = max(0, x_peak - margin), min(best_tile.shape[1], x_peak + margin + 1)
    region = best_tile[y0:y1, x0:x1]

    # Rebin
    rebinned = zoom(region, source_scale / target_scale, order=1)
    cy, cx = rebinned.shape[0] // 2, rebinned.shape[1] // 2
    half = stamp_size // 2
    psf = rebinned[cy - half:cy + half + 1, cx - half:cx + half + 1]

    psf = np.maximum(psf, 0)
    psf /= psf.sum()

    focus_label = f"f{best_f:+d}"
    pos_label = f"t{tile_row},{tile_col}" if detector_pos else "center"
    label = f"tinytim_massey_{band}_{focus_label}_{pos_label}"

    return psf.astype(np.float32), label

def build_tinytim_direct_psf(
    band: str,
    instrument: str,
    date_obs: str,
    detector_pos: Tuple[int, int],
    spectrum_type: str = "g2v",
    psf_diameter: float = 3.0,
    stamp_size: int = PSF_STAMP_SIZE,
    tinytim_bin: Optional[Path] = None,
    pixel_scale: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, str]]:
    """Generate a PSF by direct compilation with local TinyTim v7.5.

    tiny1 (parameter file) -> tiny2 (ray tracing) -> extract 61x61 center -> normalize

    Reference: dinos_i 4_Psf_Maker.ipynb workflow.

    Args:
        band: Band (e.g. "F450W", "F606W")
        instrument: "ACS/WFC", "WFPC2/PC", "WFPC2/WF2", "WFPC2/WF3", "WFPC2/WF4"
        date_obs: Observation date "YYYY-MM-DD"
        detector_pos: Detector pixel (x, y)
        spectrum_type: Spectral type ("g2v", "blackbody_6000", etc.)
        psf_diameter: PSF diameter (arcsec)
        stamp_size: Output stamp size
        tinytim_bin: TinyTim executable directory; use default if None
        pixel_scale: Target pixel scale (arcsec/pixel); rebin if different from TinyTim's native scale

    Returns:
        (psf_normalized, source_label) or None
    """
    bin_dir = tinytim_bin or get_tinytim_bin()
    if bin_dir is None:
        print(f"    [WARN] TinyTim bin not set (TINYTIM_BIN env)", flush=True)
        return None
    tiny1 = bin_dir / "tiny1"
    tiny2 = bin_dir / "tiny2"

    if not tiny1.exists() or not tiny2.exists():
        print(f"    [WARN] TinyTim binaries not found at {bin_dir}", flush=True)
        return None

    cam_cfg = TINYTIM_CAMERAS.get(instrument)
    if cam_cfg is None:
        print(f"    [WARN] TinyTim: unknown instrument {instrument}", flush=True)
        return None

    # focus correction
    z4_offset = _compute_focus_offset(date_obs, instrument)
    z4 = TINYTIM_FOCUS_CENTER.get(instrument, 0.0) + z4_offset

    # spectrum mapping
    spectrum_map = {
        "g2v": "2\n6000",           # blackbody 6000K (approx. G2V)
        "blackbody_6000": "2\n6000",
        "powerlaw_0": "3\n0",
    }
    spec_input = spectrum_map.get(spectrum_type, "2\n6000")

    # build tiny1 input sequence (reference dinos_i TinyTim_Interactive_Flow.md)
    cam_id = cam_cfg["camera_id"]
    x, y = detector_pos

    # ACS/WFC requires selecting a chip (1 or 2); WFPC2 WFC requires selecting a chip (2/3/4, see cam_cfg["wfpc2_chip"])
    if instrument == "ACS/WFC":
        chip = 1 if x < 2048 else 2
        tiny1_input = f"{cam_id}\n{chip}\n{x} {y}\n{band.lower()}\n{spec_input}\n{psf_diameter}\nN\n{z4:.3f}\ntinytim_psf\n"
    elif cam_cfg.get("wfpc2_chip") is not None:
        # WFPC2 Wide Field: after selecting 5 in the tiny1 menu, enter the chip number (2/3/4)
        chip = cam_cfg["wfpc2_chip"]
        tiny1_input = f"{cam_id}\n{chip}\n{x} {y}\n{band.lower()}\n{spec_input}\n{psf_diameter}\nN\n{z4:.3f}\ntinytim_psf\n"
    elif instrument == "WFPC2/PC":
        tiny1_input = f"{cam_id}\n{x} {y}\n{band.lower()}\n{spec_input}\n{psf_diameter}\nN\n{z4:.3f}\ntinytim_psf\n"
    else:
        tiny1_input = f"{cam_id}\n{x} {y}\n{band.lower()}\n{spec_input}\n{psf_diameter}\nN\n{z4:.3f}\ntinytim_psf\n"

    tmpdir = Path(tempfile.mkdtemp())
    try:
        param_file = tmpdir / "tinytim_psf.par"

        # Step 1: tiny1 (generate parameter file)
        env = os.environ.copy()
        env["TINYTIM"] = str(bin_dir)
        result1 = subprocess.run(
            [str(tiny1), str(param_file)],
            input=tiny1_input,
            capture_output=True, text=True,
            timeout=60, cwd=str(tmpdir),
            env=env,
        )
        if result1.returncode != 0 or not param_file.exists():
            print(f"    [WARN] TinyTim tiny1 failed: rc={result1.returncode}", flush=True)
            return None

        # Step 2: tiny2 (ray tracing)
        result2 = subprocess.run(
            [str(tiny2), str(param_file)],
            capture_output=True, text=True,
            timeout=120, cwd=str(tmpdir),
            env=env,
        )

        # find output (ACS produces *00_psf.fits, WFPC2 produces *00.fits)
        psf_candidates = list(tmpdir.glob("tinytim_psf00_psf.fits")) + list(tmpdir.glob("tinytim_psf00.fits"))
        if not psf_candidates:
            print(f"    [WARN] TinyTim no PSF output files found in {tmpdir}", flush=True)
            return None

        with fits.open(str(psf_candidates[0])) as hdul:
            psf_full = np.asarray(hdul[0].data, dtype=float)
            native_scale = float(hdul[0].header.get("PIXSCALE", 0.0) or 0.0)

        # Rebin to the target pixel scale (TinyTim output native scale vs science image scale).
        # WFPC2 WF chips are natively 0.0999"/px, but the DRZ science image is 0.0455"/px, so enlarge ~2.19x.
        if native_scale > 0 and pixel_scale and pixel_scale > 0:
            factor = native_scale / pixel_scale
            if abs(factor - 1.0) > 0.01:
                psf_full = zoom(psf_full, factor, order=1)

        # Crop stamp_size×stamp_size centered on the centroid.
        # The old code half_full-30:half_full+31 wrapped negative indices on a 30x30 array, leaving only 1/4 of the PSF.
        yy, xx = np.indices(psf_full.shape)
        tot = psf_full.sum()
        if tot > 0:
            cy = float((yy * psf_full).sum() / tot)
            cx = float((xx * psf_full).sum() / tot)
        else:
            cy, cx = psf_full.shape[0] / 2.0, psf_full.shape[1] / 2.0
        cyi, cxi = int(round(cy)), int(round(cx))

        half = stamp_size // 2
        y0, x0 = cyi - half, cxi - half
        psf_out = np.zeros((stamp_size, stamp_size), dtype=float)
        sy0, sx0 = max(0, y0), max(0, x0)
        sy1, sx1 = min(psf_full.shape[0], y0 + stamp_size), min(psf_full.shape[1], x0 + stamp_size)
        psf_out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = psf_full[sy0:sy1, sx0:sx1]

        psf_out = np.maximum(psf_out, 0)
        tot = psf_out.sum()
        if tot > 0:
            psf_out /= tot

        label = f"tinytim_direct_{band}_{instrument.replace('/','_')}_z4{z4:.3f}"
        return psf_out.astype(np.float32), label

    except Exception as e:
        print(f"    [WARN] TinyTim Direct failed: {e}", flush=True)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

