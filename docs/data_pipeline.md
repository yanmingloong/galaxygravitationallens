# Data Pipeline Usage guide

*中文版见 [data_pipeline_zh.md](data_pipeline_zh.md).*

Observational data production pipeline: preprocess raw HST data (MAST-downloaded FITS) into PyAutoLens-ready standardized products (sci / noise / psf / mask).

## 1. Overview

```
data_pipeline/
├── imaging/        # lens image processing (HST optical/NIR)
│   ├── config.py   # 16 lens parameters (SWELLS disk + SLACS ellipticals)
│   ├── paths.py    # path resolution (GALAXYLENS_DATA_ROOT environment variable)
│   ├── io/         # source file discovery + FITS loading
│   ├── stages/     # processing stages (cutout/centroid/background/cosmic_ray/noise/psf/starmask/alignment)
│   ├── output/     # output (FITS writing / diagnostic plots / logs / quality assessment)
│   └── preprocess_master.py  # main flow scheduling + CLI
├── spectra/        # spectral pipeline (planned: SDSS / KCWI / DEIMOS)
└── other_obs/      # other observations (planned: NIRC2 K'-band etc.)
```

**Design principle**: data production (data_pipeline) is separated from data consumption (lensfit). This module only does "raw → processed", producing standardized FITS; the fitting engine lensfit consumes these products. Therefore this module **only depends on astropy/photutils**, not autolens.

## 2. Environment configuration

### 2.1 Data root directory (required)

```bash
export GALAXYLENS_DATA_ROOT=/path/to/observations
```

Points to the directory containing `raw/` (MAST downloads) and `processed_v2/` (processed products). If unset, `find_source_files` raises a RuntimeError with a hint.

### 2.2 SExtractor (optional, enhanced background estimation)

```bash
export SEXTRACTOR_BIN=/path/to/sex   # optional
```

If unset, automatically searches PATH using `shutil.which("sex")`; if not found, falls back to edge sigma-clipping background estimation (fallback is built in, no need to worry).

### 2.3 TinyTim (optional, physical PSF model)

```bash
export TINYTIM_GRID_BASE=/path/to/tinytim_psf_models     # Massey grid
export TINYTIM_BIN=/path/to/tinytim                      # tiny1/tiny2 executable directory
export TINYTIM_BREATHING_CSV=/path/to/Hubble_dspace.csv  # HST breathing model
```

All three are optional: when unset, the corresponding functionality is automatically skipped (Massey grid / direct compilation / focus correction).

### 2.3 Dependencies

Core dependencies: `numpy`, `scipy`, `astropy`, `photutils`.

```bash
pip install -e ".[data]"
```

## 3. Quick start

```python
from galaxylens.data_pipeline.imaging.preprocess_master import process_lens
from galaxylens.data_pipeline.imaging.paths import get_processed_root

# Process a single lens (output to $GALAXYLENS_DATA_ROOT/processed_v2/disk/J2141-0001/v1/)
process_lens("J2141-0001", version_tag="v1", output_base=get_processed_root())
```

## 4. CLI usage

```bash
# Process a single lens
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J2141-0001 --version-tag v1

# Process the whole SWELLS sample
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --sample SWELLS --version-tag v1

# Process only the specified band + force PSF method
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J1630+4520 --bands F814W --psf-method moffat

# Dry run (only list files to be processed, no writing to disk)
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J1630+4520 --dry-run
```

## 5. Processing flow (per band)

1. `file_discovery` — find source files (FLC preferred → FLT/DRC/DRZ fallback)
2. `loaders` — load SCI + WHT + WCS
3. `cutout` — WCS cutout to 15″×15″
4. `centroid` — 2D Gaussian center fine-tuning (iteratively correct RA/Dec when offset >10px)
5. `background` — SExtractor background estimation (fallback edge sigma-clipping)
6. `cosmic_ray` — deepCR (single exposure) or sigma-clipped stacking (multiple exposures)
7. `noise` — noise map (method auto-selected by product type)
8. `psf` — PSF construction (ePSF → TinyTim → Moffat fallback, auto-comparison for best)
9. `starmask` — star mask
10. `alignment` — cross-band alignment (for multiple bands)
11. `output` — FITS writing + diagnostic plots + logs + quality assessment (8 ratings)

## 6. PSF construction (stages/psf package)

Split into submodules:

| Submodule | Responsibility |
|---|---|
| `stars.py` | field star detection (DAOStarFinder) + isolation filtering |
| `metrics.py` | EE50 (half-light radius) + second-moment ellipticity |
| `epsf.py` | photutils EPSFBuilder oversampled ePSF (preferred, captures real aberrations) |
| `analytic.py` | Moffat / Gaussian / elliptical Gaussian analytic PSF |
| `tinytim.py` | TinyTim Massey grid (ACS) + direct compilation (WFPC2) |
| `__init__.py` | `build_psf` auto-selection entry + re-export |

Automatic selection strategy: ePSF and TinyTim each generate candidates → EE50 ∈ [0.8, 3.0]px and ellipticity < 0.15 hard thresholds → qualifying ePSF preferred → TinyTim as fallback → Moffat as last resort.

## 7. Tutorial: full preprocessing of one lens

```python
import os
os.environ["GALAXYLENS_DATA_ROOT"] = "/path/to/observations"

from galaxylens.data_pipeline.imaging.paths import get_raw_root, get_processed_root
from galaxylens.data_pipeline.imaging.io.file_discovery import find_source_files
from galaxylens.data_pipeline.imaging.preprocess_master import process_lens

# 1. See which source files a lens has
files = find_source_files("J2141-0001", "F814W", product_type="FLC")
print(f"Found {len(files)} FLC files")

# 2. Full preprocessing
out = process_lens("J2141-0001", version_tag="v1", output_base=get_processed_root())
print(f"Product directory: {out}")
# Products include *_sci.fits / *_noise.fits / *_psf.fits / *_starmask.fits / processing_log.json
```

## 8. Environment variable reference

| Environment variable | Purpose | Required |
|---|---|---|
| `GALAXYLENS_DATA_ROOT` | data root directory (raw/ + processed_v2/) | Yes |
| `GALAXYLENS_CONFIG_DIR` | configs/ directory (default: repo root configs/) | No |
| `SEXTRACTOR_BIN` | SExtractor binary | No |
| `TINYTIM_GRID_BASE` | TinyTim Massey grid | No |
| `TINYTIM_BIN` | TinyTim executable directory | No |
| `TINYTIM_BREATHING_CSV` | HST breathing model CSV | No |

Lens parameters are loaded from `configs/{disk,slacs}/*.yaml` by `galaxylens.config.loader.load_lens_config()` into a `LensConfig` dataclass.
