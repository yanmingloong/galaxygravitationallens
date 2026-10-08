"""Path resolution: data root directory and external tool paths, all specified via environment variables.

Environment variables:
  GALAXYLENS_DATA_ROOT        Data root directory (contains raw/ and processed_v2/, required)
  GALAXYLENS_CONFIG_DIR       configs/ directory (defaults to the repository root configs/)
  SEXTRACTOR_BIN              SExtractor binary (optional; auto-detected via PATH if not set)
  TINYTIM_GRID_BASE           TinyTim Massey grid root directory (optional)
  TINYTIM_BIN                 TinyTim executable directory tiny1/tiny2 (optional)
  TINYTIM_BREATHING_CSV       HST breathing model CSV (optional)
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


def _env(name: str) -> Optional[str]:
    return os.environ.get(name)


def get_data_root() -> Path:
    """Data root directory (contains raw/ and processed_v2/). Raises RuntimeError if not set."""
    root = _env("GALAXYLENS_DATA_ROOT")
    if not root:
        raise RuntimeError(
            "GALAXYLENS_DATA_ROOT is not set. Point it to a directory containing raw/ and processed_v2/, e.g.:\n"
            "  export GALAXYLENS_DATA_ROOT=/mnt/d/HermesAgent/ResearchProjects/GalaxyGravLens/DataPipeline"
        )
    return Path(root)


def get_raw_root() -> Path:
    """Raw data directory (FITS files downloaded from MAST)."""
    return get_data_root() / "raw"


def get_processed_root() -> Path:
    """Processed products directory."""
    return get_data_root() / "processed_v2"


def get_sample_library_root() -> Path:
    """Sample library root directory (contains 01_raw ~ 05_exports)."""
    root = _env("GALAXYLENS_SAMPLE_LIB")
    if not root:
        raise RuntimeError(
            "GALAXYLENS_SAMPLE_LIB is not set. Point it to the sample library root directory (containing 01_raw~05_exports), e.g.:\n"
            "  export GALAXYLENS_SAMPLE_LIB=/mnt/d/HermesAgent/ResearchProjects/GalaxyGravLens/DiskLenses/SampleLibrary"
        )
    return Path(root)


def get_config_dir() -> Path:
    """configs/ directory (contains the lens YAML files for disk/ and slacs/)."""
    env = _env("GALAXYLENS_CONFIG_DIR")
    if env:
        return Path(env)
    # Default: repository root configs/ (this file is in src/galaxylens/config/, 3 levels up to the repository root)
    return Path(__file__).resolve().parents[3] / "configs"


def get_sextractor_bin() -> Optional[Path]:
    """SExtractor binary path (environment variable takes priority, otherwise auto-detected via PATH)."""
    bin_ = _env("SEXTRACTOR_BIN") or shutil.which("sex")
    return Path(bin_) if bin_ else None


def get_tinytim_grid_base() -> Optional[Path]:
    """TinyTim Massey grid root directory (returns None if not set)."""
    v = _env("TINYTIM_GRID_BASE")
    return Path(v) if v else None


def get_tinytim_bin() -> Optional[Path]:
    """TinyTim executable directory (returns None if not set)."""
    v = _env("TINYTIM_BIN")
    return Path(v) if v else None


def get_tinytim_breathing_csv() -> Optional[Path]:
    """HST breathing model CSV path (returns None if not set)."""
    v = _env("TINYTIM_BREATHING_CSV")
    return Path(v) if v else None
