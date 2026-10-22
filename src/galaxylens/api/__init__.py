"""Facade layer: user-facing API (``gl.fit`` / ``gl.load_config`` / ``gl.load_dataset``).

The core engine lives in ``lensfit``; this package is the handle users hold.
"""

from __future__ import annotations

from .config_loader import DataSpec, FitConfig, SearchSpec, load_config
from .derived import f_dm_of, m2d_within, m_star_total
from .result import Result, weighted_percentile
from .search import build_search
from .facade import fit, fit_mein, fit_positions, load_dataset

__all__ = [
    "DataSpec",
    "SearchSpec",
    "FitConfig",
    "load_config",
    "load_dataset",
    "fit",
    "fit_mein",
    "fit_positions",
    "Result",
    "weighted_percentile",
    "f_dm_of",
    "m2d_within",
    "m_star_total",
    "build_search",
]
