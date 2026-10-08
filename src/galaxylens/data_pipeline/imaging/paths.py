"""Compatibility layer: re-exports galaxylens.config.paths.

This file is kept so as not to break existing import paths within
data_pipeline/imaging; the authoritative implementation lives in
galaxylens.config.paths.
"""
from galaxylens.config.paths import (  # noqa: F401
    get_data_root,
    get_raw_root,
    get_processed_root,
    get_config_dir,
    get_sextractor_bin,
    get_tinytim_grid_base,
    get_tinytim_bin,
    get_tinytim_breathing_csv,
)
