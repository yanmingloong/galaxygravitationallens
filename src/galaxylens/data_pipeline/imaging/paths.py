"""兼容层：re-export galaxylens.config.paths。
Compatibility layer: re-exports galaxylens.config.paths.

保留本文件是为了不破坏 data_pipeline/imaging 内既有 import 路径；
权威实现见 galaxylens.config.paths。
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
