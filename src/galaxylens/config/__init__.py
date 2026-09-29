"""galaxylens 配置模块：路径解析 + YAML 配置加载。

- paths.py    环境变量路径解析（数据根 / SExtractor / TinyTim）
- schema.py   配置 dataclass（LensConfig / BandConfig）
- loader.py   YAML 加载 → dataclass
- defaults.py 处理参数默认值
"""
from galaxylens.config.paths import (  # noqa: F401
    get_data_root, get_raw_root, get_processed_root, get_config_dir,
    get_sextractor_bin,
    get_tinytim_grid_base, get_tinytim_bin, get_tinytim_breathing_csv,
)
from galaxylens.config.schema import BandConfig, LensConfig  # noqa: F401
from galaxylens.config.loader import (  # noqa: F401
    load_lens_config, list_lenses, get_bands_for_lens, get_pixel_scale,
)
from galaxylens.config.defaults import (  # noqa: F401
    CUTOUT_SIZE_ARCSEC, MASK_RADIUS_FACTOR, PSF_STAMP_SIZE,
    MIN_STAR_SNR, STAR_SHARPLO, N_STARS_PSF,
    PSF_EE50_MIN_PX, PSF_EE50_MAX_PX, PSF_ELL_MAX,
    INSTRUMENT_PIXEL_SCALE,
)

__all__ = [
    "get_data_root", "get_raw_root", "get_processed_root", "get_config_dir",
    "get_sextractor_bin",
    "get_tinytim_grid_base", "get_tinytim_bin", "get_tinytim_breathing_csv",
    "BandConfig", "LensConfig",
    "load_lens_config", "list_lenses", "get_bands_for_lens", "get_pixel_scale",
    "CUTOUT_SIZE_ARCSEC", "MASK_RADIUS_FACTOR", "PSF_STAMP_SIZE",
    "MIN_STAR_SNR", "STAR_SHARPLO", "N_STARS_PSF",
    "PSF_EE50_MIN_PX", "PSF_EE50_MAX_PX", "PSF_ELL_MAX",
    "INSTRUMENT_PIXEL_SCALE",
]
