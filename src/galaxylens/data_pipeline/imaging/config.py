"""透镜参数配置（兼容层，从 galaxylens.config 加载 YAML）。

原为硬编码 dict，现改为从 configs/{disk,slacs}/*.yaml 加载，
返回 LensConfig dataclass。本文件保留原接口名供 data_pipeline 内部调用。
"""
from galaxylens.config.loader import (  # noqa: F401
    load_lens_config,
    list_lenses,
    get_bands_for_lens,
    get_pixel_scale,
)
from galaxylens.config.defaults import (  # noqa: F401
    CUTOUT_SIZE_ARCSEC,
    MASK_RADIUS_FACTOR,
    PSF_STAMP_SIZE,
    MIN_STAR_SNR,
    STAR_SHARPLO,
    N_STARS_PSF,
    PSF_EE50_MIN_PX,
    PSF_EE50_MAX_PX,
    PSF_ELL_MAX,
    INSTRUMENT_PIXEL_SCALE,
)


def get_lens_config(lens_id):
    """获取透镜配置（返回 LensConfig dataclass）。"""
    return load_lens_config(lens_id)
