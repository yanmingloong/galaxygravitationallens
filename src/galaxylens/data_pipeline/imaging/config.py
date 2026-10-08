"""Lens parameter configuration (compatibility layer, loading YAML from
galaxylens.config).

Originally a hardcoded dict, now loads from configs/{disk,slacs}/*.yaml and
returns a LensConfig dataclass. This file keeps the original interface names
for internal calls within data_pipeline.
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
    """Get the lens configuration (returns a LensConfig dataclass)."""
    return load_lens_config(lens_id)
