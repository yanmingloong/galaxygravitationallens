"""SKIRT radiative transfer -> mock images.

Converts a selected galaxy's all-field particles to SKIRT input
(:mod:`.convert`), generates and runs SKIRT simulations (:mod:`.render`),
and turns the raw output into fake HST images with an extinction map
(:mod:`.imaging`).
"""

from .convert import (
    tng_to_skirt, load_stars, load_gas, inertia_rotation, write_source,
    write_dust, F_DUST,
)
from .config import BandConfig, MockObsConfig, J2141_BANDS, J2141_OBS
from .render import write_ski, run_skirt
from .imaging import make_fake_hst, extinction_map

__all__ = [
    "tng_to_skirt", "load_stars", "load_gas", "inertia_rotation",
    "write_source", "write_dust", "F_DUST",
    "BandConfig", "MockObsConfig", "J2141_BANDS", "J2141_OBS",
    "write_ski", "run_skirt",
    "make_fake_hst", "extinction_map",
]
