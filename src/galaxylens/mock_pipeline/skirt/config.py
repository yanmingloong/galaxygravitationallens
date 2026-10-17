"""Observation configuration for the SKIRT mock pipeline.

Separates "which galaxy to observe" (``TargetConfig``) from "how to observe
it" (this module): each band carries its SKIRT band name, HST pixel scale,
background noise and peak normalization, so switching instruments (ACS vs
WFPC2) or filters needs no code change.
"""

from dataclasses import dataclass


@dataclass
class BandConfig:
    """One mock observation band (filter + instrument + noise model).

    ``label`` is the file tag (e.g. ``"F814W"``); ``skirt_name`` is the full
    SKIRT band name (e.g. ``"HST ACS WFC F814W"`` or ``"HST WFPC2 F814W"``);
    ``pixel_scale`` is the HST pixel scale in arcsec/px.
    """

    label: str
    skirt_name: str
    pixel_scale: float
    noise_sigma: float = 8.81    # background noise sigma, e-/s
    peak: float = 4905.0         # peak normalization, e-/s


@dataclass
class MockObsConfig:
    """One mock observation setup (band set + instrument geometry)."""

    bands: list                    # list[BandConfig]
    fov: str = "40 kpc"
    num_pixels: int = 200
    azimuth: str = "90 deg"        # long axis horizontal after 3-axis alignment
    distance: str = "10 Mpc"       # mock distance (LocalUniverseCosmology)


# Default J2141 observation: ACS/WFC F814W + WFC3/IR F160W
J2141_BANDS = [
    BandConfig("F814W", "HST ACS WFC F814W", 0.1258),
    BandConfig("F160W", "HST WFC3 IR F160W", 0.13),
]
J2141_OBS = MockObsConfig(bands=J2141_BANDS)
