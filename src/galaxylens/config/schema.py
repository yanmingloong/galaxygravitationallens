"""Config dataclass definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class BandConfig:
    """Instrument and pixel-scale configuration for a single band."""
    instrument: str
    pixel_scale: float


@dataclass
class LensConfig:
    """Complete configuration for a single lens."""
    lens_id: str
    name: str
    sample: str          # "SWELLS" / "SLACS"
    ra: float
    dec: float
    zl: float
    zs: float
    bands: Dict[str, BandConfig]
    center_offset_arcsec: Tuple[float, float] = (0.0, 0.0)
