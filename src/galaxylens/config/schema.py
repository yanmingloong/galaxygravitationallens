"""配置 dataclass 定义。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class BandConfig:
    """单个波段的仪器与像元尺度配置。"""
    instrument: str
    pixel_scale: float


@dataclass
class LensConfig:
    """单个透镜的完整配置。"""
    lens_id: str
    name: str
    sample: str          # "SWELLS" / "SLACS"
    ra: float
    dec: float
    zl: float
    zs: float
    bands: Dict[str, BandConfig]
    center_offset_arcsec: Tuple[float, float] = (0.0, 0.0)
