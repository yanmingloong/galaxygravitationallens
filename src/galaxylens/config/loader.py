"""YAML 配置加载 → dataclass。 YAML config loading → dataclass."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml

from galaxylens.config.paths import get_config_dir
from galaxylens.config.schema import BandConfig, LensConfig


def _find_yaml(lens_id: str) -> Path:
    """在 configs/{disk,slacs}/ 下定位透镜的 YAML 文件。"""
    cfg_dir = get_config_dir()
    for sample in ("disk", "slacs"):
        p = cfg_dir / sample / f"{lens_id}.yaml"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"找不到透镜 {lens_id} 的配置 YAML（在 {cfg_dir}/disk 或 slacs/）"
    )


def load_lens_config(lens_id: str) -> LensConfig:
    """加载单个透镜配置。"""
    p = _find_yaml(lens_id)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    bands = {b: BandConfig(**v) for b, v in data["bands"].items()}
    return LensConfig(
        lens_id=lens_id,
        name=data["name"],
        sample=data["sample"],
        ra=float(data["ra"]),
        dec=float(data["dec"]),
        zl=float(data["zl"]),
        zs=float(data["zs"]),
        bands=bands,
        center_offset_arcsec=tuple(data.get("center_offset_arcsec", (0.0, 0.0))),
    )


SAMPLE_TO_DIR = {"SWELLS": "disk", "SLACS": "slacs"}


def list_lenses(sample: Optional[str] = None) -> List[str]:
    """列出所有或指定样本的透镜 ID。"""
    cfg_dir = get_config_dir()
    dirs = [SAMPLE_TO_DIR.get(sample, sample)] if sample else ["disk", "slacs"]
    lenses = []
    for s in dirs:
        d = cfg_dir / s
        if d.is_dir():
            lenses.extend(sorted(p.stem for p in d.glob("*.yaml")))
    return sorted(lenses)


def get_bands_for_lens(lens_id: str) -> List[str]:
    """获取透镜的可用波段列表。"""
    return list(load_lens_config(lens_id).bands.keys())


def get_pixel_scale(lens_id: str, band: str) -> float:
    """获取透镜+波段的像元尺度。"""
    cfg = load_lens_config(lens_id)
    band_cfg = cfg.bands.get(band)
    return band_cfg.pixel_scale if band_cfg is not None else 0.05
