"""路径解析：数据根目录与外部工具路径，全部由环境变量指定。
Path resolution: data root and external tool paths, all specified via environment variables.

环境变量清单:
  GALAXYLENS_DATA_ROOT        数据根目录（含 raw/ 与 processed_v2/，必需）
  GALAXYLENS_CONFIG_DIR       configs/ 目录（默认仓库根 configs/）
  SEXTRACTOR_BIN              SExtractor 二进制（可选，未设则 PATH 自动查找）
  TINYTIM_GRID_BASE           TinyTim Massey 栅格根目录（可选）
  TINYTIM_BIN                 TinyTim 可执行目录 tiny1/tiny2（可选）
  TINYTIM_BREATHING_CSV       HST 呼吸模型 CSV（可选）
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


def _env(name: str) -> Optional[str]:
    return os.environ.get(name)


def get_data_root() -> Path:
    """数据根目录（含 raw/ 与 processed_v2/）。未设置时抛 RuntimeError。"""
    root = _env("GALAXYLENS_DATA_ROOT")
    if not root:
        raise RuntimeError(
            "GALAXYLENS_DATA_ROOT 未设置。请指向含 raw/ 和 processed_v2/ 的目录，例如：\n"
            "  export GALAXYLENS_DATA_ROOT=/mnt/d/HermesAgent/ResearchProjects/GalaxyGravLens/DataPipeline"
        )
    return Path(root)


def get_raw_root() -> Path:
    """原始数据目录（MAST 下载的 FITS）。"""
    return get_data_root() / "raw"


def get_processed_root() -> Path:
    """处理产物目录。"""
    return get_data_root() / "processed_v2"


def get_sample_library_root() -> Path:
    """样本库根目录（含 01_raw ~ 05_exports）。"""
    root = _env("GALAXYLENS_SAMPLE_LIB")
    if not root:
        raise RuntimeError(
            "GALAXYLENS_SAMPLE_LIB 未设置。请指向样本库根目录（含 01_raw~05_exports），例如：\n"
            "  export GALAXYLENS_SAMPLE_LIB=/mnt/d/HermesAgent/ResearchProjects/GalaxyGravLens/DiskLenses/SampleLibrary"
        )
    return Path(root)


def get_config_dir() -> Path:
    """configs/ 目录（含 disk/ 与 slacs/ 的透镜 YAML）。"""
    env = _env("GALAXYLENS_CONFIG_DIR")
    if env:
        return Path(env)
    # 默认：仓库根 configs/（本文件在 src/galaxylens/config/，向上 3 级到仓库根）
    return Path(__file__).resolve().parents[3] / "configs"


def get_sextractor_bin() -> Optional[Path]:
    """SExtractor 二进制路径（环境变量优先，否则 PATH 自动查找）。"""
    bin_ = _env("SEXTRACTOR_BIN") or shutil.which("sex")
    return Path(bin_) if bin_ else None


def get_tinytim_grid_base() -> Optional[Path]:
    """TinyTim Massey 栅格根目录（未设置返回 None）。"""
    v = _env("TINYTIM_GRID_BASE")
    return Path(v) if v else None


def get_tinytim_bin() -> Optional[Path]:
    """TinyTim 可执行目录（未设置返回 None）。"""
    v = _env("TINYTIM_BIN")
    return Path(v) if v else None


def get_tinytim_breathing_csv() -> Optional[Path]:
    """HST 呼吸模型 CSV 路径（未设置返回 None）。"""
    v = _env("TINYTIM_BREATHING_CSV")
    return Path(v) if v else None
