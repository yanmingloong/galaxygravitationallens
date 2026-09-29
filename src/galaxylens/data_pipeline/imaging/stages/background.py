"""
背景估计与减除 Background estimation & subtraction
==============
从图像边缘区域估计天空背景，然后减除。

功能:
  1. Sigma-clipped 边缘条带统计估计背景水平
  2. 背景减除

方法:
  - estimate_background(): 边缘 sigma-clipped 中值和 RMS
  - subtract_background(): data - bg_median

处理流程:
    data → 提取边缘5px条带 → sigma_clipped_stats(3σ) → 返回 (median, rms)
         → data - median → 输出

使用示例:
  >>> from galaxylens.data_pipeline.imaging.stages.background import estimate_background, subtract_background
  >>> bg_med, bg_rms = estimate_background(sci)
  >>> sci_bgsub = subtract_background(sci, bg_med)
"""

from __future__ import annotations
from typing import Optional, Tuple
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats


# ══════════════════════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════════════════════

def estimate_background(
    data: np.ndarray,
    sigma: float = 3.0,
    edge_width: int = 5,
) -> Tuple[float, float]:
    """从边缘条带估计天空背景的 median 和 RMS。

    取图像四条边缘各 edge_width 像素宽的条带，
    做 sigma-clipped 统计，排除星系和亮源的影响。

    Args:
        data: 2D 图像
        sigma: sigma-clipping 阈值
        edge_width: 边缘条带宽度 (像素)

    Returns:
        (bg_median, bg_rms): 背景中值和 RMS

    Note:
        假设透镜位于图像中心，边缘条带主要包含天空背景。
        若透镜偏离中心太远，此方法可能低估背景。
    """
    edge_strip = np.concatenate([
        data[:edge_width, :].ravel(),
        data[-edge_width:, :].ravel(),
        data[:, :edge_width].ravel(),
        data[:, -edge_width:].ravel(),
    ])

    mean, median, std = sigma_clipped_stats(
        edge_strip, sigma=sigma, maxiters=5,
    )
    return float(median), float(std)


def subtract_background(
    data: np.ndarray,
    bg_value: float,
) -> np.ndarray:
    """从图像中减除背景水平。

    Args:
        data: 输入 2D 图像
        bg_value: 背景值 (通常来自 estimate_background 的 median)

    Returns:
        背景减除后的图像 (data - bg_value)
    """
    return data - bg_value


def process_background(
    data: np.ndarray,
    sigma: float = 3.0,
) -> Tuple[np.ndarray, float, float]:
    """一站式背景估计+减除。

    优先 SExtractor，失败回退边缘 sigma-clipping。

    Args:
        data: 输入 2D 图像
        sigma: sigma-clipping 阈值 (回退时使用)

    Returns:
        (bg_subtracted_data, bg_median, bg_rms)
    """
    bg_med, bg_rms = estimate_background_sextractor(data)
    if bg_med is not None:
        print(f"    BG (SExtractor): median={bg_med:.4f}, rms={bg_rms:.4f}", flush=True)
        return subtract_background(data, bg_med), bg_med, bg_rms

    # 回退：边缘 sigma-clipping
    bg_med, bg_rms = estimate_background(data, sigma=sigma)
    return subtract_background(data, bg_med), bg_med, bg_rms


# ══════════════════════════════════════════════════════════════════════
# 方法 2: SExtractor 背景估计 (E3, 2026-07-03)
# ══════════════════════════════════════════════════════════════════════

def estimate_background_sextractor(
    data: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """使用 SExtractor 估计背景 mean 和 sigma。

    参考 dinos_i 3_Pre-Processor.ipynb 使用 pysex。

    Args:
        data: 2D 图像

    Returns:
        (bg_mean, bg_sigma) 或 (None, None) 如果 SExtractor 不可用
    """
    try:
        # 写临时 FITS
        tmpdir = Path(tempfile.mkdtemp())
        fits_path = tmpdir / "tmp_sci.fits"
        fits.writeto(str(fits_path), data.astype(np.float32), overwrite=True)

        # 配置
        bg_path = tmpdir / "tmp_bg.fits"
        config = "\n".join([
            "CATALOG_TYPE    NONE",
            "CHECKIMAGE_TYPE BACKGROUND",
            "CHECKIMAGE_NAME " + str(bg_path),
            "BACK_SIZE       32",
            "BACK_FILTERSIZE 3",
            "DETECT_THRESH   3.0",
            "ANALYSIS_THRESH 3.0",
            "WEIGHT_TYPE     BACKGROUND",
        ])
        config_path = tmpdir / "default.sex"
        config_path.write_text(config)

        # 用绝对路径调用 sex
        sex_bin = os.environ.get("SEXTRACTOR_BIN") or shutil.which("sex")
        if sex_bin is None:
            return None, None
        result = subprocess.run(
            [sex_bin, str(fits_path), "-c", str(config_path)],
            capture_output=True, text=True, timeout=60,
            cwd=str(tmpdir),
        )
        if result.returncode != 0:
            print(f"    [WARN] SExtractor exit={result.returncode}: {result.stderr[:200]}", flush=True)

        # 读背景图
        if bg_path.exists():
            bg_data = fits.getdata(str(bg_path))
            bg_mean = float(np.mean(bg_data))
            bg_std = float(np.std(bg_data))
            return bg_mean, bg_std

        return None, None

    except Exception as e:
        print(f"    [WARN] SExtractor failed: {e}", flush=True)
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
