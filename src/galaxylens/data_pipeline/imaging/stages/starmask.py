"""
恒星杂光 Mask Stellar stray-light mask
=============
生成恒星掩模，标记场星周围区域以防止其杂光污染透镜建模。

功能:
  1. 对检测到的每颗场星生成圆形 mask
  2. 排除透镜中心区域（保护星系和弧）
  3. mask 半径 = MASK_RADIUS_FACTOR × PSF_FWHM

处理流程:
    场星列表 → 排除中心区域 → 逐星生成圆形mask → 合并输出

使用示例:
  >>> from galaxylens.data_pipeline.imaging.stages.starmask import make_starmask
  >>> mask = make_starmask(shape, field_stars, psf_fwhm_pix, center, pixel_scale)
"""

from __future__ import annotations
from typing import Optional, List, Tuple

import numpy as np

from ..config import MASK_RADIUS_FACTOR


# ══════════════════════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════════════════════

def make_starmask(
    shape: Tuple[int, int],
    star_coords: List[Tuple[float, float, float]],
    psf_fwhm_pix: float,
    lens_center: Tuple[float, float],
    pixel_scale: float,
    lens_exclusion_arcsec: float = 5.0,
    arc_positions: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """生成恒星杂光 mask。

    对每颗检测到的场星制作圆形 mask，
    排除透镜中心区域和已知弧位置。

    Args:
        shape: 图像 shape (ny, nx)
        star_coords: 场星列表 [(x, y, flux), ...]
        psf_fwhm_pix: PSF FWHM (像素)
        lens_center: 透镜中心 (cx, cy)
        pixel_scale: 像元尺度 (arcsec/pixel)
        lens_exclusion_arcsec: 中心保护半径 (角秒)
        arc_positions: 弧区额外保护坐标 [(x, y), ...]

    Returns:
        Boolean mask (True = 需要mask掉)
    """
    mask = np.zeros(shape, dtype=bool)
    cx, cy = lens_center

    lens_excl_pix = lens_exclusion_arcsec / pixel_scale
    mask_radius = int(np.ceil(MASK_RADIUS_FACTOR * psf_fwhm_pix))

    # 预生成坐标网格
    yy, xx = np.ogrid[:shape[0], :shape[1]]

    for sx, sy, flux in star_coords:
        # 跳过透镜中心区域
        if np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2) < lens_excl_pix:
            continue

        # 跳过弧区域
        if arc_positions:
            skip = False
            for ax, ay in arc_positions:
                if np.sqrt((sx - ax) ** 2 + (sy - ay) ** 2) < mask_radius:
                    skip = True
                    break
            if skip:
                continue

        # 圆形 mask
        dist = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
        mask[dist < mask_radius] = True

    return mask


def combine_masks(
    masks: List[np.ndarray],
    target_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """合并多个 mask（逻辑 OR），统一到目标 shape。

    用于合并不同波段/曝光的 starmask。

    Args:
        masks: Boolean mask 列表
        target_shape: 目标 shape，None 则取各 mask 的最小公共尺寸

    Returns:
        合并后的 Boolean mask
    """
    if not masks:
        raise ValueError("No masks to combine")

    if target_shape is None:
        shapes = [m.shape for m in masks]
        target_shape = (
            min(s[0] for s in shapes),
            min(s[1] for s in shapes),
        )

    combined = np.zeros(target_shape, dtype=bool)
    for m in masks:
        if m.shape == target_shape:
            combined = combined | m
        else:
            # center-crop to target_shape
            dy = (m.shape[0] - target_shape[0]) // 2
            dx = (m.shape[1] - target_shape[1]) // 2
            cropped = m[dy:dy + target_shape[0], dx:dx + target_shape[1]]
            combined = combined | cropped

    return combined


def get_mask_stats(mask: np.ndarray) -> dict:
    """计算 mask 的统计信息。

    Args:
        mask: Boolean mask

    Returns:
        {"n_masked": int, "fraction_pct": float}
    """
    n = int(mask.sum())
    return {
        "n_masked": n,
        "fraction_pct": 100.0 * n / mask.size if mask.size > 0 else 0.0,
    }
