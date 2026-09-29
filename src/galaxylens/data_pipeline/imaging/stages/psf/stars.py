"""场星检测与筛选（用于 PSF 构建）。"""
from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree

from ...config import MIN_STAR_SNR, STAR_SHARPLO


def detect_stars(
    data: np.ndarray,
    psf_fwhm_pix: float = 2.5,
    min_snr: float = MIN_STAR_SNR,
) -> List[Tuple[float, float, float]]:
    """用 DAOStarFinder 检测图像中的点源（场星）。

    Args:
        data: 背景减除后的 2D 图像
        psf_fwhm_pix: PSF FWHM 估计 (像素)，影响检测灵敏度
        min_snr: 最低 SNR 阈值

    Returns:
        [(x_centroid, y_centroid, flux), ...] 按流量降序排列

    Note:
        - 使用 sigma-clipped std 避免亮星系抬高检测阈值
        - photutils 3.0+ 用 x_centroid/y_centroid，旧版用 xcentroid/ycentroid
    """
    _, _, std_clipped = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    threshold = max(
        min_snr * std_clipped,
        min_snr * np.median(np.abs(data - np.median(data))) * 1.4826,
    )

    finder = DAOStarFinder(
        fwhm=psf_fwhm_pix,
        threshold=threshold,
        sharplo=STAR_SHARPLO, sharphi=3.0,
        roundlo=-0.8, roundhi=0.8,
    )

    sources = finder(data)
    if sources is None:
        return []

    sources.sort("flux", reverse=True)

    xcol = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"

    return [
        (float(s[xcol]), float(s[ycol]), float(s["flux"]))
        for s in sources
    ]

def filter_field_stars(
    stars: List[Tuple[float, float, float]],
    center: Tuple[float, float],
    exclusion_radius_pix: float = 60.0,
    max_stars: int = 20,
    data_shape: Optional[Tuple[int, int]] = None,
    min_dist: float = 5.0,
    edge_margin: float = 20.0,
) -> List[Tuple[float, float, float]]:
    """排除透镜中心区域 + 筛选孤立场星，用于 PSF 构建。

    修复 (2026-08-17): 此前只排除中心 + 截断数量，无孤立性筛选，
    星系团块/弧亮度节点/低 SNR 弥散斑被当作场星喂给 EPSFBuilder，
    ePSF 被稀释膨胀 (EE50 5-12px, 正常 ~1.1px)。

    筛选步骤:
      1. 排除透镜中心区域 (星系本身 + 弧亮度节点)
      2. 排除贴边源 (stamp 截断会污染 ePSF)
      3. 孤立性: 与最近邻源距离 >= min_dist px (KD-tree, O(n log n))
      4. 按亮度降序截断 max_stars

    Args:
        stars: detect_stars() 的输出 [(x, y, flux), ...]
        center: 透镜中心 (cx, cy)
        exclusion_radius_pix: 中心排除半径 (像素)
        max_stars: 最多保留的场星数
        data_shape: 图像 (ny, nx)，用于贴边排除；None 则跳过
        min_dist: 孤立性阈值 (像素)
        edge_margin: 距边缘的最小距离 (像素)

    Returns:
        场星列表 [(x, y, flux), ...]
    """
    cx, cy = center
    field = [
        (sx, sy, fl) for sx, sy, fl in stars
        if np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2) >= exclusion_radius_pix
    ]
    # 排除贴边源 (stamp 截断会污染 ePSF)
    if data_shape is not None:
        ny, nx = data_shape
        field = [(sx, sy, fl) for sx, sy, fl in field
                 if edge_margin <= sx <= nx - edge_margin
                 and edge_margin <= sy <= ny - edge_margin]
    if len(field) < 1:
        return []
    xs = np.array([s[0] for s in field], dtype=float)
    ys = np.array([s[1] for s in field], dtype=float)
    # 孤立性: KD-tree 求每颗源到最近其它源的距离 (O(n log n), 全幅几千星也快)
    if len(field) >= 2:
        coords = np.column_stack([xs, ys])
        dists, _ = cKDTree(coords).query(coords, k=2)
        nearest = dists[:, 1]  # 第2近 = 最近其它源 (第1是自身, 距离0)
    else:
        nearest = np.array([np.inf])
    isolated = [(sx, sy, fl) for j, (sx, sy, fl) in enumerate(field)
                if nearest[j] >= min_dist]
    if len(isolated) < 1:
        return []
    isolated.sort(key=lambda s: s[2], reverse=True)  # 按 flux 降序
    return isolated[:max_stars]

