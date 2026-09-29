"""
2D 高斯中心微调 2D Gaussian centroid refinement
==============
用二维高斯拟合 refine 透镜中心位置，纠正 WCS 指向误差。

功能:
  1. 对裁剪后图像的中心区域做 2D Gaussian 拟合
  2. 返回精修后的亚像素 centroid

原理:
  透镜星系中心是图像最亮的区域，用 Gaussian 拟合其亮度峰
  得到的 (x0, y0) 比 WCS 指向更准确。

使用示例:
  >>> from galaxylens.data_pipeline.imaging.stages.centroid import fit_centroid
  >>> cx_fit, cy_fit = fit_centroid(sci_cut, sci_cut.shape[1]//2, sci_cut.shape[0]//2)
  >>> print(f"Offset: ({cx_fit-cx:+.1f}, {cy_fit-cy:+.1f}) px")
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit


# ══════════════════════════════════════════════════════════════════════
# 内部：2D Gaussian 模型
# ══════════════════════════════════════════════════════════════════════

def _gaussian_2d(
    xy: Tuple[np.ndarray, np.ndarray],
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float,
) -> np.ndarray:
    """二维椭圆高斯函数。"""
    x, y = xy
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    return offset + amplitude * np.exp(
        -(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2)
    )


# ══════════════════════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════════════════════

def fit_centroid(
    data: np.ndarray,
    cx: float,
    cy: float,
    box_size: int = 20,
) -> Tuple[float, float]:
    """用二维高斯拟合中心亮度峰值，返回亚像素 centroid。

    在 (cx, cy) 周围的 box_size×box_size 区域内，
    用椭圆 Gaussian fit 精确定位亮度中心。

    Args:
        data: 2D 图像（应为背景减除后的数据）
        cx: 初始中心 x 坐标 (像素)
        cy: 初始中心 y 坐标 (像素)
        box_size: 拟合区域大小 (像素)

    Returns:
        (cx_fit, cy_fit): 精修后的中心坐标 (像素)

    Note:
        拟合失败时返回输入的 (cx, cy)。
    """
    half = box_size // 2
    yc, xc = int(round(cy)), int(round(cx))

    # 确保不越界
    y1, y2 = max(0, yc - half), min(data.shape[0], yc + half)
    x1, x2 = max(0, xc - half), min(data.shape[1], xc + half)

    sub = data[y1:y2, x1:x2]
    if sub.size < 9:  # 太小无法拟合
        return cx, cy

    yy, xx = np.mgrid[y1:y2, x1:x2]

    # 初始猜测
    guess = (
        float(sub.max()),
        x1 + half,
        y1 + half,
        2.0,
        2.0,
        0.0,
        float(np.median(sub)),
    )

    # 参数边界
    bounds = (
        [0, x1, y1, 0.5, 0.5, -np.pi / 2, -np.inf],
        [np.inf, x2, y2, box_size / 2, box_size / 2, np.pi / 2, np.inf],
    )

    try:
        popt, _ = curve_fit(
            _gaussian_2d,
            (xx.ravel(), yy.ravel()),
            sub.ravel(),
            p0=guess,
            bounds=bounds,
            maxfev=5000,
        )
        return float(popt[1]), float(popt[2])  # x0, y0
    except Exception:
        return cx, cy


def compute_offset(
    cx_fit: float,
    cy_fit: float,
    cx_initial: float,
    cy_initial: float,
    pixel_scale: float = 0.05,
) -> dict:
    """计算 centroid 偏移量（像素和角秒）。

    Args:
        cx_fit, cy_fit: 拟合后的 centroid
        cx_initial, cy_initial: 初始中心
        pixel_scale: 像元尺度 (arcsec/pixel)

    Returns:
        {"dx_px": float, "dy_px": float, "dx_as": float, "dy_as": float}
    """
    dx_px = cx_fit - cx_initial
    dy_px = cy_fit - cy_initial
    return {
        "dx_px": float(dx_px),
        "dy_px": float(dy_px),
        "dx_as": float(dx_px * pixel_scale),
        "dy_as": float(dy_px * pixel_scale),
    }
