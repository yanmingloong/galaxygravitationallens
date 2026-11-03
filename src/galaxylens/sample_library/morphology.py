"""Image morphology measurement for lens identification (图像形状识别).

从 FITS 图像直接测形态学参数，判断盘/椭圆。当 catalog 参数（lens_type_label /
n_sersic）缺失或不可靠时，作为识别层的图像兜底。纯 numpy + astropy（惰性 import），
不依赖 mgefit / autolens。

判据（阈值经真实样本标定，2026-11-03）：
  C = 5 * log10(r80 / r20)  —— 光集中度（主判据，最稳健）
      J2141 edge-on S0 盘 C = 1.58；J1630 SLACS 椭圆 C = 4.21
      阈值：C < 3.1 判盘、C > 3.4 判椭圆，中间区间回落轴比
  q = b / a                  —— 轴比（光度二阶矩，辅助）
      注意：透镜弧 + PSF 会把 q 拉圆（J2141/J1630 都测到 q≈0.89，不可靠），
      故 q 只在 C 中间区间作辅助，不作主判据

方法：
  - 背景：外圈 annulus 中值估计 + 减除
  - 中心：光重心（一阶矩）
  - C：圆形孔径累积通量 20% / 80% 半径
  - q/PA：光度二阶矩（惯性张量）特征值比 / 特征向量方向

用法：
  morph = measure_morphology("xxx_f814w_sci.fits")
  lens_type, basis = classify_morphology(morph)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# 光集中度阈值（需真实样本标定；初始值按 Sérsic n 的典型 C 区间）
C_DISK_MAX = 3.1      # C 低于此判盘
C_ELLIP_MIN = 3.4     # C 高于此判椭圆；中间用轴比辅助


def measure_morphology(
    fits_path,
    center=None,
    rmax_pix: Optional[float] = None,
) -> dict:
    """从 FITS 图像测形态学参数，返回 dict。

    参数：
      fits_path: 科学图像 FITS 路径（sci 帧）
      center:    透镜中心 (x, y)，缺省用光重心
      rmax_pix:  测光半径上限（像素），缺省 = min(shape)/2 * 0.8（避开弧/外围）
    """
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float64)

    if data.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {data.shape}")

    bg = _estimate_background(data)
    img = np.clip(data - bg, 0.0, None)

    if center is None:
        center = _centroid(img)
    center = (float(center[0]), float(center[1]))

    if rmax_pix is None:
        rmax_pix = float(min(data.shape)) / 2.0 * 0.8

    r20, r80, C = _concentration(img, center, rmax_pix)
    q, pa = _axis_ratio(img, center, rmax_pix)

    return {
        "C": float(C),
        "q": float(q),
        "pa_deg": float(pa),
        "r20_pix": float(r20),
        "r80_pix": float(r80),
        "center": center,
        "background": float(bg),
        "rmax_pix": float(rmax_pix),
    }


def classify_morphology(morph: dict) -> Tuple[str, str]:
    """C + q -> ``(lens_type, basis)``（disk / elliptical）。"""
    C = morph["C"]
    q = morph.get("q", 1.0)

    if C < C_DISK_MAX:
        return "disk", f"C={C:.2f}_disk_like"
    if C > C_ELLIP_MIN:
        return "elliptical", f"C={C:.2f}_bulge_like"

    # 中间区间：用轴比辅助（edge-on 盘很扁）
    if q < 0.5:
        return "disk", f"C={C:.2f}_q={q:.2f}_edge_on_disk"
    return "elliptical", f"C={C:.2f}_q={q:.2f}_ambiguous_ellip"


def _estimate_background(data: np.ndarray) -> float:
    """外圈 annulus（外 10%）中值作背景。"""
    r = _radial_grid(data)
    rmax = r.max()
    annulus = data[(r > 0.85 * rmax) & (r <= rmax)]
    return float(np.median(annulus)) if annulus.size else 0.0


def _radial_grid(data: np.ndarray) -> np.ndarray:
    ny, nx = data.shape
    yy, xx = np.indices((ny, nx))
    return np.sqrt((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2)


def _centroid(img: np.ndarray) -> Tuple[float, float]:
    """光重心（一阶矩）。"""
    ny, nx = img.shape
    yy, xx = np.indices((ny, nx))
    w = img.sum()
    if w <= 0:
        return (nx / 2.0, ny / 2.0)
    xc = (img * xx).sum() / w
    yc = (img * yy).sum() / w
    return (float(xc), float(yc))


def _concentration(img: np.ndarray, center, rmax_pix: float) -> Tuple[float, float, float]:
    """圆形孔径累积通量 -> (r20, r80, C)。"""
    ny, nx = img.shape
    yy, xx = np.indices((ny, nx))
    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)

    mask = r <= rmax_pix
    total = img[mask].sum()
    if total <= 0:
        return (np.nan, np.nan, np.nan)

    radii = np.arange(1.0, rmax_pix, 0.5)
    cum = np.array([img[r <= rr].sum() for rr in radii])
    frac = cum / total

    r20 = radii[np.argmin(np.abs(frac - 0.20))]
    r80 = radii[np.argmin(np.abs(frac - 0.80))]
    C = 5.0 * np.log10(r80 / r20) if r20 > 0 else np.nan
    return (float(r20), float(r80), float(C))


def _axis_ratio(img: np.ndarray, center, rmax_pix: float) -> Tuple[float, float]:
    """光度二阶矩（惯性张量）-> (q=b/a, PA_deg)。"""
    ny, nx = img.shape
    yy, xx = np.indices((ny, nx))
    dx = xx - center[0]
    dy = yy - center[1]
    r = np.sqrt(dx ** 2 + dy ** 2)

    w = img * (r <= rmax_pix)
    wsum = w.sum()
    if wsum <= 0:
        return (1.0, 0.0)

    Ixx = (w * dx * dx).sum() / wsum
    Iyy = (w * dy * dy).sum() / wsum
    Ixy = (w * dx * dy).sum() / wsum

    a = (Ixx + Iyy) / 2.0
    b = np.sqrt(((Ixx - Iyy) / 2.0) ** 2 + Ixy ** 2)
    lam1 = a + b
    lam2 = a - b
    q = float(np.sqrt(lam2 / lam1)) if lam1 > 0 else 1.0
    pa = float(0.5 * np.degrees(np.arctan2(2.0 * Ixy, Ixx - Iyy)))
    return (q, pa)
