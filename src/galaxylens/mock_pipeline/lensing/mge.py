"""Light MGE fit from a mock image (mgefit), the arc-building mass model's front end.

从 mock 图（SKIRT noiseless ``total.fits`` 或 fake-HST ``data.fits``）拟合光 MGE
（surface brightness），产出 ``(surf, sig_as, q_obs, total_lum)``，供
:func:`~galaxylens.mock_pipeline.lensing.arc.build_lens_galaxy` 乘 M/L 转质量 MGE。

这是 arc stage 的前置组件（之前缺失，靠外部 MGE 产物兜底）；提炼自 MF 系列的
``mf1a_mge.py``，逻辑冻结：两遍截断二阶矩定中心/eps/theta → ``sectors_photometry``
椭圆环测光 → ``fit_sectors`` MGE 拟合。``mgefit`` 惰性 import（不装则 import 本模块
仍可用，只在实际拟合时报错）。

关键口径（与 ``mf1a_mge.py`` 一致，别改）：
  - ``sigmapsf_px`` 用 PSF 二阶矩 σ 去卷积，避免 PyAutoLens 再卷一次造成双重 PSF；
  - 盘外暗恒星晕会拟合出 σ~20 kpc 宽高斯，必须截断到 R < 2.5×R50；
  - ``qbounds`` 下限 = sqrt(cos²i + q_int_min² sin²i)，保证去投影后 q_int ≥ 0.05。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def measure_geometry(img, rmax_px):
    """两遍 2D 二阶矩 -> (col_c, row_c, eps, theta_deg)。

    第一遍全图粗定位中心；第二遍只在 R<rmax_px 截断区精测 eps/theta，排除盘外
    暗恒星晕（否则 eps 被晕拉圆、或被激进阈值切薄）。
    """
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    # 第一遍：全图粗中心（温和阈值）
    sel0 = img > 0.01 * img.max()
    m0 = img[sel0].astype(np.float64)
    x0 = xx[sel0].astype(np.float64)
    y0 = yy[sel0].astype(np.float64)
    xc0 = float((x0 * m0).sum() / m0.sum())
    yc0 = float((y0 * m0).sum() / m0.sum())
    # 第二遍：截断区精测
    r2 = (xx - xc0) ** 2 + (yy - yc0) ** 2
    sel = (img > 0.05 * img.max()) & (r2 < rmax_px ** 2)
    m = img[sel].astype(np.float64)
    x = xx[sel].astype(np.float64)
    y = yy[sel].astype(np.float64)
    xc = float((x * m).sum() / m.sum())
    yc = float((y * m).sum() / m.sum())
    dx = x - xc
    dy = y - yc
    Sxx = (dx * dx * m).sum() / m.sum()
    Syy = (dy * dy * m).sum() / m.sum()
    Sxy = (dx * dy * m).sum() / m.sum()
    evals, evecs = np.linalg.eigh(np.array([[Sxx, Sxy], [Sxy, Syy]]))
    eps = float(1.0 - np.sqrt(evals[0] / evals[1]))          # 椭圆率
    theta = float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])))
    return xc, yc, eps, theta


def read_pixel_scale_kpc(skirt_params_xml: str) -> float:
    """从 SKIRT parameters.xml 读像素尺度 (kpc/px) = fieldOfViewX / numPixelsX。"""
    import re

    xml = Path(skirt_params_xml).read_text()
    fov_pc = float(re.search(r'fieldOfViewX="([\d.eE+-]+) pc"', xml).group(1))
    npx = int(re.search(r'numPixelsX="(\d+)"', xml).group(1))
    return fov_pc / 1000.0 / npx          # pc -> kpc


def read_psf_sigma_px(psf_path: str) -> float:
    """从 PSF FITS 算圆形高斯 σ（px），供 mgefit sigmapsf 去卷积。"""
    from astropy.io import fits

    psf = fits.getdata(psf_path).astype(np.float64)
    ny, nx = psf.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    psf_n = psf / psf.sum()
    var = float((r2 * psf_n).sum())
    return float(np.sqrt(var / 2.0))       # 1D 高斯 σ = sqrt(<r²>/2)


def fit_light_mge(
    fits_path,
    pixel_scale_kpc: float,
    r50_kpc: float,
    qbounds,
    kpc_per_arcsec: float,
    ngauss: int = 8,
    sigmapsf_px: float = 0.0,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """从 mock 图拟合光 MGE -> ``(surf_pc2, sig_as, q_obs, total_lum)``。

    ``surf_pc2`` 是峰值面密度（flux/pc²）；``sig_as`` 是各高斯宽度（arcsec）；
    ``q_obs`` 是投影轴比；``total_lum`` 是相对总光度（SKIRT 通量单位）。返回的
    四元组对齐 MF 系列 ``load_mge`` 的输出，直接喂给 ``build_lens_galaxy``。
    """
    import mgefit as mge
    from astropy.io import fits

    d = fits.getdata(fits_path)
    img = d[0].astype(np.float64) if d.ndim == 3 else d.astype(np.float64)

    rmax_px = 2.5 * r50_kpc / pixel_scale_kpc
    col_c, row_c, eps, theta = measure_geometry(img, rmax_px)

    mask = img > 0
    s = mge.sectors_photometry(img, eps, theta, row_c, col_c,
                               minlevel=0.0, mask=mask, plot=False)
    sel = s.radius < rmax_px

    m = mge.fit_sectors(s.radius[sel], s.angle[sel], s.counts[sel], eps,
                        ngauss=ngauss, sigmapsf=sigmapsf_px,
                        qbounds=qbounds, scale=pixel_scale_kpc,
                        plot=False, linear=False, quiet=quiet)
    total_counts, sigma_px, q_obs = m.sol

    sigma_kpc = np.asarray(sigma_px, dtype=np.float64) * pixel_scale_kpc
    sigma_pc = sigma_kpc * 1000.0
    q_obs = np.asarray(q_obs, dtype=np.float64)
    total_counts = np.asarray(total_counts, dtype=np.float64)
    surf = total_counts / (2.0 * np.pi * sigma_pc ** 2 * q_obs)   # flux/pc²
    sig_as = sigma_kpc / kpc_per_arcsec
    total_lum = float(total_counts.sum())
    return surf, sig_as, q_obs, total_lum
