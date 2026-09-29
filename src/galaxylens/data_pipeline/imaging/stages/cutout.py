"""
WCS 天文裁剪
===========
根据透镜天球坐标裁剪图像到指定视场。

功能:
  1. 基于 WCS 坐标的精确裁剪（优先）
  2. 像素坐标 fallback（WCS 缺失或失败时）
  3. 支持坐标偏移（盘星系 centroid 精调）

处理流程:
    输入: sci_data + wcs + ra/dec + cutout_size
           │
           ├─ WCS 可用？
           │   ├─ YES → Cutout2D(wcs=wcs) 天球坐标裁剪
           │   └─ NO  → 像素坐标 fallback
           │
           └─ 返回: cutout_sci, cutout_wht (可选)

使用示例:
  >>> from galaxylens.data_pipeline.imaging.stages.cutout import wcs_cutout
  >>> sci_cut = wcs_cutout(sci, wcs, ra=247.6, dec=45.3, size_arcsec=15, pixel_scale=0.05)
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u


# ══════════════════════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════════════════════

def wcs_cutout(
    data: np.ndarray,
    wcs: Optional[WCS],
    ra: float,
    dec: float,
    size_arcsec: float,
    pixel_scale: float,
    offset_arcsec: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """基于 WCS 坐标裁剪图像到指定天球位置和视场。

    Args:
        data: 输入 2D 图像
        wcs: WCS 对象（可为 None 触发 pixel fallback）
        ra: 目标赤经 (deg)
        dec: 目标赤纬 (deg)
        size_arcsec: 裁剪视场大小 (角秒)
        pixel_scale: 像元尺度 (arcsec/pixel)
        offset_arcsec: 坐标偏移量 (dRA, dDec) 角秒，用于 centroid 精调

    Returns:
        裁剪后的 2D 数组

    Note:
        当 WCS 裁剪失败时，自动回退到图像中心裁剪。
    """
    # 应用坐标偏移
    if offset_arcsec != (0.0, 0.0):
        import numpy as np
        ra_eff = ra + offset_arcsec[0] / 3600.0 / np.cos(np.deg2rad(dec))
        dec_eff = dec + offset_arcsec[1] / 3600.0
    else:
        ra_eff, dec_eff = ra, dec

    skycoord = SkyCoord(ra=ra_eff, dec=dec_eff, unit="deg", frame="icrs")
    cutout_size_pix = int(size_arcsec / pixel_scale)

    # 优先 WCS 裁剪
    if wcs is not None:
        try:
            cutout = Cutout2D(
                data, skycoord, cutout_size_pix,
                wcs=wcs, mode="trim",
            )
            return cutout.data
        except Exception:
            pass  # fallback below

    # Fallback: 像素坐标裁剪（图像中心）
    cx, cy = data.shape[1] // 2, data.shape[0] // 2
    try:
        if wcs is not None:
            xpix, ypix = wcs.world_to_pixel(skycoord)
        else:
            xpix, ypix = cx, cy
        cutout = Cutout2D(
            data, (float(xpix), float(ypix)),
            cutout_size_pix, mode="trim",
        )
        return cutout.data
    except Exception:
        # 最终fallback：直接中心裁剪
        half = cutout_size_pix // 2
        y1, y2 = max(0, cy - half), min(data.shape[0], cy + half)
        x1, x2 = max(0, cx - half), min(data.shape[1], cx + half)
        return data[y1:y2, x1:x2].copy()


def wcs_cutout_with_wht(
    sci: np.ndarray,
    wht: Optional[np.ndarray],
    wcs: Optional[WCS],
    ra: float,
    dec: float,
    size_arcsec: float,
    pixel_scale: float,
    offset_arcsec: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """同时对 SCI 和 WHT 进行 WCS 裁剪。

    Args:
        sci: 科学图像
        wht: 权重图（可为 None）
        wcs: WCS 对象
        ra, dec: 天球坐标
        size_arcsec: 视场大小
        pixel_scale: 像元尺度
        offset_arcsec: 坐标偏移

    Returns:
        (sci_cutout, wht_cutout) — wht_cutout 为 None 若输入为 None
    """
    sci_cut = wcs_cutout(sci, wcs, ra, dec, size_arcsec, pixel_scale, offset_arcsec)

    if wht is None:
        return sci_cut, None

    wht_cut = wcs_cutout(wht, wcs, ra, dec, size_arcsec, pixel_scale, offset_arcsec)
    return sci_cut, wht_cut
