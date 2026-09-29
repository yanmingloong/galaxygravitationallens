"""photutils EPSFBuilder 超采样 ePSF 构建（Anderson & King 2000）。 photutils EPSFBuilder oversampled ePSF construction (Anderson & King 2000)."""
from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import EPSFBuilder, extract_stars
from scipy.ndimage import zoom

from ...config import PSF_STAMP_SIZE


def build_epsf(
    data: np.ndarray,
    star_coords: List[Tuple[float, float, float]],
    stamp_size: int = PSF_STAMP_SIZE,
    oversampling: int = 4,
    maxiters: int = 10,
) -> Optional[Tuple[np.ndarray, str, int]]:
    """使用 photutils EPSFBuilder 构建超采样有效 PSF。

    基于 Anderson & King (2000) 方法，迭代构建 ePSF 并精修星像质心。
    最终降采样回探测器像素尺度。

    Args:
        data: 背景减除后的 2D 图像
        star_coords: 场星坐标列表 [(x, y, flux), ...]
        stamp_size: 输出 PSF 尺寸 (奇数像素)
        oversampling: 超采样因子 (默认 4)
        maxiters: 最大迭代次数 (默认 10)

    Returns:
        (psf_normalized, source_label, n_stars_used) 或 None (失败时)
    """
    if len(star_coords) == 0:
        return None

    try:
        star_table = Table(
            names=["x", "y", "flux"],
            data=[
                [s[0] for s in star_coords],
                [s[1] for s in star_coords],
                [s[2] for s in star_coords],
            ],
        )

        nddata = NDData(data=data.astype(np.float64))
        extract_size = min(stamp_size + 4, min(data.shape) // 2)
        stars = extract_stars(nddata, star_table, size=extract_size)

        if len(stars) == 0:
            return None

        builder = EPSFBuilder(
            oversampling=oversampling,
            maxiters=maxiters,
            progress_bar=False,
        )
        epsf_result, fitted_stars = builder(stars)

        epsf_data = epsf_result.data
        ds_factor = 1.0 / oversampling
        psf_ds = zoom(epsf_data, ds_factor, order=1)

        if psf_ds.shape[0] >= stamp_size:
            start = (psf_ds.shape[0] - stamp_size) // 2
            psf_out = psf_ds[start:start + stamp_size, start:start + stamp_size]
        else:
            psf_out = np.zeros((stamp_size, stamp_size), dtype=np.float64)
            start = (stamp_size - psf_ds.shape[0]) // 2
            s = psf_ds.shape[0]
            psf_out[start:start + s, start:start + s] = psf_ds

        psf_out = np.maximum(psf_out, 0)
        psf_out /= psf_out.sum()

        n_used = len(fitted_stars)
        source_label = f"epsf_{oversampling}x_{n_used}stars"

        return psf_out, source_label, n_used

    except Exception:
        return None

