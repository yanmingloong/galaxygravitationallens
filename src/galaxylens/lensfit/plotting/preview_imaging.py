"""PyAutoLens 风格的成像数据预览图（可选，需 autolens）。
PyAutoLens-style imaging data preview (optional, requires autolens).

把预处理产出的 sci/noise/psf FITS 画成 PyAutoLens 三联预览图，
供快速检查数据质量。

此模块依赖 autolens，因此放在 lensfit/plotting/ 而非 data_pipeline/
（data_pipeline 只依赖 astropy/photutils，保持依赖干净）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_imaging_preview(
    sci_path: Path,
    noise_path: Path,
    psf_path: Path,
    pixel_scale: float,
    starmask: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    cutout_size_arcsec: float = 15.0,
) -> None:
    """用 PyAutoLens 绘制 sci/noise/psf 三联预览图。

    Args:
        sci_path: SCI FITS 路径
        noise_path: 噪声图 FITS 路径
        psf_path: PSF FITS 路径
        pixel_scale: 像元尺度 (arcsec/pix)
        starmask: 恒星 mask 数组（可选，用于叠加到圆形 mask 上）
        output_path: 输出 PNG 路径（默认与 sci_path 同目录、同 stem）
        cutout_size_arcsec: 裁剪视场直径 (角秒)
    """
    import autolens as al

    # 构造 PyAutoLens Imaging 数据集
    dataset = al.Imaging.from_fits(
        data_path=sci_path,
        noise_map_path=noise_path,
        psf_path=psf_path,
        pixel_scales=pixel_scale,
    )

    # 圆形 mask + 恒星 mask 合并
    shape_native = starmask.shape if starmask is not None else dataset.shape_native
    circ_mask = al.Mask2D.circular(
        shape_native=shape_native,
        pixel_scales=pixel_scale,
        radius=cutout_size_arcsec / 2 * 0.98,
    )
    circ_native = np.array(circ_mask.native, dtype=bool)
    if starmask is not None:
        circ_native = circ_native | starmask
    combined_mask = al.Mask2D(mask=circ_native, pixel_scales=pixel_scale)
    dataset = dataset.apply_mask(mask=combined_mask)

    # 输出路径
    if output_path is None:
        output_path = sci_path.with_suffix("")
    output_dir = output_path.parent
    output_stem = output_path.name

    # 绘制 subplot（sci / noise / psf / mask）
    from autoarray.plot.mat_plot.output import Output
    from autoarray.plot.mat_plot.two_d import MatPlot2D

    mat_plot = MatPlot2D(
        output=Output(path=str(output_dir), filename=output_stem, format="png")
    )
    plotter = al.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
    plotter.subplot_dataset()
