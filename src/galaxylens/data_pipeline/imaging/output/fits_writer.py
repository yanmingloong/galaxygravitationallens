"""
FITS 标准化输出
===============
将处理后的 SCI/Noise/PSF/Mask 写入标准 FITS 文件。

功能:
  1. 按统一命名规范写入 FITS 文件
  2. SCI → float32, Noise → float32, PSF → float32, Mask → uint8
  3. 自动创建输出目录

命名规范:
  {short_name}_{band_lower}_{type}.fits
  例: j2141_f814w_sci.fits, 1630+4520_f435w_noise.fits

使用示例:
  >>> from galaxylens.data_pipeline.imaging.output.fits_writer import write_processed_band
  >>> write_processed_band(output_dir, "J2141-0001", "F814W", sci=sci, noise=noise, psf=psf)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from astropy.io import fits


# ══════════════════════════════════════════════════════════════════════
# 命名工具
# ══════════════════════════════════════════════════════════════════════

def make_short_name(lens_id: str) -> str:
    """将透镜ID转换为短文件名前缀。

    Args:
        lens_id: 如 "J2141-0001", "J1630+4520"

    Returns:
        短名: 如 "j21410001", "1630+4520"
    """
    # 去除 SDSS/SDSSJ 前缀
    name = lens_id
    for prefix in ["SDSSJ", "SDSS_", "SDSS"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.lower().replace("-", "")


def make_prefix(lens_id: str, band: str) -> str:
    """生成输出文件名前缀。

    Args:
        lens_id: 透镜标识符
        band: 波段名 (如 "F814W")

    Returns:
        前缀: 如 "1630+4520_f814w"
    """
    short = lens_id.replace("SDSSJ", "").replace("SDSS_", "")
    return f"{short.lower()}_{band.lower()}"


# ══════════════════════════════════════════════════════════════════════
# 写入接口
# ══════════════════════════════════════════════════════════════════════

def write_processed_band(
    output_dir: Path,
    lens_id: str,
    band: str,
    sci: np.ndarray,
    noise: np.ndarray,
    psf: np.ndarray,
    starmask: np.ndarray,
    cr_mask: np.ndarray | None = None,
) -> dict[str, Path]:
    """将一个透镜+波段的处理结果写入 FITS 文件。

    Args:
        output_dir: 输出目录 (如 processed/J2141-0001/v2_flc_whtonly/)
        lens_id: 透镜标识符
        band: 波段名
        sci: 背景减除后的科学图像
        noise: 噪声图 σ-map
        psf: 归一化 PSF kernel
        starmask: 恒星 mask (Boolean)
        cr_mask: 宇宙线 mask (Boolean, 可选)

    Returns:
        {type: filepath} 字典

    Output files:
        {prefix}_sci.fits, {prefix}_noise.fits, {prefix}_psf.fits,
        {prefix}_starmask.fits, [{prefix}_crmask.fits]
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = make_prefix(lens_id, band)

    files: dict[str, Path] = {}

    # SCI
    sp = output_dir / f"{prefix}_sci.fits"
    fits.writeto(sp, sci.astype(np.float32), overwrite=True)
    files["sci"] = sp

    # Noise
    np_ = output_dir / f"{prefix}_noise.fits"
    fits.writeto(np_, noise.astype(np.float32), overwrite=True)
    files["noise"] = np_

    # PSF
    pp = output_dir / f"{prefix}_psf.fits"
    fits.writeto(pp, psf.astype(np.float32), overwrite=True)
    files["psf"] = pp

    # Star mask
    mp = output_dir / f"{prefix}_starmask.fits"
    fits.writeto(mp, starmask.astype(np.uint8), overwrite=True)
    files["starmask"] = mp

    # CR mask (optional)
    if cr_mask is not None:
        cp = output_dir / f"{prefix}_crmask.fits"
        fits.writeto(cp, cr_mask.astype(np.uint8), overwrite=True)
        files["crmask"] = cp

    return files


def write_combined_starmask(
    output_dir: Path,
    lens_id: str,
    starmask: np.ndarray,
) -> Path:
    """写入合并后的恒星 mask（所有波段 OR）。

    Args:
        output_dir: 输出目录
        lens_id: 透镜标识符
        starmask: 合并后的 Boolean mask

    Returns:
        输出文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    short = make_short_name(lens_id)
    path = output_dir / f"{short}_starmask.fits"
    fits.writeto(path, starmask.astype(np.uint8), overwrite=True)
    return path
