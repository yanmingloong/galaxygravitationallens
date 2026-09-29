"""PSF 构建（拆分为子模块）。 PSF construction (split into sub-modules).

子模块:
  - stars.py    场星检测与筛选
  - metrics.py  EE50 / 椭率度量
  - epsf.py     photutils ePSF 构建
  - analytic.py Moffat / Gaussian 解析 PSF
  - tinytim.py  TinyTim（Massey 栅格 + 直接编译）

本文件提供 build_psf 自动选优入口，并 re-export 所有公开函数，
保持 `from ...stages.psf import ...` 的调用方式不变。
"""
from __future__ import annotations
from typing import Optional, List, Tuple

import numpy as np

from .stars import detect_stars, filter_field_stars
from .metrics import psf_ee50, psf_ellipticity
from .epsf import build_epsf
from .analytic import (
    build_moffat_psf,
    build_gaussian_psf,
    build_elliptical_gaussian_psf,
)
from .tinytim import (
    build_tinytim_massey_psf,
    build_tinytim_direct_psf,
    MASSEY_AVAILABLE_BANDS,
)
from ...config import PSF_EE50_MIN_PX, PSF_EE50_MAX_PX, PSF_ELL_MAX


def build_psf(
    data: np.ndarray,
    field_stars: List[Tuple[float, float, float]],
    psf_fwhm_pix: float = 2.5,
    pixel_scale: float = 0.05,
    prefer_epsf: bool = True,
    psf_method: Optional[str] = None,
    band: Optional[str] = None,
    date_obs: Optional[str] = None,
    detector_pos: Optional[Tuple[int, int]] = None,
    instrument: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """自动选择并构建 PSF (对比选优)。

    自动选择 (psf_method=None 时):
      1. 生成候选: ePSF (经验, 捕捉真实像差) + TinyTim Massey (物理保底)
      2. 算指标: EE50 + 二阶矩椭率
      3. 硬门槛筛除: EE50 ∈ [0.8, 3.0]px 且椭率 < 0.15
      4. 选优: 达标 ePSF 优先 -> 达标 TinyTim(保底) -> Moffat 兜底
    指定 psf_method 则跳过自动选择。

    Args:
        data: 背景减除后的科学图像
        field_stars: 场星列表 [(x, y, flux), ...]
        psf_fwhm_pix: PSF FWHM 估计 (像素)
        pixel_scale: 像元尺度 (arcsec/pixel)
        prefer_epsf: 是否优先尝试 ePSF
        psf_method: 强制指定方法 ("epsf", "moffat", "gaussian", "gaussian_ell",
                    "tinytim_massey", "tinytim_direct")
        band: 波段 (TinyTim 需要)
        date_obs: 观测日期 (TinyTim 聚焦修正需要)
        detector_pos: 探测器像素 (TinyTim 位置映射需要)
        instrument: 仪器名 (TinyTim direct 需要)

    Returns:
        (psf_array, source_label)
    """
    # ── 强制指定方法 ──
    if psf_method == "epsf":
        if len(field_stars) > 0:
            result = build_epsf(data, field_stars)
            if result is not None:
                psf, label, n = result
                print(f"  PSF: ePSF ({n} stars)", flush=True)
                return psf, label

    elif psf_method == "moffat":
        fwhm_as = psf_fwhm_pix * pixel_scale
        psf, label = build_moffat_psf(fwhm_arcsec=fwhm_as, pixel_scale=pixel_scale)
        print(f"  PSF: Moffat (FWHM={fwhm_as:.2f}\", beta=3.0)", flush=True)
        return psf, label

    elif psf_method == "gaussian":
        fwhm_as = psf_fwhm_pix * pixel_scale
        psf, label = build_gaussian_psf(fwhm_arcsec=fwhm_as, pixel_scale=pixel_scale)
        print(f"  PSF: Gaussian (FWHM={fwhm_as:.2f}\")", flush=True)
        return psf, label

    elif psf_method == "gaussian_ell":
        fwhm_as = psf_fwhm_pix * pixel_scale
        psf, label = build_elliptical_gaussian_psf(fwhm_arcsec=fwhm_as, pixel_scale=pixel_scale)
        print(f"  PSF: Elliptical Gaussian (FWHM={fwhm_as:.2f}\", e=0.05)", flush=True)
        return psf, label

    elif psf_method == "tinytim_massey":
        if instrument and "ACS" not in str(instrument).upper():
            print(f"  PSF: TinyTim Massey skipped (instrument={instrument}, ACS only)", flush=True)
        elif band and band in MASSEY_AVAILABLE_BANDS:
            result = build_tinytim_massey_psf(
                band=band, date_obs=date_obs or "",
                detector_pos=detector_pos, pixel_scale=pixel_scale,
            )
            if result is not None:
                psf, label = result
                print(f"  PSF: TinyTim Massey ({label})", flush=True)
                return psf, label

    elif psf_method == "tinytim_direct":
        if band and instrument:
            result = build_tinytim_direct_psf(
                band=band, instrument=instrument,
                date_obs=date_obs or "",
                detector_pos=detector_pos or (400, 400),
                pixel_scale=pixel_scale,
            )
            if result is not None:
                psf, label = result
                print(f"  PSF: TinyTim Direct ({label})", flush=True)
                return psf, label

    # ── 自动对比选优 (TinyTim 保底 + ePSF 候选, EE50/椭率门槛) ──
    candidates = []  # [(name, psf, label, ee50, ell, valid)]

    # 1. ePSF (经验候选, 捕捉真实像差)
    if prefer_epsf and len(field_stars) > 0:
        result = build_epsf(data, field_stars)
        if result is not None:
            psf, label, n_used = result
            ee50, fwhm = psf_ee50(psf)
            ell = psf_ellipticity(psf)
            ok = (PSF_EE50_MIN_PX <= ee50 <= PSF_EE50_MAX_PX) and (ell < PSF_ELL_MAX)
            candidates.append(("epsf", psf, label, ee50, ell, ok))
            print(f"  PSF candidate ePSF ({n_used} stars): EE50={ee50:.2f}px, "
                  f"ell={ell:.3f} -> {'OK' if ok else 'REJECT'}", flush=True)

    # 2. TinyTim Massey (物理保底, 不依赖场星)
    if band and band in MASSEY_AVAILABLE_BANDS:
        result = build_tinytim_massey_psf(
            band=band, date_obs=date_obs or "",
            detector_pos=detector_pos, pixel_scale=pixel_scale,
        )
        if result is not None:
            psf, label = result
            ee50, fwhm = psf_ee50(psf)
            ell = psf_ellipticity(psf)
            ok = (PSF_EE50_MIN_PX <= ee50 <= PSF_EE50_MAX_PX) and (ell < PSF_ELL_MAX)
            candidates.append(("tinytim_massey", psf, label, ee50, ell, ok))
            print(f"  PSF candidate TinyTim Massey: EE50={ee50:.2f}px, "
                  f"ell={ell:.3f} -> {'OK' if ok else 'REJECT'}", flush=True)

    # 3. 选优: 达标 ePSF 优先(捕捉真实像差) -> 达标 TinyTim(保底)
    for pref in ("epsf", "tinytim_massey"):
        for name, psf, label, ee50, ell, ok in candidates:
            if name == pref and ok:
                print(f"  PSF selected: {name} ({label}, EE50={ee50:.2f}px, ell={ell:.3f})",
                      flush=True)
                return psf, label

    # 4. Moffat 兜底 (无达标候选)
    fwhm_as = psf_fwhm_pix * pixel_scale
    psf, label = build_moffat_psf(fwhm_arcsec=fwhm_as, pixel_scale=pixel_scale)
    print(f"  PSF: Moffat fallback (FWHM={fwhm_as:.2f}\", beta=3.0)", flush=True)
    return psf, label

__all__ = [
    "detect_stars", "filter_field_stars",
    "psf_ee50", "psf_ellipticity",
    "build_epsf",
    "build_moffat_psf", "build_gaussian_psf", "build_elliptical_gaussian_psf",
    "build_tinytim_massey_psf", "build_tinytim_direct_psf",
    "build_psf",
]
