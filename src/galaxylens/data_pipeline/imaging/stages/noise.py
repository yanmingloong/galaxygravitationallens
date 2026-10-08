"""
噪声图生成 Noise map generation
==========
从 HST 权重图(WHT)和/或科学图像计算 σ-map（噪声标准差图）。

方法列表:
  1. noise_from_wht()             — CALACS标准：σ = 1/√(WHT)
  2. noise_from_wht_plus_poisson()— WHT + Poisson散粒噪声 (DRC)
  3. noise_from_wht_poisson_calib()— + 双轮σ-clipping经验标定 (盘星系final)
  4. noise_poisson_readnoise()    — Poisson + 读出噪声 (无WHT回退)
  5. noise_stacked()              — 多曝光堆叠: σ/√N

参考文献:
  - CALACS ISR 2015-03: ACS WHT weighting scheme
  - MultiDrizzle Handbook: WHT for drizzled products
  - preprocess_swells.py v1.3: empirical noise calibration method

处理流程:
    输入: sci_bgsub + wht (可选)
           │
           ├─ WHT 存在？
           │   ├─ YES → 输入类型？
           │   │   ├─ FLT/FLC → noise_from_wht()
           │   │   │              └─ 多曝光? → noise_stacked()
           │   │   └─ DRC/DRZ  → noise_from_wht_plus_poisson()
           │   │                  └─ 标定? → noise_from_wht_poisson_calibrated()
           │   └─ NO  → noise_poisson_readnoise()
           │
           └─ 输出: noise_map (σ-map, 2D float32)

使用示例:
  >>> from galaxylens.data_pipeline.imaging.stages.noise import noise_from_wht
  >>> noise = noise_from_wht(wht_data)
"""

from __future__ import annotations
from typing import Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# 方法 1: WHT-only (CALACS 标准)
# ══════════════════════════════════════════════════════════════════════

def noise_from_wht(
    wht_data: np.ndarray,
    n_exposures: int = 1,
) -> np.ndarray:
    """CALACS 标准噪声：从 WHT 权重图直接计算 σ-map。

    CALACS 管线的 WHT extension 包含所有噪声成分的逆方差：
    读出噪声 + 暗流 + 平场误差 + 泊松散粒噪声。
    因此 noise = 1/√(WHT)，无需额外加泊松项。

    Args:
        wht_data: WHT 权重图 (2D array)，单位 electrons⁻²·s²
        n_exposures: 堆叠曝光数。>1 时 noise /= √(n_exposures)

    Returns:
        噪声图 σ-map (2D float32)，单位 electrons/s

    Raises:
        ValueError: 若 wht_data 全为非正值

    Note:
        仅适用于 ACS/WFC 的 FLT/FLC 产品。
        DRC 的 WHT 来自 drizzle，含像素间相关性，此方法会低估噪声。
    """
    # 处理非正WHT值
    wht_positive = wht_data[wht_data > 0]
    if len(wht_positive) == 0:
        raise ValueError("WHT data contains no positive values")

    wht_median = float(np.median(wht_positive))
    wht_safe = np.where(wht_data > 0, wht_data, wht_median)

    noise = 1.0 / np.sqrt(np.maximum(wht_safe, 1e-10))

    # 处理无穷值
    noise = np.where(np.isfinite(noise), noise, np.median(noise[np.isfinite(noise)]))

    # 多曝光缩放
    if n_exposures > 1:
        noise = noise / np.sqrt(float(n_exposures))

    return noise.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 方法 2: WHT + Poisson (DRC 产品)
# ══════════════════════════════════════════════════════════════════════

def noise_from_wht_plus_poisson(
    wht_data: np.ndarray,
    sci_bgsub: np.ndarray,
) -> np.ndarray:
    """WHT + Poisson 散粒噪声（用于 DRC 产品）。

    DRC 的 WHT 图来自 drizzle 过程（1/σ²），
    需额外考虑源光子泊松噪声。

    警告：此方法会 double-count 天空背景噪声，
    因为 MultiDrizzle WHT 已包含天空方差。
    对 DRC 推荐使用 noise_from_wht_poisson_calibrated()。

    Args:
        wht_data: WHT 权重图
        sci_bgsub: 背景减除后的科学图像 (electrons/s)

    Returns:
        噪声图 σ-map (2D float32)
    """
    # WHT 噪声
    noise_wht = noise_from_wht(wht_data)

    # Poisson 噪声：σ_poisson = √(max(sci, 0))
    noise_poisson = np.sqrt(np.maximum(sci_bgsub, 0))

    # 平方和开方
    noise = np.sqrt(noise_wht**2 + noise_poisson**2)
    return noise.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 方法 3: WHT + Poisson + 经验标定 (盘星系 DRC final)
# ══════════════════════════════════════════════════════════════════════

def noise_from_wht_poisson_calibrated(
    wht_data: np.ndarray,
    sci_bgsub: np.ndarray,
    min_bg_pixels: int = 20,
) -> np.ndarray:
    """WHT + Poisson + 双轮 σ-clipping 经验全局缩放标定。

    修正 DRC 产品中 WHT+Poisson 双重计数天空背景的问题。

    算法：
      1. 先算 noise_raw = √(σ²_wht + σ²_poisson)
      2. 用双轮 sigma-clipping 识别背景像素
      3. 在背景区比较 SCI 的 std 与 noise_raw 的 median
      4. scale = σ_sci_bg / median(noise_raw_bg)
      5. noise_final = noise_raw × scale

    Args:
        wht_data: WHT 权重图
        sci_bgsub: 背景减除后的科学图像
        min_bg_pixels: 背景区最少像素数，低于此值跳过标定

    Returns:
        经验标定后的噪声图 σ-map (2D float32)

    References:
        preprocess_swells.py v1.3
        fix_noise_maps_disk_v2.py
    """
    # Step 1: 原始 WHT+Poisson
    noise_raw = noise_from_wht_plus_poisson(wht_data, sci_bgsub).astype(np.float64)

    # Step 2: 第一轮 sigma-clipping (MAD-based)
    sci_flat = sci_bgsub.ravel()
    med_sci = float(np.median(sci_flat))
    mad_sci = float(np.median(np.abs(sci_flat - med_sci)))
    sigma_est = mad_sci * 1.4826

    lo = med_sci - 3.0 * sigma_est
    hi = med_sci + 5.0 * sigma_est
    bg1 = (sci_bgsub > lo) & (sci_bgsub < hi)

    if bg1.sum() < min_bg_pixels:
        return noise_raw.astype(np.float32)

    # Step 3: 第二轮 sigma-clipping (在背景子集内)
    bg1_sci = sci_bgsub[bg1]
    bg_med1 = float(np.median(bg1_sci))
    bg_std1 = float(np.std(bg1_sci))

    bg2 = np.abs(sci_bgsub - bg_med1) < 2.5 * bg_std1
    bg_final = bg1 & bg2

    if bg_final.sum() < min_bg_pixels:
        return noise_raw.astype(np.float32)

    # Step 4: 缩放因子
    sci_std_bg = float(np.std(sci_bgsub[bg_final]))
    noise_med_bg = float(np.median(noise_raw[bg_final]))

    if noise_med_bg < 1e-15 or sci_std_bg < 1e-15:
        return noise_raw.astype(np.float32)

    scale = np.clip(sci_std_bg / noise_med_bg, 0.01, 100.0)
    noise = noise_raw * scale

    return noise.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Method 3.5: pure WHT + global rescaling (Barnabè-style, disk-galaxy FLC)
# ══════════════════════════════════════════════════════════════════════

def noise_from_wht_rescaled(
    wht_data: np.ndarray,
    sci_bgsub: np.ndarray,
    min_bg_pixels: int = 20,
) -> np.ndarray:
    """Pure WHT noise × single global rescaling factor (Barnabè+2012 style).

    Pure WHT noise (σ = 1/√WHT) is physically correct but too small for
    disk-galaxy Nautilus fits: model systematic errors (≳16 e⁻/s) dwarf the
    measurement noise (≈0.35 e⁻/s), blowing up χ²/px to ≈2167. Barnabè+2012
    rescales the noise by a single global factor so that reduced χ² ≈ 1. This
    preprocessing-stage approximation uses the background region to estimate
    that factor:

        scale = σ_sci_bg / median(noise_raw_bg)

    Unlike the signal-dependent bug noise (√(1/WHT + SCI)), this method scales
    the whole map uniformly and adds no double-counted Poisson term — matching
    Barnabè's single-factor assumption.

    Algorithm:
      1. noise_raw = 1/√WHT (pure WHT, no Poisson term)
      2. Two-pass sigma-clipping to identify background pixels
      3. scale = σ_sci_bg / median(noise_raw_bg)
      4. noise_final = noise_raw × scale (uniform upscale)

    Args:
        wht_data: WHT weight map.
        sci_bgsub: Background-subtracted science image.
        min_bg_pixels: Minimum background pixels; below this, skip calibration.

    Returns:
        Globally rescaled noise σ-map (2D float32).

    References:
        Barnabè+2012 (SWELLS IV, mnras0423-1073): global rescaling to reduced χ²~1
    """
    # Step 1: pure WHT noise (no Poisson term)
    noise_raw = noise_from_wht(wht_data).astype(np.float64)

    # Step 2: first-pass sigma-clipping (MAD-based)
    sci_flat = sci_bgsub.ravel()
    med_sci = float(np.median(sci_flat))
    mad_sci = float(np.median(np.abs(sci_flat - med_sci)))
    sigma_est = mad_sci * 1.4826

    lo = med_sci - 3.0 * sigma_est
    hi = med_sci + 5.0 * sigma_est
    bg1 = (sci_bgsub > lo) & (sci_bgsub < hi)

    if bg1.sum() < min_bg_pixels:
        return noise_raw.astype(np.float32)

    # Step 3: second-pass sigma-clipping
    bg1_sci = sci_bgsub[bg1]
    bg_med1 = float(np.median(bg1_sci))
    bg_std1 = float(np.std(bg1_sci))
    bg2 = np.abs(sci_bgsub - bg_med1) < 2.5 * bg_std1
    bg_final = bg1 & bg2

    if bg_final.sum() < min_bg_pixels:
        return noise_raw.astype(np.float32)

    # Step 4: global rescaling factor
    sci_std_bg = float(np.std(sci_bgsub[bg_final]))
    noise_med_bg = float(np.median(noise_raw[bg_final]))

    if noise_med_bg < 1e-15 or sci_std_bg < 1e-15:
        return noise_raw.astype(np.float32)

    scale = np.clip(sci_std_bg / noise_med_bg, 0.01, 100.0)
    noise = noise_raw * scale

    return noise.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 方法 4: Poisson + 读出噪声 (无 WHT 回退)
# ══════════════════════════════════════════════════════════════════════

def noise_poisson_readnoise(
    sci_bgsub: np.ndarray,
    bg_rms: float,
    apply_median_filter: bool = True,
    median_filter_size: int = 3,
) -> np.ndarray:
    """Poisson 散粒噪声 + 读出噪声估计（无 WHT 时的回退方案）。

    用于 WFPC2 等无 WHT extension 的产品。

    Args:
        sci_bgsub: 背景减除后的科学图像
        bg_rms: 背景 RMS（来自 estimate_background）
        apply_median_filter: 是否做中值滤波平滑（消除 drizzle 条纹）
        median_filter_size: 中值滤波窗口大小

    Returns:
        噪声图 σ-map (2D float32)
    """
    noise = np.sqrt(np.maximum(sci_bgsub, 0) + bg_rms**2)

    if apply_median_filter:
        from scipy.ndimage import median_filter as mf
        noise = mf(noise, size=median_filter_size)

    # Rescale to match background RMS
    noise_pos = noise[noise > 0]
    if len(noise_pos) > 0:
        noise_med = float(np.median(noise_pos))
        if noise_med > 1e-15:
            noise = noise * bg_rms / noise_med

    return noise.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 方法 5: 多曝光堆叠噪声
# ══════════════════════════════════════════════════════════════════════

def noise_stacked(
    noise_single: np.ndarray,
    n_exposures: int,
) -> np.ndarray:
    """多曝光堆叠后的噪声缩放：σ_stacked = σ_single / √N。

    Args:
        noise_single: 单次曝光噪声图
        n_exposures: 堆叠曝光数

    Returns:
        堆叠后的噪声图
    """
    if n_exposures <= 1:
        return noise_single
    return (noise_single / np.sqrt(float(n_exposures))).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 一站式接口：根据产品类型自动选择方法
# ══════════════════════════════════════════════════════════════════════

def compute_noise(
    sci_bgsub: np.ndarray,
    wht_data: Optional[np.ndarray],
    product_type: str,
    bg_rms: float = 0.0,
    n_exposures: int = 1,
    calibrate: bool = True,
    use_bug_noise: bool = False,
    use_rescaled: bool = False,
) -> np.ndarray:
    """根据产品类型自动选择噪声方法。

    Args:
        sci_bgsub: 背景减除后的科学图像
        wht_data: WHT 权重图 (可为 None)
        product_type: "FLT", "FLC", "DRC", "DRZ"
        bg_rms: 背景 RMS (用于无 WHT 回退)
        n_exposures: 曝光数 (>1 时会堆叠缩放)
        calibrate: 对 DRC/DRZ 是否做经验标定
        use_bug_noise: FLT/FLC 用信号相关 bug noise √(1/WHT+SCI)（吸收模型残差）
        use_rescaled: FLT/FLC 用纯 WHT × 全局 rescaling 因子（Barnabè 式）

    Returns:
        噪声图 σ-map (2D float32)
    """
    if wht_data is not None:
        if product_type in ("FLT", "FLC"):
            if use_bug_noise:
                # Signal-dependent bug noise: σ = √(1/WHT + SCI), absorbs model residuals.
                noise = noise_from_wht_plus_poisson(wht_data, sci_bgsub)
            elif use_rescaled:
                # Global rescaling: pure WHT × single factor (Barnabè-style).
                noise = noise_from_wht_rescaled(wht_data, sci_bgsub)
            else:
                noise = noise_from_wht(wht_data, n_exposures=n_exposures)
        elif product_type in ("DRC", "DRZ"):
            if calibrate:
                noise = noise_from_wht_poisson_calibrated(wht_data, sci_bgsub)
            else:
                noise = noise_from_wht_plus_poisson(wht_data, sci_bgsub)
        else:
            noise = noise_from_wht(wht_data, n_exposures=n_exposures)
    else:
        noise = noise_poisson_readnoise(sci_bgsub, bg_rms)

    return noise
