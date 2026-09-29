"""
定量质量评估
============
对处理后的数据自动计算质量指标并给出评级。

评估指标:
  - SNR_peak: 图像最大 SNR
  - SNR_center: 中心 20×20px 中值 SNR
  - noise_uniformity: 背景区噪声均匀性 (std/median)
  - PSF_FWHM: PSF 半高全宽 (arcsec)
  - PSF_ellipticity: PSF 椭率
  - CR_fraction: 宇宙线像素占比
  - mask_fraction: 恒星 mask 占比
  - bg_residual: 背景减除残差 (σ 单位)

评级体系:
  ★★★★★ — 所有通过，可直接用于建模
  ★★★★  — 1 项警告
  ★★★   — 2 项警告，建议检查
  ★★    — 3+ 项警告或1项失败，需重新处理

使用示例:
  >>> from galaxylens.data_pipeline.imaging.output.quality import assess_quality
  >>> result = assess_quality(sci, noise, psf, starmask, cr_mask)
  >>> print(result["rating"], result["stars"])
"""

from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np
from scipy.ndimage import median_filter


# ══════════════════════════════════════════════════════════════════════
# 阈值定义
# ══════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    "SNR_peak":        {"warn": 30,   "fail": 20},
    "SNR_center":      {"warn": 5,    "fail": 3},
    "noise_uniformity": {"warn": 0.3,  "fail": 0.5},
    "PSF_FWHM_min":    {"warn": 0.04, "fail": 0.02},  # arcsec (放宽: ACS实际~0.05-0.15")
    "PSF_FWHM_max":    {"warn": 0.20, "fail": 0.30},  # arcsec (放宽)
    "PSF_ellipticity": {"warn": 0.10, "fail": 0.15},
    "CR_fraction":     {"warn": 1.0,  "fail": 2.0},   # percent (默认: LA-cosmic单曝光~0.5%)
    "mask_fraction":   {"warn": 5.0,  "fail": 10.0},   # percent
    "bg_residual":     {"warn": 0.5,  "fail": 1.0},   # σ units
}

# 按宇宙线检测方法区分的 CR_fraction 阈值（percent）。
# deepCR 检出率天然高于 LA-cosmic（~1% vs ~0.5%），
# 交叉审计证实 deepCR 多检出的像素 89% 是真尖峰、过检仅 ~1.8%，
# 因此 deepCR 用更宽的阈值，避免"检出更完整"被冤枉成降级。
CR_THRESHOLDS_BY_METHOD = {
    "deepCR":    {"warn": 2.0, "fail": 4.0},
    "LA-cosmic": {"warn": 1.0, "fail": 2.0},
    "stacked":   {"warn": 0.5, "fail": 1.0},  # 多曝光堆叠后残留应极低
}


# ══════════════════════════════════════════════════════════════════════
# 指标计算
# ══════════════════════════════════════════════════════════════════════

def compute_snr_metrics(
    sci: np.ndarray,
    noise: np.ndarray,
) -> dict:
    """计算 SNR 相关指标。

    Args:
        sci: 科学图像
        noise: 噪声图

    Returns:
        {"snr_peak": float, "snr_center_median": float}
    """
    snr = np.where(noise > 0, np.abs(sci) / noise, 0)

    ny, nx = sci.shape
    cy, cx = ny // 2, nx // 2
    r = 10
    center_snr = snr[cy - r:cy + r, cx - r:cx + r]

    return {
        "snr_peak": float(np.max(snr)),
        "snr_center_median": float(np.median(center_snr)),
    }


def compute_noise_uniformity(
    noise: np.ndarray,
    sci: np.ndarray,
) -> float:
    """计算背景区噪声均匀性。

    用 sigma-clipping 识别背景区域，
    计算噪声图在背景区的 std/median 比值。
    接近 0 表示均匀。

    Args:
        noise: 噪声图
        sci: 科学图像 (用于识别背景)

    Returns:
        噪声不均匀性指数 (0-∞, 越小越好)
    """
    # 识别背景区：SCI 接近 0 的区域
    sci_flat = sci.ravel()
    med = np.median(sci_flat)
    mad = np.median(np.abs(sci_flat - med)) * 1.4826
    bg = np.abs(sci - med) < 3.0 * mad

    if bg.sum() < 20:
        return float(np.std(noise) / max(np.median(noise), 1e-10))

    bg_noise = noise[bg]
    return float(np.std(bg_noise) / max(np.median(bg_noise), 1e-10))


def compute_psf_metrics(
    psf: np.ndarray,
    pixel_scale: float = 0.05,
) -> dict:
    """计算 PSF 质量指标。

    Args:
        psf: 归一化 PSF 数组
        pixel_scale: 像元尺度 (arcsec/pixel)

    Returns:
        {"fwhm_arcsec": float, "ellipticity": float}
    """
    cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
    peak = float(psf[cy, cx])
    half = peak / 2.0

    if half <= 0:
        return {"fwhm_arcsec": float("nan"), "ellipticity": float("nan")}

    # FWHM: 从中心行估计
    row = psf[cy, :]
    above = row >= half
    crossings = np.where(np.diff(above.astype(int)))[0]
    if len(crossings) >= 2:
        f_left = crossings[0]
        f_right = crossings[-1]
        fwhm_pix = f_right - f_left
        if fwhm_pix > 0:
            fwhm_as = fwhm_pix * pixel_scale
        else:
            fwhm_as = float("nan")
    else:
        fwhm_as = float("nan")

    # Ellipticity: e = (a-b)/(a+b), 用二阶矩估计
    yy, xx = np.indices(psf.shape)
    total = max(float(psf.sum()), 1e-10)
    x2 = float(np.sum(psf * (xx - cx) ** 2) / total)
    y2 = float(np.sum(psf * (yy - cy) ** 2) / total)
    xy = float(np.sum(psf * (xx - cx) * (yy - cy)) / total)

    if x2 + y2 > 0:
        e1 = (x2 - y2) / (x2 + y2)
        e2 = 2.0 * xy / (x2 + y2)
        ellip = float(np.sqrt(e1 ** 2 + e2 ** 2))
    else:
        ellip = 0.0

    return {"fwhm_arcsec": fwhm_as, "ellipticity": ellip}


def compute_bg_residual(
    sci: np.ndarray,
    bg_rms: float,
) -> float:
    """计算背景减除残差（边缘中值，σ 单位）。

    Args:
        sci: 背景减除后的图像
        bg_rms: 背景 RMS

    Returns:
        残差 (σ 单位)，接近 0 表示背景减除干净
    """
    edge = np.concatenate([
        sci[0, :], sci[-1, :], sci[:, 0], sci[:, -1],
    ])
    return float(np.abs(np.median(edge)) / max(bg_rms, 1e-10))


# ══════════════════════════════════════════════════════════════════════
# 评估与评级
# ══════════════════════════════════════════════════════════════════════

# 指标方向性: higher_is_better = 值越大越好 (SNR)
_HIGHER_IS_BETTER = {"SNR_peak", "SNR_center"}


def _check_metric(name: str, value: float, thresholds: dict = None) -> str:
    """检查单个指标相对于阈值的状态。

    对 higher_is_better 指标: value < fail → fail, value < warn → warn
    对 lower_is_better 指标:  value > fail → fail, value > warn → warn
    PSF_FWHM: 使用专用 min/max 检查

    Args:
        thresholds: 可选，覆盖默认 THRESHOLDS（用于按方法区分的 CR 阈值）。
    """
    if np.isnan(value):
        return "unknown"

    t = thresholds if thresholds is not None else THRESHOLDS

    if name in ("PSF_FWHM_min",):
        if value < t[name]["fail"]:
            return "fail"
        if value < t[name]["warn"]:
            return "warn"
        return "pass"

    if name in ("PSF_FWHM_max",):
        if value > t[name]["fail"]:
            return "fail"
        if value > t[name]["warn"]:
            return "warn"
        return "pass"

    if name in _HIGHER_IS_BETTER:
        # 值越大越好 (SNR)
        if value < t[name]["fail"]:
            return "fail"
        if value < t[name]["warn"]:
            return "warn"
        return "pass"
    else:
        # 值越小越好 (noise, CR, mask, etc.)
        if value > t[name]["fail"]:
            return "fail"
        if value > t[name]["warn"]:
            return "warn"
        return "pass"


def _stars_from_issues(issues: list[str]) -> int:
    """从问题列表计算星级。"""
    n_fails = sum(1 for i in issues if i.endswith("(FAIL)"))
    n_warns = sum(1 for i in issues if i.endswith("(WARN)"))

    if n_fails > 0:
        return 2
    elif n_warns >= 3:
        return 2
    elif n_warns == 2:
        return 3
    elif n_warns == 1:
        return 4
    else:
        return 5


# ══════════════════════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════════════════════

def assess_quality(
    sci: np.ndarray,
    noise: np.ndarray,
    psf: np.ndarray,
    starmask: np.ndarray | None = None,
    cr_mask: np.ndarray | None = None,
    bg_rms: float = 0.0,
    pixel_scale: float = 0.05,
    cr_method: str | None = None,
) -> Dict[str, Any]:
    """一站式质量评估。

    Args:
        sci: 科学图像
        noise: 噪声图
        psf: PSF 数组
        starmask: 恒星 mask (可选)
        cr_mask: 宇宙线 mask (可选)
        bg_rms: 背景 RMS
        pixel_scale: 像元尺度
        cr_method: 宇宙线检测方法 ("deepCR" / "LA-cosmic" / "stacked" / None)。
                   用于按方法选择 CR_fraction 阈值（deepCR 检出率天然更高）。

    Returns:
        {
            "metrics": {...},
            "issues": [...],
            "stars": int (2-5),
            "rating": "★★★★★",
            "usable": bool,
        }
    """
    # 计算各项指标
    snr_m = compute_snr_metrics(sci, noise)
    noise_uni = compute_noise_uniformity(noise, sci)
    psf_m = compute_psf_metrics(psf, pixel_scale)
    bg_res = compute_bg_residual(sci, bg_rms)

    metrics = {
        "snr_peak": snr_m["snr_peak"],
        "snr_center_median": snr_m["snr_center_median"],
        "noise_uniformity": noise_uni,
        "psf_fwhm_arcsec": psf_m["fwhm_arcsec"],
        "psf_ellipticity": psf_m["ellipticity"],
        "bg_residual_sigma": bg_res,
        "cr_fraction_pct": 100.0 * float(cr_mask.sum()) / cr_mask.size
                           if cr_mask is not None and cr_mask.size > 0 else 0.0,
        "mask_fraction_pct": 100.0 * float(starmask.sum()) / starmask.size
                             if starmask is not None and starmask.size > 0 else 0.0,
    }

    # 按宇宙线检测方法选 CR 阈值（默认用 THRESHOLDS 里的通用值）
    eff_thresholds = dict(THRESHOLDS)
    if cr_method is not None and cr_method in CR_THRESHOLDS_BY_METHOD:
        eff_thresholds["CR_fraction"] = CR_THRESHOLDS_BY_METHOD[cr_method]

    # 检查阈值
    checks = [
        ("SNR_peak", metrics["snr_peak"], "peak SNR", "high"),
        ("SNR_center", metrics["snr_center_median"], "中心 SNR", "high"),
        ("noise_uniformity", metrics["noise_uniformity"], "噪声不均匀", "low"),
        ("PSF_FWHM_min", metrics["psf_fwhm_arcsec"], "PSF FWHM过低", "low"),
        ("PSF_FWHM_max", metrics["psf_fwhm_arcsec"], "PSF FWHM过高", "high"),
        ("PSF_ellipticity", metrics["psf_ellipticity"], "PSF椭率偏高", "high"),
        ("CR_fraction", metrics["cr_fraction_pct"], "宇宙线残留", "high"),
        ("mask_fraction", metrics["mask_fraction_pct"], "mask占比过高", "high"),
        ("bg_residual", metrics["bg_residual_sigma"], "背景残余", "high"),
    ]

    issues = []
    for name, value, label, direction in checks:
        status = _check_metric(name, value, eff_thresholds)
        if status == "fail":
            issues.append(f"{label}{value:.3g} (FAIL)")
        elif status == "warn":
            issues.append(f"{label}{value:.3g} (WARN)")

    stars = _stars_from_issues(issues)
    star_map = {5: "★★★★★", 4: "★★★★", 3: "★★★", 2: "★★"}

    return {
        "metrics": metrics,
        "issues": issues,
        "stars": stars,
        "rating": star_map.get(stars, "★★"),
        "usable": stars >= 4,
    }
