"""
Quantitative quality assessment
============
Automatically compute quality metrics on the processed data and assign a rating.

Metrics:
  - SNR_peak: maximum SNR of the image
  - SNR_center: median SNR of the central 20×20 px region
  - noise_uniformity: noise uniformity of the background region (std/median)
  - PSF_FWHM: PSF full width at half maximum (arcsec)
  - PSF_ellipticity: PSF ellipticity
  - CR_fraction: cosmic-ray pixel fraction
  - mask_fraction: star-mask fraction
  - bg_residual: background-subtraction residual (σ units)

Rating system:
  ★★★★★ — all checks pass; ready for modeling
  ★★★★  — 1 warning
  ★★★   — 2 warnings; inspection recommended
  ★★    — 3+ warnings or 1 failure; reprocessing required

Usage example:
  >>> from galaxylens.data_pipeline.imaging.output.quality import assess_quality
  >>> result = assess_quality(sci, noise, psf, starmask, cr_mask)
  >>> print(result["rating"], result["stars"])
"""

from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np
from scipy.ndimage import median_filter


# ══════════════════════════════════════════════════════════════════════
# Threshold definitions
# ══════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    "SNR_peak":        {"warn": 30,   "fail": 20},
    "SNR_center":      {"warn": 5,    "fail": 3},
    "noise_uniformity": {"warn": 0.3,  "fail": 0.5},
    "PSF_FWHM_min":    {"warn": 0.04, "fail": 0.02},  # arcsec (relaxed: ACS actual ~0.05-0.15")
    "PSF_FWHM_max":    {"warn": 0.20, "fail": 0.30},  # arcsec (relaxed)
    "PSF_ellipticity": {"warn": 0.10, "fail": 0.15},
    "CR_fraction":     {"warn": 1.0,  "fail": 2.0},   # percent (default: LA-cosmic single exposure ~0.5%)
    "mask_fraction":   {"warn": 5.0,  "fail": 10.0},   # percent
    "bg_residual":     {"warn": 0.5,  "fail": 1.0},   # σ units
}

# CR_fraction thresholds (percent) split by cosmic-ray detection method.
# deepCR has a naturally higher detection rate than LA-cosmic (~1% vs ~0.5%);
# cross-audit confirms that 89% of deepCR's extra detections are true spikes and over-detection is only ~1.8%,
# so deepCR uses wider thresholds to avoid penalizing "more complete detection" as a downgrade.
CR_THRESHOLDS_BY_METHOD = {
    "deepCR":    {"warn": 2.0, "fail": 4.0},
    "LA-cosmic": {"warn": 1.0, "fail": 2.0},
    "stacked":   {"warn": 0.5, "fail": 1.0},  # residual after multi-exposure stacking should be very low
}


# ══════════════════════════════════════════════════════════════════════
# Metric computation
# ══════════════════════════════════════════════════════════════════════

def compute_snr_metrics(
    sci: np.ndarray,
    noise: np.ndarray,
) -> dict:
    """Compute SNR-related metrics.

    Args:
        sci: science image
        noise: noise map

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
    """Compute noise uniformity in the background region.

    Identifies the background region via sigma-clipping,
    then computes the std/median ratio of the noise map over that region.
    A value near 0 indicates uniformity.

    Args:
        noise: noise map
        sci: science image (used to identify the background)

    Returns:
        noise non-uniformity index (0-∞, smaller is better)
    """
    # Identify the background region: where SCI is close to 0
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
    """Compute PSF quality metrics.

    Args:
        psf: normalized PSF array
        pixel_scale: pixel scale (arcsec/pixel)

    Returns:
        {"fwhm_arcsec": float, "ellipticity": float}
    """
    cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
    peak = float(psf[cy, cx])
    half = peak / 2.0

    if half <= 0:
        return {"fwhm_arcsec": float("nan"), "ellipticity": float("nan")}

    # FWHM: estimated from the central row
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

    # Ellipticity: e = (a-b)/(a+b), estimated via second moments
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
    """Compute the background-subtraction residual (edge median, in σ units).

    Args:
        sci: image after background subtraction
        bg_rms: background RMS

    Returns:
        residual (in σ units); a value near 0 indicates clean background subtraction
    """
    edge = np.concatenate([
        sci[0, :], sci[-1, :], sci[:, 0], sci[:, -1],
    ])
    return float(np.abs(np.median(edge)) / max(bg_rms, 1e-10))


# ══════════════════════════════════════════════════════════════════════
# Assessment and rating
# ══════════════════════════════════════════════════════════════════════

# Metric directionality: higher_is_better = larger is better (SNR)
_HIGHER_IS_BETTER = {"SNR_peak", "SNR_center"}


def _check_metric(name: str, value: float, thresholds: dict = None) -> str:
    """Check the status of a single metric against thresholds.

    For higher_is_better metrics: value < fail → fail, value < warn → warn
    For lower_is_better metrics: value > fail → fail, value > warn → warn
    PSF_FWHM: uses dedicated min/max checks

    Args:
        thresholds: optional, overrides the default THRESHOLDS (used for method-specific CR thresholds).
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
        # larger is better (SNR)
        if value < t[name]["fail"]:
            return "fail"
        if value < t[name]["warn"]:
            return "warn"
        return "pass"
    else:
        # smaller is better (noise, CR, mask, etc.)
        if value > t[name]["fail"]:
            return "fail"
        if value > t[name]["warn"]:
            return "warn"
        return "pass"


def _stars_from_issues(issues: list[str]) -> int:
    """Compute the star rating from the issue list."""
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
# Public interface
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
    """One-stop quality assessment.

    Args:
        sci: science image
        noise: noise map
        psf: PSF array
        starmask: star mask (optional)
        cr_mask: cosmic-ray mask (optional)
        bg_rms: background RMS
        pixel_scale: pixel scale
        cr_method: cosmic-ray detection method ("deepCR" / "LA-cosmic" / "stacked" / None).
                   Used to select the CR_fraction threshold per method (deepCR naturally has a higher detection rate).

    Returns:
        {
            "metrics": {...},
            "issues": [...],
            "stars": int (2-5),
            "rating": "★★★★★",
            "usable": bool,
        }
    """
    # Compute all metrics
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

    # Select the CR threshold per detection method (default to the generic values in THRESHOLDS)
    eff_thresholds = dict(THRESHOLDS)
    if cr_method is not None and cr_method in CR_THRESHOLDS_BY_METHOD:
        eff_thresholds["CR_fraction"] = CR_THRESHOLDS_BY_METHOD[cr_method]

    # Check thresholds
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
