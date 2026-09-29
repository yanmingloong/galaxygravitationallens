"""
宇宙线检测与处理 Cosmic ray detection & handling
================
对单次曝光 FLT 数据检测宇宙线，或通过多曝光堆叠自动剔除。

方法:
  1. detect_cosmic_rays_laplace() — LA Cosmic 简化版 (局部中值偏离)
  2. sigma_clipped_stack()        — Sigma-clipped median 堆叠 (多曝光 CR 剔除)

处理流程:
    FLT 曝光
      ├─ 单曝光 → detect_cosmic_rays_laplace() → CR mask
      └─ 多曝光 → sigma_clipped_stack() → stacked SCI + CR hitmap

参考文献:
  - van Dokkum (2001) PASP 113, 1420: LA Cosmic algorithm
  - CALACS ISR: cosmic ray rejection in ACS pipeline

使用示例:
  >>> from galaxylens.data_pipeline.imaging.stages.cosmic_ray import detect_cosmic_rays_laplace
  >>> cr_mask = detect_cosmic_rays_laplace(sci_data)
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
from scipy.ndimage import median_filter


# ══════════════════════════════════════════════════════════════════════
# 默认参数
# ══════════════════════════════════════════════════════════════════════

CR_SIGMA: float = 8.0        # CR 检测 σ 阈值
CR_LOCAL_BOX: int = 5        # 局部中值滤波窗口 (像素)


# ══════════════════════════════════════════════════════════════════════
# 方法 1: LA Cosmic 简化版
# ══════════════════════════════════════════════════════════════════════

def detect_cosmic_rays_laplace(
    data: np.ndarray,
    sigma: float = CR_SIGMA,
    box: int = CR_LOCAL_BOX,
) -> np.ndarray:
    """基于局部中值偏离检测宇宙线像素 (LA Cosmic 简化版)。

    对每个像素，比较其值与局部中值滤波的差异。
    若偏差 > sigma × local_MAD，标记为宇宙线。

    Args:
        data: 输入 2D 图像
        sigma: 检测阈值 (单位: local MAD)
        box: 局部滤波窗口大小 (像素)

    Returns:
        Boolean mask，True = 宇宙线像素
    """
    local_med = median_filter(data, size=box)
    local_mad = median_filter(np.abs(data - local_med), size=box) * 1.4826
    local_mad = np.maximum(local_mad, 1e-10)

    cr_mask = (data - local_med) > sigma * local_mad
    return cr_mask


# ══════════════════════════════════════════════════════════════════════
# 方法 2: Sigma-clipped median 堆叠
# ══════════════════════════════════════════════════════════════════════

def sigma_clipped_stack(
    images: List[np.ndarray],
    sigma: float = 3.0,
    maxiters: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sigma-clipped median 堆叠多个图像 (pseudo-drizzle CR 剔除)。

    对每个像素位置，计算各曝光的中值和 MAD，
    剔除偏离 > sigma × MAD 的值，然后取中值。

    Args:
        images: 图像列表 (多个曝光的同一区域)
        sigma: clipping 阈值
        maxiters: 最大迭代次数

    Returns:
        (stacked_image, cr_hitmap):
          stacked_image — 堆叠后的图像 (2D float64)
          cr_hitmap    — 每个像素被 clipping 的总次数 (2D int)
    """
    if len(images) == 1:
        return images[0].astype(np.float64), np.zeros_like(images[0], dtype=int)

    stack = np.stack([img.astype(np.float64) for img in images], axis=0)
    total_hitmap = np.zeros(stack.shape[1:], dtype=int)

    for iteration in range(maxiters):
        median = np.median(stack, axis=0)
        mad = 1.4826 * np.median(
            np.abs(stack - median[np.newaxis, :, :]), axis=0,
        )
        mad = np.maximum(mad, 1e-10)

        deviate = np.abs(stack - median[np.newaxis, :, :]) / mad[np.newaxis, :, :]
        mask = deviate > sigma

        if mask.sum() == 0:
            break

        total_hitmap += mask.sum(axis=0)
        stack = np.where(mask, np.nan, stack)

    result = np.nanmedian(stack, axis=0)
    return result, total_hitmap


def compute_cr_fraction(cr_mask: np.ndarray) -> float:
    """计算宇宙线像素占比。

    Args:
        cr_mask: Boolean CR mask

    Returns:
        CR 像素占比 (0-100%)
    """
    if cr_mask.size == 0:
        return 0.0
    return 100.0 * float(cr_mask.sum()) / float(cr_mask.size)


# ══════════════════════════════════════════════════════════════════════
# 方法 3: deepCR 深度学习检测 (2026-07-03)
# ══════════════════════════════════════════════════════════════════════

# deepCR 模型名映射 (参考 dinos_i)
_DEEPCR_MODEL_MAP = {
    ("ACS", "WFC", "F435W"): "ACS-WFC-F606W-2-32",
    ("ACS", "WFC", "F606W"): "ACS-WFC-F606W-2-32",
    ("ACS", "WFC", "F814W"): "ACS-WFC-F606W-2-32",
}


def detect_cosmic_rays_deepcr(
    data: np.ndarray,
    exptime: float = 1.0,
    instrument: str = "ACS",
    detector: str = "WFC",
    band: str = "F814W",
    threshold: float = 0.5,
    inpaint: bool = False,
):
    """使用 deepCR 深度学习模型检测宇宙线（可选修复）。

    Zhang & Bloom (2020): 基于 Mask R-CNN 的 HST 宇宙线检测。
    模型按仪器/波段特化，精度远高于 LA-cosmic。

    Args:
        data: 2D 图像 (electrons)
        exptime: 曝光时间 (秒)
        instrument: 仪器名 ("ACS", "WFC3")
        detector: 探测器 ("WFC", "UVIS")
        band: 波段名
        threshold: 检测阈值 (0-1, 默认 0.5, 参考 dinos_i)
        inpaint: True 时返回 (mask, 修复后图像)，修复用 5×5 中值填充
                 (deepCR 对 ACS/WFC 无专门 inpaint 模型时走 medmask)。

    Returns:
        inpaint=False: Boolean CR mask (True = 宇宙线像素)
        inpaint=True:  (Boolean CR mask, 修复后 electrons 图像)

    Note:
        如果模型不可用或 deepCR 未安装，inpaint=False 返回全 False mask；
        inpaint=True 返回 (全 False mask, 原图)。
    """
    try:
        from deepCR import deepCR
    except ImportError:
        if inpaint:
            return np.zeros(data.shape, dtype=bool), np.asarray(data)
        return np.zeros(data.shape, dtype=bool)

    model_key = (instrument, detector, band)
    model_name = _DEEPCR_MODEL_MAP.get(model_key)

    if model_name is None:
        # 尝试构造模型名
        model_name = f"{instrument}-{detector}-{band}-2-32"

    try:
        # deepCR 期望输入单位为 electrons。
        # FLC/FLT 的 SCI extension BUNIT=ELECTRONS（已经是总电子数），
        # 不能乘以 exptime（那是 electrons/s 才需要做的转换）。
        image_e = np.maximum(data, 0)

        mdl = deepCR(mask=model_name, device="CPU")
        # clean(inpaint=False) 返回单个 mask；inpaint=True 返回 (mask, inpainted) 元组。
        # inpainted 已由 deepCR 还原到输入单位 (electrons)。
        result = mdl.clean(image_e, threshold=threshold, inpaint=inpaint)

        if inpaint:
            mask, inpainted = result
            return np.asarray(mask).astype(bool), np.asarray(inpainted)
        return np.asarray(result).astype(bool)

    except Exception as e:
        print(f"    [WARN] deepCR failed ({e}), falling back to LA-cosmic", flush=True)
        if inpaint:
            return np.zeros(data.shape, dtype=bool), np.asarray(data)
        return np.zeros(data.shape, dtype=bool)
