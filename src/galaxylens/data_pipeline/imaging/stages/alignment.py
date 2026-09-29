"""
跨波段对齐
==========
以参考波段（通常 F814W）为中心，计算其他波段相对于参考的 centroid 偏移。

功能:
  1. 以 F814W 的 centroid 为基准
  2. 计算各波段的 centroid 偏移（像素和角秒）
  3. 生成对齐报告

使用场景:
  - 盘星系多波段（ACS + WFPC2 + WFC3）需要对齐
  - 椭圆星系纯 ACS 通常不需要（同一仪器，WCS 一致）

使用示例:
  >>> from galaxylens.data_pipeline.imaging.stages.alignment import compute_alignment
  >>> offsets = compute_alignment({"F814W": (cx1, cy1, 0.05),
  ...                                "F606W": (cx2, cy2, 0.0455)})
"""

from __future__ import annotations
from typing import Dict, Tuple


# ══════════════════════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════════════════════

def compute_alignment(
    band_centroids: Dict[str, Tuple[float, float, float]],
    ref_band: str = "F814W",
) -> Dict[str, dict]:
    """计算各波段相对于参考波段的 centroid 偏移。

    Args:
        band_centroids: {band: (cx, cy, pixel_scale), ...}
        ref_band: 参考波段名 (默认 F814W)

    Returns:
        {band: {"offset_px": (dx, dy), "offset_as": (dx_as, dy_as)}, ...}
        参考波段自身的偏移为 (0, 0)

    Raises:
        KeyError: 若 ref_band 不在 band_centroids 中
    """
    if ref_band not in band_centroids:
        # 参考波段不在本批波段中（如只处理 F450W/F606W 而无 F814W）时，
        # 回退到第一个可用波段作参考，而不是抛错。
        ref_band = next(iter(band_centroids))

    cx_ref, cy_ref, scale_ref = band_centroids[ref_band]

    offsets: Dict[str, dict] = {}
    for band, (cx, cy, scale) in band_centroids.items():
        if band == ref_band:
            offsets[band] = {
                "offset_px": (0.0, 0.0),
                "offset_as": (0.0, 0.0),
            }
        else:
            dx_px = float(cx - cx_ref)
            dy_px = float(cy - cy_ref)
            offsets[band] = {
                "offset_px": (dx_px, dy_px),
                "offset_as": (dx_px * scale, dy_px * scale),
            }

    return offsets


def format_alignment_report(
    offsets: Dict[str, dict],
    ref_band: str = "F814W",
) -> str:
    """将对齐偏移量格式化为可读字符串。

    Args:
        offsets: compute_alignment() 的输出
        ref_band: 参考波段名

    Returns:
        格式化后的对齐报告
    """
    # 若 ref_band 不在 offsets 中，推断为 offset=(0,0) 的参考波段
    if ref_band not in offsets:
        for b, off in offsets.items():
            if off["offset_px"] == (0.0, 0.0):
                ref_band = b
                break

    lines = [f"Cross-band alignment (ref: {ref_band}):"]
    for band, off in sorted(offsets.items()):
        dx, dy = off["offset_px"]
        dx_as, dy_as = off["offset_as"]
        if band == ref_band:
            lines.append(f"  {band}: reference")
        else:
            lines.append(
                f"  {band}: ({dx:+.1f}, {dy:+.1f}) px "
                f"= ({dx_as:+.3f}\", {dy_as:+.3f}\")"
            )
    return "\n".join(lines)
