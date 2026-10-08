"""
Cross-band alignment
====================
Centers on the reference band (usually F814W) and computes the centroid
offsets of the other bands relative to the reference.

Features:
  1. Uses the F814W centroid as the reference
  2. Computes the centroid offset of each band (in pixels and arcseconds)
  3. Generates an alignment report

Use cases:
  - Disk galaxies with multiple bands (ACS + WFPC2 + WFC3) need alignment
  - Elliptical galaxies with pure ACS usually do not (same instrument, consistent WCS)

Usage example:
  >>> from galaxylens.data_pipeline.imaging.stages.alignment import compute_alignment
  >>> offsets = compute_alignment({"F814W": (cx1, cy1, 0.05),
  ...                                "F606W": (cx2, cy2, 0.0455)})
"""

from __future__ import annotations
from typing import Dict, Tuple


# ══════════════════════════════════════════════════════════════════════
# Public interface
# ══════════════════════════════════════════════════════════════════════

def compute_alignment(
    band_centroids: Dict[str, Tuple[float, float, float]],
    ref_band: str = "F814W",
) -> Dict[str, dict]:
    """Compute the centroid offset of each band relative to the reference band.

    Args:
        band_centroids: {band: (cx, cy, pixel_scale), ...}
        ref_band: reference band name (default F814W)

    Returns:
        {band: {"offset_px": (dx, dy), "offset_as": (dx_as, dy_as)}, ...}
        The reference band's own offset is (0, 0)

    Raises:
        KeyError: if ref_band is not in band_centroids
    """
    if ref_band not in band_centroids:
        # If the reference band is not among this batch of bands (e.g. only
        # F450W/F606W are being processed without F814W), fall back to the
        # first available band as the reference instead of raising an error.
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
    """Format the alignment offsets as a human-readable string.

    Args:
        offsets: output of compute_alignment()
        ref_band: reference band name

    Returns:
        the formatted alignment report
    """
    # If ref_band is not in offsets, infer it as the band whose offset is (0,0)
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
