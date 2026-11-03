"""Merge a ray-traced arc into the mock galaxy image.

The final strong-lens mock observation = the SKIRT galaxy image plus the
arc, scaled so the arc's peak is ``arc_snr`` times the background noise RMS.
Pure NumPy + astropy; no autolens needed.
"""
from __future__ import annotations

import numpy as np


def scale_arc(arc, arc_snr, noise_rms):
    """Scale the arc so its peak is ``arc_snr * noise_rms``.

    Returns ``(arc_scaled, scale)`` where ``scale = arc_snr * noise_rms /
    arc.max()``.
    """
    arc = np.asarray(arc, dtype=np.float64)
    scale = arc_snr * noise_rms / arc.max()
    return arc * scale, float(scale)


def merge_arc(data, arc_scaled):
    """Add a (smaller, centred) arc onto the galaxy image, padding as needed.

    ``arc_scaled`` is centred and zero-padded to ``data.shape``, then added.
    """
    data = np.asarray(data, dtype=np.float64)
    arc = np.asarray(arc_scaled, dtype=np.float64)
    ny, nx = data.shape
    ay, ax = arc.shape
    pad_y = (ny - ay) // 2
    pad_x = (nx - ax) // 2
    arc_pad = np.zeros_like(data)
    arc_pad[pad_y:pad_y + ay, pad_x:pad_x + ax] = arc
    return data + arc_pad
