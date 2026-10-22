"""MGE I/O and parameter conversion.

Multi-Gaussian Expansion (MGE) decomposes a 1-D or 2-D surface-brightness
profile into a sum of Gaussians. The decomposition algorithm is Cappellari's
``mgefit``; this module provides the I/O convention and the conversion between
an MGE's (total counts L_k, sigma_k, axial ratio q_k) and the per-Gaussian
parameters consumed by mass-profile builders (intensity, sigma, ellipticity).

Each component carries its morphological label (``bulge`` or ``disk``) so
downstream models can treat the two stellar components separately (e.g. a bulge
M/L differing from the disk M/L). The B2 convention writes 6 bulge Gaussians
first, then 10 disk Gaussians; the label is preserved here rather than relying
on that positional convention.

References
----------
- Cappellari 2002, MNRAS 333, 400 (MGE definition and fitting)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Tuple

import numpy as np


class MgeComponent(NamedTuple):
    """One MGE Gaussian component with its morphological label.

    Attributes:
        component: ``"bulge"`` or ``"disk"``.
        L_k: total counts of the Gaussian.
        sigma_arcsec: Gaussian sigma in arcsec.
        q_obs: observed axial ratio (0 < q <= 1).
    """

    component: str
    L_k: float
    sigma_arcsec: float
    q_obs: float


def load_mge(csv_path: Path) -> List[MgeComponent]:
    """Load an MGE parameter CSV written by the B2 convention.

    Expected CSV header: ``component,L_k,sigma_px,sigma_arcsec,q_obs``. Returns
    one :class:`MgeComponent` per row, in file order, preserving the
    ``component`` label (bulge/disk).

    Args:
        csv_path: path to the MGE parameter CSV.

    Returns:
        List of :class:`MgeComponent` named tuples.
    """
    rows: List[MgeComponent] = []
    with open(csv_path) as f:
        for line in f:
            if line.startswith("component"):
                continue
            comp, L, _spx, sarc, q = line.strip().split(",")
            rows.append(MgeComponent(comp, float(L), float(sarc), float(q)))
    return rows


def q_to_ell_comps(q: float) -> Tuple[float, float]:
    """Convert an observed axial ratio ``q`` to ellipticity components ``(e1, e2)``.

    Uses the first ellipticity component ``e1 = -(1-q)/(1+q)`` with ``e2=0``
    (major axis aligned with the reference axis). This is the convention used
    throughout the disk-lens series (PA ~ 87 deg, negative e1).

    Args:
        q: observed axial ratio (0 < q <= 1).

    Returns:
        ``(e1, e2)`` ellipticity components.
    """
    e = (1.0 - q) / (1.0 + q)
    return (-e, 0.0)


def mge_gaussian_parameters(mge: List[MgeComponent]) -> List[dict]:
    """Convert MGE components into per-Gaussian mass-profile parameters.

    For each Gaussian the central surface brightness follows from the MGE
    total-counts normalization ``intensity = L_k / (2*pi*sigma_k^2*q_k)``
    (Cappellari 2002). The returned dicts hold ``component``, ``intensity``,
    ``sigma`` (arcsec), and ``ell_comps``, ready to feed a Gaussian mass
    profile.

    Args:
        mge: list of :class:`MgeComponent` named tuples.

    Returns:
        List of dicts with keys ``component``, ``intensity``, ``sigma``,
        ``ell_comps``.
    """
    params = []
    for c in mge:
        intensity = c.L_k / (2.0 * np.pi * c.sigma_arcsec**2 * c.q_obs)
        params.append(
            {
                "component": c.component,
                "intensity": intensity,
                "sigma": c.sigma_arcsec,
                "ell_comps": q_to_ell_comps(c.q_obs),
            }
        )
    return params
