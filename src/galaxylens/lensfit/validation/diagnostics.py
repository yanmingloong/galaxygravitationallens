"""Counterfactual diagnostics for mock closure.

Tools that answer "why did the fit miss the truth" — primarily the azimuthal
Fourier modes of the face-on stellar disk, which flag a bar / strong spiral
arms (m=2) or a lopsided disk (m=1). A significant m=2 in the inner disk is
the root cause of the ``beta_z`` bias in the MF series: a bar breaks the
axisymmetric Jeans assumption that ``jampy`` relies on.

The fit-model-dependent diagnostics (``chi2(beta_z)`` scan at the true mass,
model-vs-truth ``v_rms`` profile) live with the closure entry in
:mod:`galaxylens.lensfit.validation.closure` — they need the jampy engine.
"""
from __future__ import annotations

import numpy as np


def fourier_modes(x, y, mass, rbins, modes=(1, 2, 3, 4), min_per_bin=50):
    """Azimuthal Fourier amplitudes of the face-on stellar surface density.

    ``A_m(R) = |sum w_i exp(i m phi_i)| / sum w_i``, mass-weighted and
    radially binned (``phi = atan2(y, x)`` in the disk-aligned x-y plane).
    ``m=2`` significant -> bar or strong spiral arms; ``m=1`` -> lopsided.

    Returns ``(R_center, {m: amplitudes})`` where each amplitude array is
    ``NaN`` for bins with fewer than ``min_per_bin`` particles.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    rbins = np.asarray(rbins, dtype=np.float64)

    R = np.hypot(x, y)
    phi = np.arctan2(y, x)
    idx = np.digitize(R, rbins) - 1
    Rc = 0.5 * (rbins[:-1] + rbins[1:])
    amps = {m: np.full(len(rbins) - 1, np.nan) for m in modes}
    for b in range(len(rbins) - 1):
        sel = idx == b
        if sel.sum() < min_per_bin:
            continue
        w = mass[sel]
        W = w.sum()
        for m in modes:
            amps[m][b] = abs(w @ np.exp(1j * m * phi[sel])) / W
    return Rc, amps
