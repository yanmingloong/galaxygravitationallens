"""Velocity anisotropy ``beta_z = 1 - sigma_z^2 / sigma_R^2`` from particles.

Cylindrical anisotropy with ``z`` the disk normal and ``R`` the in-plane
cylindrical radius, matching jampy's ``beta_z`` definition. ``beta_z`` is an
intrinsic quantity (inclination-independent). Velocities are mass-weighted
with the mean removed.

Inputs are particles already aligned to the mock frame (long axis = x, disk
normal = z) with velocities relative to the stellar centre of mass. Pure
NumPy.
"""
from __future__ import annotations

import numpy as np


def beta_z(pos_s_al, vel_s_rel, m_cur, R_ap_kpc):
    """Mass-weighted ``beta_z = 1 - sigma_z^2 / sigma_R^2`` within ``R < R_ap``.

    Returns ``(beta_z, n_star)``. ``beta_z`` is ``NaN`` when fewer than 50
    particles are enclosed or ``sigma_R^2 <= 0``.
    """
    pos = np.asarray(pos_s_al, dtype=np.float64)
    vel = np.asarray(vel_s_rel, dtype=np.float64)
    m = np.asarray(m_cur, dtype=np.float64)

    x, y, _z = pos[:, 0], pos[:, 1], pos[:, 2]
    R = np.hypot(x, y)
    vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]

    # radial velocity v_R (in-plane, outward) and normal velocity v_z
    v_R = (vx * x + vy * y) / R
    sel = R < R_ap_kpc
    n = int(sel.sum())
    if n < 50:
        return float("nan"), n

    w = m[sel]
    vr_mean = np.average(v_R[sel], weights=w)
    vz_mean = np.average(vz[sel], weights=w)
    sigr2 = np.average((v_R[sel] - vr_mean) ** 2, weights=w)
    sigz2 = np.average((vz[sel] - vz_mean) ** 2, weights=w)
    if sigr2 <= 0:
        return float("nan"), n
    return float(1.0 - sigz2 / sigr2), n
