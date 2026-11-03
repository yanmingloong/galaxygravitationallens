"""Dark-matter fraction ground truth from particles.

``f_dm`` is computed in three apertures (``2.2 Rd`` / ``5 kpc`` / ``10 kpc``)
under two conventions:

- **spherical** (primary): 3-D ball of radius ``R``. Unambiguous.
- **cylindrical edge-on** (secondary): cylinder ``sqrt(x^2 + z^2) < R`` along
  the edge-on line of sight (``+y``). Matches the model's edge-on cylindrical
  aperture in ``lensfit.validation.fdm``.

Inputs are particles already aligned to the mock frame (long axis = x,
disk normal = z) and in physical units. All functions are pure NumPy.
"""
from __future__ import annotations

import numpy as np


def surface_density(R, mass, rbins):
    """Stellar surface density Sigma(R) in the face-on x-y plane (log bins).

    Returns ``(R_center, Sigma)`` with ``Sigma`` in the caller's mass unit per
    ``kpc^2``.
    """
    R = np.asarray(R, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    area = np.pi * (rbins[1:] ** 2 - rbins[:-1] ** 2)
    hist, _ = np.histogram(R, bins=rbins, weights=mass)
    return 0.5 * (rbins[:-1] + rbins[1:]), hist / area


def fit_exp_rd(R_kpc, sigma_kpc2, r50):
    """Exponential scale length from an outer-region fit.

    Fits ``log(Sigma) = -R/Rd + const`` over ``[0.5 R50, 2.5 R50]``. Returns
    ``NaN`` if fewer than 3 valid bins.
    """
    R = np.asarray(R_kpc, dtype=np.float64)
    s = np.asarray(sigma_kpc2, dtype=np.float64)
    sel = (R >= 0.5 * r50) & (R <= 2.5 * r50) & (s > 0)
    if sel.sum() < 3:
        return float("nan")
    p = np.polyfit(R[sel], np.log(s[sel]), 1)
    return float(-1.0 / p[0])


def f_dm_apertures(pos_s_al, m_cur, pos_dm_al, m_dm, apertures):
    """``f_dm`` in each aperture under both conventions.

    ``pos_s_al`` / ``pos_dm_al`` are star / DM positions in the aligned frame
    (kpc); ``m_cur`` the current stellar masses (Msun); ``m_dm`` the per-DM-
    particle mass (Msun). ``apertures`` maps ``name -> radius_kpc``. Returns
    ``{name: {"R_kpc", "f_dm_spherical", "f_dm_cyl_edgeon"}}``.
    """
    pos_s = np.asarray(pos_s_al, dtype=np.float64)
    pos_dm = np.asarray(pos_dm_al, dtype=np.float64)
    m_cur = np.asarray(m_cur, dtype=np.float64)

    r_dm_3d = np.linalg.norm(pos_dm, axis=1)
    r_s_3d = np.linalg.norm(pos_s, axis=1)
    # edge-on projection (LOS = +y): sky-plane coordinates = (x, z)
    r_dm_proj = np.hypot(pos_dm[:, 0], pos_dm[:, 2])
    r_s_proj = np.hypot(pos_s[:, 0], pos_s[:, 2])

    out = {}
    for name, rad in apertures.items():
        m_dm_sph = (r_dm_3d < rad).sum() * m_dm
        m_st_sph = m_cur[r_s_3d < rad].sum()
        m_dm_prj = (r_dm_proj < rad).sum() * m_dm
        m_st_prj = m_cur[r_s_proj < rad].sum()
        out[name] = {
            "R_kpc": round(float(rad), 3),
            "f_dm_spherical": round(float(m_dm_sph / (m_dm_sph + m_st_sph)), 4),
            "f_dm_cyl_edgeon": round(float(m_dm_prj / (m_dm_prj + m_st_prj)), 4),
        }
    return out
