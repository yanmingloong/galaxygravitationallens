"""Dark-matter fraction apertures for mock-closure comparison.

Two conventions, matching the truth conventions in
:mod:`galaxylens.mock_pipeline.truth.fdm`:

- **spherical** (primary): 3-D ball, via ``jam.mge.radial_mass``.
- **cylindrical edge-on** (secondary): edge-on cylinder
  ``sqrt(x^2 + z^2) < R`` along the line of sight, via deprojection +
  angular integration.

The cylindrical aperture is NOT ``jam.mge.cylindrical_mass`` (that is the 2-D
projected mass within a circle, a different convention). Note the argument
order of :func:`cylindrical_mass` — ``(surf, q_obs, sig_as, ...)``: the axis
ratio is argument 2 and the width is argument 3. An earlier version swapped
them, silently producing wrong cylinder masses (the 2026-10-26 fix).

All functions are pure: they take the already-built MGE arrays (stellar and
DM) plus a :class:`~galaxylens.lensfit.structures.Geometry`.

References
----------
- Cappellari 2008, MNRAS 390, 71 (oblate deprojection, eq. 14)
- Cappellari 2002, MNRAS 333, 400 (MGE mass integrals)
"""
from __future__ import annotations

import numpy as np


def deproject_q(q_obs, inc_deg):
    """Observed -> intrinsic axis ratio (oblate deprojection).

    ``q_intr = sqrt(q_obs^2 - cos^2 i) / sin i`` (Cappellari 2008 eq. 14).
    Returns NaN when the deprojection is unconstrained or invalid: a face-on
    view (i=0) or ``q_obs <= cos i`` (no oblate spheroid matches).
    """
    inc = np.radians(inc_deg)
    sin_i = np.sin(inc)
    q2 = np.asarray(q_obs, dtype=np.float64) ** 2 - np.cos(inc) ** 2
    q2 = np.where(q2 > 0, q2, np.nan)
    return np.sqrt(q2) / sin_i


def cylindrical_mass(surf_obs, q_obs, sig_as, R_as, geom, ntheta=200):
    """Edge-on cylindrical mass within projected radius ``R_as``.

    ``surf_obs`` is the peak surface density, ``q_obs`` the observed axial
    ratio, ``sig_as`` the Gaussian width (arcsec). Deprojects to the intrinsic
    axis ratio, converts the peak to its edge-on value (mass-conserving), then
    integrates the elliptical Gaussian over a circular aperture.
    """
    q_intr = deproject_q(q_obs, geom.inclination_deg)
    surf_edge = np.asarray(surf_obs, dtype=np.float64) \
        * np.asarray(q_obs, dtype=np.float64) / q_intr
    R_pc = R_as * geom.pc_per_arcsec
    sig_pc = np.asarray(sig_as, dtype=np.float64) * geom.pc_per_arcsec
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
    tot = 0.0
    for s, sg, qi in zip(surf_edge, sig_pc, q_intr):
        denom = np.cos(theta) ** 2 + np.sin(theta) ** 2 / qi ** 2
        sig_eff2 = sg ** 2 / denom
        integrand = sig_eff2 * (1.0 - np.exp(-R_pc ** 2 / (2.0 * sig_eff2)))
        tot += s * np.trapezoid(integrand, theta)
    return float(tot)


def spherical_f_dm(surf_lum, sig_s, q_s, surf_dm, sig_dm, q_obs_h, R_as, geom):
    """``f_dm`` in a 3-D ball of radius ``R_as`` (``jam.mge.radial_mass``)."""
    import jampy as jam

    m_star = jam.mge.radial_mass(
        surf_lum, sig_s, q_s, geom.inclination_deg, R_as, distance=geom.da_mpc
    )
    m_dm = jam.mge.radial_mass(
        surf_dm, sig_dm, np.full_like(surf_dm, q_obs_h), geom.inclination_deg,
        R_as, distance=geom.da_mpc,
    )
    return float(m_dm / (m_dm + m_star))


def cylindrical_f_dm(surf_lum, sig_s, q_s, surf_dm, sig_dm, q_obs_h, R_as, geom):
    """``f_dm`` in an edge-on cylinder of radius ``R_as``."""
    m_star = cylindrical_mass(surf_lum, q_s, sig_s, R_as, geom)
    m_dm = cylindrical_mass(surf_dm, np.full_like(surf_dm, q_obs_h), sig_dm,
                            R_as, geom)
    return float(m_dm / (m_dm + m_star))
