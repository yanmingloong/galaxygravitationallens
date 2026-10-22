"""Derived quantities from a joint fit: f_dm, M_2D(<R_E), and total M*.

These are the "parameter -> physical quantity" conversions the B series worked
out (``b6_fdm.py``): the dark-matter fraction and enclosed masses are recomputed
from the fitted parameters via jampy's MGE mass integrals, so users never
re-derive the CALIB bridge or the mass integrals by hand.

References
----------
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (f_dm(2.2 Rd), M_2D(<R_E))
- Cappellari 2002, MNRAS 333, 400 (MGE surface density / mass integrals)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..lensfit.structures import Geometry, StellarTemplate
from ..lensfit.models.dm.gnfw_virial import gnfw_halo_mge, halo_cosmo


def m_star_total(ml_img: float, template: StellarTemplate) -> float:
    """Total stellar mass ``M* = ml_img * calib * l2d_rel`` (Msun)."""
    return ml_img * template.calibration.calib * template.l2d_rel


def f_dm_of(
    ml_img: float,
    v_vir: float,
    gamma: float,
    c_minus2: float,
    q_h: float,
    ref_arcsec: float,
    template: StellarTemplate,
    geom: Geometry,
) -> Tuple[float, float, float]:
    """Dark-matter fraction and enclosed masses within a circular aperture.

    Stellar mass = template MGE x (ml_img x calib); halo mass = the gNFW MGE at
    (v_vir, gamma, c_-2, q_h). Both integrated with ``jam.mge.radial_mass`` (the
    spherical-aperture convention ``b6_fdm.py`` uses for f_dm).

    Returns ``(f_dm, M_star, M_dm)`` in Msun.
    """
    import jampy as jam

    c = halo_cosmo(geom.zlens, geom.zsrc)
    surf_rel, sig, q = template.mge
    surf_s = surf_rel * template.calibration.calib * ml_img

    surf_dm_sph, sig_dm, _, _, _ = gnfw_halo_mge(v_vir, gamma, c_minus2, q_h, c)
    si = np.sin(np.radians(geom.inclination_deg))
    co = np.cos(np.radians(geom.inclination_deg))
    q_obs = np.sqrt(q_h ** 2 * si ** 2 + co ** 2)
    surf_dm = surf_dm_sph / q_obs  # oblate, mass-conserving

    m_star = jam.mge.radial_mass(
        surf_s, sig, q, geom.inclination_deg, ref_arcsec, distance=geom.da_mpc
    )
    m_dm = jam.mge.radial_mass(
        surf_dm,
        sig_dm,
        np.full_like(surf_dm, q_obs),
        geom.inclination_deg,
        ref_arcsec,
        distance=geom.da_mpc,
    )
    f_dm = m_dm / (m_dm + m_star)
    return f_dm, m_star, m_dm


def m2d_within(
    ml_img: float,
    v_vir: float,
    gamma: float,
    c_minus2: float,
    q_h: float,
    ref_arcsec: float,
    template: StellarTemplate,
    geom: Geometry,
) -> float:
    """Projected 2D mass within a circular aperture (``M_2D(<R)``), in Msun.

    Uses ``jam.mge.cylindrical_mass`` (the 2-D projected convention, matching
    ``b5_vvir`` / ``b6_vvir_profile``'s M_proj(<R_E) check).
    """
    import jampy as jam

    c = halo_cosmo(geom.zlens, geom.zsrc)
    surf_rel, sig, q = template.mge
    surf_s = surf_rel * template.calibration.calib * ml_img

    surf_dm_sph, sig_dm, _, _, _ = gnfw_halo_mge(v_vir, gamma, c_minus2, q_h, c)
    si = np.sin(np.radians(geom.inclination_deg))
    co = np.cos(np.radians(geom.inclination_deg))
    q_obs = np.sqrt(q_h ** 2 * si ** 2 + co ** 2)
    surf_dm = surf_dm_sph / q_obs

    surf_pot = np.concatenate([surf_s, surf_dm])
    sig_pot = np.concatenate([sig, sig_dm])
    q_pot = np.concatenate([q, np.full_like(surf_dm, q_obs)])
    return float(jam.mge.cylindrical_mass(surf_pot, sig_pot, q_pot, ref_arcsec, distance=geom.da_mpc))
