"""Spherical NFW dark halo (v200 + c200), Trick+2016 / J1331.

The D series (J1331) uses a SPHERICAL NFW parameterized by the virial velocity
``v200`` and concentration ``c200``, in the WMAP5 cosmology (H0=71.9), NOT the
oblate gNFW (v_vir + inner_slope + q_h) of the B series (J2141). This module
holds the NFW physics and the pluggable ``MassComponent``.

Honesty boundary: the D series (J1331) is a quadruple-image system with NO
pixel-level imaging, so only the DYNAMICS side (``dyn_mge`` -> the NFW MGE for
jampy) is validated against the D4/D5 fits. The lensing deflection (``attach``)
is NOT validated and raises ``NotImplementedError``.

References
----------
- Navarro, Frenk & White 1997, ApJ 490, 493 (NFW profile)
- Wright & Brainerd 2000, ApJ 534, 34 (analytic projected NFW mass)
- Trick, van de Ven & Dutton 2016, MNRAS 463, 4594 / arXiv:1609.05477
  (v200 + c200, WMAP5 H0=71.9)
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.optimize import least_squares

from ..base import MassComponent
from ...structures import Calibration, Geometry, MGE, PriorSet

_H0 = 71.9  # WMAP5 (Dunkley+2009), Trick+2016 convention
_G = 4.302e-3  # pc (km/s)^2 / Msun


def nfw_params(v200: float, c200: float, h0: float = _H0):
    """NFW virial params -> ``(r200_pc, rs_pc, rho0)``."""
    r200_pc = (v200 / (10 * h0)) * 1e6
    rs_pc = r200_pc / c200
    rho_crit = 3 * (h0 / 1e6) ** 2 / (8 * np.pi * _G)
    delta_c = (200 / 3) * c200 ** 3 / (np.log(1 + c200) - c200 / (1 + c200))
    rho0 = rho_crit * delta_c
    return r200_pc, rs_pc, rho0


def nfw_projected_mass(R_pc, rs_pc: float, rho0: float):
    """Projected NFW mass ``M_proj(<R)`` (Wright & Brainerd 2000), R in pc."""
    R_pc = np.asarray(R_pc, dtype=float)
    x = R_pc / rs_pc
    g = np.empty_like(x, dtype=float)
    m = x < 1
    g[m] = np.log(x[m] / 2) + np.arccosh(1 / x[m]) / np.sqrt(1 - x[m] ** 2)
    m = np.isclose(x, 1)
    g[m] = 1 + np.log(0.5)
    m = x > 1
    g[m] = np.log(x[m] / 2) + np.arccos(1 / x[m]) / np.sqrt(x[m] ** 2 - 1)
    return 4 * np.pi * rho0 * rs_pc ** 3 * g


def mge_fit_cumulative(R_pc, M_nfw, ngauss: int = 10):
    """Fit the cumulative mass ``M(<R)`` with an MGE; returns ``(surf, sigma_pc)``."""
    sig_init = np.logspace(np.log10(R_pc.min() * 2), np.log10(R_pc.max() / 3), ngauss)

    def M_mge(params, R):
        surf = np.exp(params[:ngauss])
        sig = params[ngauss:]
        M = np.zeros_like(R)
        for k in range(ngauss):
            M += 2 * np.pi * sig[k] ** 2 * surf[k] * (1 - np.exp(-R ** 2 / (2 * sig[k] ** 2)))
        return M

    def resid(params):
        Mm = M_mge(params, R_pc)
        return np.log10(Mm) - np.log10(M_nfw)

    surf_init = (M_nfw[-1] / (2 * np.pi * sig_init ** 2).sum()) * np.ones(ngauss)
    p0 = np.concatenate([np.log(surf_init), sig_init])
    res = least_squares(resid, p0, max_nfev=20000, method="trf")
    surf_fit = np.exp(res.x[:ngauss])
    sig_fit = res.x[ngauss:]
    order = np.argsort(sig_fit)
    return surf_fit[order], sig_fit[order]


def nfw_halo_mge(v200: float, c200: float, pc_per_arcsec: float, ngauss: int = 10):
    """Spherical NFW MGE for dynamics; returns ``(surf, sigma_arcsec, q)``."""
    r200_pc, rs_pc, rho0 = nfw_params(v200, c200)
    R = np.logspace(0, 5.6, 80)
    M_nfw = nfw_projected_mass(R, rs_pc, rho0)
    surf_nfw, sig_nfw_pc = mge_fit_cumulative(R, M_nfw, ngauss=ngauss)
    sig_nfw_arc = sig_nfw_pc / pc_per_arcsec
    q_nfw = np.ones_like(sig_nfw_arc)
    return surf_nfw, sig_nfw_arc, q_nfw


class NFW(MassComponent):
    """Spherical NFW dark-matter component (Trick+2016 v200 + c200).

    Two free parameters: ``v200`` (km/s), ``c200`` (concentration). Only the
    dynamics MGE (``dyn_mge``) is validated by the D4/D5 fits; the deflection
    (``attach``) is not implemented because J1331 has no pixel-level imaging.
    """

    name: ClassVar[str] = "nfw"
    is_luminous: ClassVar[bool] = False

    def __init__(self, ngauss: int = 10):
        self.ngauss = ngauss

    @classmethod
    def from_config(cls, cfg):
        return cls()

    def attach(self, lens, priors: PriorSet, geom: Geometry) -> None:
        raise NotImplementedError(
            "NFW.attach (deflection) is not implemented: the D series (J1331 quad) "
            "has no pixel-level imaging, so only dyn_mge (dynamics) is validated."
        )

    def dyn_mge(self, instance, geom: Geometry, calib: Calibration) -> MGE:
        dark = instance.galaxies.lens.dark
        v200 = float(dark.v200)
        c200 = float(dark.c200)
        return nfw_halo_mge(v200, c200, geom.pc_per_arcsec, self.ngauss)

    @property
    def n_free_params(self) -> int:
        return 2  # v200, c200
