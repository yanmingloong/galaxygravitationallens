"""gNFW dark halo normalized by the virial velocity v_vir (Barnabe+2012).

One component, one source of truth: the lensing profile (kappa_s / scale_radius
for PyAutoLens deflection) and the dynamics MGE (surf / sigma for jampy) are
BOTH derived from the same ``vvir_to_rs`` call, so the two engines can never see
inconsistent halos. The deflection profile lives in
``gnfw_virial_profile.py`` (it must subclass ``autolens.mp.gNFW``); this module
holds the cosmology/geometry physics and the pluggable ``MassComponent``.

References
----------
- Wyithe, Turner & Spergel 2001, ApJ 555, 504 (gNFW density profile)
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (v_vir normalization,
  Eq. 7-11: zeta integral, delta_c, r_vir)
"""

from __future__ import annotations

from typing import ClassVar, NamedTuple

import numpy as np
from scipy.integrate import quad
from astropy.cosmology import FlatLambdaCDM

from ..base import MassComponent
from ...structures import Calibration, Geometry, MGE, PriorSet

_C_LIGHT = 2.998e5  # km/s
_G_KPC = 4.302e-6  # kpc (km/s)^2 / Msun
_H0 = 70.0
_OM0 = 0.3


class HaloCosmo(NamedTuple):
    """Cosmology/geometry constants needed by the gNFW halo (Barnabe H0=70, Om=0.3)."""

    da_mpc: float
    pc_per_arcsec: float
    kpc_per_arcsec: float
    h: float
    rho_crit_kpc3: float
    rho_crit_pc3: float
    sigma_crit: float


def halo_cosmo(zlens: float, zsrc: float, h0: float = _H0, om0: float = _OM0) -> HaloCosmo:
    """Compute the distance / density / critical-density constants at ``zlens``.

    Matches ``b5_vvir.py`` (rho_crit) and ``b6_vvir_profile.py`` (Sigma_crit),
    with H0=70, Om=0.3 (the Barnabe convention).
    """
    cosmo = FlatLambdaCDM(H0=h0, Om0=om0)
    ez = np.sqrt(om0 * (1 + zlens) ** 3 + (1 - om0))
    d_l = cosmo.angular_diameter_distance(zlens).value  # Mpc
    d_s = cosmo.angular_diameter_distance(zsrc).value
    d_ls = cosmo.angular_diameter_distance(zlens, zsrc).value
    return HaloCosmo(
        da_mpc=d_l,
        pc_per_arcsec=d_l * np.pi / 0.648,
        kpc_per_arcsec=d_l * np.pi / 0.648 / 1000.0,
        h=h0 * ez / 100.0,
        rho_crit_kpc3=2.775e11 * (h0 / 100.0) ** 2 * ez ** 2 / 1e9,
        rho_crit_pc3=2.775e11 * (h0 / 100.0) ** 2 * ez ** 2 / 1e18,
        sigma_crit=_C_LIGHT ** 2
        / (4 * np.pi * _G_KPC)
        * (d_s * 1e3)
        / ((d_l * 1e3) * (d_ls * 1e3)),
    )


def zeta(c: float, gamma: float, qh: float) -> float:
    """Barnabe Eq. 10: oblate-halo normalization integral."""
    def f(tau):
        return tau ** (2 - gamma) * (1 + tau) ** (gamma - 3) / \
            np.sqrt(1 - (1 - qh ** 2) * tau ** 2 / c ** 2)
    return quad(f, 0.0, c)[0]


def vvir_to_rs(v_vir, gamma, c_minus2, qh, h):
    """Barnabe Eq. 7-11: return ``(r_s_kpc, delta_c, r_vir_kpc)``."""
    c = (2 - gamma) * c_minus2
    z1 = zeta(c, gamma, 1.0)
    zq = zeta(c, gamma, qh)
    r_vir = v_vir / (h * np.sqrt(zq / z1))  # kpc
    r_s = r_vir / c
    delta_c = (200 / 3) * c ** 3 / z1
    return r_s, delta_c, r_vir


def gnfw_halo_mge(v_vir, gamma, c_minus2, q_h, c: HaloCosmo, ngauss=14):
    """Projected gNFW MGE for dynamics from v_vir params (no autofit instance).

    Refactors ``b5_vvir.dark_halo_mge_vvir``; returns
    ``(surf_dm_sph, sig_dm_arcsec, r_s_kpc, delta_c, r_vir_kpc)``.
    """
    import mgefit as mge

    r_s, delta_c, r_vir = vvir_to_rs(v_vir, gamma, c_minus2, q_h, c.h)
    rb_pc = r_s * 1000.0  # pc
    r_pc = np.geomspace(0.05 * c.pc_per_arcsec, rb_pc * 20, 200)
    m = r_pc / rb_pc
    rho = delta_c * c.rho_crit_pc3 / (m ** gamma * (1 + m) ** (3 - gamma))
    mh = mge.fit_1d(r_pc, rho, ngauss=ngauss, quiet=1, plot=0)
    return mh.sol[0], mh.sol[1] / c.pc_per_arcsec, r_s, delta_c, r_vir


class GNFWVirial(MassComponent):
    """gNFW dark-matter component (Barnabe v_vir parametrization).

    Four free parameters: ``v_vir``, ``c_2`` (=c_{-2}), ``inner_slope`` (gamma),
    ``q_h``. Lensing deflection (``attach``) and dynamics MGE (``dyn_mge``) share
    the same ``vvir_to_rs`` physics.
    """

    name: ClassVar[str] = "gnfw_virial"
    is_luminous: ClassVar[bool] = False

    def __init__(self, zlens, zsrc, inclination_deg, h0=_H0, om0=_OM0):
        self.zlens = zlens
        self.zsrc = zsrc
        self.inclination_deg = inclination_deg
        self._c = halo_cosmo(zlens, zsrc, h0, om0)

    @classmethod
    def from_config(cls, cfg):
        g = cfg.geometry
        return cls(g.zlens, g.zsrc, g.inclination_deg)

    def _halo_mge(self, v_vir, gamma, c_minus2, q_h, ngauss=14):
        """Projected gNFW MGE for dynamics (see ``gnfw_halo_mge``)."""
        return gnfw_halo_mge(v_vir, gamma, c_minus2, q_h, self._c, ngauss)

    def attach(self, lens, priors: PriorSet, geom: Geometry) -> None:
        import autofit as af

        from .gnfw_virial_profile import GNFWVirialVelocity

        dark = af.Model(GNFWVirialVelocity)
        dark.centre = (0.0, 0.0)
        dark.v_vir = af.UniformPrior(
            lower_limit=priors.v_vir[0], upper_limit=priors.v_vir[1]
        )
        c2 = priors.c_2
        if c2 is not None and c2.kind == "gaussian":
            dark.c_2 = af.GaussianPrior(mean=c2.mean, sigma=c2.sigma)
            dark.c_2.lower_limit = c2.lower
            dark.c_2.upper_limit = c2.upper
        else:
            lo = c2.lower if c2 is not None else 3.0
            hi = c2.upper if c2 is not None else 20.0
            dark.c_2 = af.UniformPrior(lower_limit=lo, upper_limit=hi)
        dark.inner_slope = af.UniformPrior(
            lower_limit=priors.gamma[0], upper_limit=priors.gamma[1]
        )
        dark.q_h = af.UniformPrior(lower_limit=priors.q_h[0], upper_limit=priors.q_h[1])
        dark.inclination = self.inclination_deg
        dark.overdens = 200.0
        dark.redshift_object = self.zlens
        dark.redshift_source = self.zsrc
        lens.dark = dark

    def dyn_mge(self, instance, geom: Geometry, calib: Calibration) -> MGE:
        dark = instance.galaxies.lens.dark
        v_vir = float(dark.v_vir)
        gamma = float(dark.inner_slope)
        c_minus2 = float(dark.c_2)
        q_h = float(dark.q_h)
        surf_dm_sph, sig_dm, _, _, _ = self._halo_mge(v_vir, gamma, c_minus2, q_h)
        si = np.sin(np.radians(geom.inclination_deg))
        co = np.cos(np.radians(geom.inclination_deg))
        q_obs_h = np.sqrt(q_h ** 2 * si ** 2 + co ** 2)
        surf_dm = surf_dm_sph / q_obs_h  # oblate, mass-conserving
        return surf_dm, sig_dm, np.full_like(surf_dm, q_obs_h)

    @property
    def n_free_params(self) -> int:
        return 4
