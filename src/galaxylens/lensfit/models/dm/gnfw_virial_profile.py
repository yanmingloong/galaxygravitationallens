"""PyAutoLens deflection profile for the Barnabe v_vir-normalized gNFW halo.

This module imports ``autolens`` at the top because the profile MUST subclass
``autolens.mp.gNFW``; it is imported lazily by ``GNFWVirial.attach`` so the
component module stays importable without autolens installed. The class is
top-level (stable module path) so autofit can pickle it for samplers.

Refactors ``b6_vvir_profile.gNFWVirialVelocity``: the cosmology/geometry is
computed from the constructor's ``redshift_object``/``redshift_source`` (not
module globals), with H0=70, Om0=0.3 fixed by ``halo_cosmo``.

References
----------
- Wyithe, Turner & Spergel 2001, ApJ 555, 504 (gNFW)
- Barnabe et al. 2012, MNRAS 423, 1073 (v_vir normalization, Eq. 7-11)
"""

from __future__ import annotations

import numpy as np

import autolens as al

from .gnfw_virial import halo_cosmo, vvir_to_rs


class GNFWVirialVelocity(al.mp.gNFW):
    """Elliptical gNFW halo normalized by the virial velocity v_vir (Barnabe+2012).

    Constructor parameters (the four free halo params, plus fixed geometry):
    ``v_vir`` [km/s], ``c_2`` (=c_{-2} = r_vir / r_{-2}), ``inner_slope`` (gamma),
    ``q_h`` (3-D axis ratio), ``inclination`` [deg], ``overdens`` (=200),
    ``redshift_object`` / ``redshift_source``.
    """

    def __init__(
        self,
        centre=(0.0, 0.0),
        v_vir=242.0,
        c_2=7.7,
        inner_slope=1.0,
        q_h=1.0,
        inclination=78.5,
        overdens=200.0,
        redshift_object=0.138,
        redshift_source=0.7127,
    ):
        c = halo_cosmo(redshift_object, redshift_source)
        r_s, delta_c, r_vir = vvir_to_rs(v_vir, inner_slope, c_2, q_h, c.h)
        rho_s = delta_c * c.rho_crit_kpc3  # Msun/kpc^3
        kappa_s = rho_s * r_s / c.sigma_crit
        scale_radius = r_s / c.kpc_per_arcsec  # arcsec

        # 3-D q_h -> projected q_obs -> ellipticity components (B4 convention).
        si = np.sin(np.radians(inclination))
        co = np.cos(np.radians(inclination))
        q_obs = np.sqrt(q_h ** 2 * si ** 2 + co ** 2)
        e = (1.0 - q_obs) / (1.0 + q_obs)
        ell_comps = (-e, 0.0)

        # Kept for reporting (M_vir etc.).
        self.v_vir = v_vir
        self.c_2 = c_2
        self.q_h = q_h
        self.inclination = inclination
        self.r_s_kpc = r_s
        self.r_vir_kpc = r_vir
        self.delta_c = delta_c
        self.overdens = overdens

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            kappa_s=kappa_s,
            inner_slope=inner_slope,
            scale_radius=scale_radius,
        )
