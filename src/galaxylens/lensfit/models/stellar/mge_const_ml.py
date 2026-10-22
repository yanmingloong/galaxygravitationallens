"""Constant-M/L stellar mass component from an MGE template (B6/B7 recipe).

The fixed stellar structure (Sersic bulge + exponential disk) is decomposed into
an MGE with the rbounds guard, then scaled by a single shared M/L. The CALIB
bridge (imaging flux scale vs dynamics mass scale) is applied explicitly so both
engines share the SAME stellar mass.

References
----------
- Cappellari 2002, MNRAS 333, 400 (MGE decomposition)
- Barnabe et al. 2012, MNRAS 423, 1073 (MGE x constant M/L stellar mass)
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.special import gammaincinv

from ..base import MassComponent
from ...structures import Calibration, Geometry, MGE, PriorSet, StellarTemplate
from ...mge.fit import fit_mge_1d


def stellar_mge_peak(
    pc_per_arcsec: float,
    n_bulge: float = 1.21,
    re_bulge: float = 0.26,
    q_bulge: float = 0.53,
    rd_disk: float = 1.51,
    q_disk: float = 0.31,
    disk_over_bulge: float = 4.0,
    psf_sigma_arcsec: float = 0.064,
    bulge_i0: float = 957.0,
) -> StellarTemplate:
    """Build the resolved stellar mass template (relative MGE + CALIB).

    Refactors B6/B7 ``make_stellar_mge_peak``: fits the Sersic bulge
    (n, Re, q) and exponential disk (Rd, q) with ``fit_mge_1d`` under the
    rbounds guard (sigma_min >= PSF sigma, else a sigma<PSF fake spike appears),
    and returns the relative peak surface-brightness MGE plus ``l2d_rel`` and
    ``calib``.

    ``calib = bulge_i0 / sum(surf_bulge)`` puts the template on the F814W flux
    scale; ``bulge_i0`` is the B4 Step1 F814W bulge central intensity (e-/s).
    """
    bn = gammaincinv(2 * n_bulge, 0.5)

    def sb(r_pc):
        return np.exp(-bn * ((r_pc / (re_bulge * pc_per_arcsec)) ** (1 / n_bulge) - 1))

    r_b = np.geomspace(0.003 * pc_per_arcsec, 15 * pc_per_arcsec, 200)
    psf_sig_pc = psf_sigma_arcsec * pc_per_arcsec
    tot_b, sig_b_pc = fit_mge_1d(r_b, sb(r_b), ngauss=10, psf_sigma=psf_sig_pc)
    peak_b = tot_b / (np.sqrt(2 * np.pi) * sig_b_pc)
    sig_b = sig_b_pc / pc_per_arcsec
    q_b = np.full_like(sig_b, q_bulge)

    r_d = np.geomspace(0.003 * pc_per_arcsec, 20 * pc_per_arcsec, 200)
    tot_d, sig_d_pc = fit_mge_1d(
        r_d, np.exp(-r_d / (rd_disk * pc_per_arcsec)), ngauss=8, psf_sigma=psf_sig_pc
    )
    l_b2d = float((tot_b * sig_b_pc * q_bulge).sum())
    l_d2d_raw = float((tot_d * sig_d_pc * q_disk).sum())
    peak_d = tot_d / (np.sqrt(2 * np.pi) * sig_d_pc) * (
        disk_over_bulge * l_b2d / l_d2d_raw
    )
    sig_d = sig_d_pc / pc_per_arcsec
    q_d = np.full_like(sig_d, q_disk)

    surf_rel = np.concatenate([peak_b, peak_d])
    sig = np.concatenate([sig_b, sig_d])
    q = np.concatenate([q_b, q_d])
    l2d_rel = float(
        (peak_b * 2 * np.pi * (sig_b * pc_per_arcsec) ** 2 * q_b).sum()
        + (peak_d * 2 * np.pi * (sig_d * pc_per_arcsec) ** 2 * q_d).sum()
    )
    calib = bulge_i0 / float(peak_b.sum())
    return StellarTemplate(
        mge=(surf_rel, sig, q), l2d_rel=l2d_rel, calibration=Calibration(calib)
    )


class MgeConstMl(MassComponent):
    """Stellar mass: fixed-structure MGE with a single shared constant M/L.

    One free parameter: ``ml_img`` (shared across all Gaussians via a single
    ``af.Prior`` object; one prior per Gaussian would silently turn it into N
    independent parameters).
    """

    name: ClassVar[str] = "mge_const_ml"
    is_luminous: ClassVar[bool] = True

    def __init__(self, template: StellarTemplate):
        self.template = template

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.stellar_template)

    def attach(self, lens, priors: PriorSet, geom: Geometry) -> None:
        import autolens as al
        import autofit as af

        surf_rel, sig, q = self.template.mge
        calib = self.template.calibration.calib
        ml_prior = af.LogUniformPrior(
            lower_limit=priors.ml_img[0], upper_limit=priors.ml_img[1]
        )
        for i, (sr, s, qq) in enumerate(zip(surf_rel, sig, q)):
            e = (1.0 - qq) / (1.0 + qq)
            g = af.Model(al.mp.Gaussian)
            g.centre = (0.0, 0.0)
            g.sigma = s
            g.intensity = sr * calib
            g.ell_comps = (-e, 0.0)
            g.mass_to_light_ratio = ml_prior
            setattr(lens, f"mge_{i}", g)

    def dyn_mge(self, instance, geom: Geometry, calib: Calibration) -> MGE:
        ml_img = float(instance.galaxies.lens.mge_0.mass_to_light_ratio)
        surf_rel, sig, q = self.template.mge
        surf_s = surf_rel * calib.calib * ml_img  # peak x ml_img x CALIB = Msun/pc^2
        return surf_s, sig, q

    @property
    def n_free_params(self) -> int:
        return 1
