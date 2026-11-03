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


def stellar_mge_kband(
    mge_L,
    mge_sigma_arcsec,
    mge_q,
    pixel_scale: float,
    mstar_ref: float,
    ml_ref: float,
    sigma_crit_arcsec2: float,
    pc_per_arcsec: float,
) -> StellarTemplate:
    """Build the K'-band stellar template from a literature MGE (B9 calibration).

    Reads a literature K'-band MGE (per-Gaussian TOTAL flux ``L_k``, intrinsic
    sigma in arcsec, axial ratio ``q``) and produces a :class:`StellarTemplate`
    with the TWO-sided B9 calibration:

    - ``calibration`` (dynamics bridge): ``surf_s = peak * calib * ml_img`` is
      Msun/pc^2, with ``calib = L_ref / l2d_rel`` and ``L_ref = mstar_ref /
      ml_ref``. This is the B9 ``SURF_FACTOR`` (``L_K_REF / L_rel / pc_per_as^2``).
    - ``calibration_img`` (imaging bridge): ``g.intensity = peak * calib_img`` is
      Sigma_crit-normalised (``al.mp.Gaussian``), with ``calib_img = L_ref /
      (L_rel_arcsec2 * sigma_crit_arcsec2)``. This is the B9 ``INTENSITY``
      (``stellar_mge_intensity``).

    The peak surface brightness follows the B9 ``PEAK_STAR`` convention
    ``peak = L_k / (sqrt(2 pi) * sigma_px)`` with ``sigma_px = sigma_arcsec /
    pixel_scale`` (a 1-D Gaussian peak, matching ``b9_kband_cluster.py``).

    Args:
        mge_L: per-Gaussian total flux ``L_k`` (array).
        mge_sigma_arcsec: per-Gaussian intrinsic sigma in arcsec (array).
        mge_q: per-Gaussian axial ratio (array).
        pixel_scale: pixel scale in arcsec/pixel.
        mstar_ref: reference total stellar mass (Msun), e.g. 1.32e11.
        ml_ref: reference Salpeter M/L (Msun/Lsun), e.g. 0.8 for K'-band.
        sigma_crit_arcsec2: critical surface density in Msun/arcsec^2
            (``halo_cosmo(zl, zs).sigma_crit * kpc_per_arcsec^2``).
        pc_per_arcsec: physical scale in pc/arcsec at the lens redshift.

    Returns:
        A :class:`StellarTemplate` whose ``calibration`` is the dynamics bridge
        and ``calibration_img`` the imaging bridge, with ``l2d_rel`` in pc^2 so
        ``M* = ml_img * calib * l2d_rel = ml_img * L_ref``.

    References
    ----------
    - Barnabe et al. 2012, MNRAS 423, 1073 (K'-band stellar MGE, M* = 1.32e11)
    - Cappellari 2002, MNRAS 333, 400 (MGE peak/luminosity conventions)
    """
    L = np.asarray(mge_L, dtype=float)
    sig_arc = np.asarray(mge_sigma_arcsec, dtype=float)
    q = np.asarray(mge_q, dtype=float)

    # Peak surface brightness (B9 PEAK_STAR, 1-D Gaussian peak convention).
    peak = L / (np.sqrt(2.0 * np.pi) * (sig_arc / pixel_scale))

    # Relative total 2-D light, arcsec^2 and pc^2 (for the two bridges).
    l_rel_arcsec2 = float(np.sum(peak * 2.0 * np.pi * sig_arc ** 2 * q))
    l2d_rel = float(np.sum(peak * 2.0 * np.pi * (sig_arc * pc_per_arcsec) ** 2 * q))

    L_ref = mstar_ref / ml_ref
    c_dyn = L_ref / l2d_rel  # dynamics bridge: surf_s = peak * c_dyn * ml_img (Msun/pc^2)
    c_img = L_ref / (l_rel_arcsec2 * sigma_crit_arcsec2)  # imaging bridge (Sigma_crit-normalised)

    return StellarTemplate(
        mge=(peak, sig_arc, q),
        l2d_rel=l2d_rel,
        calibration=Calibration(c_dyn),
        calibration_img=Calibration(c_img),
    )


def stellar_mge_ellip(
    pc_per_arcsec: float,
    n: float = 4.0,
    re_arcsec: float = 2.0,
    q: float = 0.8,
    psf_sigma_arcsec: float = 0.1,
    bulge_i0: float = 1000.0,
) -> StellarTemplate:
    """Single-Sersic (no disk) stellar template for elliptical lenses.

    Elliptical lenses have a single Sersic light profile (de Vaucouleurs ``n=4``
    by default) and NO disk -- the elliptical counterpart of
    :func:`stellar_mge_peak` (which is a Sersic bulge + exponential disk for
    disk lenses). Decomposes one Sersic into an MGE with the rbounds guard
    (``sigma_min >= PSF``, else a ``sigma < PSF`` fake spike appears) and returns
    a relative-peak :class:`StellarTemplate` with the single shared CALIB bridge
    (``calib = bulge_i0 / sum(peak)``), exactly the F814W convention of B7.

    Args:
        pc_per_arcsec: physical scale in pc/arcsec at the lens redshift.
        n: Sersic index (4.0 = de Vaucouleurs for a typical slow rotator).
        re_arcsec: effective radius in arcsec.
        q: observed axial ratio (``qL`` from the light fit).
        psf_sigma_arcsec: PSF Gaussian sigma in arcsec (the rbounds floor).
        bulge_i0: central surface brightness (e-/s) anchoring the CALIB bridge
            (the Sersic amplitude from the light fit).

    Returns:
        A :class:`StellarTemplate` (relative peak + CALIB, no disk).

    References
    ----------
    - Sersic 1968; de Vaucouleurs 1948 (n=4 profile)
    - Cappellari 2002, MNRAS 333, 400 (MGE decomposition + rbounds)
    - Barnabe et al. 2012, MNRAS 423, 1073 (MGE x M/L stellar mass)
    """
    bn = gammaincinv(2.0 * n, 0.5)

    def sb(r_pc):
        return np.exp(-bn * ((r_pc / (re_arcsec * pc_per_arcsec)) ** (1.0 / n) - 1.0))

    r = np.geomspace(0.003 * pc_per_arcsec, 15.0 * pc_per_arcsec, 200)
    psf_sig_pc = psf_sigma_arcsec * pc_per_arcsec
    tot, sig_pc = fit_mge_1d(r, sb(r), ngauss=10, psf_sigma=psf_sig_pc)
    peak = tot / (np.sqrt(2.0 * np.pi) * sig_pc)
    sig = sig_pc / pc_per_arcsec
    q_arr = np.full_like(sig, q)
    l2d_rel = float((peak * 2.0 * np.pi * (sig * pc_per_arcsec) ** 2 * q_arr).sum())
    calib = bulge_i0 / float(peak.sum())
    return StellarTemplate(
        mge=(peak, sig, q_arr), l2d_rel=l2d_rel, calibration=Calibration(calib)
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
        # Imaging intensity bridge (Sigma_crit-normalised). F814W shares one
        # `calibration` for both sides; B9 K'-band splits them.
        calib = (
            self.template.calibration_img.calib
            if self.template.calibration_img is not None
            else self.template.calibration.calib
        )
        if priors.ml_img_spec is not None:
            # Gaussian M/L prior (B9 K'-band: Salpeter M/L ~ N(0.8, 0.3)).
            spec = priors.ml_img_spec
            ml_prior = af.GaussianPrior(mean=spec.mean, sigma=spec.sigma)
            ml_prior.lower_limit = spec.lower
            ml_prior.upper_limit = spec.upper
        else:
            # Default: LogUniform over ml_img bounds (B7 F814W).
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
