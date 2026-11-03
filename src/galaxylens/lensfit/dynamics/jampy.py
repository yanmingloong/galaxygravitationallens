"""jampy Jeans engine (Cappellari 2020).

Reproduces B7 ``dyn_chi2``: axisymmetric JAM with ``moment='zz'`` (v_rms) plus an
optional ``V_circ(2.2 Rd)`` term, both normalized into ``chi2_dyn``.

References
----------
- Cappellari 2020, MNRAS 494, 4819 (jampy)
- Barnabe et al. 2012, MNRAS 423, 1073 (v_rms + V_circ joint constraint)
"""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .base import DynamicsEngine
from ..structures import Geometry, KinData, MGE, PriorSet


def beta_cyl(ratio: float):
    """Cylindrical-alignment approximation (Cappellari 2020, eq. 37);
    ``ratio = sigma_theta / sigma_r``.

    Note: the source B5 script's docstring cited "Cappellari 2026"; that appears
    to be a typo for the JAM cylindrical-alignment relation in Cappellari 2020.
    """
    def f(r, th):
        beta = 1 - ratio ** (-2 * np.cos(2 * th))
        dbdt = -4 * np.log(ratio) * ratio ** (-2 * np.cos(2 * th)) * np.sin(2 * th)
        return beta, dbdt
    return f


def beta_z_to_ratio(beta_z: float) -> float:
    """``ratio = sqrt(1 - beta_z)`` (inverse of ``beta_z = 1 - ratio^2``)."""
    return float(np.sqrt(1.0 - beta_z))


def ratio_to_beta_z(ratio: float) -> float:
    """``beta_z = 1 - ratio^2``."""
    return float(1.0 - ratio ** 2)


def beta_cyl_beta_z(beta_z: float):
    """Cylindrical-alignment beta from a DIRECT ``beta_z`` free parameter.

    The MF-series (mock closure) convention: the free parameter is ``beta_z``
    itself, not ``ratio = sigma_theta / sigma_r``. Converts ``beta_z -> ratio``
    and reuses :func:`beta_cyl`, so both conventions share one relation.
    """
    return beta_cyl(beta_z_to_ratio(beta_z))


def chi2_of(vrms_model, vrms_data, vrms_err) -> float:
    """Chi-squared of the model v_rms against the observed points."""
    return float(np.sum(((vrms_model - vrms_data) / vrms_err) ** 2))


class JampyEngine(DynamicsEngine):
    """jampy axisymmetric Jeans: v_rms (moment='zz') + optional V_circ."""

    name: ClassVar[str] = "jampy"
    n_free_params: ClassVar[int] = 1  # ratio (anisotropy)

    def __init__(self, anisotropy_param: str = "ratio"):
        self.anisotropy_param = anisotropy_param  # "ratio" (B) | "beta_z" (MF)

    @classmethod
    def from_config(cls, cfg):
        return cls(anisotropy_param=getattr(cfg, "dynamics_anisotropy", "ratio"))

    def attach(self, lens, priors: PriorSet) -> None:
        # Zero-mass Gaussian carrying the anisotropy parameter as its M/L
        # (B6/B7 convention): it adds no deflection, only a free parameter.
        import autolens as al
        import autofit as af

        lens.anisotropy = af.Model(al.mp.Gaussian)
        lens.anisotropy.centre = (0.0, 0.0)
        lens.anisotropy.sigma = 1.0
        lens.anisotropy.intensity = 0.0
        lens.anisotropy.ell_comps = (0.0, 0.0)
        bounds = priors.beta_z if self.anisotropy_param == "beta_z" else priors.ratio
        if bounds is None:
            raise ValueError(
                "anisotropy_param='beta_z' but priors.beta_z is not set"
            )
        lens.anisotropy.mass_to_light_ratio = af.UniformPrior(
            lower_limit=bounds[0], upper_limit=bounds[1]
        )

    def ratio_from(self, instance) -> float:
        return float(instance.galaxies.lens.anisotropy.mass_to_light_ratio)

    def chi2(
        self,
        model_lum: MGE,
        model_pot: MGE,
        ratio: float,
        geom: Geometry,
        kin: KinData,
    ) -> Tuple[float, dict]:
        import jampy as jam

        inc = geom.inclination_deg
        da = geom.da_mpc
        if self.anisotropy_param == "beta_z":
            ratio = beta_z_to_ratio(ratio)
        surf_pot, sig_pot, q_pot = model_pot

        chi2_dyn = 0.0
        info = {}
        if len(kin.vrms) > 0:
            out = jam.axi.proj(
                model_lum,
                model_pot,
                inc,
                0.0,
                da,
                kin.xbin,
                kin.ybin,
                beta=beta_cyl(ratio),
                gamma=0.0,
                kappa=None,
                ml=1.0,
                moment="zz",
                pixsize=0.0,
                quiet=True,
                plot=False,
                nang=15,
                nlos=40,
            )
            chi2_vrms = chi2_of(out.model, kin.vrms, kin.vrms_err)
            info["chi2_vrms"] = chi2_vrms
            info["vrms_model"] = np.asarray(out.model)
            chi2_dyn += chi2_vrms

        if kin.vcirc is not None:
            if geom.rd_arcsec is None:
                raise ValueError(
                    "kin.vcirc is set but geometry.rd_arcsec is None; "
                    "V_circ(2.2 Rd) needs the disk scale length"
                )
            vcirc = jam.mge.vcirc(
                surf_pot,
                sig_pot,
                q_pot,
                inc,
                0.0,
                da,
                np.array([2.2 * geom.rd_arcsec]),
            )[0]
            chi2_ha = ((vcirc - kin.vcirc) / kin.vcirc_err) ** 2
            chi2_dyn += chi2_ha
            info["vcirc"] = float(vcirc)
            info["chi2_vcirc"] = float(chi2_ha)

        if kin.aperture_vrms is not None:
            # Single luminosity-weighted aperture dispersion sigma_ap (elliptical
            # lenses): jam.axi.proj convolves the model v_rms with the pixsize
            # (aperture size) + sigmapsf (PSF) kernel, so out.model at the centre
            # IS the light-weighted sigma_ap. The circular fibre is approximated
            # by a square aperture of side 2*radius (jampy's pixsize convention).
            ap_side = 2.0 * kin.aperture_radius
            psf_sig = kin.kin_psf_sigma or 0.0
            out_ap = jam.axi.proj(
                model_lum,
                model_pot,
                inc,
                0.0,
                da,
                np.array([0.0]),
                np.array([0.0]),
                beta=beta_cyl(ratio),
                gamma=0.0,
                kappa=None,
                ml=1.0,
                moment="zz",
                pixsize=ap_side,
                sigmapsf=psf_sig,
                normpsf=1.0,
                quiet=True,
                plot=False,
                nang=15,
                nlos=40,
            )
            sigma_ap_model = float(np.asarray(out_ap.model)[0])
            chi2_ap = ((sigma_ap_model - kin.aperture_vrms) / kin.aperture_vrms_err) ** 2
            chi2_dyn += chi2_ap
            info["sigma_ap"] = sigma_ap_model
            info["chi2_ap"] = float(chi2_ap)

        return chi2_dyn, info
