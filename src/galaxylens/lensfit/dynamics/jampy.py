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


def chi2_of(vrms_model, vrms_data, vrms_err) -> float:
    """Chi-squared of the model v_rms against the observed points."""
    return float(np.sum(((vrms_model - vrms_data) / vrms_err) ** 2))


class JampyEngine(DynamicsEngine):
    """jampy axisymmetric Jeans: v_rms (moment='zz') + optional V_circ."""

    name: ClassVar[str] = "jampy"
    n_free_params: ClassVar[int] = 1  # ratio (anisotropy)

    @classmethod
    def from_config(cls, cfg):
        return cls()

    def attach(self, lens, priors: PriorSet) -> None:
        # Zero-mass Gaussian carrying ``ratio`` as its M/L (B6/B7 convention):
        # it adds no deflection, only a free parameter for the anisotropy.
        import autolens as al
        import autofit as af

        lens.anisotropy = af.Model(al.mp.Gaussian)
        lens.anisotropy.centre = (0.0, 0.0)
        lens.anisotropy.sigma = 1.0
        lens.anisotropy.intensity = 0.0
        lens.anisotropy.ell_comps = (0.0, 0.0)
        lens.anisotropy.mass_to_light_ratio = af.UniformPrior(
            lower_limit=priors.ratio[0], upper_limit=priors.ratio[1]
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
        surf_pot, sig_pot, q_pot = model_pot

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
        info = {"chi2_vrms": chi2_vrms, "vrms_model": np.asarray(out.model)}
        chi2_dyn = chi2_vrms

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

        return chi2_dyn, info
