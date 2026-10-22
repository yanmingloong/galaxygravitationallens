"""jampy Jeans engine with a counter-rotating core (KDC) + direct beta_z + PSF.

The D-series (J1331) dynamics differs from the B-series (J2141) engine in three
ways, all of which this engine encodes:

1. The anisotropy is ``beta_z`` DIRECTLY (``beta=betaz``), not the cylindrical
   ``ratio = sqrt(1 - beta_z)`` used by the B series.
2. A counter-rotating core modulates the rotation: ``kappa(R) = kappa_prime *
   (R^2 - R0^2) / (R^2 + R0^2)``, so ``v_rot = kappa(R) * V_kappa1`` and
   ``v_rms = sqrt(sigma_los^2 + v_rot^2)`` (two ``jam.axi.proj`` calls:
   ``moment='zz'`` for sigma_los, ``moment='z'`` with ``kappa=1`` for V_kappa1).
3. A PSF / seeing convolution (``normpsf``/``sigmapsf``) is applied.

References
----------
- Cappellari 2020, MNRAS 494, 4819 (jampy)
- Trick, van de Ven & Dutton 2016, MNRAS 463, 4594 / arXiv:1609.05477
  (counter-rotating core, tangential beta_z = -0.4)
"""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .base import DynamicsEngine
from ..structures import Geometry, KinData, MGE, PriorSet


class JampyKdcEngine(DynamicsEngine):
    """jampy axisymmetric Jeans with a counter-rotating core (J1331).

    One free parameter: ``beta_z`` (the anisotropy, direct). ``kappa_prime`` and
    ``r0_kappa`` (the KDC amplitude and zero-radius) and the seeing FWHM are
    fixed observational/structural values.
    """

    name: ClassVar[str] = "jampy_kdc"
    n_free_params: ClassVar[int] = 1  # beta_z

    def __init__(
        self,
        kappa_prime: float = 0.76,
        r0_kappa_arcsec: float = 2.2,
        seeing_fwhm_arcsec: float = 1.1,
    ):
        self.kappa_prime = kappa_prime
        self.r0_kappa = r0_kappa_arcsec
        # jampy's sigmapsf is the Gaussian sigma, not the FWHM
        self.sig_seeing = seeing_fwhm_arcsec / (2 * np.sqrt(2 * np.log(2)))

    @classmethod
    def from_config(cls, cfg):
        return cls()

    def attach(self, lens, priors: PriorSet) -> None:
        # Zero-mass Gaussian carrying beta_z as its M/L (same carrier hack as the
        # B-series ratio, but the value is beta_z directly, not ratio).
        import autolens as al
        import autofit as af

        lens.anisotropy = af.Model(al.mp.Gaussian)
        lens.anisotropy.centre = (0.0, 0.0)
        lens.anisotropy.sigma = 1.0
        lens.anisotropy.intensity = 0.0
        lens.anisotropy.ell_comps = (0.0, 0.0)
        lens.anisotropy.mass_to_light_ratio = af.UniformPrior(
            lower_limit=priors.beta_z[0], upper_limit=priors.beta_z[1]
        )

    def ratio_from(self, instance) -> float:
        """Return ``beta_z`` DIRECTLY (J1331 convention), not ratio=sqrt(1-beta_z)."""
        return float(instance.galaxies.lens.anisotropy.mass_to_light_ratio)

    def chi2(
        self,
        model_lum: MGE,
        model_pot: MGE,
        beta_z: float,
        geom: Geometry,
        kin: KinData,
    ) -> Tuple[float, dict]:
        import jampy as jam

        inc = geom.inclination_deg
        da = geom.da_mpc

        # sigma_los (no rotation, kappa=0, moment='zz')
        osig = jam.axi.proj(
            model_lum_proj=model_lum,
            model_pot_proj=model_pot,
            inc=inc,
            mbh=0.0,
            distance=da,
            xbin=kin.xbin,
            ybin=kin.ybin,
            beta=beta_z,
            kappa=0.0,
            normpsf=1.0,
            sigmapsf=self.sig_seeing,
            moment="zz",
            quiet=True,
            plot=False,
        )
        # V_kappa1 (kappa=1 rotation curve, moment='z')
        ov = jam.axi.proj(
            model_lum_proj=model_lum,
            model_pot_proj=model_pot,
            inc=inc,
            mbh=0.0,
            distance=da,
            xbin=kin.xbin,
            ybin=kin.ybin,
            beta=beta_z,
            kappa=1.0,
            normpsf=1.0,
            sigmapsf=self.sig_seeing,
            moment="z",
            quiet=True,
            plot=False,
        )

        R = np.asarray(kin.xbin)
        kappa_R = self.kappa_prime * (R ** 2 - self.r0_kappa ** 2) / (R ** 2 + self.r0_kappa ** 2)
        vrot = kappa_R * ov.model
        vrms = np.sqrt(osig.model ** 2 + vrot ** 2)

        chi2 = float(np.sum(((vrms - kin.vrms) / kin.vrms_err) ** 2))
        info = {
            "vrms_model": np.asarray(vrms),
            "sigma_los": np.asarray(osig.model),
            "vrot": np.abs(np.asarray(vrot)),
        }
        return chi2, info
