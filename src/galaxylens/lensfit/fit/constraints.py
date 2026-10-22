"""Pluggable lensing constraints: pixel-level imaging vs point-level.

The joint likelihood couples a lensing term and a dynamics term,

    logL = -0.5 * (w_lens * chi2_lens + w_dyn * chi2_dyn),

and the lensing chi-squared is supplied by a pluggable
:class:`LensingConstraint`. This lets a single ``JointAnalysis`` drive a
pixel-level arc fit (Barnabe J2141, an extended arc) and a point-level quad fit
(Trick J1331, four point-like images) by swapping the constraint alone -- the
mass components, dynamics engine, sampler and facade stay unchanged.

Three flavours are planned (each backed by a real fit):

- ``ImagingConstraint``: pixel-level arc fit via ``autolens.AnalysisImaging``
  (B series / J2141, extended arc). VALIDATED by B7.
- ``MeinConstraint``: single-value Einstein-mass constraint ``M_2D(<Rein)``
  against the measured Einstein mass (D5 / J1331 quad). VALIDATED by D5.
- ``PositionConstraint``: four image positions fitted with the scale-free
  potential (D1 / J1331 quad). VALIDATED by D1.

Important honesty boundary: J1331 is a quadruple-image system (four point-like
QSO images, no extended arc), so its lensing information is point-level -- image
positions and the Einstein mass -- NOT pixel-level imaging. The D-series
constraints therefore exercise the point-level path only; the deflection
(``attach``) side of the D-series mass components is NOT validated by any
pixel-level fit.

References
----------
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (arc imaging fit)
- Trick, van de Ven & Dutton 2016, MNRAS 463, 4594 / arXiv:1609.05477
  (quad image positions + Einstein mass)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..structures import Calibration, Geometry, MGE


def m2d_within(mge: MGE, rein_arcsec: float, pc_per_arcsec: float) -> float:
    """Projected mass of an MGE within ``rein_arcsec`` (aperture, Msun).

    Each Gaussian contributes ``2 pi (sigma_pc)^2 q surf (1 - exp(-R^2/2 sigma^2))``
    with ``sigma_pc = sigma_arcsec * pc_per_arcsec`` and the exponential argument
    evaluated in arcsec. The ``pc_per_arcsec`` factor converts the MGE width from
    arcsec to pc so the surface density (Msun/pc^2) integrates to a mass.

    NOTE (D5 unit pitfall): the original D5 ``M_star`` formula omitted the
    ``pc_per_arcsec**2`` factor (it squared ``sigma`` in arcsec against a surface
    density in Lsun/pc^2), underweighting M_star by ~4e6 and effectively turning
    the Einstein-mass constraint into "M_nfw ~= Mein". This helper is the
    CORRECTED form.
    """
    surf, sigma_arcsec, q = mge
    sigma_arcsec = np.asarray(sigma_arcsec, dtype=float)
    surf = np.asarray(surf, dtype=float)
    q = np.asarray(q, dtype=float)
    sigma_pc = sigma_arcsec * pc_per_arcsec
    enclosed = 1.0 - np.exp(-(rein_arcsec ** 2) / (2.0 * sigma_arcsec ** 2))
    return float(np.sum(2.0 * np.pi * sigma_pc ** 2 * q * surf * enclosed))


class LensingConstraint(ABC):
    """How the lensing observation constrains the mass model.

    ``chi2`` returns the plain (unreduced) lensing chi-squared of a fitted
    ``instance``; ``n_data_points`` is the number of lensing data points, used
    by the joint analysis for the imaging degrees-of-freedom.
    """

    name: ClassVar[str] = ""

    @abstractmethod
    def chi2(
        self,
        instance,
        mass_components,
        geom: Geometry,
        calib: Calibration,
    ) -> float:
        """Lensing chi-squared of a fitted ``instance``."""

    @property
    @abstractmethod
    def n_data_points(self) -> int:
        """Number of lensing data points."""


class ImagingConstraint(LensingConstraint):
    """Pixel-level arc fit via ``autolens.AnalysisImaging`` (Barnabe J2141).

    Wraps the galaxy-subtracted, noise-rescaled ``al.Imaging`` produced by the
    facade's dataset loader. ``n_unmasked`` (unmasked pixel count, set after
    masking) is the lensing data-point count.
    """

    name: ClassVar[str] = "imaging"

    def __init__(self, dataset):
        import autolens as al

        self.dataset = dataset
        self._analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
        self.n_unmasked = None  # set by the caller after masking

    def chi2(self, instance, mass_components, geom, calib) -> float:
        return self._analysis.fit_from(instance=instance).chi_squared

    @property
    def n_data_points(self) -> int:
        if self.n_unmasked is None:
            raise RuntimeError(
                "ImagingConstraint.n_unmasked is unset; set it after masking"
            )
        return self.n_unmasked


class MeinConstraint(LensingConstraint):
    """Single-value Einstein-mass constraint (D5 / J1331 quad).

    ``chi2 = ((M_2D(<Rein) - Mein) / (frac_err * Mein))^2`` where ``M_2D`` is the
    projected mass of the mass-model MGE within ``rein_arcsec`` (see
    :func:`m2d_within`). One data point.

    This is the point-level lensing term of the J1331 joint fit: the four image
    positions were digested into a single Einstein mass by the D1 scale-free fit,
    so the D5 joint likelihood constrains the mass model against that one number
    rather than against pixel-level imaging.
    """

    name: ClassVar[str] = "mein"

    def __init__(self, rein_arcsec: float, mein_msun: float, mein_frac_err: float = 0.10):
        self.rein_arcsec = rein_arcsec
        self.mein = float(mein_msun)
        self.mein_err = mein_frac_err * self.mein

    def chi2(self, instance, mass_components, geom, calib) -> float:
        m2d = 0.0
        for comp in mass_components:
            mge = comp.dyn_mge(instance, geom, calib)
            m2d += m2d_within(mge, self.rein_arcsec, geom.pc_per_arcsec)
        return ((m2d - self.mein) / self.mein_err) ** 2

    @property
    def n_data_points(self) -> int:
        return 1
