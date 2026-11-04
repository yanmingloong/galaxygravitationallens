"""Radial (per-Gaussian) M/L stellar mass component (D4/D5 / J1331).

The D series uses a NON-constant M/L: the bulge and the inner/outer disk each
carry their own Salpeter M/L (Trick+2016 Table 7; bulge ~4.7, inner disk ~2.2,
outer disk ~4.7). The luminous tracer (light) therefore differs from the mass,
so ``lum_mge`` (light) and ``dyn_mge`` (mass) are distinct.

Honesty boundary: the D series (J1331) has no pixel-level imaging, so only the
dynamics side (``dyn_mge``/``lum_mge``) is validated against D4/D5. The lensing
deflection (``attach``) is not validated and raises ``NotImplementedError``.

References
----------
- Cappellari 2002, MNRAS 333, 400 (MGE)
- Trick, van de Ven & Dutton 2016, MNRAS 463, 3151 / arXiv:1609.05477
  (Salpeter bulge/disk M/L, Table 7)
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..base import MassComponent
from ...structures import Calibration, Geometry, MGE, PriorSet


class MgeRadialMl(MassComponent):
    """Stellar mass: fixed-structure MGE with a per-Gaussian (radial) M/L.

    ``lum_mge`` holds the light MGE ``(surf_lum, sigma, q)`` in Lsun/pc^2 (the
    I0,i peak surface brightness, Trick+2016 Table 3). ``ml_array`` is the
    per-Gaussian M/L, so the mass MGE is ``surf_lum * ml_array`` (Msun/pc^2).
    Zero free parameters in the validated D5 3-param MAP (the M/L is fixed).
    """

    name: ClassVar[str] = "mge_radial_ml"
    is_luminous: ClassVar[bool] = True

    def __init__(self, lum_mge: MGE, ml_array):
        surf, sig, q = lum_mge
        self._surf_lum = np.asarray(surf, dtype=float)
        self._sig = np.asarray(sig, dtype=float)
        self._q = np.asarray(q, dtype=float)
        self.ml_array = np.asarray(ml_array, dtype=float)
        if self.ml_array.shape != self._surf_lum.shape:
            raise ValueError("ml_array must match the MGE Gaussian count")

    @classmethod
    def from_config(cls, cfg):
        if cfg.stellar_radial is None:
            raise ValueError(
                "mge_radial_ml requires cfg.stellar_radial (a RadialStellarTemplate "
                "holding the absolute lum_mge + ml_array); set it in the J1331 recipe"
            )
        return cls(cfg.stellar_radial.lum_mge, cfg.stellar_radial.ml_array)

    def attach(self, lens, priors: PriorSet, geom: Geometry) -> None:
        raise NotImplementedError(
            "MgeRadialMl.attach (deflection) is not implemented: the D series has "
            "no pixel-level imaging, so only the dynamics side is validated."
        )

    def lum_mge(self, instance, geom: Geometry, calib: Calibration) -> MGE:
        return self._surf_lum, self._sig, self._q

    def dyn_mge(self, instance, geom: Geometry, calib: Calibration) -> MGE:
        return self._surf_lum * self.ml_array, self._sig, self._q

    @property
    def n_free_params(self) -> int:
        return 0  # fixed M/L in the validated D5 3-param MAP
