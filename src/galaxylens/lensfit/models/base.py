"""Component interfaces for the pluggable lensfit model layer.

``MassComponent`` unifies dark matter and stellar mass: both produce a
deflection field (via PyAutoLens) for imaging and an MGE surface density for
dynamics, so the assembler just iterates components and never branches on
"dark vs stellar". ``SourceModel`` produces light-only source profiles.

References
----------
- Cappellari 2002, MNRAS 333, 400 (MGE surface density convention)
- Barnabe et al. 2012, MNRAS 423, 1073 (mass model: gNFW + MGE x M/L)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ..structures import Calibration, Geometry, MGE, PriorSet


class MassComponent(ABC):
    """A mass component: dark matter OR stellar, unified behind one interface.

    In imaging both contribute deflection; in dynamics both contribute an MGE
    surface density to the potential. The stellar component additionally acts as
    the luminous tracer (``is_luminous``), so its MGE feeds ``model_lum`` in the
    Jeans engine as well as ``model_pot``.
    """

    name: ClassVar[str] = ""
    is_luminous: ClassVar[bool] = False

    @classmethod
    def from_config(cls, cfg):
        """Build this component from a ``PipelineConfig`` (zero-special-case
        instantiation for the assembler)."""
        raise NotImplementedError

    @abstractmethod
    def attach(self, lens, priors: PriorSet, geom: Geometry) -> None:
        """Attach this component's model + priors to the lens ``af.Model``."""

    @abstractmethod
    def dyn_mge(self, instance, geom: Geometry, calib: Calibration) -> MGE:
        """Return this component's ``(surf, sigma, q_obs)`` MGE from a fitted
        ``instance``. ``surf`` is a physical surface density (Msun/pc^2)."""

    def lum_mge(self, instance, geom: Geometry, calib: Calibration) -> MGE:
        """Luminous-tracer MGE (the light distribution feeding ``model_lum``).

        Defaults to :meth:`dyn_mge` -- valid when the M/L is constant, so light
        and mass share the same shape (Barnabe J2141). Components with a radial
        (non-constant) M/L override this to return the light MGE, which differs
        from the mass MGE (Trick J1331)."""
        return self.dyn_mge(instance, geom, calib)

    @property
    @abstractmethod
    def n_free_params(self) -> int:
        """Number of free parameters this component contributes."""


class SourceModel(ABC):
    """A light-only source model (single Sersic today, multi later)."""

    name: ClassVar[str] = ""

    @classmethod
    def from_config(cls, cfg):
        """Build this source model from a ``PipelineConfig``."""
        raise NotImplementedError

    @abstractmethod
    def attach(self, source, priors: PriorSet) -> None:
        """Attach the source light profile(s) + priors to the source ``af.Model``."""

    @property
    @abstractmethod
    def n_free_params(self) -> int:
        """Number of free parameters this source model contributes."""
