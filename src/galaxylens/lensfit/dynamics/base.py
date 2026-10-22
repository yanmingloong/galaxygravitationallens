"""Dynamics-engine interface (jampy today, AGAMA later).

References
----------
- Cappellari 2020, MNRAS 494, 4819 (jampy)
- Vasiliev 2019, MNRAS 482, 1525 (AGAMA)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Tuple

from ..structures import Geometry, KinData, MGE, PriorSet


class DynamicsEngine(ABC):
    """Jeans/dynamics engine behind one interface.

    ``chi2`` takes the luminous tracer MGE and the total potential MGE plus the
    explicit ``ratio = sqrt(1 - beta_z)`` and returns ``(chi2_dyn, info)``. The
    ``ratio`` is a dynamics-only free parameter: ``attach`` places its carrier
    (a zero-mass Gaussian holding ``ratio`` as its M/L, the B6/B7 convention) on
    the lens model, so each engine owns its own kinematic parameters and the
    assembler never hardcodes jampy specifics.
    """

    name: ClassVar[str] = ""
    n_free_params: ClassVar[int] = 0

    @classmethod
    def from_config(cls, cfg):
        """Build this engine from a ``PipelineConfig``."""
        raise NotImplementedError

    def attach(self, lens, priors: PriorSet) -> None:
        """Attach engine-specific free parameters (e.g. the ratio carrier).
        Default is a no-op: engines with no kinematic free parameters override."""

    @abstractmethod
    def ratio_from(self, instance) -> float:
        """Read the fitted ``ratio = sqrt(1 - beta_z)`` from ``instance``."""

    @abstractmethod
    def chi2(
        self,
        model_lum: MGE,
        model_pot: MGE,
        ratio: float,
        geom: Geometry,
        kin: KinData,
    ) -> Tuple[float, dict]:
        """Compute the dynamics chi-squared. Returns ``(chi2, info_dict)``."""
