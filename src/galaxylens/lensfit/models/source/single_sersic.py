"""Single-Sersic source model (light only).

References
----------
- Barnabe et al. 2012, MNRAS 423, 1073 (single-Sersic source galaxy)
"""

from __future__ import annotations

from typing import ClassVar, Optional

from ..base import SourceModel
from ...structures import PriorSet, SourcePriors


class SingleSersic(SourceModel):
    """A single Sersic source galaxy (7 free parameters, B7 priors)."""

    name: ClassVar[str] = "single_sersic"

    def __init__(self, source_priors: Optional[SourcePriors] = None):
        self.source_priors = source_priors or SourcePriors()

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.source_priors)

    def attach(self, source, priors: PriorSet) -> None:
        import autolens as al
        import autofit as af

        sp = self.source_priors
        source.bulge = af.Model(al.lp.Sersic)
        source.bulge.centre_0 = af.GaussianPrior(mean=0.0, sigma=sp.centre_sigma)
        source.bulge.centre_1 = af.GaussianPrior(mean=0.0, sigma=sp.centre_sigma)
        source.bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=sp.ell_sigma)
        source.bulge.ell_comps.ell_comps_0.lower_limit = -sp.ell_bound
        source.bulge.ell_comps.ell_comps_0.upper_limit = sp.ell_bound
        source.bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=sp.ell_sigma)
        source.bulge.ell_comps.ell_comps_1.lower_limit = -sp.ell_bound
        source.bulge.ell_comps.ell_comps_1.upper_limit = sp.ell_bound
        source.bulge.intensity = af.LogUniformPrior(
            lower_limit=sp.intensity_bounds[0], upper_limit=sp.intensity_bounds[1]
        )
        source.bulge.effective_radius = af.UniformPrior(
            lower_limit=sp.effective_radius_bounds[0],
            upper_limit=sp.effective_radius_bounds[1],
        )
        source.bulge.sersic_index = af.UniformPrior(
            lower_limit=sp.sersic_index_bounds[0],
            upper_limit=sp.sersic_index_bounds[1],
        )

    @property
    def n_free_params(self) -> int:
        return 7
