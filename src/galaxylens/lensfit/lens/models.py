"""Lens model builders for disk-galaxy lens fitting.

Provides the Barnabe-style disk-lens model: a generalized NFW dark halo
(``GNFWVirialVelocity``, v_vir-normalized with a free c_{-2}, inner slope gamma,
and 3-D axis ratio q_h) plus a Multi-Gaussian Expansion stellar mass with a
shared, spatially-constant M/L, plus a Sersic source. The stellar light is
subtracted in a separate step (see ``subtract.py``) but the stellar MASS still
acts as a lens, so it must be included here -- subtracting the light and
dropping the mass is the classic mistake that leaves the halo holding only ~14%
of the Einstein mass.

This is the PURE-LENSING builder (B4 convention). The halo is parameterized the
same way as the joint fit (v_vir / c_{-2} / gamma / q_h, see
``lensfit.models.dm.gnfw_virial``), but there is no dynamics engine or
anisotropy parameter. The gamma prior default is wider ([0.2, 2.0]) than the
joint fit's [0.3, 1.5] because lensing alone constrains gamma weakly.

References
----------
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (SWELLS IV:
  gNFW + MGE x M/L disk-lens model, galaxy-subtracted)
- Wyithe, Turner & Spergel 2001, ApJ 555, 504 (gNFW)
- Dutton & Maccio 2014 (c-M relation, the optional c_2 prior)
- Cappellari 2002, MNRAS 333, 400 (MGE)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from ..mge.io import MgeComponent
    from ..structures import PriorSpec


def build_lens_model(
    zlens: float,
    zsrc: float,
    mge: List["MgeComponent"],
    ml_bounds: Tuple[float, float] = (0.1, 20.0),
    v_vir_bounds: Tuple[float, float] = (150.0, 350.0),
    gamma_bounds: Tuple[float, float] = (0.2, 2.0),
    c_2_prior: Optional["PriorSpec"] = None,
    q_h_bounds: Tuple[float, float] = (0.4, 1.0),
    inclination_deg: float = 78.5,
):
    """Build the Barnabe-style disk-lens model (gNFWVirialVelocity + MGE x M/L + Sersic).

    Returns an ``autofit.Collection`` with a lens galaxy (dark
    ``GNFWVirialVelocity`` + N MGE stellar-mass Gaussians sharing one M/L prior)
    and a Sersic source galaxy. The 12 free parameters are: M/L (1, shared),
    gNFW v_vir + c_2 + inner_slope + q_h (4), source Sersic (7).

    Args:
        zlens: lens redshift.
        zsrc: source redshift.
        mge: list of ``(L_k, sigma_arcsec, q_obs)`` MGE components (see
            ``lensfit.mge.load_mge``).
        ml_bounds: ``(lower, upper)`` for the shared LogUniform M/L prior.
        v_vir_bounds: ``(lower, upper)`` for the Uniform v_vir prior [km/s].
        gamma_bounds: ``(lower, upper)`` for the Uniform inner_slope prior.
        c_2_prior: optional :class:`~lensfit.structures.PriorSpec` for the
            concentration c_{-2} (None = Uniform [3, 20]; a Gaussian PriorSpec
            encodes the Dutton & Maccio 2014 c-M relation).
        q_h_bounds: ``(lower, upper)`` for the Uniform 3-D halo axis ratio q_h.
        inclination_deg: fixed line-of-sight inclination (maps q_h -> ell_comps).

    Returns:
        ``autofit.Collection(galaxies=Collection(lens=..., source=...))``.

    Note:
        ``autolens``/``autofit``/``GNFWVirialVelocity`` are imported lazily.
    """
    import autolens as al
    import autofit as af

    from ..mge.io import mge_gaussian_parameters
    from ..models.dm.gnfw_virial_profile import GNFWVirialVelocity

    lens = af.Model(al.Galaxy, redshift=zlens)
    lens.dark = af.Model(GNFWVirialVelocity)
    lens.dark.centre = (0.0, 0.0)
    lens.dark.v_vir = af.UniformPrior(
        lower_limit=v_vir_bounds[0], upper_limit=v_vir_bounds[1]
    )
    if c_2_prior is not None and c_2_prior.kind == "gaussian":
        lens.dark.c_2 = af.GaussianPrior(mean=c_2_prior.mean, sigma=c_2_prior.sigma)
        lens.dark.c_2.lower_limit = c_2_prior.lower
        lens.dark.c_2.upper_limit = c_2_prior.upper
    else:
        lo = c_2_prior.lower if c_2_prior is not None else 3.0
        hi = c_2_prior.upper if c_2_prior is not None else 20.0
        lens.dark.c_2 = af.UniformPrior(lower_limit=lo, upper_limit=hi)
    lens.dark.inner_slope = af.UniformPrior(
        lower_limit=gamma_bounds[0], upper_limit=gamma_bounds[1]
    )
    lens.dark.q_h = af.UniformPrior(lower_limit=q_h_bounds[0], upper_limit=q_h_bounds[1])
    lens.dark.inclination = inclination_deg
    lens.dark.overdens = 200.0
    lens.dark.redshift_object = zlens
    lens.dark.redshift_source = zsrc

    # Stellar mass: fixed-structure MGE Gaussians sharing a single constant M/L.
    # The shared prior must be ONE object -- one prior per Gaussian would turn
    # M/L into N independent free parameters.
    ml_prior = af.LogUniformPrior(lower_limit=ml_bounds[0], upper_limit=ml_bounds[1])
    for i, g in enumerate(mge_gaussian_parameters(mge)):
        gg = af.Model(al.mp.Gaussian)
        gg.centre = (0.0, 0.0)
        gg.sigma = g["sigma"]
        gg.intensity = g["intensity"]
        gg.ell_comps = g["ell_comps"]
        gg.mass_to_light_ratio = ml_prior
        setattr(lens, f"mge_{i}", gg)

    source = af.Model(al.Galaxy, redshift=zsrc, bulge=al.lp.Sersic)
    source.bulge.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    source.bulge.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    source.bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    source.bulge.ell_comps.ell_comps_0.lower_limit = -0.95
    source.bulge.ell_comps.ell_comps_0.upper_limit = 0.95
    source.bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    source.bulge.ell_comps.ell_comps_1.lower_limit = -0.95
    source.bulge.ell_comps.ell_comps_1.upper_limit = 0.95
    source.bulge.intensity = af.LogUniformPrior(lower_limit=0.01, upper_limit=10.0)
    source.bulge.effective_radius = af.UniformPrior(lower_limit=0.05, upper_limit=3.0)
    source.bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=6.0)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))
