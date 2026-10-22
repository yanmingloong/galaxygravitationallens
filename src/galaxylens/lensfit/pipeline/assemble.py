"""Config -> model assembly (zero physics).

Maps component NAMES in a ``PipelineConfig`` to component classes, instantiates
them via ``from_config``, attaches them to the lens/source ``af.Model``, and
builds the ``JointAnalysis``. All physics lives in the components; this module
only reads configuration and wires pieces together.
"""

from __future__ import annotations

from ..fit.joint import joint_analysis
from ..fit.constraints import ImagingConstraint
from ..dynamics.jampy import JampyEngine
from ..dynamics.jampy_kdc import JampyKdcEngine
from ..models.dm.gnfw_virial import GNFWVirial
from ..models.dm.nfw import NFW
from ..models.stellar.mge_const_ml import MgeConstMl
from ..models.stellar.mge_radial_ml import MgeRadialMl
from ..models.source.single_sersic import SingleSersic

# name -> component class. Adding a component = add an import + a dict entry.
_COMPONENT_REGISTRY = {
    GNFWVirial.name: GNFWVirial,
    NFW.name: NFW,
    MgeConstMl.name: MgeConstMl,
    MgeRadialMl.name: MgeRadialMl,
    SingleSersic.name: SingleSersic,
    JampyEngine.name: JampyEngine,
    JampyKdcEngine.name: JampyKdcEngine,
}


def _component(name: str, cfg):
    try:
        cls = _COMPONENT_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown component {name!r}; registered: {sorted(_COMPONENT_REGISTRY)}"
        ) from None
    return cls.from_config(cfg)


def _build(cfg):
    import autolens as al
    import autofit as af

    mass_components = [_component(cfg.stellar, cfg), _component(cfg.dark, cfg)]
    source_component = _component(cfg.source, cfg)
    engine = _component(cfg.dynamics, cfg)

    lens = af.Model(al.Galaxy, redshift=cfg.geometry.zlens)
    for comp in mass_components:
        comp.attach(lens, cfg.priors, cfg.geometry)
    engine.attach(lens, cfg.priors)

    source = af.Model(al.Galaxy, redshift=cfg.geometry.zsrc)
    source_component.attach(source, cfg.priors)

    collection = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    return collection, mass_components, source_component, engine


def build_collection(cfg):
    """Build the ``af.Collection`` model from a ``PipelineConfig`` (no dataset)."""
    return _build(cfg)[0]


def assemble(cfg, dataset, n_unmasked=None):
    """Build the model ``af.Collection`` and the ``JointAnalysis``.

    ``dataset`` is the (galaxy-subtracted, noise-rescaled) ``al.Imaging`` that
    the imaging branch of the likelihood fits; it is produced by the facade
    layer's dataset loader (step 2), not stored in the config. ``n_unmasked`` is
    the unmasked-pixel count (the imaging data-point count) produced by the same
    loader; it is set on the imaging constraint here so callers never poke into
    the analysis internals.

    The pixel-level imaging branch is the default lensing constraint;
    point-level constraints (Mein / image positions) are wired once the
    D-series components land.
    """
    collection, mass_components, source_component, engine = _build(cfg)
    constraint = ImagingConstraint(dataset)
    if n_unmasked is not None:
        constraint.n_unmasked = n_unmasked
    analysis = joint_analysis(
        constraint,
        mass_components,
        engine,
        cfg.geometry,
        cfg.kinematics,
        cfg.stellar_template.calibration,
        source_component.n_free_params,
    )
    return collection, analysis
