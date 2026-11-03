"""Joint lensing + dynamics analysis (Barnabe-style B6/B7 likelihood).

The ``JointAnalysis`` is built by ``joint_analysis`` (a factory) so that the
heavy ``autolens``/``autofit`` imports stay lazy; the class body is defined
inside the factory because its base class is chosen at import time: an
``ImagingConstraint`` (pixel-level arc) subclasses ``al.AnalysisImaging``, while
a point-level constraint (Mein / image positions) subclasses ``af.Analysis``.

References
----------
- Barnabe et al. 2012, MNRAS 423, 1073 (joint lens + dynamics likelihood)
"""

from __future__ import annotations

import numpy as np

from ..structures import Calibration, Geometry, KinData, MGE


def _concat_mge(parts) -> MGE:
    """Concatenate ``(surf, sigma, q_obs)`` MGE triples along the Gaussian axis."""
    surf = np.concatenate([p[0] for p in parts])
    sig = np.concatenate([p[1] for p in parts])
    q = np.concatenate([p[2] for p in parts])
    return surf, sig, q


def joint_analysis(
    lens_constraint,
    mass_components,
    engine,
    geometry: Geometry,
    kinematics: KinData,
    calibration: Calibration,
    n_source: int,
):
    """Build the joint-fit analysis from a pluggable ``LensingConstraint``.

    The resulting object is completely generic: it iterates the mass components
    to assemble the potential MGE (and the luminous tracer MGE), calls the
    dynamics engine, and combines the lensing and dynamics chi-squared with the
    B6/B7 reduced-chi2 normalization
    ``-0.5 * (chi2_lens/nu_lens + chi2_dyn/nu_dyn)``. It knows nothing about
    gNFW vs NFW, jampy vs AGAMA, or pixel-level vs point-level lensing -- the
    constraint supplies the lensing chi2, the engine supplies the dynamics chi2.

    Degrees of freedom follow B7 exactly:
    ``nu_lens = n_lens_data - n_source`` and
    ``nu_dyn = n_points - n_mass_params``.
    """
    import autofit as af
    import autolens as al

    from .constraints import ImagingConstraint

    base = (
        al.AnalysisImaging
        if isinstance(lens_constraint, ImagingConstraint)
        else af.Analysis
    )

    class _JointAnalysis(base):
        __module__ = __name__
        __qualname__ = "JointAnalysis"

        def __init__(
            self,
            lens_constraint,
            mass_components,
            engine,
            geometry,
            kinematics,
            calibration,
            n_source,
        ):
            if base is al.AnalysisImaging:
                super().__init__(dataset=lens_constraint.dataset, use_jax=False)
            else:
                super().__init__()
            self.lens_constraint = lens_constraint
            self.mass_components = mass_components
            self.engine = engine
            self.geometry = geometry
            self.kinematics = kinematics
            self.calibration = calibration
            self.n_source = n_source

        @property
        def n_mass_params(self) -> int:
            return sum(c.n_free_params for c in self.mass_components) + self.engine.n_free_params

        def log_likelihood_function(self, instance):
            chi2_lens = self.lens_constraint.chi2(
                instance, self.mass_components, self.geometry, self.calibration
            )

            lum_parts = []
            pot_parts = []
            for comp in self.mass_components:
                mge = comp.dyn_mge(instance, self.geometry, self.calibration)
                pot_parts.append(mge)
                if comp.is_luminous:
                    lum_parts.append(comp.lum_mge(instance, self.geometry, self.calibration))
            model_lum = _concat_mge(lum_parts)
            model_pot = _concat_mge(pot_parts)

            ratio = self.engine.ratio_from(instance)
            chi2_dyn, _ = self.engine.chi2(
                model_lum, model_pot, ratio, self.geometry, self.kinematics
            )

            nu_lens = self.lens_constraint.n_data_points - self.n_source
            # A single aperture dispersion (elliptical, 1 point) is under-constrained
            # by n_mass_params, so n_points - n_mass_params can be <= 0; a negative
            # reduced-chi2 degree of freedom would flip the dynamics chi2 sign and
            # blow up log L (J1630 first cluster run: log Z -> +18456). Clamp at 1.
            nu_dyn = max(1, self.kinematics.n_points - self.n_mass_params)
            return -0.5 * (chi2_lens / nu_lens + chi2_dyn / nu_dyn)

    # Give the class a stable module path so multiprocessing can pickle it
    # (a local class has no importable ``module.qualname``). Register it in this
    # module's namespace: on Linux (fork) workers inherit the registration; on
    # spawn platforms the dynamic base + lazy import prevent a module-level
    # definition, so single-core remains the fallback there.
    globals()["JointAnalysis"] = _JointAnalysis

    return _JointAnalysis(
        lens_constraint,
        mass_components,
        engine,
        geometry,
        kinematics,
        calibration,
        n_source,
    )
