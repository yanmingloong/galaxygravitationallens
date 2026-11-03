"""Mock-closure validation: compare fitted ``f_dm`` to particle truth.

This package computes the fitted dark-matter fraction under the same aperture
conventions as :mod:`galaxylens.mock_pipeline.truth.fdm` and quantifies the
bias against the particle truth, turning a joint fit into a closure test.
"""
from .fdm import (
    deproject_q, cylindrical_mass, spherical_f_dm, cylindrical_f_dm,
)
from .bias import bias, f_dm_bias_table
from .corner import plot_corner
from .prior_sensitivity import ml_prior_reweight, ess
from .diagnostics import fourier_modes
from .closure import sample_f_dm, closure_f_dm

__all__ = [
    "deproject_q", "cylindrical_mass", "spherical_f_dm", "cylindrical_f_dm",
    "bias", "f_dm_bias_table",
    "plot_corner", "ml_prior_reweight", "ess",
    "fourier_modes", "sample_f_dm", "closure_f_dm",
]
