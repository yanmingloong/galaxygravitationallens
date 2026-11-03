"""galaxylens: gravitational lens modeling library for strong lensing of disk and elliptical galaxies.

galaxylens: strong gravitational lens modeling for disk and elliptical galaxies.

Usage::

    import galaxylens as gl
    cfg = gl.load_config("configs/fit/j2141_barnabe.yaml")
    result = gl.fit(cfg)
    print(result.beta_z, result.m_star, result.f_dm(2.2 * cfg.pipeline.geometry.rd_arcsec))
"""

from __future__ import annotations

from galaxylens.api import (
    Pipeline,
    Stage,
    analyze,
    fit,
    fit_mein,
    fit_positions,
    identify,
    load_config,
    load_dataset,
)
from galaxylens.mock_pipeline import build_mock_pipeline as mock

__version__ = "0.1.0"

__all__ = [
    "fit",
    "fit_mein",
    "fit_positions",
    "load_config",
    "load_dataset",
    "analyze",
    "identify",
    "mock",
    "Pipeline",
    "Stage",
    "__version__",
]
