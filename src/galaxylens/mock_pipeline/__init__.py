"""Mock data production and selection (simulated disk-lens analogs).

This package produces simulated data **with ground truth**, distinct from the
observational ``sample_library`` (real survey catalogs, no ground truth):

- ``selection/`` — TNG disk-lens analog selection (scan -> L1/L2/L3 -> catalog).
- ``skirt/``    — SKIRT radiative transfer -> mock images.
- ``truth/``    — ground-truth computation shared by selection and validation.
- ``lensing/``  — strong-lens arc ray-tracing on the true mass model.
- ``orchestrate/`` — the mock-closure Pipeline (select -> truth -> render -> arc
  -> fit -> closure), each stage still callable on its own.

Mock fitting reuses ``lensfit`` and compares against the ground truth from
``truth/`` in ``lensfit/validation/``.
"""

from .orchestrate import build_mock_pipeline

__all__ = ["build_mock_pipeline"]
