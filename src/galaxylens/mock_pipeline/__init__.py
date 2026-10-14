"""Mock data production and selection (simulated disk-lens analogs).

This package produces simulated data **with ground truth**, distinct from the
observational ``sample_library`` (real survey catalogs, no ground truth):

- ``selection/`` — TNG disk-lens analog selection (scan -> L1/L2/L3 -> catalog).
- ``skirt/``    — SKIRT radiative transfer -> mock images (pending).
- ``truth/``    — ground-truth computation shared by selection and validation.

Mock fitting later reuses ``lensfit`` and compares against the ground truth
from ``truth/`` in ``lensfit/validation/``.
"""
