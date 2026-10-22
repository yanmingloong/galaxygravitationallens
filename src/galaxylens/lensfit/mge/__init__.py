"""Multi-Gaussian Expansion: fit, I/O, and deprojection (Cappellari 2002)."""

from __future__ import annotations

from .deproject import deproject, project_back
from .fit import fit_mge_1d
from .io import MgeComponent, load_mge, mge_gaussian_parameters, q_to_ell_comps

__all__ = [
    "MgeComponent",
    "load_mge",
    "q_to_ell_comps",
    "mge_gaussian_parameters",
    "fit_mge_1d",
    "deproject",
    "project_back",
]
