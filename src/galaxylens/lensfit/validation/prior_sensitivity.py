"""Prior-centre sensitivity via importance reweighting.

The mock-closure "finding five" tool: shift the M/L prior centre by ``delta``
dex and reweight the existing posterior instead of re-running the fit. This is
only reliable when the existing posterior covers the shifted centre (the ESS
must stay above a few percent); see :func:`ess`.

References
----------
- Importance reweighting: a standard nested-sampling post-processing step
  (e.g. the "prior sensitivity" tests common in mock-closure studies).
"""
from __future__ import annotations

import numpy as np


def ml_prior_reweight(lg_ml, w, lg_ml_center, sigma_dex=0.15):
    """Reweight samples by a Gaussian M/L prior centred at ``lg_ml_center``.

    The original prior is uniform (weight 1); the new prior is
    ``N(lg_ml_center, sigma_dex)``. Returns the renormalized weights.
    """
    lg_ml = np.asarray(lg_ml, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ratio = np.exp(-0.5 * ((lg_ml - lg_ml_center) / sigma_dex) ** 2)
    w_new = w * ratio
    s = w_new.sum()
    return w_new / s if s > 0 else w_new


def ess(w):
    """Kish effective sample size of the (unnormalized) weights."""
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))
