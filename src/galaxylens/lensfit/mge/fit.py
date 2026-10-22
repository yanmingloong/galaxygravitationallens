"""1-D MGE fitting wrapper around Cappellari's ``mgefit.fit_1d``.

The single most important pitfall in MGE fitting is guarded here: ``fit_1d``
must be given an explicit ``rbounds`` lower bound at the PSF sigma. Without it,
the fitter over-decomposes a Sersic flat core (n ~ 1) into a Gaussian with
``sigma < PSF`` -- a numerically sharp, physically unresolved spike. That spike
is only ~2% of the light but its central surface density scales as 1/sigma^2,
so it injects an unphysically dense core into any downstream mass model (in the
B-series it drove the stellar M/L to the prior floor and opened a 23% Einstein
mass deficit). Constraining ``sigma_min >= PSF sigma`` removes it; the price is
a larger central residual (~14%), which is honest.

References
----------
- Cappellari 2002, MNRAS 333, 400 (MGE fitting)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def fit_mge_1d(
    radius: np.ndarray,
    profile: np.ndarray,
    ngauss: int,
    psf_sigma: float,
    rbounds: Optional[Tuple[float, float]] = None,
    inner_slope: float = 2.0,
    outer_slope: float = 3.0,
    linear: bool = False,
    negative: bool = False,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 1-D profile with an MGE, guarding sigma_min >= PSF sigma.

    Thin wrapper over ``mgefit.fit_1d`` that injects a PSF-limited lower bound
    on the Gaussian sigmas unless the caller supplies explicit ``rbounds``.

    Args:
        radius: logarithmically sampled radii (same units as ``psf_sigma``).
        profile: profile values at ``radius``.
        ngauss: number of Gaussians to fit.
        psf_sigma: PSF Gaussian sigma in the same units as ``radius``. Used as
            the default ``rbounds`` lower bound to prevent ``sigma < PSF`` spikes.
        rbounds: explicit ``(sigma_min, sigma_max)`` override. Defaults to
            ``(psf_sigma, radius.max())``.
        inner_slope: forced inner power-law slope (``mgefit`` default 2).
        outer_slope: forced outer power-law slope (``mgefit`` default 3).
        linear: prioritize speed over accuracy (skip sigma optimization).
        negative: allow negative Gaussians.
        quiet: suppress ``mgefit`` console output.

    Returns:
        ``(total_counts, sigma)`` arrays, each of length ``ngauss`` (sorted by
        increasing sigma).

    Note:
        ``mgefit`` is imported lazily so this module is importable without the
        ``mgefit`` dependency installed.
    """
    import mgefit

    if rbounds is None:
        rbounds = (psf_sigma, float(np.max(radius)))

    m = mgefit.fit_1d(
        radius,
        profile,
        ngauss=ngauss,
        inner_slope=inner_slope,
        outer_slope=outer_slope,
        linear=linear,
        negative=negative,
        plot=False,
        quiet=quiet,
        rbounds=rbounds,
    )
    total_counts, sigma = m.sol
    return np.asarray(total_counts), np.asarray(sigma)
