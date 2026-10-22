"""Noise rescale for galaxy-subtracted lens fitting.

Barnabe's method: after subtracting the lens-galaxy light, the residual image
still carries systematic residuals (dust lane, imperfect bulge/disk subtraction)
that the raw measurement noise does not cover. The standard fix is to scale the
noise map by ``f = sqrt(reduced_chi2)`` before the arc fit, so that the fit's
chi2/px lands near 1.

Without this rescale the arc fit sees chi2/px ~ 30, the posterior collapses
into an over-narrow peak, and Nautilus can grind for days without converging
(observed: 2 days, no samples.csv written).

References
----------
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (SWELLS IV,
  galaxy-subtracted lens modelling with reduced-chi2 rescaling)
"""

from __future__ import annotations

import math


def noise_rescale_factor(chi2: float, n_unmasked: int) -> float:
    """Compute the noise rescale factor ``f = sqrt(chi2 / n_unmasked)``.

    Args:
        chi2: the chi-squared of the step whose residuals are to be absorbed
            (e.g. the Step-1 galaxy-subtraction chi2).
        n_unmasked: number of unmasked image pixels used in that chi2.

    Returns:
        ``f = sqrt(reduced chi2)``. Multiply the noise map by ``f`` so the
        subsequent fit's chi2/px is ~1.

    Raises:
        ValueError: if ``n_unmasked <= 0``.
    """
    if n_unmasked <= 0:
        raise ValueError(f"n_unmasked must be positive, got {n_unmasked}")
    return math.sqrt(chi2 / n_unmasked)
