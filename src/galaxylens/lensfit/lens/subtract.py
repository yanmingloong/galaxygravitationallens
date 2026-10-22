"""Galaxy-subtraction step: fit and subtract the lens-galaxy light (scipy).

Step 1 of Barnabe's galaxy-subtracted lens modelling: fit the lens-galaxy
stellar light with a Sersic bulge + exponential disk, then subtract it, leaving
the source arc (plus residual systematics) for the arc fit.

This uses ``scipy.optimize.least_squares`` rather than a nested sampler. The
bulge structure must be FIXED (R_e, n, ellipticity to literature values) --
only the bulge intensity and the disk (I / R_e / ellipticity) plus the centre
are free. If the bulge structure is left free, the bulge-disk degeneracy drives
the bulge to the prior edge AND makes a Nautilus bound covariance singular
(``LinAlgError: Singular matrix``).

References
----------
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (galaxy-subtracted
  lens modelling)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def subtract_stellar_light(
    dataset,
    zlens: float,
    bulge_re: float = 0.26,
    bulge_n: float = 1.21,
    bulge_e1: float = -0.036,
    x0: Tuple[float, ...] = (2.5, 1.5, 2.53, -0.527, 0.0, 0.0, 0.0),
    lo: Tuple[float, ...] = (1.0, -1.0, 0.5, -0.9, -0.4, -1.0, -1.0),
    hi: Tuple[float, ...] = (5.0, 4.0, 6.0, -0.1, 0.4, 1.0, 1.0),
    max_nfev: int = 3000,
):
    """Fit the lens-galaxy light (Sersic bulge + exponential disk) and subtract it.

    The fitted parameters are ``[log10 bulge_I, log10 disk_I, disk_Re, disk_e1,
    disk_e2, centre_0, centre_1]``; the bulge structure (R_e, n, e1) is fixed.

    Args:
        dataset: a masked/oversampled ``autolens.Imaging`` dataset.
        zlens: lens redshift.
        bulge_re: fixed bulge effective radius (arcsec).
        bulge_n: fixed bulge Sersic index.
        bulge_e1: fixed bulge ellipticity component e1.
        x0: initial guess for the 7 free parameters.
        lo: lower bounds for the 7 free parameters.
        hi: upper bounds for the 7 free parameters.
        max_nfev: max least_squares function evaluations.

    Returns:
        ``(light_galaxy, chi2)`` where ``light_galaxy`` is the best-fit
        ``autolens.Galaxy`` and ``chi2`` is the fit chi-squared (used later for
        the noise rescale).

    Note:
        ``autolens`` and ``scipy`` are imported lazily.
    """
    import autolens as al
    from scipy.optimize import least_squares

    x0 = np.asarray(x0, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    def make_light(p):
        bulge = al.lp.Sersic(
            centre=(p[5], p[6]),
            ell_comps=(bulge_e1, 0.0),
            intensity=10.0 ** p[0],
            effective_radius=bulge_re,
            sersic_index=bulge_n,
        )
        disk = al.lp.Sersic(
            centre=(p[5], p[6]),
            ell_comps=(p[3], p[4]),
            intensity=10.0 ** p[1],
            effective_radius=p[2],
            sersic_index=1.0,
        )
        return al.Galaxy(redshift=zlens, bulge=bulge, disk=disk)

    def residuals(p):
        tracer = al.Tracer(galaxies=[make_light(p)])
        fit = al.FitImaging(dataset=dataset, tracer=tracer)
        return np.asarray(fit.normalized_residual_map)

    res = least_squares(residuals, x0, bounds=(lo, hi), max_nfev=max_nfev)
    light = make_light(res.x)
    chi2 = 2.0 * res.cost
    return light, chi2
