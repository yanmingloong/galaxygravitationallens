"""Scale-free lensing: quad image positions -> Rein/Mein (D1 / Trick+2016).

Fits the four image positions of a quadruple-image system with a scale-free
potential ``psi(R', theta) = R'^alpha F(theta)`` (Evans & Witt 2003), ``alpha=1``
fixed (flat rotation curve). This is the POINT-LEVEL lensing of J1331: the source
is a point-like QSO (four images), so the lensing information is the image
positions, not a pixel-level arc. The fit yields the Einstein radius
``Rein = a0/2`` and the Einstein mass ``Mein = pi Sigma_crit Rein^2``.

References
----------
- Evans & Witt 2003, MNRAS 345, 1351 (scale-free potential)
- Kochanek 1991, ApJ 373, 354 (source-plane chi2 with magnification weighting)
- Trick, van de Ven & Dutton 2016, MNRAS 463, 3151 / arXiv:1609.05477
  (WMAP5 cosmology, Table 4 image positions, Table 5 results)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class ScaleFreeFit:
    """Best-fit scale-free lens: params + derived Einstein quantities."""

    a0: float
    a2: float
    b2: float
    xs: float
    ys: float
    rein_arcsec: float
    rein_kpc: float
    mein_1e10: float
    mcrit_1e10: float
    chi2: float
    dof: int


def _F(theta, a0, a2, b2):
    return a0 / 2.0 + a2 * np.cos(2 * theta) + b2 * np.sin(2 * theta)


def _dF(theta, a0, a2, b2):
    return -2 * a2 * np.sin(2 * theta) + 2 * b2 * np.cos(2 * theta)


def _ddF(theta, a0, a2, b2):
    return -4 * a2 * np.cos(2 * theta) - 4 * b2 * np.sin(2 * theta)


def deflection(x, y, a0, a2, b2, alpha: float = 1.0):
    """Deflection angle ``alpha(theta)`` (lens eq. ``beta = theta - alpha(theta)``)."""
    R = np.hypot(x, y)
    theta = np.arctan2(y, x)
    ct, st = x / R, y / R
    F, dF = _F(theta, a0, a2, b2), _dF(theta, a0, a2, b2)
    Rfac = R ** (alpha - 1.0)
    return (Rfac * (alpha * ct * F - st * dF),
            Rfac * (alpha * st * F + ct * dF))


def magnification(x, y, a0, a2, b2):
    """Magnification tensor ``M = (I - Hessian psi)^{-1}``."""
    R = np.hypot(x, y)
    theta = np.arctan2(y, x)
    u, v = x / R, y / R
    g = (_F(theta, a0, a2, b2) + _ddF(theta, a0, a2, b2)) / R
    H = np.array([[g * v * v, -g * u * v], [-g * u * v, g * u * u]])
    return np.linalg.inv(np.eye(2) - H)


def _residuals(params, theta_obs, err_arcsec):
    a0, a2, b2, xs, ys = params
    beta = np.array([xs, ys])
    resid = []
    for x, y in theta_obs:
        ax, ay = deflection(x, y, a0, a2, b2)
        beta_i = np.array([x - ax, y - ay])
        d = magnification(x, y, a0, a2, b2) @ (beta - beta_i)
        resid += [d[0] / err_arcsec, d[1] / err_arcsec]
    return np.array(resid)


def fit_image_positions(theta_obs, err_arcsec, p0=None):
    """Fit the scale-free lens to the image positions. Returns ``(params, chi2)``."""
    if p0 is None:
        p0 = np.array([1.8, 0.0, 0.0, 0.1, 0.1])
    r = least_squares(
        _residuals, p0, args=(theta_obs, err_arcsec),
        xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=5000,
    )
    return r.x, float(np.sum(r.fun ** 2))


def rein_mein(a0, a2, b2, kpc_per_arcsec: float, sigma_crit: float):
    """``Rein = a0/2``; ``Mein = pi Sigma_crit Rein^2``; ``Mcrit`` (critical curve)."""
    rein_arcsec = a0 / 2.0
    rein_kpc = rein_arcsec * kpc_per_arcsec
    mein = np.pi * sigma_crit * rein_kpc ** 2 / 1e10
    th = np.linspace(0, 2 * np.pi, 2000)
    ff = _F(th, a0, a2, b2) + _ddF(th, a0, a2, b2)
    mcrit = (sigma_crit / 2.0) * np.trapezoid(ff ** 2, th) * kpc_per_arcsec ** 2 / 1e10
    return rein_arcsec, rein_kpc, mein, mcrit


def sigma_crit(zlens: float, zsrc: float, h0: float = 71.9, om0: float = 0.258) -> float:
    """``Sigma_crit = c^2/(4 pi G) Ds/(Dd Dds)`` in Msun/kpc^2 (WMAP5)."""
    from astropy import units as u
    from astropy.constants import G as Gconst, c as cconst
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=h0, Om0=om0)
    Dd = cosmo.angular_diameter_distance(zlens)
    Ds = cosmo.angular_diameter_distance(zsrc)
    Dds = cosmo.angular_diameter_distance_z1z2(zlens, zsrc)
    sc = (cconst ** 2 / (4 * np.pi * Gconst)) * (Ds / (Dd * Dds))
    return sc.to(u.Msun / u.kpc ** 2).value
