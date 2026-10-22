"""Morphological measures from mock images and aligned star particles.

Two ellipticity conventions exist in this project — **do not confuse them**:

- **projected ellipticity** (2026-10-14 unified convention):
  ``1 - sqrt(lambda_min/lambda_max)`` from the intensity-weighted covariance
  of the projected image, with no threshold. Because the image includes
  bulge/halo light, nuclear-dominated galaxies look rounder. This is the
  quantity reported in the mock morphology table.
- **intrinsic axis ratio**: ``sigma_z/sigma_x`` of aligned particles (the
  3D disk thickness along the disk normal). It is *not* a projected
  ellipticity and must not be compared to one.

Extinction is ``A = -2.5 log10(sum_dust / sum_nodust)`` on SKIRT noiseless
output (with/without dust).

All functions are pure NumPy — file I/O (FITS / .dat loading) lives in the
caller.
"""

import numpy as np


def half_mass_radius(R, mass, frac=0.5):
    """Radius enclosing ``frac`` of the total mass (sorted-radius cumsum)."""
    R = np.asarray(R, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    idx = np.argsort(R)
    c = np.cumsum(mass[idx])
    if c[-1] <= 0:
        return 0.0
    return float(R[idx[np.searchsorted(c, c[-1] * frac)]])


def projected_ellipticity(image):
    """Projected ellipticity 1 - sqrt(lambda_min/lambda_max) of a 2D image.

    Uses the intensity-weighted covariance (second moments) of the whole
    image with no threshold. ``image`` is a 2D array (NaNs are zeroed).
    """
    img = np.nan_to_num(np.asarray(image, dtype=np.float64))
    img = np.squeeze(img)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    tot = img.sum()
    if tot <= 0:
        return 0.0
    cx = (xx * img).sum() / tot
    cy = (yy * img).sum() / tot
    dx = xx - cx
    dy = yy - cy
    cxx = (dx * dx * img).sum() / tot
    cyy = (dy * dy * img).sum() / tot
    cxy = (dx * dy * img).sum() / tot
    ev = np.linalg.eigvalsh([[cxx, cxy], [cxy, cyy]])
    if ev[1] <= 0:
        return 0.0
    return float(1.0 - np.sqrt(ev[0] / ev[1]))


def major_axis_angle(image):
    """Major-axis position angle (deg, CCW from the image x-axis).

    Uses the intensity-weighted covariance eigen-decomposition: the major
    axis is the eigenvector of the *larger* eigenvalue. This removes the
    +/-90 deg ambiguity of the classical ``0.5 * arctan2(2 mxy, mxx - myy)``
    formula, which returns a principal-axis direction but cannot tell major
    from minor without comparing the variances along the two candidate axes.

    ``image`` is a 2D array (NaNs zeroed, negatives clipped). Returns the
    angle in degrees in (-90, 90]; the axis is bidirectional so +180 deg is
    the same axis. For an unrotated SKIRT output this is the roll correction
    (``roll = -angle`` to bring the major axis horizontal).
    """
    img = np.nan_to_num(np.asarray(image, dtype=np.float64))
    img = np.squeeze(img)
    img = np.maximum(img, 0)
    tot = img.sum()
    if tot <= 0:
        return 0.0
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    cx = (xx * img).sum() / tot
    cy = (yy * img).sum() / tot
    dx = xx - cx
    dy = yy - cy
    mxx = (dx * dx * img).sum() / tot
    myy = (dy * dy * img).sum() / tot
    mxy = (dx * dy * img).sum() / tot
    cov = np.array([[mxx, mxy], [mxy, myy]])
    _evals, evecs = np.linalg.eigh(cov)   # ascending eigenvalues
    major = evecs[:, 1]                    # larger eigenvalue -> long axis
    ang = np.degrees(np.arctan2(major[1], major[0]))
    if ang > 90:
        ang -= 180
    elif ang <= -90:
        ang += 180
    return float(ang)


def intrinsic_axis_ratios(x, y, z, m):
    """Mass-weighted axis ratios (sigma_y/sigma_x, sigma_z/sigma_x, sigma_z/sigma_y).

    ``x``/``y``/``z`` are particle coordinates in an aligned frame (x =
    disk long axis, z = disk normal) and ``m`` the masses. The intrinsic
    disk flattening is ``1 - sigma_z/sigma_x``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    mtot = m.sum()
    mx = (x * m).sum() / mtot
    my = (y * m).sum() / mtot
    mz = (z * m).sum() / mtot
    sx = np.sqrt((m * (x - mx) ** 2).sum() / mtot)
    sy = np.sqrt((m * (y - my) ** 2).sum() / mtot)
    sz = np.sqrt((m * (z - mz) ** 2).sum() / mtot)
    return (sy / sx, sz / sx, sz / sy)


def extinction(sum_dust, sum_nodust):
    """Dust extinction A = -2.5 log10(sum_dust / sum_nodust) (mag).

    ``sum_dust`` / ``sum_nodust`` are total fluxes of the SKIRT noiseless
    output with and without dust.
    """
    return float(-2.5 * np.log10(sum_dust / sum_nodust))
