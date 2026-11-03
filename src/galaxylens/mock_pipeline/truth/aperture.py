"""Aperture velocity dispersion ``sigma_ap`` from stellar particles.

The elliptical-mock ground truth: the OBSERVED velocity dispersion within a
circular aperture on the sky. Unlike :mod:`.anisotropy` (intrinsic, inclination-
independent ``beta_z``), ``sigma_ap`` is a projected quantity and therefore
depends on the inclination.

Conventions (agreed in ``docs/roadmap``, elliptical-mock spec):

* **mass-weighted** (constant M/L approximation): each particle's weight is its
  stellar mass ``m_cur``.
* **dispersion, not rms**: ``sigma = sqrt(<v_los^2> - <v_los>^2)`` (mean
  removed), NOT ``v_rms = sqrt(<v_los^2>)`` which would fold in the mean
  streaming. For a slow-rotator (elliptical) the two nearly coincide; for a disk
  they do not.
* **circular aperture** on the sky: select ``R_sky < R_ap`` where ``R_sky`` is
  the projected radius. A square aperture (jampy's ``pixsize = 2*R_ap``) covers
  ~27% more area — do not mix the two.
* **no PSF convolution** in v1 (``R_ap >> PSF``; add later if needed).

Projection: particles are in the mock frame (x = major axis, z = disk normal),
velocities relative to the stellar centre of mass. Inclination ``i`` is the
angle between the LOS and the disk normal (0 = face-on, 90 = edge-on), the disk
tilted about the x-axis so the major axis stays projected. Then
``v_los = -v_y sin i + v_z cos i`` and ``R_sky = sqrt(x^2 + (y cos i + z sin i)^2)``.
The ``-v_y`` sign is irrelevant for the dispersion on an axisymmetric system
(``<v_y v_z> = 0``) but the exact cross-term sign must be checked against jampy
when the elliptical mock is actually run.

Pure NumPy; no file I/O.
"""
from __future__ import annotations

import numpy as np

MIN_PARTICLES = 50  # same floor as anisotropy.beta_z


def sigma_ap(pos_s_al, vel_s_rel, m_cur, incl_deg, R_ap_kpc):
    """Mass-weighted ``sigma_ap`` within a circular sky aperture ``R_sky < R_ap``.

    ``pos_s_al`` (N, 3) kpc, mock frame (x = major axis, z = disk normal);
    ``vel_s_rel`` (N, 3) km/s relative to the stellar COM; ``m_cur`` (N,) Msun;
    ``incl_deg`` inclination; ``R_ap_kpc`` aperture radius in the SAME units as
    ``pos_s_al`` (convert arcsec -> kpc via ``pc_per_arcsec`` in the caller).

    Returns ``(sigma_ap_km_s, n_star)``; ``sigma_ap`` is ``NaN`` when fewer than
    ``MIN_PARTICLES`` are enclosed.
    """
    pos = np.asarray(pos_s_al, dtype=np.float64)
    vel = np.asarray(vel_s_rel, dtype=np.float64)
    m = np.asarray(m_cur, dtype=np.float64)

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    _vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]

    si = np.sin(np.radians(incl_deg))
    co = np.cos(np.radians(incl_deg))

    v_los = -vy * si + vz * co
    R_sky = np.hypot(x, y * co + z * si)

    sel = R_sky < R_ap_kpc
    n = int(sel.sum())
    if n < MIN_PARTICLES:
        return float("nan"), n

    w = m[sel]
    v_mean = np.average(v_los[sel], weights=w)
    sigma2 = np.average((v_los[sel] - v_mean) ** 2, weights=w)
    return float(np.sqrt(sigma2)), n
