"""Disk-galaxy kinematics from stellar particles (circularity method).

Ground-truth computation shared by `mock_pipeline.selection` (L3 disk
criterion) and `lensfit.validation` (mock truth comparison). All functions
are pure NumPy — no file I/O. Particle loading (HDF5 -> physical units)
lives in `selection/download.py`; SKIRT conversion lives in `skirt/`.

Circularity convention (Abadi+2003 style)::

    kappa = j_z / (R * v_c(R))

where j_z is the specific angular momentum along the disk axis, R the
cylindrical radius about that axis, and v_c the circular velocity from the
enclosed mass. A particle is ``disk`` if ``kappa >= kappa_thresh``
(default 0.7). ``f_disk`` is the mass-weighted disk fraction and the
bulge-to-disk ratio is ``B/D = (1 - f_disk) / f_disk``.

The disk axis is the angular-momentum axis of the disk particles
(iterated once from the total angular momentum), NOT the total angular
momentum and NOT the inertia tensor — those give different axes for
perturbed or near-spherical systems.
"""

import numpy as np

from .cosmology import GRAV, H

KAPPA_THRESH = 0.7


def unwrap(pos_raw, box):
    """Unwrap periodic-boundary coordinates to a contiguous cloud.

    ``pos_raw`` is modified in place and returned; ``box`` is the simulation
    box size in the same units as ``pos_raw`` (comoving kpc/h).
    """
    med = np.median(pos_raw, axis=0)
    for i in range(3):
        pos_raw[:, i] -= box * np.round((pos_raw[:, i] - med[i]) / box)
    return pos_raw


def com_iterative(pos, mass, n_iter=3, shrink=0.5):
    """Iterative shrinking-sphere center of mass (robust to outliers)."""
    c = np.average(pos, axis=0, weights=mass)
    rad = np.percentile(np.linalg.norm(pos - c, axis=1), 90)
    for _ in range(n_iter):
        d = np.linalg.norm(pos - c, axis=1)
        m = d < rad
        if m.sum() < 100:
            break
        c = np.average(pos[m], axis=0, weights=mass[m])
        rad *= shrink
    return c


def angular_momentum(pos, vel, mass, center):
    """Total angular momentum about ``center``."""
    return np.sum(mass[:, None] * np.cross(pos - center, vel), axis=0)


def circularity(pos, vel, mass, center, axis):
    """Circularity kappa and cylindrical radius R for each particle.

    Returns ``(kappa, R_cyl)`` where ``kappa`` is clipped to [-2, 2] and
    ``R_cyl`` is the cylindrical radius about ``axis``. The enclosed-mass
    circular velocity is computed via sorted-radius cumulative mass
    interpolation.
    """
    r = pos - center
    u = axis / np.linalg.norm(axis)
    z = np.dot(r, u)
    R = np.sqrt(np.sum(r ** 2, axis=1) - z ** 2)
    jz = np.sum(np.cross(r, vel) * u, axis=1)
    rm = np.linalg.norm(r, axis=1)
    s = np.argsort(rm)
    me = np.cumsum(mass[s])
    mu = np.maximum(np.interp(rm, rm[s], me), 1e-10)
    vc = np.sqrt(GRAV * mu / np.maximum(rm, 1e-3))
    return np.clip(jz / np.maximum(rm * vc, 1e-10), -2, 2), R


def disk_axis(pos, vel, mass, center, kappa_thresh=KAPPA_THRESH):
    """Disk angular-momentum axis (unit vector), iterated once.

    First pass uses the total angular momentum as a trial axis; the final
    axis is the angular momentum of the particles with kappa >= threshold
    about that trial axis. This is the ``L_hat_disk`` convention.
    """
    l_tot = angular_momentum(pos, vel, mass, center)
    u_tot = l_tot / np.linalg.norm(l_tot)
    k1, _ = circularity(pos, vel, mass, center, u_tot)
    disk = k1 >= kappa_thresh
    l_disk = angular_momentum(pos[disk], vel[disk], mass[disk], center)
    return l_disk / np.linalg.norm(l_disk)


def disk_fraction(kappa, mass, kappa_thresh=KAPPA_THRESH):
    """Return ``(f_disk, bulge_to_disk)`` from circularity and masses.

    ``f_disk`` is the mass-weighted disk fraction; ``B/D`` is the
    bulge-to-disk ratio (``inf`` when no disk particle exists).
    """
    disk = kappa >= kappa_thresh
    m_total = mass.sum()
    m_disk = mass[disk].sum()
    f_disk = m_disk / m_total if m_total > 0 else 0.0
    bd = (m_total - m_disk) / m_disk if m_disk > 0 else np.inf
    return f_disk, bd


def counter_rotation_angle(pos, vel, mass, center, axis=None,
                           kappa_thresh=KAPPA_THRESH,
                           core_R=3.0, outer_R=(6.0, 20.0)):
    """Angle (deg) between core and outer-disk angular momenta (KDC test).

    A coherent counter-rotating core (KDC) has its angular momentum
    anti-parallel to the outer disk -> angle ~ 180 deg. A weak bias is
    90-140 deg; co-rotation is < 90 deg. Note a mild reversed rotation
    (~20 km/s) in a hot bulge is NOT a KDC: random bulge orbits give ~50%
    counter-rotating particles, so a weak bias must not be read as coherent
    counter-rotation (the 359159 false-positive lesson).

    ``axis`` defaults to the disk angular-momentum axis (``disk_axis``);
    cylindrical radius R is measured about it. ``core`` = R < core_R;
    ``outer`` = outer_R[0] < R < outer_R[1]. ``pos``/``vel``/``mass`` are
    physical units (kpc, km/s, Msun), already unwrapped.
    """
    if axis is None:
        axis = disk_axis(pos, vel, mass, center, kappa_thresh)
    u = axis / np.linalg.norm(axis)
    r = pos - center
    R = np.sqrt(np.sum(r ** 2, axis=1) - np.dot(r, u) ** 2)
    core = R < core_R
    outer = (R > outer_R[0]) & (R < outer_R[1])
    if core.sum() == 0 or outer.sum() == 0:
        return 0.0
    l_core = angular_momentum(pos[core], vel[core], mass[core], center)
    l_outer = angular_momentum(pos[outer], vel[outer], mass[outer], center)
    cosang = np.dot(l_core, l_outer) / max(
        np.linalg.norm(l_core) * np.linalg.norm(l_outer), 1e-30)
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))


def disk_thickness(pos, center, axis, R_cyl, disk_mask, inner_R=20.0):
    """Median |z| of disk particles within cylindrical radius < inner_R kpc.

    Robust to the heavy tails that inflate std(|z|); this is the adopted
    disk-thickness measure.
    """
    r = pos - center
    z_disk = np.abs(np.dot(r[disk_mask], axis))
    inner = R_cyl[disk_mask] < inner_R
    return float(np.median(z_disk[inner])) if inner.sum() > 0 else 0.0


def disk_kinematics(pos, vel, mass, center, kappa_thresh=KAPPA_THRESH,
                    inner_R=20.0):
    """Full disk-kinematics summary from physical stellar particles.

    ``pos``/``vel``/``mass`` must already be in physical units (kpc,
    km/s, Msun) and unwrapped. Returns a dict with the ground-truth
    quantities used by both L3 selection and mock validation::

        f_disk, B_over_D, L_hat (unit vector), disk_thickness_med_kpc,
        disk_Re_kpc, re_kpc, mstar_msun, axis_angle_deg, kdc_angle_deg
    """
    l_tot = angular_momentum(pos, vel, mass, center)
    u_tot = l_tot / np.linalg.norm(l_tot)

    axis = disk_axis(pos, vel, mass, center, kappa_thresh)
    kappa, R_cyl = circularity(pos, vel, mass, center, axis)
    disk = kappa >= kappa_thresh

    f_disk, bd = disk_fraction(kappa, mass, kappa_thresh)
    thick = disk_thickness(pos, center, axis, R_cyl, disk, inner_R)

    re_disk = half_mass_radius(R_cyl[disk], mass[disk])
    re_all = half_mass_radius(np.linalg.norm(pos - center, axis=1), mass)

    ang = np.degrees(np.arccos(np.clip(np.abs(np.dot(u_tot, axis)), 0, 1)))

    # counter-rotating core diagnostic (KDC): angle between core (R<3 kpc)
    # and outer-disk (6<R<20 kpc) angular momenta about the disk axis.
    kdc = counter_rotation_angle(pos, vel, mass, center, axis=axis)

    # B/D is inf when no disk particle exists (round() would overflow);
    # collapse to the 999.0 sentinel used by the reference analyze().
    bd_val = round(float(bd), 2) if np.isfinite(bd) else 999.0

    return {
        "f_disk": round(float(f_disk), 4),
        "B_over_D": bd_val,
        "L_hat": [round(float(x), 5) for x in axis],
        "disk_thickness_med_kpc": round(float(thick), 2),
        "disk_Re_kpc": round(float(re_disk), 1),
        "re_kpc": round(float(re_all), 1),
        "mstar_msun": round(float(mass.sum()), 2),
        "axis_angle_deg": round(float(ang), 1),
        "kdc_angle_deg": round(float(kdc), 1),
    }


# Local import to avoid a circular dependency (measures.half_mass_radius).
from .measures import half_mass_radius  # noqa: E402
