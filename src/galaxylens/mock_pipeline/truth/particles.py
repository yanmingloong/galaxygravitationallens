"""Particle loading and mock-frame alignment (inertia tensor).

Loads TNG stellar / dark-matter particle cutouts into physical units and
rotates them into the mock observation frame. The alignment convention
(``unwrap`` -> ``com_iterative`` -> ``inertia_rotation``, weighted by
``GFM_InitialMass``) matches the SKIRT mock frame produced by
``mock_pipeline.skirt.convert`` and is frozen: changing it would drift the
truth quantities away from the mock image coordinate system, silently
invalidating every mock-closure comparison downstream.

Units (all read from the HDF5 header attributes, never hard-coded)::

    H          = Header.HubbleParam
    A          = Header.Time            (= 1/(1+z))
    box        = Header.BoxSize
    pos_kpc    = Coordinates * A / H
    vel_km_s   = Velocities * sqrt(A)
    mass_msun  = Masses / H * 1e10
    m_dm_msun  = MassTable[1] / H * 1e10   (DM mass is uniform; no Masses field)

References
----------
- Springel et al. 2018, MNRAS 475, 676 (TNG unit conventions)
- Nelson et al. 2019, ComAC 6, 2 (TNG particle data model)
"""
from __future__ import annotations

from typing import Tuple

import h5py
import numpy as np


def load_particles(
    star_path: str, dm_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, float, float, float, float, float]:
    """Load star + DM cutouts and convert to physical units.

    Returns ``(pos_s_kpc, vel_s_phys, m_cur_msun, m_init_msun, pos_dm_kpc,
    m_dm, H, A, box, z)``. Star positions are unwrapped in place (periodic
    boundary); velocities/masses are the raw physical values. ``pos_s`` is
    already unwrapped but NOT yet centred/rotated; callers apply
    ``kinematics.com_iterative`` + :func:`inertia_rotation`.
    """
    with h5py.File(star_path, "r") as f:
        hdr = dict(f["Header"].attrs)
        pt4 = f["PartType4"]
        pos_s = pt4["Coordinates"][:].astype(np.float64)
        vel_s = pt4["Velocities"][:].astype(np.float64)
        m_cur = pt4["Masses"][:].astype(np.float64)
        m_init = pt4["GFM_InitialMass"][:].astype(np.float64)
    with h5py.File(dm_path, "r") as f:
        pos_dm = f["PartType1"]["Coordinates"][:].astype(np.float64)

    H = float(hdr["HubbleParam"])
    A = float(hdr["Time"])
    box = float(hdr["BoxSize"])
    z = float(hdr["Redshift"])
    mass_table = np.asarray(hdr["MassTable"], dtype=np.float64)  # 1e10 Msun/h

    # unwrap: reuse the shared helper to keep one authoritative implementation
    from .kinematics import unwrap

    pos_s = unwrap(pos_s.copy(), box)
    pos_dm = unwrap(pos_dm.copy(), box)

    pos_s_kpc = pos_s * A / H
    pos_dm_kpc = pos_dm * A / H
    vel_s_phys = vel_s * np.sqrt(A)          # km/s
    m_cur_msun = m_cur / H * 1e10
    m_init_msun = m_init / H * 1e10
    m_dm = mass_table[1] / H * 1e10          # Msun per particle

    return (pos_s_kpc, vel_s_phys, m_cur_msun, m_init_msun, pos_dm_kpc,
            m_dm, H, A, box, z)


def inertia_rotation(pos_rel: np.ndarray, mass: np.ndarray):
    """Principal-axes rotation from the mass-weighted inertia tensor.

    ``pos_rel`` are positions relative to the centre; ``mass`` the particle
    masses. Returns ``(R, evals)`` where ``evals`` is ascending
    ``[min, mid, max]`` and ``R`` has rows ``[long axis, mid axis, normal]``
    so that ``pos @ R.T`` lands the long axis on x and the disk normal on z.
    """
    r = np.asarray(pos_rel, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    S = np.einsum("i,ij,ik->jk", m, r, r) / m.sum()
    evals, evecs = np.linalg.eigh(S)              # ascending [min, mid, max]
    R = np.vstack([evecs[:, 2], evecs[:, 1], evecs[:, 0]])
    return R, evals
