"""L3 particle verification (3-field cutout -> disk kinematics).

Downloads the 3-field stellar cutout and computes the ground-truth disk
quantities via ``truth.kinematics.disk_kinematics``. This is the L3 stage of
the project's ``l3_verify_cluster.py``, with the per-particle computation
delegated to the shared truth module (so L3 selection and mock validation
use identical conventions).
"""

import os

import h5py
import numpy as np

from ..truth.cosmology import BOX_BY_SIM, H, snap_to_a
from ..truth.kinematics import com_iterative, disk_kinematics, unwrap


def load_stars(hdf5, sim, snap):
    """Load the 3-field stellar cutout -> physical (pos kpc, vel km/s, mass Msun)."""
    with h5py.File(hdf5, "r") as f:
        pt = f["PartType4"]
        pos_raw = pt["Coordinates"][:].astype(np.float64)
        vel_raw = pt["Velocities"][:].astype(np.float64)
        mass_raw = pt["Masses"][:].astype(np.float64)
    a = snap_to_a(snap)
    box = BOX_BY_SIM.get(sim, 75000.0)
    pos = unwrap(pos_raw.copy(), box) * a / H     # physical kpc
    vel = vel_raw * np.sqrt(a)                     # physical km/s
    mass = mass_raw / H * 1e10                     # Msun
    return pos, vel, mass


def verify_one(sim, snap, sid, stars_dir):
    """Download (3-field) and verify one subhalo; return a result dict.

    On success the dict carries the ``disk_kinematics`` fields plus sim,
    snap, subhalo_id and n_particles; on failure it carries ``error``.
    """
    from .download import download_cutout
    hdf5 = os.path.join(stars_dir, f"{sim}_snap{snap}_id{sid}_stars.hdf5")
    if not (os.path.exists(hdf5) and os.path.getsize(hdf5) > 100_000):
        if not download_cutout(sim, snap, sid, "stars",
                               "Coordinates,Velocities,Masses", stars_dir):
            return {"snap": snap, "subhalo_id": sid, "error": "download failed"}
    try:
        pos, vel, mass = load_stars(hdf5, sim, snap)
    except Exception as e:
        return {"snap": snap, "subhalo_id": sid, "error": f"load failed: {e}"}

    center = com_iterative(pos, mass)
    res = disk_kinematics(pos, vel, mass, center)
    res["sim"] = sim
    res["snap"] = snap
    res["subhalo_id"] = sid
    res["n_particles"] = len(mass)
    return res
