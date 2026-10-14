"""TNG particles -> SKIRT input files (with disk-axis alignment).

Converts the all-field stellar and gas cutouts into the ASCII ``.dat`` files
that SKIRT's ``ParticleSource`` / ``ParticleMedium`` consume::

    {label}_stars.dat   x y z h M_init Z age      (pc, pc, pc, pc, Msun, -, yr)
    {label}_dust.dat    x y z h M_dust             (pc, pc, pc, pc, Msun)

Disk-axis alignment uses the **inertia tensor** (three-axis rotation: disk
normal -> +z, long axis -> +x), computed directly from the particles. It does
**not** read the ``catalog.csv`` L_hat column — that column came from the
legacy ``rescan_disk_galaxies.py`` (old convention) and is not used here.

Dust follows the Torrey ISM prescription: ``M_dust = f_dust * Z_gas * M_gas``
with an ISM temperature-density cut, and ``f_dust = 0.2``.
"""

import os

import h5py
import numpy as np

from ..truth.cosmology import BOX_BY_SIM, H, cosmic_time, snap_to_a
from ..truth.kinematics import com_iterative, unwrap

F_DUST = 0.2
GAMMA = 5.0 / 3.0
MU = 0.6                       # mean molecular weight (approx.)
MP = 1.6726e-27                # kg
KB = 1.3806e-23                # J/K
# T[K] = CONST_T * u[(km/s)^2]
CONST_T = (GAMMA - 1) * MU * MP / KB * 1e6


def inertia_rotation(pos_rel, mass):
    """Three-axis alignment from the inertia tensor.

    Returns ``(R, evals)`` where the smallest eigenvector (shortest axis,
    the disk normal) maps to +z, the largest (long axis) to +x, and the
    middle to +y; ``pos @ R.T`` then has the long axis horizontal along x
    and the normal along z. ``evals`` is ascending [min, mid, max].
    """
    r = pos_rel
    S = np.einsum("i,ij,ik->jk", mass, r, r) / mass.sum()
    evals, evecs = np.linalg.eigh(S)          # ascending
    R = np.vstack([evecs[:, 2], evecs[:, 1], evecs[:, 0]])
    return R, evals


def load_stars(path, sim, snap):
    """Load all-field star cutout -> (pos_kpc, m_init, Z, age_yr, h_pc)."""
    a = snap_to_a(snap)
    box = BOX_BY_SIM.get(sim, 75000.0)
    with h5py.File(path, "r") as f:
        pt = f["PartType4"]
        pos_raw = pt["Coordinates"][:].astype(np.float64)
        m_init_raw = pt["GFM_InitialMass"][:].astype(np.float64)
        Z = pt["GFM_Metallicity"][:].astype(np.float64)
        a_form = pt["GFM_StellarFormationTime"][:].astype(np.float64)
        hsml_raw = pt["StellarHsml"][:].astype(np.float64)
    pos_raw = unwrap(pos_raw.copy(), box)
    pos_kpc = pos_raw * a / H
    m_init = m_init_raw / H * 1e10
    h_pc = hsml_raw * a / H * 1e3
    age_gyr = cosmic_time(a) - cosmic_time(np.clip(a_form, 1e-4, a))
    age_yr = age_gyr * 1e9
    return pos_kpc, m_init, Z, age_yr, h_pc


def load_gas(path, sim, snap):
    """Load all-field gas cutout -> (pos_kpc, m_gas, Z, T_K, rho_raw)."""
    a = snap_to_a(snap)
    box = BOX_BY_SIM.get(sim, 75000.0)
    with h5py.File(path, "r") as f:
        pt = f["PartType0"]
        pos_raw = pt["Coordinates"][:].astype(np.float64)
        rho_raw = pt["Density"][:].astype(np.float64)
        Z = pt["GFM_Metallicity"][:].astype(np.float64)
        u = pt["InternalEnergy"][:].astype(np.float64)
        m_raw = pt["Masses"][:].astype(np.float64)
    pos_raw = unwrap(pos_raw.copy(), box)
    pos_kpc = pos_raw * a / H
    m_gas = m_raw / H * 1e10
    T = CONST_T * u                              # K
    return pos_kpc, m_gas, Z, T, rho_raw


def write_source(out, pos_rel_kpc, m_init, Z, age_yr, h_pc):
    """Write the ParticleSource .dat (positions rotated, relative, in pc)."""
    pos = pos_rel_kpc * 1e3                     # kpc -> pc
    n = len(m_init)
    with open(out, "w") as f:
        f.write("# column 1: x (pc)\n# column 2: y (pc)\n# column 3: z (pc)\n"
                "# column 4: h (pc)\n# column 5: M_init (Msun)\n"
                "# column 6: Z (1)\n# column 7: t (yr)\n")
        for i in range(n):
            f.write(f"{pos[i, 0]:.4f} {pos[i, 1]:.4f} {pos[i, 2]:.4f} "
                    f"{h_pc[i]:.4f} {m_init[i]:.6e} {Z[i]:.6e} {age_yr[i]:.4f}\n")
    return n


def write_dust(out, pos_rel_kpc, m_gas, Z, T, rho_raw, a):
    """Write the ParticleMedium .dat with the Torrey ISM dust prescription.

    Returns ``(n_dust, n_hot)``. ISM cut: log(T/K) < 6 + 0.25 log(rho_raw).
    """
    pos = pos_rel_kpc * 1e3                     # kpc -> pc
    ism = np.log10(np.maximum(T, 1e-2)) < 6 + 0.25 * np.log10(np.maximum(rho_raw, 1e-10))
    m_dust = F_DUST * Z * m_gas                 # Msun
    rho_phys = rho_raw * 1e10 * H ** 2 * a ** -3
    h_kpc = np.cbrt(m_gas / np.maximum(rho_phys, 1e-10))
    h_pc = h_kpc * 1e3
    sel = ism & (m_dust > 0)
    n = int(sel.sum())
    with open(out, "w") as f:
        f.write("# column 1: x (pc)\n# column 2: y (pc)\n# column 3: z (pc)\n"
                "# column 4: h (pc)\n# column 5: M_dust (Msun)\n")
        for i in np.where(sel)[0]:
            f.write(f"{pos[i, 0]:.4f} {pos[i, 1]:.4f} {pos[i, 2]:.4f} "
                    f"{h_pc[i]:.4f} {m_dust[i]:.6e}\n")
    return n, int((~ism).sum())


def tng_to_skirt(stars_h5, gas_h5, label, sim, snap, outdir, f_dust=F_DUST):
    """Convert one galaxy's particles to SKIRT .dat files.

    Writes ``{outdir}/{label}_stars.dat`` and ``{outdir}/{label}_dust.dat``.
    Returns a dict with n_stars, n_dust, M_star, M_dust, flat (the intrinsic
    flattening lambda3/lambda1). ``f_dust`` is the dust-to-metal mass
    fraction (default 0.2).
    """
    a = snap_to_a(snap)
    os.makedirs(outdir, exist_ok=True)

    pos_kpc, m_init, Z, age_yr, h_pc = load_stars(stars_h5, sim, snap)
    center = com_iterative(pos_kpc, m_init)

    R, evals = inertia_rotation(pos_kpc - center, m_init)
    flat = evals[0] / evals[2]

    pos_star_rel = (pos_kpc - center) @ R.T
    n_star = write_source(f"{outdir}/{label}_stars.dat", pos_star_rel,
                          m_init, Z, age_yr, h_pc)

    pos_g, m_gas, Z_g, T, rho_raw = load_gas(gas_h5, sim, snap)
    pos_gas_rel = (pos_g - center) @ R.T
    n_dust, n_hot = write_dust(f"{outdir}/{label}_dust.dat", pos_gas_rel,
                               m_gas, Z_g, T, rho_raw, a)

    M_star = m_init.sum()
    ism = np.log10(np.maximum(T, 1e-2)) < 6 + 0.25 * np.log10(np.maximum(rho_raw, 1e-10))
    M_dust = (f_dust * Z_g * m_gas * ism).sum()

    return {"label": label, "n_stars": n_star, "n_dust": n_dust,
            "M_star": M_star, "M_dust": M_dust, "flat": flat}
