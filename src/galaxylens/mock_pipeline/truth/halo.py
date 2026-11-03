"""gNFW halo ground-truth fit from dark-matter particles.

Fits a generalized NFW profile ``(rho_s, r_s, gamma)`` to the spherical DM
density, measures the halo flattening ``q_h`` from the DM inertia tensor, and
converts to the fit parameterization ``(v_vir, c_-2)``. The conversion is the
INVERSE of ``lensfit.models.dm.gnfw_virial.vvir_to_rs``, so the truth halo and
the fit halo share one parameterization (a fit-parameterization-aware truth:
this module bridges the TNG particles and the lensfit gNFW model).

References
----------
- Wyithe, Turner & Spergel 2001, ApJ 555, 504 (gNFW v_vir parameterization)
- Barnabe et al. 2012, MNRAS 423, 1073 (Eq. 7-11 zeta normalization)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def gnfw_log_density(r, log_rho_s, log_rs, gamma):
    """log(gNFW 3-D density) with log-parameterized ``(rho_s, r_s)``.

    Used as the curve-fit model; returns ``ln(rho(r))``.
    """
    rho_s = np.exp(log_rho_s)
    rs = np.exp(log_rs)
    return np.log(rho_s) - gamma * np.log(r / rs) \
        - (3.0 - gamma) * np.log(1.0 + r / rs)


def solve_c_from_delta(delta_c, gamma):
    """Solve ``c = r_vir / r_s`` from ``delta_c = (200/3) c^3 / zeta(c, gamma, 1)``.

    Binary search over ``c in [1, 200]``.
    """
    from galaxylens.lensfit.models.dm.gnfw_virial import zeta

    def f(c):
        return (200.0 / 3.0) * c ** 3 / zeta(c, gamma, 1.0) - delta_c

    lo, hi = 1.0, 200.0
    flo, fhi = f(lo), f(hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm * flo > 0:
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return 0.5 * (lo + hi)


def fit_halo(pos_dm_al, m_dm, zlens, zsrc=0.0):
    """Fit gNFW to DM particles and convert to ``(v_vir, c_-2, q_h)``.

    ``pos_dm_al`` is the DM particle positions in the aligned mock frame
    (kpc); ``m_dm`` the per-particle DM mass (Msun). ``zlens`` / ``zsrc`` set
    the cosmology (H0=70, Om=0.3 — the fit convention). Returns a dict with
    the fit and self-check masses.
    """
    from galaxylens.lensfit.models.dm.gnfw_virial import halo_cosmo, zeta

    c = halo_cosmo(zlens, zsrc)

    r3 = np.linalg.norm(pos_dm_al, axis=1)
    rmax = np.percentile(r3, 95)
    rbins = np.logspace(np.log10(max(1.0, r3.min() + 1e-6)), np.log10(rmax), 25)
    rcent = 0.5 * (rbins[:-1] + rbins[1:])
    vol = (4.0 / 3.0) * np.pi * (rbins[1:] ** 3 - rbins[:-1] ** 3)
    hist, _ = np.histogram(r3, bins=rbins)
    rho_obs = hist * m_dm / vol                  # Msun/kpc^3
    ok = hist >= 50                              # statistically sufficient bins
    rho_obs, rcent = rho_obs[ok], rcent[ok]

    p0 = [np.log(rho_obs.max()), np.log(np.median(rcent)), 1.0]
    popt, _ = curve_fit(gnfw_log_density, rcent, np.log(rho_obs), p0=p0,
                        bounds=([-60, np.log(0.5), 0.1], [20, np.log(200), 2.8]),
                        maxfev=20000)
    log_rho_s, log_rs, gamma = popt
    rho_s, rs = np.exp(log_rho_s), np.exp(log_rs)

    # q_h: DM flattening in the disk-aligned frame (inertia tensor)
    sdm = np.einsum("i,ij,ik->jk", np.full(len(pos_dm_al), m_dm),
                    pos_dm_al, pos_dm_al)
    sdm /= (len(pos_dm_al) * m_dm)
    evdm = np.linalg.eigh(sdm)[0]                # ascending [min, mid, max]
    q_h = float(np.sqrt(max(evdm[0], 1e-6) / max(evdm[2], 1e-6)))

    # (rho_s, r_s, gamma, q_h) -> (v_vir, c_-2)
    delta_c = rho_s / c.rho_crit_kpc3
    cc = solve_c_from_delta(delta_c, gamma)
    c_minus2 = cc / (2.0 - gamma)
    r_vir = cc * rs
    z1 = zeta(cc, gamma, 1.0)
    zq = zeta(cc, gamma, q_h)
    v_vir = r_vir * c.h * np.sqrt(zq / z1)       # km/s

    m_vir_gnfw = (4.0 / 3.0) * np.pi * r_vir ** 3 * 200.0 * c.rho_crit_kpc3
    m_dm_total = len(pos_dm_al) * m_dm

    return {
        "rho_s_msun_kpc3": float(rho_s),
        "r_s_kpc": float(rs),
        "gamma": float(gamma),
        "q_h": q_h,
        "v_vir_km_s": float(v_vir),
        "c_minus2": float(c_minus2),
        "r_vir_kpc": float(r_vir),
        "delta_c": float(delta_c),
        "m_vir_gnfw_msun": float(m_vir_gnfw),
        "m_dm_total_msun": float(m_dm_total),
        "fit_n_bins": int(ok.sum()),
        "fit_rmax_kpc": float(rmax),
    }
