"""Strong-lens mock production: lens mass model + arc ray-tracing.

Builds the true-mass lens galaxy (stellar MGE in convergence units + a
``v_vir``-normalized gNFW halo), derives the Einstein radius, and ray-traces a
background Sersic source into an arc image. This is the mock counterpart of
the observed strong lens: the same mass model that ``lensfit`` fits, so the
mock's ``theta_E`` and arc are consistent with the fit parametrization.

All ``autolens`` imports are lazy; running these functions requires autolens.

References
----------
- Barnabe et al. 2012, MNRAS 423, 1073 (gNFW v_vir parametrization)
- Wyithe, Turner & Spergel 2001, ApJ 555, 504 (gNFW deflection)
"""
from __future__ import annotations

import numpy as np


def build_lens_galaxy(surf_mass_kpc2, sig_as, q_obs, v_vir, c_minus2, gamma,
                      q_h, zlens, zsrc, inclination):
    """Build the true-mass lens galaxy (stellar MGE + gNFW halo).

    ``surf_mass_kpc2`` / ``sig_as`` / ``q_obs`` are the stellar MGE (peak mass
    surface density in Msun/kpc^2, width in arcsec, observed axis ratio). The
    halo is a ``GNFWVirialVelocity`` at ``(v_vir, c_minus2, gamma, q_h)``.
    Returns ``(lens, mstar_model_msun)``, the latter a self-check (the total
    stellar mass implied by the MGE).
    """
    import autolens as al
    from galaxylens.lensfit.models.dm.gnfw_virial import halo_cosmo
    from galaxylens.lensfit.models.dm.gnfw_virial_profile import (
        GNFWVirialVelocity,
    )

    c = halo_cosmo(zlens, zsrc)
    surf_kappa = np.asarray(surf_mass_kpc2, dtype=np.float64) / c.sigma_crit

    gaussians = []
    for s, sig, q in zip(surf_kappa, np.asarray(sig_as), np.asarray(q_obs)):
        e = (1.0 - q) / (1.0 + q)
        gaussians.append(al.mp.Gaussian(
            centre=(0.0, 0.0), ell_comps=(-e, 0.0),
            intensity=float(s), sigma=float(sig), mass_to_light_ratio=1.0,
        ))
    dark = GNFWVirialVelocity(
        v_vir=v_vir, c_2=c_minus2, inner_slope=gamma, q_h=q_h,
        inclination=inclination,
    )
    kwargs = {f"mge_{i}": g for i, g in enumerate(gaussians)}
    kwargs["dark"] = dark
    lens = al.Galaxy(redshift=zlens, **kwargs)

    mstar_model = sum(
        float(g.intensity) * 2 * np.pi * float(g.sigma) ** 2 * g.axis_ratio()
        for g in gaussians
    ) * c.sigma_crit * c.kpc_per_arcsec ** 2
    return lens, float(mstar_model)


def einstein_radius(lens, npx=300, half=2.5):
    """Einstein radius (arcsec) where the mean convergence ``kappa_bar = 1``.

    Returns ``(theta_E_arcsec, kappa_bar_at_theta_E)``.
    """
    import autolens as al

    pix = 2 * half / npx
    grid = al.Grid2D.uniform(shape_native=(npx, npx), pixel_scales=pix)
    kappa = np.asarray(lens.convergence_2d_from(grid=grid).native)
    xc = (np.arange(npx) - (npx - 1) / 2) * pix
    X, Y = np.meshgrid(xc, xc)
    R = np.hypot(X, Y)
    order = np.argsort(R.ravel())
    R_sort = R.ravel()[order]
    k_sort = kappa.ravel()[order]
    k_cummean = np.cumsum(k_sort) / np.arange(1, len(k_sort) + 1)  # equal-area
    idx = np.argmin(np.abs(k_cummean - 1.0))
    return float(R_sort[idx]), float(k_cummean[idx])


def make_arc(lens, beta_as, zsrc, pixel_scale=0.05, npx=121,
             re_src=0.2, n_src=1.0):
    """Ray-trace a Sersic source at ``beta_as`` into an arc image (2-D array)."""
    import autolens as al

    source = al.Galaxy(
        redshift=zsrc,
        light=al.lp.Sersic(
            centre=(beta_as, 0.0), ell_comps=(0.0, 0.0),
            intensity=1.0, effective_radius=re_src, sersic_index=n_src,
        ),
    )
    tracer = al.Tracer(galaxies=[lens, source])
    grid = al.Grid2D.uniform(shape_native=(npx, npx), pixel_scales=pixel_scale)
    return np.asarray(tracer.image_2d_from(grid=grid).native)
