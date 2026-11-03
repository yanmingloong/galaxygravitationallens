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


def subtract_stellar_light_sersic(
    dataset,
    zlens,
    re_arcsec: float = 2.012,
    n: float = 4.0,
    q: float = 0.834,
    pa_deg: float = 18.68,
    max_nfev: int = 3000,
):
    """Fit a single-Sersic lens-galaxy light profile and subtract it (elliptical lenses).

    The elliptical-lens counterpart of :func:`subtract_stellar_light`: an
    elliptical galaxy has no disk, so the whole stellar light is ONE Sersic.
    The structure (``re_arcsec`` / ``n`` / ``q`` / ``pa_deg``) is FIXED to
    literature values (SLACS_light_params for J1630) and only the intensity +
    centre are free -- the same "fix the structure or the bulge-disk degeneracy
    drives the fit to the prior edge" rule documented on
    :func:`subtract_stellar_light`.

    Fitted parameters: ``[log10 intensity, centre_0, centre_1]``.

    Args:
        dataset: a masked/oversampled ``autolens.Imaging`` dataset.
        zlens: lens redshift.
        re_arcsec: fixed Sersic effective radius (arcsec).
        n: fixed Sersic index (4.0 = de Vaucouleurs).
        q: fixed axial ratio.
        pa_deg: fixed position angle (degrees).
        max_nfev: max least_squares function evaluations.

    Returns:
        ``(light_galaxy, chi2)`` -- the best-fit single-Sersic galaxy and the
        fit chi-squared (used for the noise rescale).
    """
    import autolens as al
    from scipy.optimize import least_squares

    # Axial ratio + PA -> autolens ell_comps (e1, e2).
    ell = (1.0 - q) / (1.0 + q)
    pa = np.radians(pa_deg)
    e1 = ell * np.cos(2.0 * pa)
    e2 = ell * np.sin(2.0 * pa)

    def make_light(p):
        sersic = al.lp.Sersic(
            centre=(p[1], p[2]),
            ell_comps=(e1, e2),
            intensity=10.0 ** p[0],
            effective_radius=re_arcsec,
            sersic_index=n,
        )
        return al.Galaxy(redshift=zlens, bulge=sersic)

    def residuals(p):
        tracer = al.Tracer(galaxies=[make_light(p)])
        fit = al.FitImaging(dataset=dataset, tracer=tracer)
        return np.asarray(fit.normalized_residual_map)

    # [log10 I_e, c0, c1]; centre free within +/-0.4" around the image centre.
    # NOTE the intensity is I_e (effective-radius surface brightness), NOT the
    # central surface brightness: for a de Vaucouleurs n=4 the central I(0) =
    # I_e * exp(b_n) ~ I_e * 2141, so a typical elliptical (I(0) ~ 250 cts/s/px)
    # has I_e ~ 0.1 -- the log10 lower bound must be far below 0.
    c = 0.0
    x0 = np.array([-1.0, c, c])
    lo = np.array([-3.0, c - 0.4, c - 0.4])
    hi = np.array([3.0, c + 0.4, c + 0.4])
    res = least_squares(residuals, x0, bounds=(lo, hi), max_nfev=max_nfev)
    light = make_light(res.x)
    chi2 = 2.0 * res.cost
    return light, chi2


def subtract_stellar_light_mge_agn(
    sci,
    noise,
    badmask,
    mge_L,
    mge_sigma_arcsec,
    mge_q,
    psf_sigma_px: float,
    mask_radius_arcsec: float,
    pixel_scale: float,
    position_angle_deg: float = 90.0,
    centre_px=None,
    max_nfev: int = 2000,
):
    """Subtract stellar light with a literature MGE + a PSF-shaped AGN point source (B9 K'-band).

    The B9 final subtraction recipe. A literature (fixed-shape) MGE, convolved
    with the PSF, is scaled to the data; a PSF-shaped point source is then added
    to absorb the unresolved compact nucleus (AGN / nuclear star cluster) at the
    galaxy centre, which is NARROWER than the PSF and cannot be removed by any
    smooth profile. Without the point source the centre leaves a positive spike
    + negative ring (reduced chi2 ~1450 within r<3 px); with it the subtraction
    reduced chi2 drops from ~2.58 to ~1.11 (B9 local verification).

    The fitted parameters are ``[a_mge, log10(I_pt), c0, c1]``:
      * ``a_mge`` -- overall scalar normalisation of the fixed MGE;
      * ``log10(I_pt)`` -- point-source peak flux, log10-parametrized to keep it
        positive and stable across orders of magnitude;
      * ``(c0, c1)`` -- point-source centre, constrained to +/- 6 px around the
        image centre so least_squares cannot misplace the point source onto a
        bright arc image or field star.

    The point-source SHAPE is fixed to the PSF (a Gaussian with
    ``sigma = psf_sigma_px``), NOT left free: an unresolved source convolved with
    the PSF is the PSF itself, and giving the width a free parameter would make
    it degenerate with the bulge core sigma (the same B2 pitfall).

    MGE units: ``mge_L`` is each Gaussian's TOTAL flux ``L_k`` (the mgefit
    ``sol[0]`` convention), NOT the peak surface density; the peak is
    ``L_k / (2 pi sigma_obs^2 q)`` with ``sigma_obs = sqrt(sigma_int^2 +
    psf_sigma^2)``.

    Args:
        sci: science image (2D array), counts.
        noise: noise map (2D array), same shape as ``sci``.
        badmask: bad-pixel mask (2D array); ``1`` = good/unmasked, ``0`` = bad.
        mge_L: per-Gaussian total flux ``L_k`` (array).
        mge_sigma_arcsec: per-Gaussian intrinsic sigma in arcsec (array).
        mge_q: per-Gaussian observed axial ratio (array).
        psf_sigma_px: PSF Gaussian sigma in pixels.
        mask_radius_arcsec: fitting-mask radius in arcsec (subtraction is fitted
            and evaluated only inside this radius).
        pixel_scale: pixel scale in arcsec per pixel.
        position_angle_deg: rotation of the MGE major axis (degrees, measured
            the same way as the B9 script's ``th``).
        centre_px: ``(c0, c1)`` image centre in pixels; defaults to the array
            centre ``(N/2, N/2)``.
        max_nfev: max ``least_squares`` function evaluations.

    Returns:
        ``(subtracted, reduced_chi2)`` where ``subtracted`` is the sky- and
        stellar-subtracted image (bad/masked pixels zeroed) and ``reduced_chi2``
        is the subtraction reduced chi-squared over the good pixels (the value
        fed to ``noise_rescale_factor``).

    Note:
        ``scipy`` is imported lazily; only ``numpy`` is needed at module import.
    """
    from scipy.optimize import least_squares

    sci = np.asarray(sci, dtype=float)
    noise = np.asarray(noise, dtype=float)
    badmask = np.asarray(badmask)
    mge_L = np.asarray(mge_L, dtype=float)
    mge_sigma_arcsec = np.asarray(mge_sigma_arcsec, dtype=float)
    mge_q = np.asarray(mge_q, dtype=float)

    ny, nx = sci.shape
    half = ny // 2
    if centre_px is None:
        centre_px = (float(half), float(half))
    c0_default, c1_default = float(centre_px[0]), float(centre_px[1])

    # Rotated coordinates (position angle) for the MGE.
    yy, xx = np.mgrid[0:ny, 0:nx]
    x = xx - c0_default
    y = yy - c1_default
    th = np.radians(position_angle_deg)
    xp = x * np.cos(th) + y * np.sin(th)
    yp = -x * np.sin(th) + y * np.cos(th)

    # MGE convolved with the PSF: sigma_obs^2 = sigma_int^2 + psf_sigma^2.
    mge = np.zeros_like(sci)
    for L, sig_arc, q in zip(mge_L, mge_sigma_arcsec, mge_q):
        sig_int_px = sig_arc / pixel_scale
        sig_obs = np.sqrt(sig_int_px ** 2 + psf_sigma_px ** 2)
        mge += (L / (2.0 * np.pi * sig_obs ** 2 * q)) * np.exp(
            -(xp ** 2 + yp ** 2 / q ** 2) / (2.0 * sig_obs ** 2)
        )

    sky = float(np.median(sci[badmask == 1]))
    img = sci - sky

    rpx = np.sqrt((xx - c0_default) ** 2 + (yy - c1_default) ** 2)
    good = (badmask == 1) & (rpx < mask_radius_arcsec / pixel_scale)

    def psf_im(c0, c1, I):
        # Point source = delta convolved with the PSF = PSF-shaped Gaussian.
        return I * np.exp(
            -((xx - c0) ** 2 + (yy - c1) ** 2) / (2.0 * psf_sigma_px ** 2)
        )

    # Initial MGE normalisation (least-squares of the MGE alone vs the data).
    a0 = float(np.sum(mge[good] * img[good]) / np.sum(mge[good] ** 2))

    def resid_of(p):
        model = p[0] * mge + psf_im(p[2], p[3], 10.0 ** p[1])
        return (img - model)[good]

    p0 = np.array([a0, 2.0, c0_default, c1_default])
    res = least_squares(
        resid_of,
        p0,
        bounds=(
            [0.0, -1.0, c0_default - 6.0, c1_default - 6.0],
            [1e4, 5.0, c0_default + 6.0, c1_default + 6.0],
        ),
        max_nfev=max_nfev,
    )
    a_mge, I_pt = res.x[0], 10.0 ** res.x[1]
    c0, c1 = res.x[2], res.x[3]

    model = a_mge * mge + psf_im(c0, c1, I_pt)
    subtracted = img - model
    subtracted[~good] = 0.0

    red = float(np.sum((subtracted[good] / noise[good]) ** 2) / good.sum())
    return subtracted, red
