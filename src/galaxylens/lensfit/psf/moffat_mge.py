"""Multi-Gaussian Moffat MGE PSF construction (paper Table 2 style).

Builds a 2D PSF image from a multi-Gaussian expansion (MGE) of a Moffat
profile: amplitudes ``G_j`` and Gaussian sigmas ``delta_j`` (arcsec). This is
the PSF representation of the WFPC2 F450W/F814W observations (Trick et al.
2016 Table 2) and is shared by mock convolution (``mock_pipeline.skirt``)
and fitting deconvolution (``lensfit.mge``) so the two sides use the same PSF.

References
----------
- Trick et al. 2016, MNRAS 462, 470 (Table 2 four-Gaussian Moffat MGE)
- Cappellari 2002, MNRAS 333, 400 (MGE formalism)
"""

import numpy as np

# Trick et al. 2016 Table 2: (G_j weight, delta_j sigma in arcsec) of the
# WFPC2 F450W/F814W PSF radial profile, fitted with four circular Gaussians.
TRICK_2016_TABLE2 = [
    (0.184, 0.038),
    (0.485, 0.085),
    (0.222, 0.169),
    (0.109, 0.487),
]


def moffat_mge_psf(weights, sigmas_arcsec, pixel_scale, shape=81):
    """2D PSF image from a multi-Gaussian Moffat MGE expansion.

    ``weights`` and ``sigmas_arcsec`` are the MGE amplitudes and Gaussian
    sigmas (arcsec); ``pixel_scale`` is arcsec/pixel; ``shape`` is the
    square output size (odd recommended, PSF centred). Returns a normalised
    PSF (sum = 1).

    For the WFPC2 Table 2 PSF pass ``weights``/``sigmas_arcsec`` from
    ``TRICK_2016_TABLE2`` (unpacked) with the target pixel scale.
    """
    weights = np.asarray(weights, dtype=np.float64)
    sigmas_arcsec = np.asarray(sigmas_arcsec, dtype=np.float64)
    yy, xx = np.mgrid[0:shape, 0:shape]
    c = (shape - 1) / 2
    psf = np.zeros((shape, shape), dtype=np.float64)
    for gj, dj in zip(weights, sigmas_arcsec):
        s = dj / pixel_scale            # Gaussian sigma in pixels
        psf += gj * np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * s ** 2)) \
            / (2 * np.pi * s ** 2)
    return psf / psf.sum()
