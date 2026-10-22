"""MGE deprojection: 2-D surface brightness -> 3-D luminosity density.

For an oblate axisymmetric system viewed at inclination ``i``, the observed
axial ratio ``q_obs`` deprojects to an intrinsic axial ratio ``q_intr`` via
(Cappellari 2002):

    q_intr = sqrt((q_obs^2 - cos^2 i) / sin^2 i)

Requiring ``q_obs >= cos(i)`` (otherwise ``q_intr`` is imaginary). L_k and
sigma_k are unchanged by deprojection.

References
----------
- Cappellari 2002, MNRAS 333, 400 (eq. 12-13, oblate-axisymmetric deprojection)
"""

from __future__ import annotations

import numpy as np


def deproject(q_obs: np.ndarray, incl_deg: float) -> np.ndarray:
    """Deproject observed axial ratios ``q_obs`` to intrinsic ``q_intr``.

    Args:
        q_obs: observed (projected) axial ratios.
        incl_deg: inclination angle in degrees.

    Returns:
        Intrinsic axial ratios.

    Raises:
        ValueError: if any ``q_obs < cos(incl_deg)``, which would make the
            intrinsic axial ratio imaginary. Check the B2 qbounds constraint.
    """
    q_obs = np.asarray(q_obs, dtype=float)
    cosi = np.cos(np.radians(incl_deg))
    sini2 = np.sin(np.radians(incl_deg)) ** 2
    q2 = (q_obs**2 - cosi**2) / sini2
    if np.any(q2 < 0):
        bad = q_obs[q2 < 0]
        raise ValueError(
            f"deprojection failed: {np.sum(q2 < 0)} Gaussian(s) have "
            f"q_obs < cos({incl_deg} deg)={cosi:.4f} (q_obs={bad}); "
            f"intrinsic axial ratio would be imaginary."
        )
    return np.sqrt(q2)


def project_back(q_intr: np.ndarray, incl_deg: float) -> np.ndarray:
    """Forward-project intrinsic axial ratios back to observed (validation helper).

    Inverse of :func:`deproject`; used to verify the deprojection round-trips
    exactly.

    Args:
        q_intr: intrinsic axial ratios.
        incl_deg: inclination angle in degrees.

    Returns:
        Reconstructed observed axial ratios.
    """
    q_intr = np.asarray(q_intr, dtype=float)
    cosi = np.cos(np.radians(incl_deg))
    sini2 = np.sin(np.radians(incl_deg)) ** 2
    return np.sqrt(q_intr**2 * sini2 + cosi**2)
