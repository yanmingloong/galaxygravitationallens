"""Ground-truth computation for mock data (shared by selection and validation).

This package holds the pure physical/morphological computations that both
`mock_pipeline.selection` (L3 disk criterion) and `lensfit.validation`
(mock truth comparison) rely on, so the two never diverge in convention:

- :mod:`~galaxylens.mock_pipeline.truth.kinematics` — circularity method
  (f_disk / B/D / disk axis / thickness) from stellar particles.
- :mod:`~galaxylens.mock_pipeline.truth.measures` — half-mass radius,
  projected ellipticity (covariance convention), intrinsic axis ratios,
  extinction.
- :mod:`~galaxylens.mock_pipeline.truth.cosmology` — Planck 2015 distances,
  SIS Einstein radius, cosmic time, TNG unit conversions.
"""

from .cosmology import (
    H, GRAV, C_KMS, ARCSEC_PER_RAD, H0, OMEGA_M, OMEGA_L,
    SNAP_Z, SNAP_TO_A, BOX_BY_SIM,
    E_z, comoving_distance, angular_diameter_distance, kpc_per_arcsec,
    cosmic_time, Dls_over_Ds, theta_E_SIS,
    tng_mass_to_msun, tng_length_to_kpc, tng_length_to_physical_kpc,
    snap_to_a, snap_to_z,
)
from .kinematics import (
    KAPPA_THRESH, unwrap, com_iterative, angular_momentum, circularity,
    disk_axis, disk_fraction, disk_thickness, disk_kinematics,
)
from .measures import (
    half_mass_radius, projected_ellipticity, intrinsic_axis_ratios,
    extinction,
)

__all__ = [
    # cosmology
    "H", "GRAV", "C_KMS", "ARCSEC_PER_RAD", "H0", "OMEGA_M", "OMEGA_L",
    "SNAP_Z", "SNAP_TO_A", "BOX_BY_SIM",
    "E_z", "comoving_distance", "angular_diameter_distance", "kpc_per_arcsec",
    "cosmic_time", "Dls_over_Ds", "theta_E_SIS",
    "tng_mass_to_msun", "tng_length_to_kpc", "tng_length_to_physical_kpc",
    "snap_to_a", "snap_to_z",
    # kinematics
    "KAPPA_THRESH", "unwrap", "com_iterative", "angular_momentum",
    "circularity", "disk_axis", "disk_fraction", "disk_thickness",
    "disk_kinematics",
    # measures
    "half_mass_radius", "projected_ellipticity", "intrinsic_axis_ratios",
    "extinction",
]
