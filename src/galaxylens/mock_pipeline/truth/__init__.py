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
- :mod:`~galaxylens.mock_pipeline.truth.particles` — HDF5 particle loading
  (physical units) + inertia-tensor mock-frame alignment.
- :mod:`~galaxylens.mock_pipeline.truth.fdm` — dark-matter fraction ground
  truth (spherical / cylindrical edge-on apertures).
- :mod:`~galaxylens.mock_pipeline.truth.anisotropy` — velocity anisotropy
  ``beta_z = 1 - sigma_z^2 / sigma_R^2``.
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
    counter_rotation_angle,
)
from .measures import (
    half_mass_radius, projected_ellipticity, intrinsic_axis_ratios,
    extinction, major_axis_angle,
)
from .particles import load_particles, inertia_rotation
from .fdm import surface_density, fit_exp_rd, f_dm_apertures
from .anisotropy import beta_z
from .halo import gnfw_log_density, solve_c_from_delta, fit_halo

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
    "disk_kinematics", "counter_rotation_angle",
    # measures
    "half_mass_radius", "projected_ellipticity", "intrinsic_axis_ratios",
    "extinction", "major_axis_angle",
    # particles
    "load_particles", "inertia_rotation",
    # fdm
    "surface_density", "fit_exp_rd", "f_dm_apertures",
    # anisotropy
    "beta_z",
    # halo
    "gnfw_log_density", "solve_c_from_delta", "fit_halo",
]
