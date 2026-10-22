"""Cosmology, physical constants, and TNG unit conversions.

Cosmology is flat Lambda-CDM **Planck 2015** (H0=67.74, Omega_m=0.3089,
Omega_Lambda=0.6911), matching the project's adopted convention (used in the
theta_E cross-section re-derivation and SKIRT cosmic-time conversion).
Angular-diameter integrals use Simpson quadrature, so no astropy dependency
is required.

All distances are in Mpc/kpc or arcsec as noted. TNG unit conversions follow
the IllustrisTNG conventions (mass in 1e10 Msun/h, length in comoving kpc/h).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
H = 0.6774                  # little-h for TNG unit conversion (h = H0/100)
GRAV = 4.302e-6             # gravitational constant, kpc (km/s)^2 / Msun
C_KMS = 3.0e5               # speed of light, km/s
ARCSEC_PER_RAD = 206265.0   # arcsec per radian

# ---------------------------------------------------------------------------
# Planck 2015 flat Lambda-CDM
# ---------------------------------------------------------------------------
H0 = 67.74                  # km/s/Mpc
OMEGA_M = 0.3089
OMEGA_L = 0.6911
# H0 in 1/Gyr (exact conversion from km/s/Mpc)
H0_GYR = H0 / 3.08567758e19 * 3.15576e16

# ---------------------------------------------------------------------------
# TNG snapshot <-> redshift / scale factor
# ---------------------------------------------------------------------------
SNAP_Z = {99: 0.0, 95: 0.0485, 91: 0.0994, 90: 0.1099, 88: 0.1418, 84: 0.1973,
          78: 0.3, 67: 0.5, 59: 0.7}
SNAP_TO_A = {snap: 1.0 / (1.0 + z) for snap, z in SNAP_Z.items()}

# TNG box sizes, comoving kpc / h
BOX_BY_SIM = {"TNG100-1": 75000.0, "TNG50-1": 35000.0, "TNG300-1": 205000.0}


# ---------------------------------------------------------------------------
# Cosmology
# ---------------------------------------------------------------------------
def E_z(z):
    """Dimensionless Hubble parameter E(z) for flat Lambda-CDM."""
    return np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


def comoving_distance(z, n=2000):
    """Line-of-sight comoving distance (Mpc) by Simpson quadrature.

    D_c(z) = (c/H0) * int_0^z dz' / E(z').
    """
    z = float(z)
    if z <= 0:
        return 0.0
    zs = np.linspace(0.0, z, n)
    integ = 1.0 / E_z(zs)
    h = z / (n - 1)
    integral = (h / 3.0) * (integ[0] + integ[-1]
                            + 4.0 * integ[1::2].sum()
                            + 2.0 * integ[2:-1:2].sum())
    return C_KMS / H0 * integral


def angular_diameter_distance(z):
    """Angular-diameter distance D_A(z) (Mpc) in flat Lambda-CDM."""
    return comoving_distance(z) / (1.0 + z)


def kpc_per_arcsec(z):
    """Physical kpc subtended by 1 arcsec at redshift z.

    kpc/arcsec = D_A(z) * 1e3 / 206265. Used to convert surface-brightness
    profile scales (arcsec) to physical scales (kpc), e.g. J2141 z=0.138
    gives 2.517 kpc/arcsec.
    """
    return angular_diameter_distance(z) * 1000.0 / ARCSEC_PER_RAD


def cosmic_time(a):
    """Age of the universe (Gyr) at scale factor a, flat Lambda-CDM.

    t(a) = 2/(3 H0 sqrt(Omega_L)) * asinh(sqrt(Omega_L/Omega_m) a^1.5).
    Vectorized: accepts a scalar or an array of scale factors.
    """
    a = np.asarray(a, dtype=np.float64)
    x = np.sqrt(OMEGA_L / OMEGA_M) * a ** 1.5
    return 2.0 / (3.0 * H0_GYR * np.sqrt(OMEGA_L)) * np.arcsinh(x)


def Dls_over_Ds(z_lens, z_src):
    """Angular-diameter distance ratio D_ls(z_l, z_s) / D_s(0, z_s).

    In flat Lambda-CDM this equals (D_c(z_s) - D_c(z_l)) / D_c(z_s),
    the key geometric factor in the SIS Einstein-radius formula.
    """
    d_cl = comoving_distance(z_lens)
    d_cs = comoving_distance(z_src)
    return (d_cs - d_cl) / d_cs if d_cs > 0 else 0.0


def theta_E_SIS(sigma_kms, z_lens, z_src):
    """SIS Einstein radius (arcsec): theta_E = 4 pi (sigma/c)^2 * Dls/Ds * 206265.

    A single-component isothermal sphere gives the total-mass constraint;
    for disk+halo systems the true theta_E is within roughly +/-20% of the
    SIS value.

    Cosmology is Planck 2015 (module convention). The project's legacy
    ``utils.py`` used astropy Planck18; the two differ by ~0.5% (H0 67.4 vs
    67.74), and this module unifies onto the Planck-2015 convention used by
    the theta_E cross-section re-derivation and SKIRT cosmic time.
    """
    ratio = Dls_over_Ds(z_lens, z_src)
    return 4.0 * np.pi * (sigma_kms / C_KMS) ** 2 * ratio * ARCSEC_PER_RAD


# ---------------------------------------------------------------------------
# TNG unit conversions
# ---------------------------------------------------------------------------
def tng_mass_to_msun(mass_tng, h=H):
    """TNG mass (1e10 Msun/h) -> physical Msun."""
    return mass_tng / h * 1e10


def tng_length_to_kpc(length_tng, h=H):
    """TNG length (comoving kpc/h) -> comoving kpc."""
    return length_tng / h


def tng_length_to_physical_kpc(length_tng, snap, h=H):
    """TNG length (comoving kpc/h) -> physical kpc at a snapshot.

    physical = comoving / h * a, with a = 1/(1+z) the scale factor.
    (ckpc/h is "ckpc per h", so divide by h, matching tng_mass_to_msun.)
    """
    a = SNAP_TO_A.get(snap, 1.0)
    return length_tng * a / h


def snap_to_a(snap):
    """Scale factor a = 1/(1+z) for a TNG snapshot.

    Raises KeyError for a snapshot not in SNAP_Z (instead of silently
    returning a=1.0 = z=0, which would corrupt the unit conversion).
    """
    if snap in SNAP_TO_A:
        return SNAP_TO_A[snap]
    raise KeyError(
        f"snapshot {snap} not in SNAP_Z ({sorted(SNAP_Z)}); "
        "add it to SNAP_Z or pass a known snapshot")


def snap_to_z(snap):
    """Redshift for a TNG snapshot."""
    return SNAP_Z.get(snap, 0.0)
