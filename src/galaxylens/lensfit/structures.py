"""Core data structures shared by every lensfit model, engine, and assembler.

These are the "common language" of the componentized lensfit layer (see the
refinement blueprint ``docs/refinement_plan_zh.md``). They carry only data and
prior *specifications*; no engine (autolens / autofit / jampy) is imported here,
so the module stays importable without the heavy dependencies installed.

References
----------
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (SWELLS IV: the
  geometry, kinematics, and gNFW parametrization these structures describe)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

# A Multi-Gaussian Expansion triple: (surf, sigma, q_obs). ``surf`` is the
# Gaussian central surface density (physical mass surface density for a
# potential MGE, or relative peak surface brightness for a fixed stellar
# template); ``sigma`` is in arcsec; ``q_obs`` is the observed axial ratio.
MGE = Tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Geometry:
    """Fixed observational geometry (not a free parameter).

    Attributes:
        zlens: lens redshift.
        zsrc: source redshift.
        inclination_deg: line-of-sight inclination in degrees (i=78.5 for J2141,
            70 for J1331).
        da_mpc: angular-diameter distance to the lens in Mpc.
        pc_per_arcsec: physical scale in pc per arcsec at the lens redshift.
        re_arcsec: circularized Einstein radius in arcsec (for f_dm / M_2D
            aperture estimates; optional).
        rd_arcsec: disk scale length in arcsec (for V_circ(2.2 Rd) and f_dm
            apertures; optional).
    """

    zlens: float
    zsrc: float
    inclination_deg: float
    da_mpc: float
    pc_per_arcsec: float
    re_arcsec: Optional[float] = None
    rd_arcsec: Optional[float] = None


@dataclass(frozen=True)
class KinData:
    """Observed kinematics: stellar v_rms points plus an optional V_circ.

    ``xbin``/``ybin`` are the projected positions (arcsec) where ``vrms`` is
    measured; ``vrms_err`` are the 1-sigma uncertainties. ``vcirc``/``vcirc_err``
    are an optional Halpha (or other tracer) circular-velocity constraint at
    ``2.2 * rd_arcsec`` (km/s).

    For ELLIPTICAL lenses the kinematic observable is usually a SINGLE
    luminosity-weighted aperture velocity dispersion ``sigma_ap`` (e.g. a
    SDSS/SLACS fibre). The optional ``aperture_radius`` / ``aperture_vrms`` /
    ``aperture_vrms_err`` / ``kin_psf_sigma`` fields carry it; the Jampy engine
    computes the model ``sigma_ap`` inside that aperture via the ``pixsize`` +
    ``sigmapsf`` convolution of ``jam.axi.proj``.
    """

    xbin: np.ndarray
    ybin: np.ndarray
    vrms: np.ndarray
    vrms_err: np.ndarray
    vcirc: Optional[float] = None
    vcirc_err: Optional[float] = None
    # Elliptical-lens aperture dispersion (optional):
    aperture_radius: Optional[float] = None  # circular aperture radius, arcsec
    aperture_vrms: Optional[float] = None  # sigma_ap, km/s
    aperture_vrms_err: Optional[float] = None  # km/s
    kin_psf_sigma: Optional[float] = None  # kinematics PSF sigma, arcsec

    @property
    def n_points(self) -> int:
        """Number of kinematic data points (v_rms + optional V_circ + aperture)."""
        n = len(self.vrms)
        if self.vcirc is not None:
            n += 1
        if self.aperture_vrms is not None:
            n += 1
        return n


@dataclass(frozen=True)
class Calibration:
    """Imaging/dynamics M/L bridge (the B6/B7 CALIB pitfall, fixed in one place).

    ``calib`` converts the relative stellar surface-brightness template to the
    imaging flux scale: ``intensity_imaging = surf_rel * calib``, and the
    dynamics mass density ``surf_s = surf_rel * calib * ml_img``. The single free
    M/L (``ml_img``) is then the SAME parameter in both engines.
    """

    calib: float


@dataclass
class PriorSpec:
    """c-M relation prior: None=Uniform, otherwise a Gaussian centered on the
    relation (Dutton & Maccio 2014: mean 5.9, 0.11 dex -> sigma 1.49 linear)."""

    kind: str = "gaussian"
    mean: float = 5.9
    sigma: float = 1.49
    lower: float = 3.0
    upper: float = 20.0


@dataclass
class PriorSet:
    """Priors for the shared mass/kinematics parameters.

    ``ml_img`` is a LogUniform bounds tuple; ``v_vir``, ``gamma``, ``q_h``,
    ``ratio`` are Uniform bounds; ``c_2`` is a ``PriorSpec`` (uniform, or the
    D&M c-M Gaussian).

    D-series (J1331) fields: ``v200``/``c200`` (spherical NFW) and ``beta_z``
    (tangential anisotropy, used directly by the KDC engine) are optional and
    only populated by a J1331 recipe.
    """

    ml_img: Tuple[float, float]
    v_vir: Tuple[float, float]
    gamma: Tuple[float, float]
    q_h: Tuple[float, float]
    ratio: Tuple[float, float]
    c_2: Optional[PriorSpec] = None
    ml_img_spec: Optional[PriorSpec] = None  # Gaussian M/L prior (B9 K'-band: N(0.8, 0.3))
    v200: Optional[Tuple[float, float]] = None
    c200: Optional[Tuple[float, float]] = None
    beta_z: Optional[Tuple[float, float]] = None


@dataclass
class SourcePriors:
    """Priors for the Sersic source (single-Sersic, B7 values as defaults)."""

    centre_sigma: float = 0.3
    ell_sigma: float = 0.3
    ell_bound: float = 0.95
    intensity_bounds: Tuple[float, float] = (0.01, 10.0)
    effective_radius_bounds: Tuple[float, float] = (0.05, 3.0)
    sersic_index_bounds: Tuple[float, float] = (0.5, 6.0)


@dataclass(frozen=True)
class StellarTemplate:
    """Resolved stellar mass template: relative MGE + calibration.

    Produced once per target by ``stellar_mge_peak`` (the B6/B7 Sersic +
    exponential recipe with the rbounds guard). ``mge`` is
    ``(surf_rel, sig_arcsec, q_obs)`` in RELATIVE peak units; ``l2d_rel`` is the
    relative total 2D light (``sum(surf * 2*pi * (sig*pc_per_arcsec)^2 * q)``)
    used for the M* anchor ``M* = ml_img * calib * l2d_rel``; ``calibration`` is
    the CALIB value.
    """

    mge: MGE
    l2d_rel: float
    calibration: Calibration
    # Optional SEPARATE imaging bridge: al.mp.Gaussian intensity = surf_rel *
    # calibration_img (Sigma_crit-normalised). None means the two sides share
    # `calibration` (the F814W convention). The B9 K'-band calibration splits
    # them: `calibration` is the dynamics surface-density bridge (Msun/pc^2 per
    # ml_img), `calibration_img` is the imaging intensity bridge.
    calibration_img: Optional[Calibration] = None


@dataclass(frozen=True)
class RadialStellarTemplate:
    """D-series (J1331) resolved stellar template: ABSOLUTE light MGE + per-Gaussian M/L.

    Unlike ``StellarTemplate`` (relative peak units + CALIB, the B-series
    convention), the D series uses the absolute peak surface brightness I0,i
    (Lsun/pc^2, Trick+2016 Table 3) directly, scaled by a per-Gaussian Salpeter
    M/L. No CALIB bridge: the light MGE (``lum_mge``) and the mass MGE
    (``lum_mge * ml_array``) differ only by the radial M/L.
    """

    lum_mge: MGE  # (surf_lum, sigma_arcsec, q) in Lsun/pc^2
    ml_array: np.ndarray  # per-Gaussian M/L


@dataclass(frozen=True)
class MeinSpec:
    """Einstein-mass (point-level) lensing constraint (D5 / J1331 quad)."""

    rein_arcsec: float
    mein_msun: float
    frac_err: float = 0.10


@dataclass
class PipelineConfig:
    """One YAML = one reproducible fitting recipe (Barnabe-style).

    Holds the per-target geometry, kinematics, stellar template + calibration,
    component NAMES (dark/stellar/source/dynamics), and the priors. ``assemble``
    maps names to component classes and constructs the model from this object.
    """

    geometry: Geometry
    kinematics: KinData
    dark: str  # "gnfw_virial" | "gnfw_ludlow" | "nfw" | ...
    stellar: str  # "mge_const_ml" | "mge_radial_ml" | ...
    source: str  # "single_sersic" | ...
    dynamics: str  # "jampy" | "agama" | ...
    priors: PriorSet
    dynamics_anisotropy: str = "ratio"  # jampy anisotropy param: "ratio" | "beta_z"
    stellar_template: Optional[StellarTemplate] = None
    stellar_radial: Optional[RadialStellarTemplate] = None
    lens_constraint: str = "imaging"  # "imaging" | "mein"
    mein: Optional[MeinSpec] = None
    source_priors: SourcePriors = field(default_factory=SourcePriors)
    source_mesh_shape: Tuple[int, int] = (15, 15)  # pixelized source mesh (nx, ny)
    source_reg: str = "constant"  # pixelized source regularization flavour
    source_reg_coeff_bounds: Tuple[float, float] = (1e-6, 1e6)  # LogUniform
    noise_rescale: bool = True
    subtract_stellar: bool = True
