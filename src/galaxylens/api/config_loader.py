"""YAML config -> PipelineConfig (+ data/search specs) for the facade layer.

The fitting-recipe YAML is a DIFFERENT schema from the lens-catalog YAML under
``config/`` (which holds name/ra/dec/zl/zs/bands). A fit recipe encodes the
geometry, kinematics, component names, priors, stellar template, data paths and
search settings for one reproducible Barnabe-style joint fit.

References
----------
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (the joint-fit recipe
  these configs encode)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from astropy.cosmology import FlatLambdaCDM

from ..lensfit.structures import (
    Geometry,
    KinData,
    PipelineConfig,
    PriorSet,
    PriorSpec,
    SourcePriors,
)
from ..lensfit.models.stellar.mge_const_ml import (
    stellar_mge_peak,
    stellar_mge_kband,
    stellar_mge_ellip,
)

_H0 = 70.0
_OM0 = 0.3


@dataclass
class DataSpec:
    """Input FITS paths and imaging settings for one fit."""

    data_path: str
    noise_map_path: str
    psf_path: str
    starmask_path: Optional[str] = None
    pixel_scale: float = 0.05
    mask_radius: float = 5.0
    oversample: int = 1
    # Subtraction path (subtract_method):
    subtract_method: str = "sersic"  # "sersic" (B4 bulge+disk) | "sersic_single" (elliptical) | "mge_agn" (B9)
    # "sersic_single" (elliptical): single-Sersic structure, fixed to literature.
    sersic_re_arcsec: Optional[float] = None
    sersic_n: Optional[float] = None
    sersic_q: Optional[float] = None
    sersic_pa_deg: Optional[float] = None
    # "mge_agn" (B9 K'-band subtraction path):
    mge_params_path: Optional[str] = None  # mge L/sigma/q CSV (mge_agn)
    badmask_path: Optional[str] = None  # KAI badmask, 1=good (mge_agn sky RMS)
    psf_sigma_px: Optional[float] = None  # point-source PSF sigma, px (mge_agn)
    position_angle_deg: float = 90.0  # MGE major-axis rotation (mge_agn)


@dataclass
class SearchSpec:
    """Sampler settings (local or slurm backend)."""

    backend: str = "local"  # "local" | "slurm"
    name: str = "fit"
    n_live: int = 200
    number_of_cores: int = 1
    output_root: Optional[str] = None  # None -> GALAXYLENS_OUTPUT_ROOT
    # slurm-only (rendered into the SBATCH template when backend == "slurm")
    partition: Optional[str] = None
    account: Optional[str] = None
    conda_env_path: Optional[str] = None
    workdir: Optional[str] = None  # cluster project root (cd before running)
    cpus_per_task: int = 1
    mem: Optional[str] = None  # e.g. "64G"
    time_hours: float = 12.0


@dataclass
class FitConfig:
    """A complete fitting recipe: core pipeline + data + search settings."""

    pipeline: PipelineConfig
    data: DataSpec
    search: SearchSpec
    subtract_stellar: bool = True
    noise_rescale: bool = True


def _geometry(d: dict) -> Geometry:
    cosmo = FlatLambdaCDM(H0=_H0, Om0=_OM0)
    zl = float(d["zlens"])
    zs = float(d["zsrc"])
    da = cosmo.angular_diameter_distance(zl).value
    return Geometry(
        zlens=zl,
        zsrc=zs,
        inclination_deg=float(d["inclination_deg"]),
        da_mpc=da,
        pc_per_arcsec=da * np.pi / 0.648,
        re_arcsec=d.get("re_arcsec"),
        rd_arcsec=d.get("rd_arcsec"),
    )


def _kinematics(d: dict) -> KinData:
    vrms_raw = d.get("v_rms", [])
    vrms = (
        np.asarray(vrms_raw, dtype=float).reshape(-1, 3)
        if len(vrms_raw)
        else np.zeros((0, 3))
    )
    return KinData(
        xbin=vrms[:, 0],
        ybin=np.zeros(len(vrms)),
        vrms=vrms[:, 1],
        vrms_err=vrms[:, 2],
        vcirc=d.get("vcirc"),
        vcirc_err=d.get("vcirc_err"),
        aperture_radius=d.get("aperture_radius"),
        aperture_vrms=d.get("aperture_vrms"),
        aperture_vrms_err=d.get("aperture_vrms_err"),
        kin_psf_sigma=d.get("kin_psf_sigma"),
    )


def _priors(d: dict) -> PriorSet:
    c2 = d.get("c_2")
    ml_spec = d.get("ml_img_spec")
    return PriorSet(
        ml_img=tuple(d["ml_img"]),
        v_vir=tuple(d["v_vir"]),
        gamma=tuple(d["gamma"]),
        q_h=tuple(d["q_h"]),
        ratio=tuple(d["ratio"]),
        c_2=PriorSpec(**c2) if c2 else None,
        ml_img_spec=PriorSpec(**ml_spec) if ml_spec else None,
    )


def _source_priors(d: dict) -> SourcePriors:
    return SourcePriors(**(d or {}))


def _load_mge_csv(path) -> "tuple":
    """Load an MGE ``(L_k, sigma_arcsec, q)`` CSV (B2 ``mge_params.csv``).

    Columns: ``<name>, L_k, sigma_px, sigma_arcsec, q`` (header skipped). ``L_k``
    is each Gaussian's TOTAL flux, not the peak surface density.
    """
    import csv

    Ls, sigs, qs = [], [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            _, L, _sp_px, sp_arc, q = row
            Ls.append(float(L))
            sigs.append(float(sp_arc))
            qs.append(float(q))
    return np.array(Ls), np.array(sigs), np.array(qs)


def load_config(path) -> FitConfig:
    """Load a fitting-recipe YAML into a :class:`FitConfig`.

    The geometry's ``da_mpc``/``pc_per_arcsec`` and the stellar template are
    computed here (cosmology H0=70, Om0=0.3; ``stellar_mge_peak`` with the
    rbounds guard), so the YAML only carries the per-target observables.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    geom = _geometry(raw["geometry"])
    kin = _kinematics(raw["kinematics"])
    comp = raw["components"]
    priors = _priors(raw["priors"])
    src_priors = _source_priors(raw.get("source_priors", {}))
    src_pix = raw.get("source_pixelization", {}) or {}

    st = raw["stellar_template"]
    if st.get("type") == "mge_csv":
        # B9 K'-band literature MGE: build the two-sided calibrated template
        # from mge_params.csv (mstar_ref / ml_ref / Sigma_crit bridge).
        from ..lensfit.models.dm.gnfw_virial import halo_cosmo
        from ..config.paths import get_data_root

        mge_path = Path(st["mge_params_path"])
        if not mge_path.is_absolute():
            mge_path = get_data_root() / mge_path
        Ls, sigs, qs = _load_mge_csv(mge_path)
        hc = halo_cosmo(geom.zlens, geom.zsrc)
        sigma_crit_as2 = hc.sigma_crit * hc.kpc_per_arcsec ** 2
        template = stellar_mge_kband(
            Ls, sigs, qs,
            pixel_scale=float(st["pixel_scale"]),
            mstar_ref=float(st.get("mstar_ref", 1.32e11)),
            ml_ref=float(st.get("ml_ref", 0.8)),
            sigma_crit_arcsec2=sigma_crit_as2,
            pc_per_arcsec=geom.pc_per_arcsec,
        )
    elif st.get("type") == "ellip":
        # Elliptical lens: single Sersic (no disk), de Vaucouleurs n=4 default.
        template = stellar_mge_ellip(
            pc_per_arcsec=geom.pc_per_arcsec,
            n=float(st.get("n", 4.0)),
            re_arcsec=float(st.get("re_arcsec", 2.0)),
            q=float(st.get("q", 0.8)),
            psf_sigma_arcsec=float(st.get("psf_sigma_arcsec", 0.1)),
            bulge_i0=float(st.get("bulge_i0", 1000.0)),
        )
    else:
        template = stellar_mge_peak(
            pc_per_arcsec=geom.pc_per_arcsec,
            n_bulge=st.get("n_bulge", 1.21),
            re_bulge=st.get("re_bulge", 0.26),
            q_bulge=st.get("q_bulge", 0.53),
            rd_disk=st.get("rd_disk", 1.51),
            q_disk=st.get("q_disk", 0.31),
            disk_over_bulge=st.get("disk_over_bulge", 4.0),
            psf_sigma_arcsec=st.get("psf_sigma_arcsec", 0.064),
            bulge_i0=st.get("bulge_i0", 957.0),
        )

    pipeline = PipelineConfig(
        geometry=geom,
        kinematics=kin,
        dark=comp["dark"],
        stellar=comp["stellar"],
        source=comp["source"],
        dynamics=comp["dynamics"],
        priors=priors,
        stellar_template=template,
        source_priors=src_priors,
        source_mesh_shape=tuple(int(x) for x in src_pix.get("mesh_shape", (15, 15))),
        source_reg=src_pix.get("reg", "constant"),
        source_reg_coeff_bounds=tuple(float(x) for x in src_pix.get("coeff_bounds", (1e-6, 1e6))),
    )

    data_raw = raw["data"]
    data = DataSpec(
        data_path=data_raw["data_path"],
        noise_map_path=data_raw["noise_map_path"],
        psf_path=data_raw["psf_path"],
        starmask_path=data_raw.get("starmask_path"),
        pixel_scale=data_raw.get("pixel_scale", 0.05),
        mask_radius=data_raw.get("mask_radius", 5.0),
        oversample=data_raw.get("oversample", 1),
        subtract_method=data_raw.get("subtract_method", "sersic"),
        sersic_re_arcsec=data_raw.get("sersic_re_arcsec"),
        sersic_n=data_raw.get("sersic_n"),
        sersic_q=data_raw.get("sersic_q"),
        sersic_pa_deg=data_raw.get("sersic_pa_deg"),
        mge_params_path=data_raw.get("mge_params_path"),
        badmask_path=data_raw.get("badmask_path"),
        psf_sigma_px=data_raw.get("psf_sigma_px"),
        position_angle_deg=data_raw.get("position_angle_deg", 90.0),
    )

    search = SearchSpec(**(raw.get("search", {}) or {}))

    return FitConfig(
        pipeline=pipeline,
        data=data,
        search=search,
        subtract_stellar=raw.get("subtract_stellar", True),
        noise_rescale=raw.get("noise_rescale", True),
    )
