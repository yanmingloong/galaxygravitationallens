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
from ..lensfit.models.stellar.mge_const_ml import stellar_mge_peak

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
    vrms = np.asarray(d["v_rms"], dtype=float)
    return KinData(
        xbin=vrms[:, 0],
        ybin=np.zeros(len(vrms)),
        vrms=vrms[:, 1],
        vrms_err=vrms[:, 2],
        vcirc=d.get("vcirc"),
        vcirc_err=d.get("vcirc_err"),
    )


def _priors(d: dict) -> PriorSet:
    c2 = d.get("c_2")
    return PriorSet(
        ml_img=tuple(d["ml_img"]),
        v_vir=tuple(d["v_vir"]),
        gamma=tuple(d["gamma"]),
        q_h=tuple(d["q_h"]),
        ratio=tuple(d["ratio"]),
        c_2=PriorSpec(**c2) if c2 else None,
    )


def _source_priors(d: dict) -> SourcePriors:
    return SourcePriors(**(d or {}))


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

    st = raw["stellar_template"]
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
    )

    search = SearchSpec(**(raw.get("search", {}) or {}))

    return FitConfig(
        pipeline=pipeline,
        data=data,
        search=search,
        subtract_stellar=raw.get("subtract_stellar", True),
        noise_rescale=raw.get("noise_rescale", True),
    )
