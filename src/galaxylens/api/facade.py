"""Facade: the user-facing entry points (``gl.fit`` / ``gl.analyze`` / ``gl.identify`` / ``gl.mock``).

"Core is the engine, facade is the handle": this module only wires the core
layer together; all physics lives in ``lensfit``. The heavy dependencies
(autolens / autofit) are imported lazily inside each function, so
``import galaxylens`` stays light.

References
----------
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (the fitting recipe
  these entry points run)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .config_loader import FitConfig, load_config as _load_config


def load_config(path) -> FitConfig:
    """Load a fitting-recipe YAML into a :class:`FitConfig` (see ``api.config_loader``)."""
    return _load_config(path)


def _output_root(search_spec) -> Path:
    if search_spec.output_root:
        return Path(search_spec.output_root)
    root = os.environ.get("GALAXYLENS_OUTPUT_ROOT")
    if not root:
        raise RuntimeError(
            "GALAXYLENS_OUTPUT_ROOT is not set and search.output_root is None; "
            "set one of them to a directory for fit outputs"
        )
    return Path(root)


def load_dataset(cfg: FitConfig, out_dir: Optional[Path] = None):
    """Load the masked, galaxy-subtracted (optional), rescaled ``al.Imaging``.

    Returns ``(dataset, n_unmasked)`` ready for ``gl.fit``.
    """
    from .dataset import load_dataset as _load_imaging

    return _load_imaging(
        cfg.data,
        cfg.pipeline.geometry.zlens,
        subtract_stellar=cfg.subtract_stellar,
        noise_rescale=cfg.noise_rescale,
        out_dir=out_dir,
    )


def _find_samples_csv(output_root: Path, name: str) -> Path:
    """Locate the most recently written samples.csv under output_root/name/..."""
    candidates = list(output_root.glob(f"{name}/*/files/samples.csv"))
    if not candidates:
        raise FileNotFoundError(f"no samples.csv found under {output_root}/{name}/...")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _submit_slurm(cfg: FitConfig, config_path: Optional[str]) -> dict:
    """Render + write + submit the slurm job for a ``backend: slurm`` recipe."""
    from .search import submit_slurm

    if config_path is None:
        raise ValueError(
            "backend='slurm' requires config_path (the YAML the cluster runner "
            "loads); pass it to gl.fit(cfg, config_path=...)"
        )
    data_root = os.environ.get("GALAXYLENS_DATA_ROOT", "")
    out_root = _output_root(cfg.search)
    return submit_slurm(cfg.search, out_root, config_path, data_root)


def fit(cfg: FitConfig, config_path: Optional[str] = None):
    """Run the full joint fit: dataset -> assemble -> sample -> :class:`Result`.

    ``backend: local`` runs Nautilus in-process and returns a :class:`Result`
    exposing ``beta_z`` / ``m_star`` / ``f_dm(...)``. ``backend: slurm`` renders
    + writes + submits an SBATCH script (via :func:`submit_slurm`) and returns a
    dict with ``script_path`` / ``job_id``; the cluster runner
    (``galaxylens.api.runner``) then loads the recipe and runs the fit locally
    on the cluster.
    """
    if cfg.search.backend == "slurm":
        return _submit_slurm(cfg, config_path)
    if cfg.search.backend != "local":
        raise ValueError(
            f"unknown backend={cfg.search.backend!r}; expected 'local' or 'slurm'"
        )

    from ..lensfit.pipeline.assemble import assemble
    from .result import Result
    from .search import build_search

    out_root = _output_root(cfg.search)
    out_root.mkdir(parents=True, exist_ok=True)

    dataset, n_unmasked = load_dataset(cfg, out_dir=out_root)

    collection, analysis = assemble(cfg.pipeline, dataset, n_unmasked)

    search = build_search(cfg.search, out_root)
    result = search.fit(model=collection, analysis=analysis)

    samples_csv = _find_samples_csv(out_root, cfg.search.name)
    return Result(samples_csv, cfg.pipeline.stellar_template, cfg.pipeline.geometry)


def fit_positions(
    theta_obs,
    err_arcsec,
    zlens: float = 0.113,
    zsrc: float = 0.254,
    kpc_per_arcsec: float = 2.006,
    p0=None,
):
    """Fit the four image positions of a quad with a scale-free potential (D1).

    Point-level lensing for a quadruple-image system (Trick+2016 J1331): the
    source is a point-like QSO, so the lensing information is the image
    positions, not a pixel-level arc. Returns a :class:`ScaleFreeFit` with the
    Einstein radius / mass. ``theta_obs`` is a (4, 2) array of image positions
    relative to the lens centre (arcsec); ``err_arcsec`` the positional error.

    References: Evans & Witt 2003; Trick, van de Ven & Dutton 2016 (arXiv:1609.05477).
    """
    from ..lensfit.lens.scale_free import (
        ScaleFreeFit,
        fit_image_positions,
        rein_mein,
        sigma_crit,
    )

    sc = sigma_crit(zlens, zsrc)
    params, chi2 = fit_image_positions(theta_obs, err_arcsec, p0)
    a0, a2, b2, xs, ys = params
    rein_a, rein_kpc, mein, mcrit = rein_mein(a0, a2, b2, kpc_per_arcsec, sc)
    return ScaleFreeFit(
        a0=a0, a2=a2, b2=b2, xs=xs, ys=ys,
        rein_arcsec=rein_a, rein_kpc=rein_kpc,
        mein_1e10=mein, mcrit_1e10=mcrit,
        chi2=chi2, dof=2 * len(theta_obs) - 5,
    )


def fit_mein(
    geometry,
    kinematics,
    lum_mge,
    ml_array,
    rein_arcsec: float,
    mein_msun: float,
    mein_frac_err: float = 0.10,
    kappa_prime: float = 0.76,
    r0_kappa_arcsec: float = 2.2,
    seeing_fwhm_arcsec: float = 1.1,
    v200_init: float = 240.0,
    c200_init: float = 8.0,
    beta_z_init: float = -0.4,
):
    """Point-level joint fit (D5/J1331): dynamics + Einstein mass, scipy MAP.

    Fits ``(v200, c200, beta_z)`` of a spherical NFW + radial-M/L stellar MGE +
    counter-rotating-core jampy dynamics, constrained by the 9 v_rms points and
    the single Einstein mass ``M_2D(<Rein) = Mein +- frac_err`` (the CORRECTED
    aperture-mass form, see ``m2d_within``). Uses the D5 priors
    ``v200 ~ N(200, 40)`` and the Maccio+2008 WMAP5 c-M relation.

    Returns a dict with the MAP ``v200`` / ``c200`` / ``beta_z`` and the
    ``chi2_dyn`` / ``chi2_lens`` breakdown.
    """
    import numpy as np
    from scipy.optimize import minimize
    from types import SimpleNamespace

    from ..lensfit.models.dm.nfw import NFW
    from ..lensfit.models.stellar.mge_radial_ml import MgeRadialMl
    from ..lensfit.dynamics.jampy_kdc import JampyKdcEngine
    from ..lensfit.fit.constraints import MeinConstraint
    from ..lensfit.structures import Calibration

    _H0 = 71.9
    _RHO_CRIT = 1.43e-7

    stellar = MgeRadialMl(lum_mge, ml_array)
    dm = NFW()
    engine = JampyKdcEngine(
        kappa_prime=kappa_prime,
        r0_kappa_arcsec=r0_kappa_arcsec,
        seeing_fwhm_arcsec=seeing_fwhm_arcsec,
    )
    mein = MeinConstraint(rein_arcsec, mein_msun, mein_frac_err)

    calib = Calibration(1.0)

    def _instance(v200, c200):
        return SimpleNamespace(galaxies=SimpleNamespace(lens=SimpleNamespace(
            dark=SimpleNamespace(v200=v200, c200=c200),
        )))

    def neg_log_prob(theta):
        v200, logc, betaz = theta
        c200 = 10.0 ** logc
        if not (50 <= v200 <= 450 and 1 <= c200 <= 50 and -0.5 <= betaz <= 0.5):
            return 1e9
        inst = _instance(v200, c200)
        lum = stellar.lum_mge(inst, geometry, calib)
        mass = stellar.dyn_mge(inst, geometry, calib)
        dm_mge = dm.dyn_mge(inst, geometry, calib)
        model_pot = (
            np.concatenate([mass[0], dm_mge[0]]),
            np.concatenate([mass[1], dm_mge[1]]),
            np.concatenate([mass[2], dm_mge[2]]),
        )
        chi2_dyn, _ = engine.chi2(lum, model_pot, betaz, geometry, kinematics)
        chi2_lens = mein.chi2(inst, [stellar, dm], geometry, calib)
        # D5 priors: v200 ~ N(200, 40) + Maccio+2008 WMAP5 c-M relation
        lp = 0.5 * ((v200 - 200.0) / 40.0) ** 2
        r200_pc = v200 / (10 * _H0) * 1e6
        m200 = (4 * np.pi / 3) * 200 * _RHO_CRIT * r200_pc ** 3
        logc_mean = 0.830 - 0.098 * np.log10((_H0 / 100.0) * m200 / 1e12)
        lp += 0.5 * ((np.log10(c200) - logc_mean) / 0.105) ** 2
        return 0.5 * chi2_dyn + 0.5 * chi2_lens + lp

    res = minimize(
        neg_log_prob,
        np.array([v200_init, np.log10(c200_init), beta_z_init]),
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 0.01, "fatol": 0.1},
    )
    v200, c200, beta_z = res.x[0], 10.0 ** res.x[1], res.x[2]

    # recompute the chi2 breakdown at the MAP
    inst = _instance(v200, c200)
    lum = stellar.lum_mge(inst, geometry, calib)
    mass = stellar.dyn_mge(inst, geometry, calib)
    dm_mge = dm.dyn_mge(inst, geometry, calib)
    model_pot = (
        np.concatenate([mass[0], dm_mge[0]]),
        np.concatenate([mass[1], dm_mge[1]]),
        np.concatenate([mass[2], dm_mge[2]]),
    )
    chi2_dyn, _ = engine.chi2(lum, model_pot, beta_z, geometry, kinematics)
    chi2_lens = mein.chi2(inst, [stellar, dm], geometry, calib)

    return {
        "v200": float(v200),
        "c200": float(c200),
        "beta_z": float(beta_z),
        "chi2_dyn": float(chi2_dyn),
        "chi2_lens": float(chi2_lens),
        "neg_log_prob": float(res.fun),
    }


def analyze(cfg: FitConfig, samples_csv=None, out_dir=None, corner: bool = True, verbose: bool = True):
    """Analyze a finished fit: posterior summary + derived + prior-edge check + corner.

    The post-fit "analyze" step, also callable on its own (manual mode). Reads the
    weighted posterior off ``samples.csv`` (auto-located under the fit's output
    dir if not given) and returns a dict with ``params`` / ``derived`` / ``edges``
    (see :mod:`galaxylens.api.analysis`).
    """
    from .analysis import analyze as _analyze

    return _analyze(cfg, samples_csv=samples_csv, out_dir=out_dir, corner=corner, verbose=verbose)


def identify(name: str, catalog=None):
    """识别透镜类型 + 建模方案（识别层入口，可独立调用）。

    ``gl.identify("J2141-0001")`` -> :class:`~galaxylens.sample_library.identify.LensPlan`
    包含 ``lens_type`` / ``constraint`` / 建模模块选择 / ``template_yaml``。这是
    "识别 -> 建模 -> 拟合" 全自动链路的第一环：拿 ``LensPlan.template_yaml`` 填参数
    生成配方，再交给 ``gl.fit``。
    """
    from ..sample_library.identify import identify as _identify

    return _identify(name, catalog=catalog)


def identify_from_image(fits_path, name=None, center=None, rmax_pix=None):
    """从图像测形态 -> 分类 -> :class:`LensPlan`（图像形状识别入口）。

    当 catalog 参数（lens_type_label/n_sersic）缺失或不可靠时，直接对 FITS 图像
    测光集中度 C + 轴比 q，判盘/椭圆（C < 3.1 盘、C > 3.4 椭圆）。返回 LensPlan，
    ``constraint`` 默认 ``imaging``（延展弧）。这是识别层的图像兜底，不依赖 catalog。
    """
    from ..sample_library.identify import plan_modeling
    from ..sample_library.morphology import classify_morphology, measure_morphology

    morph = measure_morphology(fits_path, center=center, rmax_pix=rmax_pix)
    lens_type, basis = classify_morphology(morph)
    if name is None:
        name = str(Path(fits_path).stem)
    return plan_modeling(name, lens_type, "imaging", f"image_morphology:{basis}")
