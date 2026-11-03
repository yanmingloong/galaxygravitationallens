"""Mock-closure orchestrator: select -> truth -> render -> arc -> fit -> closure.

A thin :class:`~galaxylens.api.pipeline.Pipeline` over the existing
``mock_pipeline`` modules. Manual mode is unchanged -- call ``selection.run``,
``skirt.*``, ``lensing.*``, ``validation.closure`` directly. This module only
wires them in order and carries the intermediate products through ``ctx``, so
an end-to-end mock-closure run is one call while every stage stays reachable.

The full chain needs TNG data (API key + particle HDF5) and the SKIRT binary,
so each stage imports its heavy deps lazily and fails with a clear message when
a required ``ctx`` input is missing.

ctx contract (all keys optional until the stage that needs them):
    target, data_root, outdir, workers, l3_workers   -- selection inputs
    candidate_index                                  -- which catalog row (pick)
    star_h5, dm_h5, gas_h5                           -- particle HDF5 paths
    apertures                                        -- {name: radius_kpc} (truth/closure)
    zlens, zsrc, inclination, R_ap_kpc               -- geometry for truth/arc
    label, band_cfg, obs, px_mock, anchor, crop_px   -- SKIRT render inputs
    skirt_bin                                        -- SKIRT binary
    surf_mass_kpc2, sig_as, q_obs                    -- stellar MGE for the arc
    beta_as                                          -- source offset for the arc
    mock_cfg_path                                    -- fit-recipe YAML for the mock
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from galaxylens.api.pipeline import Pipeline, Stage


def _select(ctx: Dict[str, Any]):
    from .selection.run import run_selection

    return run_selection(
        ctx["target"],
        ctx["data_root"],
        ctx["outdir"],
        workers=ctx.get("workers", 1),
        l3_workers=ctx.get("l3_workers", 4),
    )


def _pick(ctx: Dict[str, Any]):
    catalog = ctx.get("select")
    if not catalog:
        raise RuntimeError("select stage produced no candidates; run it first")
    idx = int(ctx.get("candidate_index", 0))
    if idx >= len(catalog):
        raise IndexError(f"candidate_index {idx} out of range ({len(catalog)} rows)")
    return catalog[idx]


def _truth(ctx: Dict[str, Any]):
    cand = ctx["pick"]
    star_h5 = ctx.get("star_h5")
    dm_h5 = ctx.get("dm_h5")
    if not star_h5 or not dm_h5:
        raise RuntimeError("truth stage needs ctx['star_h5'] and ctx['dm_h5']")
    from .truth.particles import load_particles
    from .truth.kinematics import com_iterative
    from .truth.fdm import f_dm_apertures
    from .truth.halo import fit_halo
    from .truth.anisotropy import beta_z

    pos_s, vel_s, m_cur, _, pos_dm, m_dm, *_ = load_particles(star_h5, dm_h5)

    center = com_iterative(pos_s, m_cur)
    R, _ = inertia_rotation_from(pos_s - center, m_cur)
    pos_s_al = (pos_s - center) @ R.T
    pos_dm_al = (pos_dm - center) @ R.T
    vel_cen = com_iterative(vel_s, m_cur)
    vel_rel = vel_s - vel_cen

    apertures = ctx.get("apertures") or {"2kpc": 2.0, "5kpc": 5.0}
    fdm = f_dm_apertures(pos_s_al, m_cur, pos_dm_al, m_dm, apertures)
    halo = fit_halo(pos_dm_al, m_dm, ctx["zlens"], ctx.get("zsrc", 0.0))
    bz = beta_z(pos_s_al, vel_rel, m_cur, ctx.get("R_ap_kpc", 2.0))
    return {"f_dm": fdm, "halo": halo, "beta_z": bz}


def _render(ctx: Dict[str, Any]):
    cand = ctx["pick"]
    star_h5 = ctx.get("star_h5")
    if not star_h5:
        raise RuntimeError("render stage needs ctx['star_h5']")
    from .skirt.convert import tng_to_skirt
    from .skirt.render import write_ski, run_skirt

    label = ctx.get("label", f"{ctx['target']}_{cand['subhalo_id']}")
    outdir = ctx["outdir"]
    tng_to_skirt(star_h5, ctx.get("gas_h5", star_h5), label, cand["sim"],
                 cand["snap"], outdir)
    ski_kw = {"incl": ctx.get("incl", "78.5 deg")}
    if "obs" in ctx:
        ski_kw["obs"] = ctx["obs"]
    dust_ski, _ = write_ski(label, ctx["band_cfg"], outdir, **ski_kw)
    run_skirt(dust_ski, skirt_bin=ctx.get("skirt_bin"), outdir=outdir)
    return {"label": label}


def _arc(ctx: Dict[str, Any]):
    from .lensing.arc import build_lens_galaxy, einstein_radius, make_arc

    truth = ctx.get("truth")
    halo = truth.get("halo") if truth else None
    lens, mstar = build_lens_galaxy(
        ctx["surf_mass_kpc2"], ctx["sig_as"], ctx["q_obs"],
        ctx.get("v_vir", halo.get("v_vir") if halo else None),
        ctx.get("c_minus2", halo.get("c_minus2") if halo else None),
        ctx.get("gamma", halo.get("gamma") if halo else None),
        ctx.get("q_h", halo.get("q_h") if halo else None),
        ctx["zlens"], ctx["zsrc"], ctx["inclination"],
    )
    theta_e, _ = einstein_radius(lens)
    arc = make_arc(lens, ctx.get("beta_as", 0.3), ctx["zsrc"])
    return {"lens": lens, "theta_E": theta_e, "arc": arc, "mstar_model": mstar}


def _fit(ctx: Dict[str, Any]):
    cfg_path = ctx.get("mock_cfg_path")
    if cfg_path is None:
        raise RuntimeError("fit stage needs ctx['mock_cfg_path'] (mock fit-recipe YAML)")
    from galaxylens.api.facade import fit, load_config

    return fit(load_config(cfg_path))


def _closure(ctx: Dict[str, Any]):
    from ..lensfit.validation.closure import closure_f_dm

    r = ctx.get("mock_fit")
    truth = ctx.get("truth")
    if r is None or truth is None:
        raise RuntimeError("closure stage needs ctx['mock_fit'] and ctx['truth']")

    # theta = (lg_ml, v_vir, gamma, c_minus2, q_h, beta_z) from the fit samples.
    # lg_ml convention: MF-mock bakes the Sigma_crit bridge into lg_ml (calib=1,
    # lg_ml = log10(ml_img * ml_to_dyn)); see closure.sample_f_dm docstring. The
    # exact ml_to_dyn factor is recipe-specific, so the caller may pre-supply
    # ctx["theta"]/ctx["weights"] instead.
    if "theta" in ctx and "weights" in ctx:
        theta, w = ctx["theta"], ctx["weights"]
    else:
        def col(key):
            return r.df[r._column(key)].to_numpy(dtype=float)

        lg_ml = np.log10(col("ml_img"))
        theta = np.column_stack([lg_ml, col("v_vir"), col("gamma"),
                                 col("c_minus2"), col("q_h"), 1.0 - col("ratio") ** 2])
        w = r.weights

    return closure_f_dm(theta, w, r.template, r.geom, truth["f_dm"], ctx["apertures"])


def inertia_rotation_from(pos_rel: np.ndarray, mass: np.ndarray):
    from .truth.particles import inertia_rotation

    return inertia_rotation(pos_rel, mass)


def build_mock_pipeline(
    target: str,
    data_root: str,
    outdir: str,
    **ctx_overrides,
) -> Pipeline:
    """Build the mock-closure pipeline. ``ctx_overrides`` seed the ``ctx`` dict."""
    ctx0: Dict[str, Any] = {
        "target": target,
        "data_root": data_root,
        "outdir": outdir,
    }
    ctx0.update(ctx_overrides)

    stages = [
        Stage("select", _select, "find TNG disk-lens analogs"),
        Stage("pick", _pick, "choose one candidate", depends=["select"]),
        Stage("truth", _truth, "particle truth (f_dm / halo / beta_z)", depends=["pick"]),
        Stage("render", _render, "SKIRT convert + render -> fake HST", depends=["pick"]),
        Stage("make_arc", _arc, "ray-trace a background arc", depends=["pick"]),
        Stage("mock_fit", _fit, "gl.fit the mock", depends=["render", "make_arc"]),
        Stage("closure", _closure, "f_dm bias vs truth", depends=["mock_fit", "truth"]),
    ]
    return Pipeline(f"mock:{target}", stages)
