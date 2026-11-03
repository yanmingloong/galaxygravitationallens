#!/usr/bin/env python3
"""Recipe regression: the B9 K'-band pixelized-source YAML loads and assembles.

Checks the j2141_b9_kband_pixelized.yaml recipe end-to-end at the CONFIG level
(no fit run): the pixelized source is wired, the Gaussian M/L prior parses, the
mge_agn subtraction flags parse, and the assembled model has 7 explicit
parameters (6 mass + 1 regularization coefficient) with n_source = 225.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# The mge_csv stellar template reads mge_params.csv relative to GALAXYLENS_DATA_ROOT.
_B2_MGE = Path(
    "/mnt/d/HermesAgent/ResearchProjects/GalaxyGravLens/DiskLenses/AutoLensFit"
    "/B_series/outputs/b2_mge"
)
os.environ.setdefault("GALAXYLENS_DATA_ROOT", str(_B2_MGE))

from galaxylens.api.config_loader import load_config, _load_mge_csv
from galaxylens.lensfit.pipeline.assemble import build_collection, _component

cfg = load_config(REPO / "configs/fit/j2141_b9_kband_pixelized.yaml")
p = cfg.pipeline

assert p.source == "pixelized", p.source
assert p.dark == "gnfw_virial" and p.stellar == "mge_const_ml" and p.dynamics == "jampy"
assert p.source_mesh_shape == (15, 15), p.source_mesh_shape
assert all(isinstance(x, float) for x in p.source_reg_coeff_bounds), p.source_reg_coeff_bounds
assert p.priors.ml_img_spec is not None
assert abs(p.priors.ml_img_spec.mean - 0.8) < 1e-9
assert cfg.data.subtract_method == "mge_agn"
assert cfg.data.psf_sigma_px is not None

col = build_collection(p)
print(f"[b9-recipe] source={p.source} prior_count={col.prior_count} (expect 7)")
assert col.prior_count == 7

src = _component("pixelized", p)
print(f"[b9-recipe] n_free_params={src.n_free_params} n_source={src.n_source} (expect 1/225)")
assert src.n_free_params == 1 and src.n_source == 225

print("[b9-recipe] coeff_bounds =", p.source_reg_coeff_bounds,
      "ml_img_spec =", p.priors.ml_img_spec)

# ---- K'-band stellar-template calibration (stellar_mge_kband) ----
from galaxylens.lensfit.models.stellar.mge_const_ml import stellar_mge_kband
from galaxylens.lensfit.models.dm.gnfw_virial import halo_cosmo

MGEP = _B2_MGE / "mge_params.csv"
if MGEP.exists():
    Ls, sigs, qs = _load_mge_csv(MGEP)
    hc = halo_cosmo(0.138, 0.7127)
    sigma_crit_as2 = hc.sigma_crit * hc.kpc_per_arcsec ** 2
    tmpl = stellar_mge_kband(
        Ls, sigs, qs, pixel_scale=0.039686,
        mstar_ref=1.32e11, ml_ref=0.8,
        sigma_crit_arcsec2=sigma_crit_as2, pc_per_arcsec=hc.pc_per_arcsec,
    )
    L_ref = 1.32e11 / 0.8  # = 1.65e11
    # M* = ml_img * calib * l2d_rel = ml_img * L_ref
    assert abs(tmpl.calibration.calib * tmpl.l2d_rel - L_ref) / L_ref < 1e-9
    # two bridges: calib_img * Sigma_crit == calib * pc_per_arcsec^2
    lhs = tmpl.calibration_img.calib * sigma_crit_as2
    rhs = tmpl.calibration.calib * hc.pc_per_arcsec ** 2
    assert abs(lhs - rhs) / rhs < 1e-9
    print(f"[kband-template] calib*l2d_rel = L_ref = {L_ref:.4e}  OK")
    print("[kband-template] img-bridge * Sigma_crit == dyn-bridge * pc^2  OK")
else:
    print("[kband-template] SKIP: mge_params.csv not present")

print("\nB9 recipe regression passed.")
