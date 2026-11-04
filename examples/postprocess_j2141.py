#!/usr/bin/env python3
"""Post-process the pulled-back J2141 B7 joint-fit result (stage 3 acceptance).

Reads the weighted ``samples.csv`` (no autofit result object -> no OOM), prints
the physical quantities, and draws the repo's own weighted corner plot (Agg
backend, so it also works headless).

Run after scp-ing the cluster output dir, e.g.:
    python examples/postprocess_j2141.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

import galaxylens as gl  # noqa: E402
from galaxylens.api.result import Result  # noqa: E402
from galaxylens.api.plotting.corner import plot_corner  # noqa: E402

# ---- paths (adjust samples_csv if you pulled the dir elsewhere) ----
SAMPLES_CSV = REPO / "outputs/j2141_b7_joint/<run_hash>/files/samples.csv"
CONFIG = REPO / "configs/fit/j2141_barnabe.yaml"
OUT_DIR = REPO / "outputs/j2141_b7_joint"

# ---- load recipe (for the stellar template + geometry) ----
cfg = gl.load_config(CONFIG)
result = Result(SAMPLES_CSV, cfg.pipeline.stellar_template, cfg.pipeline.geometry)


def fmt(arr):
    lo, med, hi = arr
    return f"{med:.3g} [-{med - lo:.2g} +{hi - med:.2g}]"


print("=== J2141 B7 joint fit (gl.fit, stage 3) weighted posterior ===")
print(f"beta_z        = {fmt(result.beta_z)}")
print(f"c_-2          = {result.c_minus2}")
print(f"gamma         = {result.gamma}")
print(f"v_vir [km/s]  = {result.v_vir}")
print(f"q_h           = {result.q_h}")
print(f"c_200         = {fmt(result.c200)}")
print(f"M* [Msun]     = {fmt(result.m_star)}")

rd = cfg.pipeline.geometry.rd_arcsec
print(f"f_dm(2.2Rd)   = {fmt(result.f_dm(2.2 * rd))}   (aperture {2.2 * rd:.3g}\")")

# ---- weighted corner plot ----
OUT_DIR.mkdir(parents=True, exist_ok=True)
corner_path = OUT_DIR / "j2141_b7_corner.png"
plot_corner(
    SAMPLES_CSV,
    corner_path,
    param_keys=("ml_img", "v_vir", "c_minus2", "gamma", "q_h", "ratio"),
    param_labels=("ml_img", "v_vir", "c_-2", "gamma", "q_h", "ratio"),
    title="J2141 B7 joint posterior (weighted)",
)
print(f"\ncorner plot -> {corner_path}")
