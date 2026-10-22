#!/usr/bin/env python3
"""Regression check: facade layer (api/) against B7 anchors.

Run with autolens_jax. Covers the locally-testable facade pieces (no FITS data):
  1. config_loader: configs/fit/j2141_barnabe.yaml -> FitConfig (round-trip).
  2. derived: m_star_total / f_dm_of at the B7 median-pdf params.
  3. result: weighted_percentile + beta_z + m_star on a synthetic samples.csv.
  4. search: build_search(local) -> af.Nautilus (construction smoke).

The data-dependent path (dataset galaxy-subtract + gl.fit end-to-end) needs the
F814W FITS (cluster-only), exercised in stage 3.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]  # repo root (portable)
sys.path.insert(0, str(REPO / "src"))

from galaxylens.api import derived
from galaxylens.api.config_loader import load_config
from galaxylens.api.derived import f_dm_of, m_star_total
from galaxylens.api.result import Result, weighted_percentile
from galaxylens.api.search import build_search

# ---- 1. config round-trip ----
cfg = load_config(REPO / "configs/fit/j2141_barnabe.yaml")
p = cfg.pipeline
print(f"[1] geometry: zl={p.geometry.zlens}, zs={p.geometry.zsrc}, "
      f"i={p.geometry.inclination_deg}, rd={p.geometry.rd_arcsec}")
print(f"    components: {p.dark}/{p.stellar}/{p.source}/{p.dynamics}")
print(f"    kin n_points={p.kinematics.n_points}, vcirc={p.kinematics.vcirc}")
print(f"    priors c_2 kind={p.priors.c_2.kind} mean={p.priors.c_2.mean}")
print(f"    search backend={cfg.search.backend}, n_live={cfg.search.n_live}")
assert p.geometry.zlens == 0.138 and p.geometry.zsrc == 0.7127
assert p.geometry.inclination_deg == 78.5
assert p.dark == "gnfw_virial" and p.stellar == "mge_const_ml"
assert p.dynamics == "jampy" and p.source == "single_sersic"
assert p.kinematics.n_points == 12  # 11 v_rms + 1 vcirc
assert p.priors.c_2.kind == "gaussian" and abs(p.priors.c_2.mean - 5.9) < 1e-9
assert cfg.search.backend == "local"

# ---- 2. derived (B7 median-pdf params) ----
B7 = dict(
    ml_img=13.575937273609815,
    v_vir=212.26679152167054,
    c_minus2=5.095233792763365,
    gamma=0.77151875639625,
    q_h=0.69384859965147,
)
tpl = p.stellar_template
mstar = m_star_total(B7["ml_img"], tpl)
calib_l2d = tpl.calibration.calib * tpl.l2d_rel
print(f"[2] M* = {mstar:.3e} Msun  (calib*l2d_rel={calib_l2d:.4e})")
# local template gives calib*l2d_rel ~1.07e10 (cluster 1.058e10), so M* ~1.45e11
assert abs(calib_l2d - 1.058e10) / 1.058e10 < 0.02
fdm, mstar_ap, mdm = f_dm_of(
    B7["ml_img"], B7["v_vir"], B7["gamma"], B7["c_minus2"], B7["q_h"],
    2.2 * p.geometry.rd_arcsec, tpl, p.geometry,
)
print(f"    f_dm(2.2Rd) = {fdm:.3f}  (M_star={mstar_ap:.3e}, M_dm={mdm:.3e})")
# Point value at the B7 median params; the B6 published 0.23 is the *posterior
# median* of f_dm (a nonlinear function of gamma/c_2), 1-sigma [0.13, 0.39], so a
# single-point comparison is not exact -- assert the physical plausibility band.
assert 0.15 <= fdm <= 0.30
assert mstar_ap > 0 and mdm > 0

# ---- 3. result (synthetic samples.csv) ----
rng = np.random.RandomState(0)
n = 200
ml = rng.uniform(10, 16, n)
vv = rng.uniform(180, 250, n)
gm = rng.uniform(0.5, 1.0, n)
c2 = rng.uniform(4, 7, n)
qh = rng.uniform(0.5, 0.9, n)
ratio = rng.uniform(0.5, 0.8, n)
w = rng.uniform(0.5, 1.5, n)
df = pd.DataFrame(
    {
        "galaxies.lens.mge_0.mass_to_light_ratio": ml,
        "galaxies.lens.dark.v_vir": vv,
        "galaxies.lens.dark.c_2": c2,
        "galaxies.lens.dark.inner_slope": gm,
        "galaxies.lens.dark.q_h": qh,
        "galaxies.lens.anisotropy.mass_to_light_ratio": ratio,
        "weight": w,
    }
)
csv_path = Path(tempfile.mkstemp(suffix=".csv")[1])
df.to_csv(csv_path, index=False)

res = Result(csv_path, tpl, p.geometry)
bz_med = res.beta_z[1]
bz_expected = weighted_percentile(1.0 - ratio ** 2, w, [16, 50, 84])
print(f"[3] beta_z weighted median = {bz_med:.4f} (hand-weighted {bz_expected[1]:.4f})")
assert abs(bz_med - bz_expected[1]) < 1e-9

mstar_med = res.m_star[1]
mstar_expected = weighted_percentile(derived.m_star_total(ml, tpl), w, [16, 50, 84])
assert abs(mstar_med - mstar_expected[1]) < 1e-6

ml_med = res.ml_img.median
ml_expected = weighted_percentile(ml, w, [16, 50, 84])[1]
print(f"    ml_img median = {ml_med:.4f} (hand-weighted {ml_expected:.4f})")
assert abs(ml_med - ml_expected) < 1e-9

c200_med = res.c200[1]
c200_expected = weighted_percentile((2.0 - gm) * c2, w, [16, 50, 84])[1]
print(f"    c200 median = {c200_med:.3f} (hand-weighted {c200_expected:.3f})")
assert abs(c200_med - c200_expected) < 1e-9

# ---- 4. search local backend smoke ----
out_root = Path(tempfile.mkdtemp())
search = build_search(cfg.search, out_root)
print(f"[4] search backend={cfg.search.backend} constructed: {type(search).__name__}")
assert search is not None

print("\nALL facade (api) regression checks passed.")
