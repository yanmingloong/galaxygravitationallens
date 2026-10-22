#!/usr/bin/env python3
"""Regression check: lensfit.lens model builder uses the B7 v_vir parametrization.

Run with autolens_jax. Verifies build_lens_model produces a 12-parameter model
with the v_vir-normalized GNFWVirialVelocity halo: prior_count=12, shared M/L
across 16 Gaussians, GNFWVirialVelocity dark + Sersic source.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root (portable)
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from galaxylens.lensfit.lens import build_lens_model
from galaxylens.lensfit.mge import load_mge

MGE_CSV = Path(
    "/mnt/d/HermesAgent/ResearchProjects/GalaxyGravLens/DiskLenses/AutoLensFit/B_series/"
    "outputs/b2_mge/mge_params.csv"
)

mge = load_mge(MGE_CSV)
model = build_lens_model(zlens=0.138, zsrc=0.7127, mge=mge)

print(f"prior_count = {model.prior_count}  (expect 12 = gNFW 4 + M/L 1 + source 7)")
assert model.prior_count == 12

# M/L shared: all mge_* Gaussians reference ONE prior object
lens = model.galaxies.lens
ml_ids = {id(getattr(lens, f"mge_{i}").mass_to_light_ratio) for i in range(16)}
print(f"distinct M/L prior objects across 16 Gaussians = {len(ml_ids)}  (expect 1)")
assert len(ml_ids) == 1

# dark profile type + source type
print(f"dark = {lens.dark.cls.__name__}  (expect GNFWVirialVelocity)")
print(f"source = {model.galaxies.source.bulge.cls.__name__}  (expect Sersic)")
assert lens.dark.cls.__name__ == "GNFWVirialVelocity"
assert "Sersic" in model.galaxies.source.bulge.cls.__name__

# dark priors: v_vir / c_2 / inner_slope / q_h (B7 v_vir parametrization)
d = lens.dark
print(f"v_vir prior = [{d.v_vir.lower_limit:.1f}, {d.v_vir.upper_limit:.1f}]")
print(f"c_2 prior = [{d.c_2.lower_limit:.1f}, {d.c_2.upper_limit:.1f}]  (Uniform default)")
print(f"inner_slope prior = [{d.inner_slope.lower_limit:.1f}, {d.inner_slope.upper_limit:.1f}]")
print(f"q_h prior = [{d.q_h.lower_limit:.2f}, {d.q_h.upper_limit:.2f}]")
assert d.v_vir.lower_limit == 150.0 and d.v_vir.upper_limit == 350.0
assert d.c_2.lower_limit == 3.0 and d.c_2.upper_limit == 20.0
assert d.inner_slope.lower_limit == 0.2 and d.inner_slope.upper_limit == 2.0
assert d.q_h.lower_limit == 0.4 and d.q_h.upper_limit == 1.0

print("\nALL lens model regression checks passed.")
