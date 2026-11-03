"""Regression: gl.identify (lens identification layer).

Verifies the identify layer maps the three anchor lenses to the right modeling
plan:
    J2141-0001  disk  x imaging   -> j2141_barnabe.yaml
    J1331+3628  disk  x positions -> j1331_trick.yaml   (quad, known-name list)
    elliptical  (DEV row)          -> j1630_ellip.yaml  (ellip branch)

Honest boundary recorded here: J1630+4520 in the shipped catalog has
lens_type_label = None (it was never tagged DEV), so auto-identify falls back
to the disk default -- fixing that is the elliptical-library upgrade, not an
identify bug. The ellip branch is exercised with a hand-built DEV row.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

import galaxylens as gl
from galaxylens.sample_library.identify import (
    LensPlan,
    identify_constraint,
    identify_type,
    plan_modeling,
)


def test_identify_j2141_disk():
    p = gl.identify("J2141-0001")
    assert p.lens_type == "disk", p.basis
    assert p.constraint == "imaging"
    assert p.template_yaml == "configs/fit/j2141_barnabe.yaml"
    assert p.dynamics == "jampy_kdc"
    print(f"[identify] J2141 -> disk/jampy_kdc ({p.basis})")


def test_identify_j1331_quad():
    p = gl.identify("J1331+3628")
    assert p.constraint == "positions", p.basis
    assert p.template_yaml == "configs/fit/j1331_trick.yaml"
    print(f"[identify] J1331 -> quad/positions ({p.basis})")


def test_identify_elliptical_branch():
    # Ellip branch: DEV label -> elliptical/sersic_single/jampy (J1630 route)
    row = pd.Series({"name": "SDSS J1630+4520", "lens_type_label": "DEV"})
    lt, tb = identify_type(row)
    assert lt == "elliptical", tb
    c, cb = identify_constraint(row, lt)
    assert c == "imaging", cb
    plan = plan_modeling("SDSS J1630+4520", lt, c, tb)
    assert plan.template_yaml == "configs/fit/j1630_ellip.yaml"
    assert plan.subtract_method == "sersic_single"
    assert plan.stellar_template == "ellip"
    assert plan.dynamics == "jampy"
    print(f"[identify] DEV row -> elliptical/jampy ({plan.basis})")


def test_identify_j1630_elliptical():
    # After the ellip-library upgrade: J1630 has a DEV label in the elliptical
    # modeling table, auto-identifies as elliptical
    p = gl.identify("J1630+4520")
    assert p.lens_type == "elliptical", p.basis
    assert p.constraint == "imaging"
    assert p.template_yaml == "configs/fit/j1630_ellip.yaml"
    assert p.subtract_method == "sersic_single"
    print(f"[identify] J1630 -> elliptical/jampy ({p.basis})")


def test_identify_elliptical_modeling_table():
    # Samples in the elliptical modeling table (enrich_elliptical output) should
    # identify as elliptical
    from galaxylens.sample_library.identify import load_catalog

    cat = load_catalog()
    modeling = cat[cat["lens_type_label"].astype(str) == "DEV"]
    assert len(modeling) == 14, f"expected 14 DEV in the elliptical modeling table, got {len(modeling)}"
    # spot-check (3 Dinos full + 2 KCWI-only)
    n_ellip = 0
    for nm in ["J1402+6321", "J1630+4520", "J2303+1422", "J0037-0942", "J0330-0020"]:
        p = gl.identify(nm)
        assert p.lens_type == "elliptical", f"{nm} -> {p.lens_type}"
        n_ellip += 1
    print(f"[identify] {len(modeling)} DEV in the elliptical modeling table, {n_ellip} spot-checked all elliptical")


def test_write_recipe():
    # write_recipe copies a template + overrides params, emits a loadable YAML
    from galaxylens.sample_library.identify import write_recipe

    row = pd.Series({"name": "SDSS J1630+4520", "lens_type_label": "DEV"})
    lt, tb = identify_type(row)
    plan = plan_modeling("SDSS J1630+4520", lt, "imaging", tb)
    out = write_recipe(plan, {"search.name": "j1630_identify_test"}, Path("/tmp/j1630_test.yaml"))
    assert out.exists()
    cfg = gl.load_config(str(out))
    assert cfg.search.name == "j1630_identify_test"
    out.unlink()
    print("[identify] write_recipe -> valid FitConfig YAML")


if __name__ == "__main__":
    test_identify_j2141_disk()
    test_identify_j1331_quad()
    test_identify_elliptical_branch()
    test_identify_j1630_elliptical()
    test_identify_elliptical_modeling_table()
    test_write_recipe()
    print("ALL verify_identify checks passed")
