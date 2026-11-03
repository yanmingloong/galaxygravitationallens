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
    # 椭圆分支：DEV 标签 -> elliptical/sersic_single/jampy（J1630 路线）
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


def test_identify_j1630_honest_boundary():
    # 诚实边界：catalog 里 J1630 的 type=None，自动识别落默认盘；
    # 这是数据缺漏（椭圆库升级补 DEV 标签），不是识别层 bug。
    p = gl.identify("J1630+4520")
    assert "no_morphology_label_default_disk" in p.basis, p.basis
    print(f"[identify] J1630 (catalog type=None) -> {p.lens_type} ({p.basis})")


def test_write_recipe():
    # write_recipe 从模板复制 + 覆盖参数，产出可 load 的 YAML
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
    test_identify_j1630_honest_boundary()
    test_write_recipe()
    print("ALL verify_identify checks passed")
