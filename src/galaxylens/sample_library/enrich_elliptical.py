"""Enrich the elliptical sample library into a modelable subset.

椭圆库升级：把 modeling_input_elliptical.csv（614 个 DEV 的"排除名单"，只有
name/z/mag）升级成"可建模椭圆样本表" modeling_input_elliptical_modeling.csv。

数据源（都在 GalaxyGravLens 项目内、本地）：
  - SLACS_all_params.csv   Dinos 3 个：zl/zs/sigma/theta_E/γ/Re/n（完整）
  - SLACS_light_params.csv Dinos 3 个：q(=qL) / PA(=phi_L) / amp_sersic_1
  - slacs_kcwi_kinematics_table.csv  KCWI 14 个：sigma_ap(=VD_reff) / reff / ellipticity

字段对齐 j1630_ellip.yaml 的输入（z_lens/z_source/sigma_km_s/theta_E_arcsec/
gamma/re_arcsec/n_sersic/q/pa_deg）。sigma_ap 取 KCWI 的 VD_reff（j1630_ellip.yaml
的 aperture_vrms=267 即此来源，非 Dinos 的 sigma_v=283）。

data_complete 标注参数完整度：
  full       = Dinos 3 个（theta_E/γ/n/q/PA 全齐，可直接生成配方）
  sigma_only = KCWI-only 11 个（有 sigma_ap/Re/q，缺 theta_E/γ/n——theta_E 缺失
               意味着无法钉住内区质量，需从 SLACS Einstein 半径文献另补）

用法：
  python -m galaxylens.sample_library.enrich_elliptical
  或指定数据源/输出： enrich_elliptical(dinos_dir=..., kcwi_path=..., out_path=...)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

# 输出列（对齐 j1630_ellip.yaml 的输入字段）
OUTPUT_COLUMNS = [
    "name", "lens_type_label", "z_lens", "z_source",
    "sigma_km_s", "sigma_err",
    "theta_E_arcsec", "gamma",
    "re_arcsec", "n_sersic", "q", "pa_deg",
    "amp_sersic_1", "source", "data_complete",
]


def _norm(name) -> str:
    name = str(name).lower().strip().replace(" ", "").replace("_", "")
    name = re.sub(r"^sdssj?", "j", name)
    if not name.startswith("j"):
        name = "j" + name
    return name.replace("+", "").replace("-", "")


def _default_paths():
    repo = Path(__file__).resolve().parents[3]          # galaxygravitationallens/
    proj = repo.parent                                   # GalaxyGravLens/
    dinos = proj / "EllipticalLenses" / "dinos_params"
    kcwi = proj / "SpectraPipeline" / "03_catalogs" / "slacs_kcwi_kinematics_table.csv"
    out = repo / "sample_library" / "modeling_input_elliptical_modeling.csv"
    return dinos, kcwi, out


def enrich_elliptical(
    dinos_dir: Optional[Path] = None,
    kcwi_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """组装可建模椭圆表，返回 DataFrame 并写 CSV。"""
    dinos_dir, kcwi_path, out_path = (
        (dinos_dir, kcwi_path, out_path)
        if dinos_dir and kcwi_path and out_path
        else _default_paths()
    )
    dinos_dir, kcwi_path, out_path = Path(dinos_dir), Path(kcwi_path), Path(out_path)

    allp = pd.read_csv(dinos_dir / "SLACS_all_params.csv")
    light = pd.read_csv(dinos_dir / "SLACS_light_params.csv")
    kcwi = pd.read_csv(kcwi_path)

    # Dinos 3 个：按 normalize 名字建 dict（避免 merge 列名冲突）
    allp_by_name = {_norm(r["name"]): r for _, r in allp.iterrows()}
    light_by_name = {_norm(r["lens_name"]): r for _, r in light.iterrows()}

    rows = []
    for _, k in kcwi.iterrows():
        name = str(k["obj\\_name"])
        nrm = _norm(name)

        d_all = allp_by_name.get(nrm)
        d_light = light_by_name.get(nrm)
        has_dinos = d_all is not None

        row = {
            "name": name,
            "lens_type_label": "DEV",           # 椭圆（识别层据此判 elliptical）
            "z_lens": float(k["zlens"]),
            "z_source": float(d_all["z_source"]) if has_dinos else None,
            "sigma_km_s": float(k["VD\\_reff"]),       # KCWI Re 孔径弥散
            "sigma_err": float(k["dVD\\_reff"]),
            "theta_E_arcsec": float(d_all["theta_E"]) if has_dinos else None,
            "gamma": float(d_all["gamma"]) if has_dinos else None,
            "re_arcsec": float(k["reff"]),
            "n_sersic": float(d_all["n_sersic_1"]) if has_dinos else 4.0,
            "q": float(d_light["qL"]) if d_light is not None else 1.0 - float(k["ellipticity"]),
            "pa_deg": float(d_light["phi_L"]) if d_light is not None else None,
            "amp_sersic_1": float(d_all["amp_sersic_1"]) if has_dinos else None,
            "source": "dinos+kcwi" if has_dinos else "kcwi",
            "data_complete": "full" if has_dinos else "sigma_only",
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n_full = (df["data_complete"] == "full").sum()
    n_sig = (df["data_complete"] == "sigma_only").sum()
    print(f"可建模椭圆：{len(df)} 个（{n_full} 个完整 Dinos + {n_sig} 个 KCWI-only）")
    print(f"完整（thetaE/γ/n/q/PA 全）：{', '.join(df[df['data_complete']=='full']['name'])}")
    print(f"缺 thetaE/γ（sigma_only）：{', '.join(df[df['data_complete']=='sigma_only']['name'])}")
    print(f"Saved: {out_path}")
    return df


if __name__ == "__main__":
    enrich_elliptical()
