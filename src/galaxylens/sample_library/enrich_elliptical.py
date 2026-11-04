"""Enrich the elliptical sample library into a modelable subset.

Upgrade ``modeling_input_elliptical.csv`` (a 614-DEV "exclusion list" with only
name/z/mag) into the modelable elliptical table
``modeling_input_elliptical_modeling.csv``.

Data sources (provided by the user via GALAXYLENS_SAMPLE_LIB or explicit args):
  - SLACS_all_params.csv     Dinos 3: zl/zs/sigma/theta_E/gamma/Re/n (complete)
  - SLACS_light_params.csv   Dinos 3: q(=qL) / PA(=phi_L) / amp_sersic_1
  - slacs_kcwi_kinematics_table.csv  KCWI 14: sigma_ap(=VD_reff) / reff / ellipticity

Fields align with j1630_ellip.yaml's inputs (z_lens/z_source/sigma_km_s/
theta_E_arcsec/gamma/re_arcsec/n_sersic/q/pa_deg). sigma_ap takes the KCWI
VD_reff (j1630_ellip.yaml's aperture_vrms=267 comes from here, NOT Dinos'
sigma_v=283).

data_complete marks parameter completeness:
  full       = Dinos 3 (theta_E/gamma/n/q/PA all present, recipe-ready)
  sigma_only = KCWI-only 11 (has sigma_ap/Re/q, missing theta_E/gamma/n -- a
               missing theta_E cannot pin the inner mass, needs the SLACS
               Einstein-radius literature)

Usage:
  python -m galaxylens.sample_library.enrich_elliptical
  or with explicit sources/output: enrich_elliptical(dinos_dir=..., kcwi_path=..., out_path=...)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd

# Output columns (aligned with j1630_ellip.yaml's input fields)
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
    root = os.environ.get("GALAXYLENS_SAMPLE_LIB")
    if not root:
        raise RuntimeError(
            "GALAXYLENS_SAMPLE_LIB is not set. Point it to the sample-library root "
            "containing dinos_params/ and slacs_kcwi_kinematics_table.csv, or pass "
            "dinos_dir / kcwi_path / out_path explicitly."
        )
    root = Path(root)
    dinos = root / "dinos_params"
    kcwi = root / "slacs_kcwi_kinematics_table.csv"
    repo = Path(__file__).resolve().parents[3]          # galaxygravitationallens/
    out = repo / "sample_library" / "modeling_input_elliptical_modeling.csv"
    return dinos, kcwi, out


def enrich_elliptical(
    dinos_dir: Optional[Path] = None,
    kcwi_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Assemble the modelable elliptical table, return the DataFrame and write CSV."""
    dinos_dir, kcwi_path, out_path = (
        (dinos_dir, kcwi_path, out_path)
        if dinos_dir and kcwi_path and out_path
        else _default_paths()
    )
    dinos_dir, kcwi_path, out_path = Path(dinos_dir), Path(kcwi_path), Path(out_path)

    allp = pd.read_csv(dinos_dir / "SLACS_all_params.csv")
    light = pd.read_csv(dinos_dir / "SLACS_light_params.csv")
    kcwi = pd.read_csv(kcwi_path)

    # Dinos 3: index by normalized name via a dict (avoid merge column clashes)
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
            "lens_type_label": "DEV",           # elliptical (identification keys on it)
            "z_lens": float(k["zlens"]),
            "z_source": float(d_all["z_source"]) if has_dinos else None,
            "sigma_km_s": float(k["VD\\_reff"]),       # KCWI Re-aperture dispersion
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
    print(f"modelable ellipticals: {len(df)} ({n_full} full Dinos + {n_sig} KCWI-only)")
    print(f"complete (thetaE/gamma/n/q/PA all): {', '.join(df[df['data_complete']=='full']['name'])}")
    print(f"missing thetaE/gamma (sigma_only): {', '.join(df[df['data_complete']=='sigma_only']['name'])}")
    print(f"Saved: {out_path}")
    return df


if __name__ == "__main__":
    enrich_elliptical()
