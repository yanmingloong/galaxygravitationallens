#!/usr/bin/env python3
"""
One-shot download of the Simard+2011 full catalog, cached locally (v2 — fixed column names).

Simard+2011 (J/ApJS/196/11/table3): ~1.1M SDSS galaxies
Actual VizieR column names (not as assumed in the documentation):
  _RA, _DE     — coordinates (J2000 deg)
  ng, e_ng     — Sersic index
  e, e_e       — ellipticity e = 1 - b/a
  phi, e_phi   — position angle (deg)
  Rhlr         — r-band half-light radius (kpc)
  Rchl_r       — r-band circular half-light radius (kpc)
  Scale        — kpc/arcsec conversion factor
  objID        — SDSS object ID

Mapping to schema:
  n_sersic      ← ng
  n_sersic_err  ← e_ng
  q_disk        ← 1 - e   (axis ratio b/a)
  PA_disk_deg   ← phi
  Re_arcsec     ← Rhlr / Scale  (kpc → arcsec)

First download ~18 min, ~50MB CSV. enrich.py uses the cache automatically afterwards.
"""

import os, sys, time
import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from astroquery.vizier import Vizier

CACHE_DIR = os.path.join(PROJECT_ROOT, "01_raw", "simard2011")
CACHE_PATH = os.path.join(CACHE_DIR, "simard2011_cache.csv")
VIZIER_ID = "J/ApJS/196/11/table3"

os.makedirs(CACHE_DIR, exist_ok=True)

def main():
    """Download the Simard+2011 full catalog and cache it (one-shot, ~18 min, 50MB)."""
    print(f"Downloading Simard+2011 ({VIZIER_ID})...")
    print(f"  ~1.1M galaxies, estimated 15-20 minutes")
    t0 = time.time()

    Vizier.ROW_LIMIT = -1
    Vizier.TIMEOUT = 1200  # 20 minutes

    try:
        catalog_list = Vizier.get_catalogs(VIZIER_ID)
        if not catalog_list or len(catalog_list) == 0:
            print("ERROR: VizieR returned no data")
            sys.exit(1)

        table = catalog_list[0]
        n_total = len(table)
        print(f"  Downloaded {n_total} rows in {time.time()-t0:.0f}s")
        print(f"  Actual columns: {list(table.columns)}")

        # Target columns (VizieR name → cache name)
        COLUMN_MAP = {
            "_RA":      "RA_deg",
            "_DE":      "DE_deg",
            "objID":    "objID",
            "z":        "z_sdss",
            "ng":       "n_sersic",
            "e_ng":     "n_sersic_err",
            "e":        "ellipticity",
            "e_e":      "ellipticity_err",
            "phi":      "PA_deg",
            "e_phi":    "PA_err",
            "Rhlr":     "Re_kpc",
            "Rchl_r":   "Re_circ_kpc",
            "Scale":    "scale_kpc_per_arcsec",
            "gg2d":     "mag_g",
            "rg2d":     "mag_r",
            "ggMag":    "M_g_abs",
            "rgMag":    "M_r_abs",
        }

        # Check which columns actually exist
        available = {k: v for k, v in COLUMN_MAP.items() if k in table.colnames}
        missing = [k for k in COLUMN_MAP if k not in table.colnames]
        print(f"  Found {len(available)}/{len(COLUMN_MAP)} columns")
        if missing:
            print(f"  Missing: {missing}")

        # Convert row by row
        rows = []
        batch_report = 100_000
        for i, row in enumerate(table):
            if (i + 1) % batch_report == 0:
                elapsed = time.time() - t0
                print(f"    Converting: {i+1}/{n_total} ({elapsed:.0f}s, {len(rows)} valid so far)...")

            d = {}
            for viz_col, cache_col in available.items():
                try:
                    val = row[viz_col]
                    if val is not None and not np.ma.is_masked(val):
                        d[cache_col] = float(val)
                except (KeyError, ValueError, TypeError):
                    pass

            # Keep only rows with coordinates
            if "RA_deg" in d and "DE_deg" in d:
                rows.append(d)

        df = pd.DataFrame(rows)
        print(f"  Converted {len(df)} valid galaxies (dropped {n_total - len(df)} without coordinates)")

        # Derived fields: q_disk = 1 - e, Re_arcsec = Re_kpc / scale
        if "ellipticity" in df.columns:
            df["q_disk"] = 1.0 - df["ellipticity"]
        if "Re_kpc" in df.columns and "scale_kpc_per_arcsec" in df.columns:
            df["Re_arcsec"] = df["Re_kpc"] / df["scale_kpc_per_arcsec"]

        # Ensure numeric types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.to_csv(CACHE_PATH, index=False)
        size_mb = os.path.getsize(CACHE_PATH) / 1e6
        print(f"\n  Saved to {CACHE_PATH} ({size_mb:.1f} MB)")
        print(f"  Final columns: {list(df.columns)}")
        print(f"  Galaxies: {len(df)}")

        # Statistics
        print(f"\n  Quick stats:")
        for col in ["n_sersic", "q_disk", "Re_arcsec", "PA_deg"]:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    print(f"    {col:15s}: {vals.min():.3f} - {vals.max():.3f} "
                          f"(median {vals.median():.3f}, {len(vals)}/{len(df)} non-null)")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print("Now re-run enrich.py --query-network to use this cache.")


if __name__ == "__main__":
    main()

