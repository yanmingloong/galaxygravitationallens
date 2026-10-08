#!/usr/bin/env python3
"""
Phase E — modeling export
====================
Read the latest enriched catalog from 04_enriched/, filter by Tier, and export for use with PyAutoLens/AGAMA.

Export subsets:
- modeling_input_tier0.csv  — Tier 0 confirmed disk galaxies (ready for modeling)
- modeling_input_tier01.csv — Tier 0+1 (high-confidence candidates)
- modeling_input_sample.csv — Tier 0+1+2 (for sample statistics)
- modeling_input_elliptical.csv — elliptical lens comparison group
"""

import os
import sys
import numpy as np
import pandas as pd


from .config import (
    SCHEMA_COLUMNS, MODELING_CORE_COLS, DIR_ENRICHED, DIR_EXPORTS,
)
from .enrich import current_enriched_path


# ================================================================
# Export functions
# ================================================================

def export_modeling_subsets(input_path: str = None,
                            output_dir: str = DIR_EXPORTS) -> dict:
    """
    Export modeling subsets.

    Args:
        input_path: enriched CSV path (default: latest version)
        output_dir: output directory

    Returns:
        dict: {subset name: (path, row count)}
    """
    if input_path is None:
        input_path = current_enriched_path()
        # fallback to latest symlink
        latest = os.path.join(DIR_ENRICHED, "tier012_latest.csv")
        if os.path.exists(latest):
            input_path = latest

    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found. Run enrich.py first.")
        return {}

    df = pd.read_csv(input_path)
    print(f"Loaded: {len(df)} systems from {os.path.basename(input_path)}")

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # ── Tier 0 only ──
    tier0 = df[df["disk_tier"] == 0].copy()
    if len(tier0) > 0:
        path = os.path.join(output_dir, "modeling_input_tier0.csv")
        tier0 = _select_export_cols(tier0)
        tier0.to_csv(path, index=False)
        results["tier0"] = (path, len(tier0))
        print(f"\n  Tier 0 (confirmed disk): {len(tier0):>4d} → modeling_input_tier0.csv")
        if verbose_cols := [c for c in ["sigma_km_s", "theta_E_arcsec", "logMstar_total"]
                            if c in tier0.columns]:
            for col in verbose_cols:
                n = tier0[col].notna().sum()
                print(f"    {col}: {n}/{len(tier0)} have data")

    # ── Tier 0+1 ──
    tier01 = df[df["disk_tier"].isin([0, 1])].copy()
    if len(tier01) > 0:
        path = os.path.join(output_dir, "modeling_input_tier01.csv")
        tier01 = _select_export_cols(tier01)
        tier01.to_csv(path, index=False)
        results["tier01"] = (path, len(tier01))
        print(f"\n  Tier 0+1 (high-confidence): {len(tier01):>4d} → modeling_input_tier01.csv")

    # ── Tier 0+1+2 ──
    tier012 = df[df["disk_tier"].isin([0, 1, 2])].copy()
    if len(tier012) > 0:
        path = os.path.join(output_dir, "modeling_input_sample.csv")
        tier012 = _select_export_cols(tier012)
        tier012.to_csv(path, index=False)
        results["sample"] = (path, len(tier012))
        print(f"\n  Tier 0+1+2 (full sample):  {len(tier012):>4d} → modeling_input_sample.csv")

    # ── Elliptical comparison group ──
    ellipticals = df[df["lens_type_label"].str.upper().isin(
        ["DEV", "ELLIPTICAL", "DEVAUCOULEURS", "ELL"]
    )].copy()
    if len(ellipticals) > 0:
        path = os.path.join(output_dir, "modeling_input_elliptical.csv")
        ellipticals = _select_export_cols(ellipticals)
        ellipticals.to_csv(path, index=False)
        results["elliptical"] = (path, len(ellipticals))
        print(f"\n  Elliptical comparison:     {len(ellipticals):>4d} → modeling_input_elliptical.csv")

    # ── Full catalog core field export ──
    full = _select_export_cols(df, core_only=True)
    path = os.path.join(output_dir, "modeling_input_full.csv")
    full.to_csv(path, index=False)
    results["full"] = (path, len(full))
    print(f"\n  Full catalog (core cols): {len(full):>4d} → modeling_input_full.csv")

    return results


def _select_export_cols(df: pd.DataFrame, core_only: bool = False) -> pd.DataFrame:
    """Select export columns: keep only meaningful schema columns (drop all-empty columns)."""
    cols = MODELING_CORE_COLS if core_only else SCHEMA_COLUMNS
    available = [c for c in cols if c in df.columns]
    result = df[available].copy()
    # Drop all-empty columns
    result = result.dropna(axis=1, how="all")
    return result


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase E: Export modeling subsets")
    parser.add_argument("--input", type=str, default=None,
                       help="Enriched CSV path (default: latest)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: 05_exports/)")
    args = parser.parse_args()

    print("Phase E — Modeling Export")
    print("=" * 60)

    results = export_modeling_subsets(
        input_path=args.input,
        output_dir=args.output_dir or DIR_EXPORTS,
    )

    print(f"\n{'='*50}")
    print("Export complete.")
    for name, (path, count) in results.items():
        print(f"  {name:15s}: {count:>5d} systems → {os.path.basename(path)}")
