#!/usr/bin/env python3
"""
Phase A — JWST COWLS lens candidate catalog
======================================
COSMOS-Web Lens Survey (COWLS): 100+ strong lens candidates discovered in the deepest JWST NIRCam imaging.

References:
- COWLS I (Mercier+2025, MNRAS 543, 203): found >100 candidates
- COWLS II (Mahler+2025, MNRAS 544, L8): detailed analysis of 17 spectacular lenses
- COWLS III (Hogg+2025): predicted ~100 lenses
- GitHub: https://github.com/Jammy2211/COWLS_COSMOS_Web_Lens_Survey

Data:
- catalogue.csv: 439 candidates (with visual scores, redshift, θE, magnitudes, magnification)
- Each candidate has a separate folder (JWST FITS + PyAutoLens modeling results)

Usage:
    python -m galaxylens.sample_library.download.cowls              # statistics on existing data
    python -m galaxylens.sample_library.download.cowls --fetch      # download catalogue.csv from GitHub
    python -m galaxylens.sample_library.download.cowls --force      # overwrite existing
"""

import os
import sys
import argparse

from ..config import RAW_DIR

COWLS_DIR = os.path.join(RAW_DIR, "cowls")
COWLS_PATH = os.path.join(COWLS_DIR, "cowls_catalogue.csv")

# GitHub raw URL for catalogue.csv
COWLS_CATALOGUE_URL = (
    "https://raw.githubusercontent.com/Jammy2211/"
    "COWLS_COSMOS_Web_Lens_Survey/main/catalogue.csv"
)
COWLS_REPO_URL = "https://github.com/Jammy2211/COWLS_COSMOS_Web_Lens_Survey"
COWLS_REF = "Mercier+2025, MNRAS 543, 203; Mahler+2025, MNRAS 544, L8"


def download_cowls(force=False):
    """Download COWLS catalogue.csv from GitHub"""
    import requests

    if os.path.exists(COWLS_PATH) and not force:
        print(f"  Already exists: {COWLS_PATH}")
        print(f"  Use --force to re-download")
        return True

    os.makedirs(COWLS_DIR, exist_ok=True)

    print(f"  Downloading from GitHub...")
    print(f"  URL: {COWLS_CATALOGUE_URL}")

    try:
        resp = requests.get(COWLS_CATALOGUE_URL, timeout=60)
        if resp.status_code == 200:
            with open(COWLS_PATH, "w", encoding="utf-8") as f:
                f.write(resp.text)
            size_kb = os.path.getsize(COWLS_PATH) / 1024
            n_lines = resp.text.count("\n")
            print(f"  ✓ Saved: {COWLS_PATH} ({size_kb:.0f} KB, ~{n_lines} lines)")
            return True
        elif resp.status_code == 404:
            print(f"  ✗ File not found on GitHub (repo may have moved)")
            print(f"  → Check: {COWLS_REPO_URL}")
            return False
        else:
            print(f"  ✗ HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def stats_cowls():
    """Statistics for the COWLS catalog"""
    import pandas as pd

    if not os.path.exists(COWLS_PATH):
        print(f"  ✗ Catalog not found: {COWLS_PATH}")
        print(f"  Run with --fetch to download")
        return False

    df = pd.read_csv(COWLS_PATH, low_memory=False)
    print(f"\n  📊 COWLS JWST Lens Candidates")
    print(f"  {'─'*40}")
    print(f"  File:        {os.path.basename(COWLS_PATH)}")
    print(f"  Rows:        {len(df)}")
    print(f"  Columns:     {len(df.columns)}")

    # Score distribution (S00-S12)
    if "score" in df.columns:
        print(f"\n  Score distribution (S00=lowest → S12=highest):")
        for s in sorted(df["score"].dropna().unique()):
            n = len(df[df["score"] == s])
            star = " ★" if s in ("S10", "S11", "S12") else ""
            print(f"    {s}: {n:>4d}{star}")

    # Redshift
    if "lens_cw_photo_z_med" in df.columns:
        z = df["lens_cw_photo_z_med"].dropna()
        if len(z) > 0:
            print(f"\n  Photometric redshift (COSMOS-Web):")
            print(f"    Range:  {z.min():.2f} – {z.max():.2f}")
            print(f"    Median: {z.median():.2f}")
            print(f"    Count:  {len(z)}/{len(df)}")

    if "lens_spec_z" in df.columns:
        n_spec = df["lens_spec_z"].notna().sum()
        print(f"    With spec-z: {n_spec}")

    # Einstein radius
    if "einstein_radius" in df.columns:
        te = df["einstein_radius"].dropna()
        if len(te) > 0:
            print(f"\n  Einstein radius (arcsec):")
            print(f"    Range:  {te.min():.2f} – {te.max():.2f}")
            print(f"    Median: {te.median():.2f}")
            print(f"    Count:  {len(te)}/{len(df)}")

    # Stellar mass
    if "lens_cw_stmass_med" in df.columns:
        sm = df["lens_cw_stmass_med"].dropna()
        if len(sm) > 0:
            print(f"\n  Stellar mass log(M*/M⊙):")
            print(f"    Range:  {sm.min():.1f} – {sm.max():.1f}")
            print(f"    Median: {sm.median():.1f}")

    # Coordinates
    if "ra" in df.columns and "dec" in df.columns:
        has_coord = df["ra"].notna().sum()
        print(f"\n  Coordinates: {has_coord}/{len(df)}")

    print(f"\n  📎 Full data + FITS images: {COWLS_REPO_URL}")
    print(f"  PyAutoLens models included for all candidates")

    return True


def main():
    parser = argparse.ArgumentParser(description="Phase A: JWST COWLS lens catalog")
    parser.add_argument("--fetch", action="store_true", help="Download from GitHub")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    print("Phase A — JWST COWLS Lens Candidates")
    print("=" * 50)
    print(f"  Reference: {COWLS_REF}")
    print(f"  Repo:      {COWLS_REPO_URL}")

    if args.fetch or args.force:
        if download_cowls(force=args.force):
            stats_cowls()
        else:
            print(f"\n  💡 Manual download:")
            print(f"     Visit: {COWLS_REPO_URL}")
            print(f"     Download catalogue.csv → {COWLS_PATH}")
    else:
        if os.path.exists(COWLS_PATH):
            stats_cowls()
        else:
            print(f"\n  No catalog found.")
            print(f"  Run with --fetch to download from GitHub")


if __name__ == "__main__":
    main()
