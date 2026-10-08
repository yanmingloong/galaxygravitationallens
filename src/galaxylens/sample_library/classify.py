#!/usr/bin/env python3
"""
Phase C — Disk Galaxy Classification
=====================
Read Phase B's master.csv and assign a Tier (0/1/2/3/X) to each lens system.
Output: 03_identified/master_classified.csv

Tier classification logic:
    Tier 0 — Known disk galaxy (paper-confirmed + B/D decomposition present)
    Tier 1 — EXP/disky morphology + Grade A/B + has spec-z
    Tier 2 — SER morphology / suspected disk + has redshift
    Tier 3 — galaxy type but no morphology information
    Tier X — elliptical/cluster/QSO (excluded)
"""

import os
import sys
import numpy as np
import pandas as pd


from .config import (
    SCHEMA_COLUMNS, MASTER_PATH, MASTER_CLASSIFIED_PATH,
    TIER0_NAMES, TIER_X_LENS_TYPES, TIER_X_SPEC_CLASSES, GRADE_ORDER,
)
from .utils import normalize_name, safe_float


# ================================================================
# Tier classifier
# ================================================================

def classify_tier(row: pd.Series) -> tuple:
    """
    Classify a single lens-system row into a Tier.

    Returns:
        (tier: int|str, basis: str) — tier is 0/1/2/3/'X', basis is a short classification rationale
    """
    name = str(row.get("name", ""))
    lens_type = str(row.get("lens_type_label", "")).strip().upper()
    grade = str(row.get("grade", "")).strip().upper()
    spec_class = str(row.get("spec_class", "")).strip().upper()
    n_sersic = safe_float(row.get("n_sersic"))
    b_d_ratio = safe_float(row.get("B_D_ratio"))
    z_lens = safe_float(row.get("z_lens"))

    # ── Tier X: explicitly excluded ──
    if lens_type in TIER_X_LENS_TYPES:
        return "X", f"excluded_by_type:{lens_type}"
    if spec_class in TIER_X_SPEC_CLASSES:
        return "X", f"excluded_by_spec:{spec_class}"
    if lens_type == "DEV":
        return "X", "deVaucouleurs_profile"

    # ── Tier 0: confirmed disk galaxy ──
    if _is_tier0(name, row):
        if not np.isnan(b_d_ratio):
            return 0, f"known_disk_lens_B/D={b_d_ratio:.2f}"
        return 0, "known_disk_lens"

    # ── Tier 1: high-confidence candidate ──
    if lens_type in ("EXP", "DISKY", "EXPONENTIAL"):
        if grade in ("A", "B") and not np.isnan(z_lens):
            return 1, f"EXP_morphology_grade={grade}"
        return 1, "EXP_morphology"

    # ── Tier 2: possible disk galaxy ──
    if lens_type in ("SER", "SERSIC"):
        if not np.isnan(z_lens):
            # Sérsic n auxiliary check
            if not np.isnan(n_sersic):
                if n_sersic < 2.0:
                    return 2, f"SER_n={n_sersic:.1f}_disk-like"
                return 3, f"SER_n={n_sersic:.1f}_bulge-like"
            return 2, "SER_morphology"
        return 3, "SER_no_redshift"

    # ── galaxy lens color auxiliary ──
    if "GALAXY" in lens_type:
        g_r = safe_float(row.get("g_r_color"))
        if not np.isnan(g_r) and g_r < 0.8:
            return 2, f"galaxy_blue_g-r={g_r:.2f}"
        return 3, "galaxy_no_morphology"

    # ── default ──
    if grade in ("A", "B"):
        return 3, f"generic_grade_{grade}"
    return "X", "low_confidence"


def _is_tier0(name: str, row: pd.Series) -> bool:
    """
    Determine whether this is a Tier 0 (confirmed disk galaxy).

    Must satisfy: (1) the name matches the known disk-galaxy list; **and**
            (2) there is B/D decomposition data or velocity dispersion
                (confirming a non-empty match).

    A name match alone is not enough to confirm — the same system may have
    data of differing quality across catalogs; only the ones that actually
    carry rich parameters qualify as Tier 0.
    """
    name_norm = normalize_name(name)

    # First: does the name match a known disk galaxy
    name_is_known = False
    for t0_name in TIER0_NAMES:
        t0_norm = normalize_name(t0_name)
        if t0_norm in name_norm or name_norm in t0_norm:
            name_is_known = True
            break

    if not name_is_known:
        return False

    # Second: are the parameters needed for modeling present
    b_d = safe_float(row.get("B_D_ratio"))
    sigma = safe_float(row.get("sigma_km_s"))
    logM = safe_float(row.get("logMstar_total"))

    has_rich_data = (
        (not np.isnan(b_d) and not np.isnan(sigma)) or
        (not np.isnan(logM) and not np.isnan(sigma))
    )

    return has_rich_data


# ================================================================
# Reclassification (Phase C↔D feedback loop)
# ================================================================

def reclassify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-run the classifier over the already-classified catalog.
    Used after Phase D enrichment — new parameters may change the Tier.

    Returns:
        pd.DataFrame: with updated disk_tier and tier_basis
    """
    old_tiers = df["disk_tier"].copy() if "disk_tier" in df.columns else pd.Series(["X"] * len(df))

    tiers = []
    bases = []
    changed = 0

    for idx, row in df.iterrows():
        tier, basis = classify_tier(row)
        tiers.append(tier)
        bases.append(basis)

        old = old_tiers.iloc[idx] if "disk_tier" in df.columns else None
        if str(old) != str(tier):
            changed += 1

    df["disk_tier"] = pd.Series(tiers, index=df.index)
    df["tier_basis"] = pd.Series(bases, index=df.index)

    if changed > 0:
        print(f"  Reclassify: {changed} systems changed tier")

    return df


# ================================================================
# Main flow
# ================================================================

def classify_master(input_path: str = MASTER_PATH,
                    output_path: str = MASTER_CLASSIFIED_PATH,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Read master.csv → classify Tier → sort → output classified.

    Args:
        input_path: input master.csv path
        output_path: output path
        verbose: print statistics

    Returns:
        pd.DataFrame: the classified catalog
    """
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found. Run build.py first.")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    master = pd.read_csv(input_path)
    print(f"Loaded: {len(master)} systems from {input_path}")

    # ── classify ──
    tiers = []
    bases = []
    for idx, row in master.iterrows():
        t, b = classify_tier(row)
        tiers.append(t)
        bases.append(b)

    master["disk_tier"] = pd.Series(tiers, index=master.index)
    master["tier_basis"] = pd.Series(bases, index=master.index)

    # ── sort ──
    master = sort_catalog(master)

    # ── statistics ──
    if verbose:
        print(f"\n{'='*50}")
        print(f"Tier Distribution:")
        for t in [0, 1, 2, 3, "X"]:
            n = len(master[master["disk_tier"] == t])
            print(f"  Tier {t}: {n:>6d}")

        # modeling-ready
        ready = master[master["disk_tier"].isin([0, 1])]
        print(f"\n  Modeling-ready (Tier 0+1): {len(ready)}")
        print(f"  Tier 0 confirmed disk lenses: {len(master[master['disk_tier']==0])}")

    # ── save ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    return master


def sort_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by disk_tier ASC, grade_rank ASC, sigma_km_s DESC.
    """
    # create numeric grade column
    df["_grade_rank"] = df["grade"].map(GRADE_ORDER).fillna(99)

    # create numeric tier column ("X" → 99)
    df["_tier_rank"] = df["disk_tier"].apply(
        lambda x: 99 if str(x) == "X" else int(x)
    )

    df = df.sort_values(
        ["_tier_rank", "_grade_rank", "sigma_km_s", "theta_E_arcsec"],
        ascending=[True, True, False, False],
        na_position="last"
    ).reset_index(drop=True)

    # clean up auxiliary columns
    df = df.drop(columns=["_grade_rank", "_tier_rank"], errors="ignore")

    return df


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase C: Classify disk lenses")
    parser.add_argument("--reclassify", action="store_true",
                       help="Re-run classification on existing classified catalog")
    parser.add_argument("--input", type=str, default=None, help="Input CSV path")
    args = parser.parse_args()

    print("Phase C — Disk Lens Classification")
    print("=" * 60)

    if args.reclassify:
        input_path = args.input or MASTER_CLASSIFIED_PATH
        if os.path.exists(input_path):
            df = pd.read_csv(input_path)
            df = reclassify(df)
            df = sort_catalog(df)
            df.to_csv(input_path, index=False)
            print(f"Re-classified and saved: {input_path}")
        else:
            print(f"ERROR: {input_path} not found")
    else:
        classify_master(input_path=args.input or MASTER_PATH)
