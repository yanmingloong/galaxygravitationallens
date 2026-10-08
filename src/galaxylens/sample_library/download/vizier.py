#!/usr/bin/env python3
"""
Phase A — download SLACS/CASTLES/CLASS catalogs from VizieR
===================================================
Use astroquery to pull the lens catalogs of three surveys from VizieR,
storing them in 01_raw/{slacs,castles,class}/.

Usage:
    python -m galaxylens.sample_library.download.vizier              # download all (skip existing)
    python -m galaxylens.sample_library.download.vizier --force      # force overwrite existing
    python -m galaxylens.sample_library.download.vizier --check      # only check, don't download
    python -m galaxylens.sample_library.download.vizier --source slacs  # download only a specified source
"""

import os
import sys
import argparse

from ..config import RAW_DIR


CATALOGS = {
    "slacs": {
        "vizier_id": "J/ApJ/638/703",
        "subdir": "slacs",
        "filename": "slacs_vizier.csv",
        "description": "SLACS: Sloan Lens ACS Survey (Bolton+2006)",
    },
    "castles": {
        "vizier_id": "J/ApJ/667/714",  # CASTLES lens database
        "subdir": "castles",
        "filename": "castles_vizier.csv",
        "description": "CASTLES: CfA-Arizona Space Telescope Lens Survey",
    },
    "class": {
        "vizier_id": "VIII/72",
        "subdir": "class",
        "filename": "class_vizier.csv",
        "description": "CLASS: Cosmic Lens All-Sky Survey (Myers+2003)",
    },
}


def download_catalog(key, cfg, force=False):
    """Download a single catalog to 01_raw/{subdir}/{filename}"""
    from astroquery.vizier import Vizier

    out_dir = os.path.join(RAW_DIR, cfg["subdir"])
    out_path = os.path.join(out_dir, cfg["filename"])

    if os.path.exists(out_path) and not force:
        import pandas as pd
        try:
            df = pd.read_csv(out_path, low_memory=False, nrows=5)
            print(f"  ✓ Already exists: {out_path} ({len(df.columns)} cols, sample ok)")
            return True
        except Exception:
            print(f"  ⚠ Existing file corrupted, re-downloading...")

    os.makedirs(out_dir, exist_ok=True)

    print(f"  Downloading {cfg['description']} ({cfg['vizier_id']})...")
    Vizier.ROW_LIMIT = -1
    Vizier.TIMEOUT = 120

    try:
        tables = Vizier.get_catalogs(cfg["vizier_id"])
        if not tables:
            print(f"  ✗ No tables returned")
            return False

        all_dfs = []
        for i, t in enumerate(tables):
            df = t.to_pandas()
            print(f"    Table {i}: {len(df)} rows, {len(df.columns)} cols")
            all_dfs.append(df)

        if len(all_dfs) == 1:
            result = all_dfs[0]
        else:
            import pandas as pd
            result = pd.concat(all_dfs, ignore_index=True)
            print(f"    Merged: {len(result)} total rows")

        result.to_csv(out_path, index=False)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  ✓ Saved: {out_path} ({size_kb:.0f} KB, {len(result)} rows)")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def check_catalog(key, cfg):
    """Check whether the catalog exists and is readable"""
    out_path = os.path.join(RAW_DIR, cfg["subdir"], cfg["filename"])
    if not os.path.exists(out_path):
        print(f"  ✗ MISSING: {out_path}")
        return False
    import pandas as pd
    try:
        df = pd.read_csv(out_path, low_memory=False)
        print(f"  ✓ {cfg['description']}: {len(df)} rows, {len(df.columns)} cols → {out_path}")
        print(f"    Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
        return True
    except Exception as e:
        print(f"  ✗ CORRUPTED: {out_path} ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Phase A: Download lens catalogs from VizieR")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--check", action="store_true", help="Only check, don't download")
    parser.add_argument("--source", type=str, choices=list(CATALOGS.keys()),
                       help="Download only one source (slacs/castles/class)")
    args = parser.parse_args()

    sources = [args.source] if args.source else list(CATALOGS.keys())

    print("Phase A — VizieR Catalog Download")
    print("=" * 50)

    ok = 0
    for key in sources:
        cfg = CATALOGS[key]
        print(f"\n[{key}] {cfg['description']}")
        if args.check:
            if check_catalog(key, cfg):
                ok += 1
        else:
            if download_catalog(key, cfg, force=args.force):
                ok += 1

    print(f"\n{'='*50}")
    print(f"Result: {ok}/{len(sources)} sources OK")


if __name__ == "__main__":
    main()
