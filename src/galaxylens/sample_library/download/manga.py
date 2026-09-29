#!/usr/bin/env python3
"""
Phase A — MaNGA SILO 透镜候选体星表 Phase A — MaNGA SILO lens candidate catalog
======================================
SDSS-IV MaNGA IFU 巡天的 Spectroscopic Identification of Lensing Objects (SILO)。
从 MaNGA ~10,000 星系中通过 IFU 光谱探测背景发射线发现强透镜候选体。

参考:
- Talbot+2018 (MNRAS 477): 首批 38 候选体，来自 2,812 MaNGA 星系
- Talbot+2022 (MNRAS 515, 4953): 扩展至 94 候选体 (8 likely + 17 probable + 69 possible)
- SDSS DR17 VAC: https://data.sdss.org/sas/dr17/manga/spectro/lensing/silo/

亮点:
- >50% 候选体 θE > 1.7"（比 SLACS 更大）
- IFU 二维运动学可用 (MaNGA DAP: σ, V/σ maps)
- 适合 lensing+dynamics 联合建模

用法:
    python -m galaxylens.sample_library.download.manga              # 统计已有数据
    python -m galaxylens.sample_library.download.manga --fetch      # 从 SDSS SAS 下载
    python -m galaxylens.sample_library.download.manga --force      # 覆盖已有
"""

import os
import sys
import argparse

from ..config import RAW_DIR

MANGA_DIR = os.path.join(RAW_DIR, "manga")
MANGA_PATH_CSV = os.path.join(MANGA_DIR, "manga_silo_candidates.csv")
MANGA_PATH_FITS = os.path.join(MANGA_DIR, "silo_manga_detections.fits")

# SDSS DR17 SILO VAC — 直接可下载的 FITS 文件
MANGA_FITS_URL = (
    "https://data.sdss.org/sas/dr17/manga/spectro/lensing/silo/"
    "v3_1_1/1.0.4/silo_manga_detections-1.0.4.fits"
)
MANGA_INDEX_URL = (
    "https://data.sdss.org/sas/dr17/manga/spectro/lensing/silo/"
    "v3_1_1/1.0.4/"
)

MANGA_REF = "Talbot et al. 2022, MNRAS 515, 4953"
MANGA_DOI = "https://doi.org/10.1093/mnras/stac2233"


def download_manga(force=False):
    """从 SDSS SAS 下载 MaNGA SILO FITS 文件并转为 CSV"""
    import requests

    target = MANGA_PATH_CSV if MANGA_PATH_CSV else MANGA_PATH_FITS
    if os.path.exists(MANGA_PATH_CSV) and not force:
        print(f"  Already exists: {MANGA_PATH_CSV}")
        return True

    os.makedirs(MANGA_DIR, exist_ok=True)

    print(f"  Downloading from SDSS SAS...")
    print(f"  URL: {MANGA_FITS_URL}")

    # Step 1: Download FITS
    try:
        resp = requests.get(MANGA_FITS_URL, timeout=300, stream=True)
        if resp.status_code != 200:
            print(f"  ✗ HTTP {resp.status_code}")
            print(f"  → Browse: {MANGA_INDEX_URL}")
            return False

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(MANGA_PATH_FITS, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (5 * 1024 * 1024) < 8192:
                    pct = 100 * downloaded / total
                    print(f"    ... {pct:.0f}% ({downloaded/1e6:.1f} MB)", flush=True)

        size_mb = os.path.getsize(MANGA_PATH_FITS) / 1e6
        print(f"  ✓ Downloaded: {size_mb:.1f} MB")

    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False

    # Step 2: Convert FITS → CSV
    print(f"  Converting FITS → CSV...")
    try:
        from astropy.table import Table
        table = Table.read(MANGA_PATH_FITS)
        df = table.to_pandas()
        df.to_csv(MANGA_PATH_CSV, index=False)
        print(f"  ✓ Saved CSV: {MANGA_PATH_CSV} ({len(df)} rows, {len(df.columns)} cols)")
        return True
    except ImportError:
        print(f"  ⚠ astropy not available, keeping FITS file only")
        print(f"  FITS saved to: {MANGA_PATH_FITS}")
        print(f"  Install astropy: pip install astropy")
        return True
    except Exception as e:
        print(f"  ✗ FITS conversion failed: {e}")
        print(f"  FITS file saved to: {MANGA_PATH_FITS}")
        return True


def stats_manga():
    """统计 MaNGA SILO 星表"""
    import pandas as pd

    path = MANGA_PATH_CSV
    if not os.path.exists(path):
        print(f"  ✗ CSV not found: {path}")
        if os.path.exists(MANGA_PATH_FITS):
            print(f"  FITS file exists: {MANGA_PATH_FITS}")
            print(f"  Run: pip install astropy && python -m galaxylens.sample_library.download.manga --force")
        else:
            print(f"  Run with --fetch to download")
        return False

    df = pd.read_csv(path, low_memory=False)
    print(f"\n  📊 MaNGA SILO Lens Candidates")
    print(f"  {'─'*40}")
    print(f"  File:        {os.path.basename(path)}")
    print(f"  Rows:        {len(df)}")
    print(f"  Columns:     {len(df.columns)}")

    # Auto-detect key columns
    for label, patterns in [
        ("RA", ["ra", "raj2000", "_ra", "RA"]),
        ("Dec", ["dec", "dej2000", "_de", "DEC"]),
        ("Redshift", ["z", "redshift", "z_lens", "Z"]),
        ("VelDisp", ["sigma", "veldisp", "vd", "VDISP"]),
        ("Grade", ["grade", "class", "flag", "likely", "quality"]),
        ("Theta_E", ["theta", "einstein", "re", "THETA_E"]),
    ]:
        col = next((c for c in df.columns if any(p.lower() in c.lower() for p in patterns)), None)
        if col:
            non_null = df[col].notna().sum()
            print(f"  {label:12s}: {non_null}/{len(df)} → column '{col}'")

    # Grade distribution if available
    grade_col = next((c for c in df.columns if "grade" in c.lower() or "flag" in c.lower() or "class" in c.lower()), None)
    if grade_col:
        print(f"\n  Grade distribution:")
        for g, n in df[grade_col].value_counts().head(10).items():
            print(f"    {str(g)[:25]:25s}: {n}")

    print(f"\n  📎 Reference: {MANGA_REF}")
    print(f"  MaNGA DAP:   https://www.sdss.org/dr17/manga/manga-data/")
    print(f"  FITS source: {MANGA_INDEX_URL}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Phase A: MaNGA SILO lens catalog")
    parser.add_argument("--fetch", action="store_true", help="Download from SDSS SAS")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    print("Phase A — MaNGA SILO Lens Candidates")
    print("=" * 50)
    print(f"  Reference: {MANGA_REF}")
    print(f"  Data:      {MANGA_INDEX_URL}")

    if args.fetch or args.force:
        if download_manga(force=args.force):
            stats_manga()
        else:
            print(f"\n  💡 Manual download — open in browser:")
            print(f"     {MANGA_INDEX_URL}")
            print(f"     Download: silo_manga_detections-1.0.4.fits")
            print(f"     Save to:  {MANGA_PATH_FITS}")
    else:
        if os.path.exists(MANGA_PATH_CSV) or os.path.exists(MANGA_PATH_FITS):
            stats_manga()
        else:
            print(f"\n  No catalog found.")
            print(f"  Run with --fetch to download (FITS → CSV auto-convert)")


if __name__ == "__main__":
    main()
