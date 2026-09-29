#!/usr/bin/env python3
"""
Phase A — lenscat.org 星表获取与统计
======================================
lenscat.org 是 30+ 巡天的聚合星表，提供 CSV 导出。
本脚本支持:
- 自动从 lenscat.org 下载最新 CSV（需要网络）
- 统计类型/分级/红移分布
- 保存 lenscat_catalog.csv 到 01_raw/lenscat/

用法:
    python -m galaxylens.sample_library.download.lenscat              # 统计已有数据
    python -m galaxylens.sample_library.download.lenscat --fetch      # 从 lenscat.org 下载最新 CSV
    python -m galaxylens.sample_library.download.lenscat --input custom.csv  # 统计指定文件
"""

import os
import sys
import argparse

from ..config import RAW_DIR, RAW_SOURCES


LENSCAT_DIR = os.path.join(RAW_DIR, "lenscat")
LENSCAT_PATH = RAW_SOURCES["lenscat"]["files"]["catalog"]
LENSCAT_URL = "https://lenscat.org/catalog.csv"  # 标准导出 URL


def download_lenscat(force=False):
    """从 lenscat.org 下载最新星表"""
    import requests
    import shutil

    if os.path.exists(LENSCAT_PATH) and not force:
        print(f"  Lenscat catalog already exists: {LENSCAT_PATH}")
        print(f"  Use --force to re-download")
        return True

    os.makedirs(LENSCAT_DIR, exist_ok=True)

    print(f"  Downloading from {LENSCAT_URL} ...")
    try:
        resp = requests.get(LENSCAT_URL, timeout=120, stream=True)
        if resp.status_code == 200:
            with open(LENSCAT_PATH, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
            size_kb = os.path.getsize(LENSCAT_PATH) / 1024
            print(f"  ✓ Saved: {LENSCAT_PATH} ({size_kb:.0f} KB)")
            return True
        else:
            print(f"  ✗ HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print(f"  → Please download manually from {LENSCAT_URL}")
        print(f"  → Save to: {LENSCAT_PATH}")
        return False


def stats_lenscat(input_path=None):
    """统计 lenscat 星表"""
    import pandas as pd

    path = input_path or LENSCAT_PATH
    if not os.path.exists(path):
        print(f"  ✗ Catalog not found: {path}")
        print(f"  Run with --fetch to download, or provide --input")
        return False

    df = pd.read_csv(path, low_memory=False)
    print(f"\n  📊 lenscat.org Catalog Statistics")
    print(f"  {'─'*40}")
    print(f"  File:        {os.path.basename(path)}")
    print(f"  Size:        {os.path.getsize(path)/1024:.0f} KB")
    print(f"  Total rows:  {len(df)}")
    print(f"  Columns:     {len(df.columns)}")

    # Type distribution
    if "type" in df.columns:
        print(f"\n  Type distribution:")
        for t, n in df["type"].value_counts().items():
            print(f"    {str(t):15s}: {n:>6d}")

    # Grading distribution
    if "grading" in df.columns:
        print(f"\n  Grading distribution:")
        for g, n in df["grading"].value_counts().items():
            print(f"    {str(g):15s}: {n:>6d}")

    # Redshift coverage
    if "zlens" in df.columns:
        has_z = df["zlens"].notna() & (df["zlens"] != "-") & (df["zlens"] != "")
        print(f"\n  Redshift coverage:")
        print(f"    With lens z:  {has_z.sum():>6d} ({100*has_z.sum()/len(df):.1f}%)")

    # Galaxy-type (most relevant for disk lenses)
    if "type" in df.columns:
        galaxy = df[df["type"] == "galaxy"]
        print(f"\n  Galaxy lenses (relevant for disk search):")
        print(f"    Total:        {len(galaxy)}")
        if "grading" in df.columns:
            for g in ["confident", "probable", "candidate"]:
                n = len(galaxy[galaxy["grading"] == g])
                print(f"    {g:15s}: {n}")

    print(f"\n  ✓ Statistics complete")
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase A: Fetch and analyze lenscat.org catalog")
    parser.add_argument("--fetch", action="store_true", help="Download latest catalog from lenscat.org")
    parser.add_argument("--force", action="store_true", help="Force re-download even if exists")
    parser.add_argument("--input", type=str, help="Use a custom CSV file")
    parser.add_argument("--stats-only", action="store_true", help="Only show statistics, no download")
    args = parser.parse_args()

    print("Phase A — lenscat.org Catalog")
    print("=" * 50)

    if args.input:
        stats_lenscat(args.input)
    elif args.fetch:
        if download_lenscat(force=args.force):
            stats_lenscat()
    else:
        # Default: show stats on existing catalog
        if os.path.exists(LENSCAT_PATH):
            stats_lenscat()
        else:
            print(f"\n  No catalog found. Options:")
            print(f"    python -m galaxylens.sample_library.download.lenscat --fetch     # Download")
            print(f"    python -m galaxylens.sample_library.download.lenscat --input <file>  # Use existing")


if __name__ == "__main__":
    main()
