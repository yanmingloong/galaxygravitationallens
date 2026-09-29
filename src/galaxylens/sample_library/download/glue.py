#!/usr/bin/env python3
"""
Phase A — GLUE 候选体星表获取与统计 Phase A — GLUE candidate catalog fetch & statistics
======================================
GLUE (Gravitational Lensing from UNIONS and Euclid) 发布在 Storfer+2025。
本脚本验证 GLUE 数据的存在性、输出统计、帮助获取最新数据。

用法:
    python -m galaxylens.sample_library.download.glue              # 统计已有数据
    python -m galaxylens.sample_library.download.glue --validate   # 验证文件完整性
    python -m galaxylens.sample_library.download.glue --input custom.csv  # 统计指定文件

注意: GLUE 数据来自合作组/论文，无公开自动下载 URL。
      如果文件缺失，需手动从论文补充材料获取。
"""

import os
import sys
import argparse

from ..config import RAW_DIR, RAW_SOURCES


GLUE_DIR = os.path.join(RAW_DIR, "glue")
GLUE_PATH = RAW_SOURCES["GLUE"]["files"]["candidates"]

# GLUE 论文信息
GLUE_REF = "Storfer et al. 2025, arXiv:XXXX.XXXXX"
GLUE_DOI = "https://ui.adsabs.harvard.edu/abs/2025arXiv...Storfer"  # 待确认
GLUE_EXPECTED_COLS = [
    "sys_name", "ra", "dec", "grade", "type",
    "mag_g", "mag_r", "mag_i", "mag_z",
    "speczd_sdss", "speczd_dr1", "photzd", "speczs",
]


def stats_glue(input_path=None):
    """统计 GLUE 候选体星表"""
    import pandas as pd

    path = input_path or GLUE_PATH
    if not os.path.exists(path):
        print(f"  ✗ GLUE catalog not found: {path}")
        print(f"\n  📎 GLUE data reference: {GLUE_REF}")
        print(f"     URL: {GLUE_DOI}")
        print(f"\n  → Please obtain the CSV from the paper's supplementary material")
        print(f"  → Save to: {GLUE_PATH}")
        return False

    df = pd.read_csv(path, low_memory=False)
    print(f"\n  📊 GLUE Candidate Statistics")
    print(f"  {'─'*40}")
    print(f"  File:        {os.path.basename(path)}")
    print(f"  Size:        {os.path.getsize(path)/1024:.0f} KB")
    print(f"  Total rows:  {len(df)}")
    print(f"  Columns:     {len(df.columns)}")

    # Grade distribution
    if "grade" in df.columns:
        print(f"\n  Grade distribution:")
        for g, n in df["grade"].value_counts().items():
            bar = "█" * min(int(30 * n / len(df)), 30)
            print(f"    {str(g):6s}: {n:>5d}  {bar}")

    # Type distribution
    if "type" in df.columns:
        print(f"\n  Type distribution:")
        for t, n in df["type"].value_counts().items():
            print(f"    {str(t):15s}: {n:>5d}")

    # Redshift availability
    print(f"\n  Redshift availability:")
    for col, label in [
        ("speczd_sdss", "SDSS spec-z"),
        ("speczd_dr1", "DESI spec-z"),
        ("photzd", "photo-z"),
        ("speczs", "source spec-z"),
    ]:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"    {label:15s}: {n:>5d} ({100*n/len(df):.1f}%)")

    # Magnitude coverage
    print(f"\n  Photometry coverage:")
    for col in ["mag_g", "mag_r", "mag_i", "mag_z"]:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"    {col:15s}: {n:>5d} ({100*n/len(df):.1f}%)")

    print(f"\n  ✓ Statistics complete")
    return True


def validate_glue(input_path=None):
    """验证 GLUE 数据完整性"""
    import pandas as pd

    path = input_path or GLUE_PATH
    if not os.path.exists(path):
        print(f"  ✗ Not found: {path}")
        return False

    df = pd.read_csv(path, low_memory=False)
    print(f"\n  🔍 GLUE Data Validation")
    print(f"  {'─'*40}")

    issues = []

    # Check required columns
    missing_cols = [c for c in GLUE_EXPECTED_COLS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    else:
        print(f"  ✓ All {len(GLUE_EXPECTED_COLS)} expected columns present")

    # Check for null coordinates
    if "ra" in df.columns and "dec" in df.columns:
        null_coord = df["ra"].isna().sum() + df["dec"].isna().sum()
        if null_coord > 0:
            issues.append(f"{null_coord} rows missing coordinates")
        else:
            print(f"  ✓ No missing coordinates")

    # Check grade values
    if "grade" in df.columns:
        valid_grades = {"A", "B", "C", "X", "a", "b", "c", "x"}
        grades = set(df["grade"].dropna().unique())
        unknown = grades - valid_grades
        if unknown:
            issues.append(f"Unknown grade values: {unknown}")
        else:
            print(f"  ✓ All grades valid: {sorted(grades)}")

    # Summary
    if issues:
        print(f"\n  ⚠ {len(issues)} issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  ✓ All validations passed")

    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Phase A: GLUE catalog statistics and validation")
    parser.add_argument("--validate", action="store_true", help="Validate data integrity")
    parser.add_argument("--input", type=str, help="Use a custom CSV file")
    args = parser.parse_args()

    print("Phase A — GLUE Candidate Catalog")
    print("=" * 50)
    print(f"  Reference: {GLUE_REF}")

    input_path = args.input or GLUE_PATH

    if args.validate:
        ok = validate_glue(input_path)
    else:
        ok = stats_glue(input_path)

    if not ok:
        print(f"\n  💡 Tip: Run with --validate to check data integrity")


if __name__ == "__main__":
    main()
