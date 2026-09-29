#!/usr/bin/env python3
"""
Phase C — 盘星系识别 Phase C — disk galaxy classification
=====================
读取 Phase B 的 master.csv，对每个透镜系统分配 Tier 0/1/2/3/X。
输出: 03_identified/master_classified.csv

Tier 分类逻辑:
    Tier 0 — 已知盘星系（论文确认 + B/D 分解存在）
    Tier 1 — EXP/disky 形态 + Grade A/B + 有 spec-z
    Tier 2 — SER 形态 / 疑似盘 + 有 redshift
    Tier 3 — galaxy 类型但无形态信息
    Tier X — 椭圆/星系团/QSO（排除）
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
# Tier 分类器
# ================================================================

def classify_tier(row: pd.Series) -> tuple:
    """
    对单行透镜系统分类 Tier。

    Returns:
        (tier: int|str, basis: str) — tier 为 0/1/2/3/'X'，basis 为分类依据简述
    """
    name = str(row.get("name", ""))
    lens_type = str(row.get("lens_type_label", "")).strip().upper()
    grade = str(row.get("grade", "")).strip().upper()
    spec_class = str(row.get("spec_class", "")).strip().upper()
    n_sersic = safe_float(row.get("n_sersic"))
    b_d_ratio = safe_float(row.get("B_D_ratio"))
    z_lens = safe_float(row.get("z_lens"))

    # ── Tier X: 明确排除 ──
    if lens_type in TIER_X_LENS_TYPES:
        return "X", f"excluded_by_type:{lens_type}"
    if spec_class in TIER_X_SPEC_CLASSES:
        return "X", f"excluded_by_spec:{spec_class}"
    if lens_type == "DEV":
        return "X", "deVaucouleurs_profile"

    # ── Tier 0: 确认盘星系 ──
    if _is_tier0(name, row):
        if not np.isnan(b_d_ratio):
            return 0, f"known_disk_lens_B/D={b_d_ratio:.2f}"
        return 0, "known_disk_lens"

    # ── Tier 1: 高置信候选 ──
    if lens_type in ("EXP", "DISKY", "EXPONENTIAL"):
        if grade in ("A", "B") and not np.isnan(z_lens):
            return 1, f"EXP_morphology_grade={grade}"
        return 1, "EXP_morphology"

    # ── Tier 2: 可能盘星系 ──
    if lens_type in ("SER", "SERSIC"):
        if not np.isnan(z_lens):
            # Sérsic n 辅助判断
            if not np.isnan(n_sersic):
                if n_sersic < 2.0:
                    return 2, f"SER_n={n_sersic:.1f}_disk-like"
                return 3, f"SER_n={n_sersic:.1f}_bulge-like"
            return 2, "SER_morphology"
        return 3, "SER_no_redshift"

    # ── galaxy 透镜色辅助 ──
    if "GALAXY" in lens_type:
        g_r = safe_float(row.get("g_r_color"))
        if not np.isnan(g_r) and g_r < 0.8:
            return 2, f"galaxy_blue_g-r={g_r:.2f}"
        return 3, "galaxy_no_morphology"

    # ── 默认 ──
    if grade in ("A", "B"):
        return 3, f"generic_grade_{grade}"
    return "X", "low_confidence"


def _is_tier0(name: str, row: pd.Series) -> bool:
    """
    判断是否为 Tier 0（确认盘星系）。

    必须满足：(1) 名称匹配已知盘星系列表；**且**
            (2) 有 B/D 分解数据或速度弥散度（确认非空匹配）。

    单独名称匹配不足以确认——同一系统在不同星表中可能有
    不同质量的数据，只有实际拥有丰富参数的才是 Tier 0。
    """
    name_norm = normalize_name(name)

    # 首先：是否匹配已知盘星系名称
    name_is_known = False
    for t0_name in TIER0_NAMES:
        t0_norm = normalize_name(t0_name)
        if t0_norm in name_norm or name_norm in t0_norm:
            name_is_known = True
            break

    if not name_is_known:
        return False

    # 其次：是否有建模必要参数
    b_d = safe_float(row.get("B_D_ratio"))
    sigma = safe_float(row.get("sigma_km_s"))
    logM = safe_float(row.get("logMstar_total"))

    has_rich_data = (
        (not np.isnan(b_d) and not np.isnan(sigma)) or
        (not np.isnan(logM) and not np.isnan(sigma))
    )

    return has_rich_data


# ================================================================
# 重分类（Phase C↔D 反馈循环）
# ================================================================

def reclassify(df: pd.DataFrame) -> pd.DataFrame:
    """
    对已分类星表重新运行分类器。
    用于 Phase D 丰富后——新参数可能改变 Tier。

    Returns:
        pd.DataFrame: 含更新后的 disk_tier 和 tier_basis
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
# 主流程
# ================================================================

def classify_master(input_path: str = MASTER_PATH,
                    output_path: str = MASTER_CLASSIFIED_PATH,
                    verbose: bool = True) -> pd.DataFrame:
    """
    读取 master.csv → 分类 Tier → 排序 → 输出 classified。

    Args:
        input_path: 输入 master.csv 路径
        output_path: 输出路径
        verbose: 打印统计

    Returns:
        pd.DataFrame: 分类后的星表
    """
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found. Run build.py first.")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    master = pd.read_csv(input_path)
    print(f"Loaded: {len(master)} systems from {input_path}")

    # ── 分类 ──
    tiers = []
    bases = []
    for idx, row in master.iterrows():
        t, b = classify_tier(row)
        tiers.append(t)
        bases.append(b)

    master["disk_tier"] = pd.Series(tiers, index=master.index)
    master["tier_basis"] = pd.Series(bases, index=master.index)

    # ── 排序 ──
    master = sort_catalog(master)

    # ── 统计 ──
    if verbose:
        print(f"\n{'='*50}")
        print(f"Tier Distribution:")
        for t in [0, 1, 2, 3, "X"]:
            n = len(master[master["disk_tier"] == t])
            print(f"  Tier {t}: {n:>6d}")

        # 建模就绪
        ready = master[master["disk_tier"].isin([0, 1])]
        print(f"\n  Modeling-ready (Tier 0+1): {len(ready)}")
        print(f"  Tier 0 confirmed disk lenses: {len(master[master['disk_tier']==0])}")

    # ── 保存 ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    return master


def sort_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 disk_tier ASC, grade_rank ASC, sigma_km_s DESC 排序。
    """
    # 创建 grade 数值列
    df["_grade_rank"] = df["grade"].map(GRADE_ORDER).fillna(99)

    # 创建 tier 数值列（"X" → 99）
    df["_tier_rank"] = df["disk_tier"].apply(
        lambda x: 99 if str(x) == "X" else int(x)
    )

    df = df.sort_values(
        ["_tier_rank", "_grade_rank", "sigma_km_s", "theta_E_arcsec"],
        ascending=[True, True, False, False],
        na_position="last"
    ).reset_index(drop=True)

    # 清理辅助列
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
