#!/usr/bin/env python3
"""
盘星系引力透镜样本库 — 全流程编排
====================================
一条命令跑通 Phase B→C→D→E。

用法:
    python pipeline.py              # 全流程（无网络查询）
    python pipeline.py --network    # 含 SDSS/VizieR 查询
    python pipeline.py --from phase_c  # 从指定 Phase 开始
    python pipeline.py --add sample.csv --source paper  # 新增样本
"""

import os
import sys
import argparse
import pandas as pd


from .config import (
    PROJECT_ROOT, MASTER_PATH, MASTER_CLASSIFIED_PATH,
    SCHEMA_COLUMNS,
)
from .build import build_master
from .classify import classify_master, reclassify, sort_catalog
from .enrich import enrich, get_latest_version, next_version
from .export import export_modeling_subsets


def run_full_pipeline(query_network: bool = False, start_from: str = "phase_b"):
    """运行完整 B→C→D→E 流程。"""

    print("╔" + "═" * 58 + "╗")
    print("║  盘星系引力透镜样本库 — 全流程构建                ║")
    print("╚" + "═" * 58 + "╝")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Network queries: {'ON' if query_network else 'OFF'}")
    print()

    # ── Phase B: 标准化去重 ──
    if start_from in ("phase_b",):
        print("█ Phase B: Build Master Catalog")
        print("─" * 60)
        master = build_master()
        if len(master) == 0:
            print("ERROR: No data loaded. Aborting.")
            return
    else:
        print("█ Phase B: Skipped (using existing master.csv)")
        if not os.path.exists(MASTER_PATH):
            print(f"ERROR: {MASTER_PATH} not found. Run with --from phase_b")
            return
        master = pd.read_csv(MASTER_PATH)

    # ── Phase C: 盘星系识别 ──
    if start_from in ("phase_b", "phase_c"):
        print(f"\n█ Phase C: Classify Disk Lenses")
        print("─" * 60)
        classified = classify_master()
    else:
        print("█ Phase C: Skipped (using existing classified)")
        if not os.path.exists(MASTER_CLASSIFIED_PATH):
            print(f"ERROR: {MASTER_CLASSIFIED_PATH} not found.")
            return
        classified = pd.read_csv(MASTER_CLASSIFIED_PATH)

    # ── Phase D: 参数丰富 ──
    if start_from in ("phase_b", "phase_c", "phase_d"):
        print(f"\n█ Phase D: Enrich Parameters")
        print("─" * 60)
        enriched = enrich(query_network=query_network)
    else:
        print("█ Phase D: Skipped")
        from .enrich import current_enriched_path
        epath = current_enriched_path()
        if not os.path.exists(epath):
            print(f"ERROR: No enriched catalog found.")
            return
        enriched = pd.read_csv(epath)

    # ── Phase E: 建模导出 ──
    if start_from in ("phase_b", "phase_c", "phase_d", "phase_e"):
        print(f"\n█ Phase E: Export Modeling Subsets")
        print("─" * 60)
        export_modeling_subsets()

    # ── Final summary ──
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Master catalog:       {len(master):>6,d} systems → 02_merged/")
    print(f"Classified:           {len(classified):>6,d} systems → 03_identified/")
    print(f"Enriched v{get_latest_version()}:         {len(enriched):>6,d} systems → 04_enriched/")
    print(f"Modeling exports:                       → 05_exports/")

    # Tier summary
    if "disk_tier" in enriched.columns:
        for t in [0, 1, 2, 3, "X"]:
            n = len(enriched[enriched["disk_tier"] == t])
            tier_label = {0: "Tier 0 (confirmed disk)", 1: "Tier 1 (high confidence)",
                         2: "Tier 2 (possible)", 3: "Tier 3 (pending)", "X": "Tier X (excluded)"}[t]
            print(f"  {tier_label:35s}: {n:>6,d}")


def add_new_sample(filepath: str, source: str = "manual"):
    """
    增量添加新样本。

    Args:
        filepath: 新样本 CSV（需至少含 name, ra_deg, dec_deg 列）
        source: 来源标签
    """
    print(f"Adding new sample: {filepath} (source: {source})")

    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found")
        return

    new = pd.read_csv(filepath)

    # 确保基本列存在
    required = ["name", "ra_deg", "dec_deg"]
    missing = [c for c in required if c not in new.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return

    # 填充 schema 列
    for col in SCHEMA_COLUMNS:
        if col not in new.columns:
            new[col] = None
    new["source_catalog"] = source

    # 加载已有 master
    if not os.path.exists(MASTER_PATH):
        print("No existing master catalog. Running full build first.")
        new.to_csv(MASTER_PATH, index=False)
        run_full_pipeline()
        return

    master = pd.read_csv(MASTER_PATH)
    print(f"  Existing master: {len(master)} systems")
    print(f"  New entries:     {len(new)} systems")

    # 去重
    from .build import deduplicate
    combined = pd.concat([master, new[SCHEMA_COLUMNS]], ignore_index=True)
    combined = deduplicate(combined)
    print(f"  After dedup:     {len(combined)} systems")

    # 保存
    combined.to_csv(MASTER_PATH, index=False)
    print(f"\n  Updated: {MASTER_PATH}")

    # 重新分类
    from .classify import classify_master
    classify_master()

    # 重新丰富
    enrich(query_network=False)

    # 导出
    export_modeling_subsets()

    print("\nIncremental update complete.")


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Disk Lens Sample Library Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                     # Full B→C→D→E (no network)
  python pipeline.py --network           # Include SDSS/Vizier queries
  python pipeline.py --from phase_d      # Skip B, C; run D, E
  python pipeline.py --add new.csv --source paper   # Add new sample
        """
    )
    parser.add_argument("--network", action="store_true",
                       help="Enable SDSS/VizieR queries (needs network)")
    parser.add_argument("--from", dest="start_from", type=str,
                       choices=["phase_b", "phase_c", "phase_d", "phase_e"],
                       default="phase_b",
                       help="Start from specific phase (default: phase_b)")
    parser.add_argument("--add", type=str, metavar="FILE",
                       help="Add new sample CSV to existing catalog")
    parser.add_argument("--source", type=str, default="manual",
                       help="Source label for --add (default: manual)")

    args = parser.parse_args()

    if args.add:
        add_new_sample(args.add, args.source)
    else:
        run_full_pipeline(query_network=args.network, start_from=args.start_from)
