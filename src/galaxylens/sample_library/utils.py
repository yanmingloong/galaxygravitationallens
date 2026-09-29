#!/usr/bin/env python3
"""
盘星系引力透镜样本库 — 共享工具函数 Disk lens sample library — shared utilities
=====================================
名称归一化、坐标匹配、去重逻辑。
"""

import re
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u

from .config import DEDUP_COORD_RADIUS_ARCSEC


# ================================================================
# 名称归一化
# ================================================================

def normalize_name(name: str) -> str:
    """
    将透镜名称归一化为统一的比较形式。

    规则:
    - 转小写
    - 去掉所有空格、下划线
    - 去掉 "SDSS " / "SDSS" 前缀（比较用）
    - 去掉 +/- 符号（J2141-0001 → j21410001）

    Examples:
        "SDSS J0820+4847" → "j08204847"
        "SDSSJ0820+4847"  → "j08204847"
        "J0820+4847"      → "j08204847"
    """
    name = str(name).lower().strip()
    # 去掉空格和下划线
    name = name.replace(" ", "").replace("_", "")
    # 去 SDSS 前缀
    name = re.sub(r'^sdssj?', 'j', name)
    # 确保以 j 开头
    if not name.startswith('j'):
        name = 'j' + name
    # 去 +/- 符号
    name = name.replace('+', '').replace('-', '')
    return name


def names_match(name1: str, name2: str) -> bool:
    """判断两个名称是否指同一系统。"""
    return normalize_name(name1) == normalize_name(name2)


# ================================================================
# 坐标匹配
# ================================================================

def coord_match(ra1, dec1, ra2, dec2) -> float:
    """
    计算两组坐标的角距离（arcsec）。

    Returns:
        float: 角距离 (arcsec)；若任一为 NaN 返回 inf
    """
    if any(pd.isna(v) for v in [ra1, dec1, ra2, dec2]):
        return float('inf')
    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
    return c1.separation(c2).arcsec


def find_matches(catalog_a: pd.DataFrame, catalog_b: pd.DataFrame,
                 radius_arcsec: float = DEDUP_COORD_RADIUS_ARCSEC) -> list:
    """
    用 astropy 做批量坐标交叉匹配。

    Args:
        catalog_a: 源星表（待匹配），至少含 ra_deg, dec_deg
        catalog_b: 目标星表（已有），至少含 ra_deg, dec_deg
        radius_arcsec: 匹配半径

    Returns:
        list of (idx_a, idx_b, sep_arcsec) 匹配对
    """
    if len(catalog_a) == 0 or len(catalog_b) == 0:
        return []

    coords_a = SkyCoord(ra=catalog_a["ra_deg"].values * u.deg,
                        dec=catalog_a["dec_deg"].values * u.deg)
    coords_b = SkyCoord(ra=catalog_b["ra_deg"].values * u.deg,
                        dec=catalog_b["dec_deg"].values * u.deg)

    idx_b, sep2d, _ = match_coordinates_sky(coords_a, coords_b)

    matches = []
    for i, (j, sep) in enumerate(zip(idx_b, sep2d)):
        if sep.arcsec < radius_arcsec:
            matches.append((i, j, sep.arcsec))
    return matches


# ================================================================
# 去重仲裁
# ================================================================

SOURCE_PRIORITY = {
    "SWELLS": 0,   # 最高 — 有 B/D 分解
    "SLACS": 1,    # 论文验证
    "CASTLES": 2,
    "lenscat": 3,
    "GLUE": 4,
}
DEFAULT_PRIORITY = 99


def resolve_duplicate(existing: pd.Series, new: pd.Series) -> pd.Series:
    """
    去重仲裁：两行指同一系统时，选择保留哪一行。

    规则:
    1. 按 source_catalog 优先级
    2. 同源按 grade (A > B > C)
    3. 合并 cross_refs（跨源对照名）

    Returns:
        pd.Series: 合并后的最佳行
    """
    existing_pri = SOURCE_PRIORITY.get(str(existing.get("source_catalog", "")), DEFAULT_PRIORITY)
    new_pri = SOURCE_PRIORITY.get(str(new.get("source_catalog", "")), DEFAULT_PRIORITY)

    if existing_pri < new_pri:
        keep, other = existing.copy(), new
    elif new_pri < existing_pri:
        keep, other = new.copy(), existing
    else:
        # 同源：比较 grade
        from .config import GRADE_ORDER
        eg = GRADE_ORDER.get(str(existing.get("grade", "")), 99)
        ng = GRADE_ORDER.get(str(new.get("grade", "")), 99)
        if eg <= ng:
            keep, other = existing.copy(), new
        else:
            keep, other = new.copy(), existing

    # 合并 cross_refs
    refs = set()
    for r in [existing.get("name", ""), new.get("name", ""),
              existing.get("cross_refs", ""), new.get("cross_refs", "")]:
        if pd.notna(r) and str(r).strip():
            refs.add(str(r).strip())
    keep["cross_refs"] = "|".join(sorted(refs))

    return keep


# ================================================================
# 辅助函数
# ================================================================

def safe_float(val, default=np.nan):
    """安全转换为 float。"""
    if pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def flag_if_present(val, flag_true: str, flag_false: str = "") -> str:
    """有值返回 flag_true，无值返回 flag_false。"""
    return flag_true if pd.notna(val) and str(val).strip() not in ("", "-", "nan") else flag_false
