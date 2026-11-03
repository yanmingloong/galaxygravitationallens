"""Lens identification: catalog row -> lens type -> modeling plan -> recipe.

识别层（第 4 项）：classify.py 只做"盘/非盘"筛选（椭圆被排进 Tier X），这里补上
正向识别——从 catalog 行推断"透镜类型（盘/椭圆）× 透镜约束（弧/四重像）"，映射到
建模模块组合 + 配方模板。输出 :class:`LensPlan`，再调 :func:`write_recipe` 生成
YAML，交给已有的 ``gl.fit``，即可串起"识别 -> 建模 -> 拟合"。

识别依据（按优先级）：
    1. ``lens_type_label`` —— survey 里已有的形态学标签（DEV=椭圆，SER/EXP/S0=盘）
    2. ``n_sersic``        —— Sérsic index 辅助（n<2 盘、n>=3 椭圆；catalog 里基本没填）
    3. ``B_D_ratio``       —— 有 bulge/disk 分解则判盘
    4. ``n_images`` / 已知名单 —— 四重像（点级约束，n_images 数据缺失时靠已知名单）

三个已知样本的映射（回归测试的锚点）：
    J2141-0001  disk  × imaging   -> j2141_barnabe.yaml (bulge+disk + jampy_kdc + 弧)
    J1331+3628  disk  × positions -> j1331_trick.yaml   (点级四重像 fit_positions)
    J1630+4520  ellip × imaging   -> j1630_ellip.yaml   (单 Sérsic + jampy + sigma_ap)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 内联 normalize_name / safe_float（不 import .utils，因为 utils -> config 会触发
# GALAXYLENS_SAMPLE_LIB 环境变量检查 + 创建目录；识别层只读 catalog CSV，应当轻）。
def _normalize_name(name: str) -> str:
    name = str(name).lower().strip()
    name = name.replace(" ", "").replace("_", "")
    name = re.sub(r"^sdssj?", "j", name)
    if not name.startswith("j"):
        name = "j" + name
    return name.replace("+", "").replace("-", "")


def _safe_float(val, default=np.nan):
    if pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

# 椭圆 / 盘的形态学标签（对应 classify.py 的 TIER_X_LENS_TYPES 与盘 tier）
ELLIPTICAL_LABELS = {"DEV", "DEVAUCOULEURS", "ELLIPTICAL", "DEV2"}
DISK_LABELS = {"SER", "SERSIC", "EXP", "EXPONENTIAL", "DISKY", "S0"}

# 已知四重像名单（n_images 在 catalog 里几乎没填，跟 TIER0_NAMES 同做法靠已知名单）
QUAD_NAMES = ["J1331+3628"]

# 仓库内 product CSV（README 声明这里是 modeling_input 产品位置）。
# 读 full（覆盖全部 tier，含四重像/特殊样本）+ elliptical（补 DEV 椭圆标签）；
# sample/tier01 是 full 的子集，不必重复读。
_CATALOG_FILES = [
    "modeling_input_full.csv",
    "modeling_input_elliptical.csv",
]


@dataclass
class LensPlan:
    """透镜识别结果：类型 + 建模模块选择 + 配方模板。

    ``stellar_template`` / ``subtract_method`` / ``dynamics`` / ``source_model``
    为 ``None`` 表示该组合不需要对应模块（如点级四重像不做恒星减除）。
    """

    name: str
    lens_type: str                  # disk / elliptical
    constraint: str                 # imaging / positions / mein
    stellar_template: Optional[str]  # disk / ellip
    subtract_method: Optional[str]   # sersic / sersic_single
    dynamics: Optional[str]          # jampy_kdc / jampy
    source_model: Optional[str]      # sersic / pixelized
    template_yaml: str               # 配方模板（相对 configs/fit/ 或绝对路径）
    basis: str                       # 识别依据（人类可读）

    def asdict(self) -> dict:
        return asdict(self)


def _is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def identify_type(row: pd.Series) -> tuple[str, str]:
    """从 catalog 行推断透镜类型 ``(lens_type, basis)``。

    ``lens_type`` 是形态分类：``disk`` / ``elliptical``（四重像是约束、在
    :func:`identify_constraint` 里单独判，不在这里）。
    """
    lt = str(row.get("lens_type_label", "") or "").strip().upper()
    n_sersic = _safe_float(row.get("n_sersic"))
    b_d = _safe_float(row.get("B_D_ratio"))

    if lt in ELLIPTICAL_LABELS:
        return "elliptical", f"lens_type={lt}"

    if lt in DISK_LABELS:
        return "disk", f"lens_type={lt}"

    # Sérsic n 辅助（n<2 盘、n>=3 椭圆）
    if not _is_nan(n_sersic):
        if n_sersic < 2.0:
            return "disk", f"n_sersic={n_sersic:.1f}_disk_like"
        if n_sersic >= 3.0:
            return "elliptical", f"n_sersic={n_sersic:.1f}_bulge_like"

    # 有 bulge/disk 分解则判盘
    if not _is_nan(b_d):
        return "disk", f"B/D={b_d:.2f}"

    # 默认盘（样本库本就是盘筛选产物；type=None 时无法判断形态，诚实标注）
    return "disk", "no_morphology_label_default_disk"


def identify_constraint(row: pd.Series, lens_type: str) -> tuple[str, str]:
    """推断透镜约束类型 ``(constraint, basis)``：imaging / positions / mein。

    四重像（n_images>=4）走点级 ``positions``；延展弧走 ``imaging``。n_images
    数据缺失时靠已知四重像名单兜底。
    """
    name = str(row.get("name", "") or "")
    nname = _normalize_name(name)

    if any(_normalize_name(qn) in nname for qn in QUAD_NAMES):
        return "positions", "known_quad"

    n_images = _safe_float(row.get("n_images"))
    if not _is_nan(n_images) and n_images >= 4:
        return "positions", f"n_images={int(n_images)}"

    return "imaging", "extended_arc"


def plan_modeling(name: str, lens_type: str, constraint: str, basis: str) -> LensPlan:
    """类型 × 约束 -> 建模模块组合 + 配方模板。"""
    if constraint == "positions":
        # 点级四重像：无恒星减除/动力学，靠像位置（Trick+2016 J1331 路线）
        return LensPlan(
            name, lens_type, "positions", None, None, None, None,
            "configs/fit/j1331_trick.yaml", f"point_level_quad:{basis}",
        )

    if lens_type == "elliptical":
        # 椭圆透镜：单 Sérsic 减恒星 + jampy 各向同性 + sigma_ap（J1630 路线）
        return LensPlan(
            name, "elliptical", "imaging", "ellip", "sersic_single",
            "jampy", "sersic", "configs/fit/j1630_ellip.yaml",
            f"elliptical_sigma_ap_arc:{basis}",
        )

    # 盘透镜：bulge+disk 减恒星 + jampy_kdc 旋转 + v_rms 剖面 + 完整弧（Barnabe 路线）
    return LensPlan(
        name, "disk", "imaging", "disk", "sersic",
        "jampy_kdc", "sersic", "configs/fit/j2141_barnabe.yaml",
        f"disk_bulge_disk_arc:{basis}",
    )


def load_catalog(catalog_dir=None) -> pd.DataFrame:
    """读 modeling_input 产品 CSV 并合并去重。

    ``catalog_dir`` 缺省为仓库内 ``sample_library/``（git 内、随时可用）；也可传
    ``GALAXYLENS_SAMPLE_LIB/05_exports`` 的权威源目录。
    """
    if catalog_dir is None:
        catalog_dir = Path(__file__).resolve().parents[3] / "sample_library"
    catalog_dir = Path(catalog_dir)

    frames = []
    for fn in _CATALOG_FILES:
        fp = catalog_dir / fn
        if fp.exists():
            frames.append(pd.read_csv(fp))

    if not frames:
        raise FileNotFoundError(
            f"no modeling_input CSVs under {catalog_dir}; expected "
            f"{_CATALOG_FILES}"
        )
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["name"], keep="first")
    # 预计算 normalize 名字（向量化一次，供 _lookup 用）
    merged["_norm"] = merged["name"].astype(str).map(_normalize_name)
    return merged


def _lookup(catalog: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """按名字（normalize 后子串匹配）在 catalog 里找一行。"""
    target = _normalize_name(name)
    if not target:
        return None
    hits = catalog[catalog["_norm"].str.contains(target, regex=False, na=False)]
    if len(hits) == 0:
        return None
    return hits.iloc[0]


def identify(name: str, catalog: Optional[pd.DataFrame] = None) -> LensPlan:
    """按名字识别透镜 -> :class:`LensPlan`（不加载重依赖，可独立调用）。"""
    if catalog is None:
        catalog = load_catalog()

    row = _lookup(catalog, name)
    if row is None:
        raise LookupError(
            f"lens '{name}' not found in the sample catalog; "
            f"pass a catalog= DataFrame or set the right catalog_dir"
        )

    lens_type, type_basis = identify_type(row)
    constraint, constr_basis = identify_constraint(row, lens_type)
    basis = f"{type_basis}; {constr_basis}"
    return plan_modeling(name, lens_type, constraint, basis)


def write_recipe(plan: LensPlan, params: dict, out_path) -> Path:
    """从 ``plan.template_yaml`` 复制模板 + 覆盖 ``params`` 里的键，写 YAML。

    ``params`` 支持点号嵌套键（``"search.name"`` -> ``cfg["search"]["name"]``），
    也支持整块 dict（``{"search": {"name": ...}}``）。复杂参数（数据路径、几何、
    先验）仍需调用方给全。返回写出路径。
    """
    import yaml

    out_path = Path(out_path)
    tpl_rel = Path(plan.template_yaml)
    if not tpl_rel.is_absolute():
        tpl_rel = Path(__file__).resolve().parents[3] / tpl_rel
    if not tpl_rel.exists():
        raise FileNotFoundError(f"recipe template not found: {tpl_rel}")

    with open(tpl_rel, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    for k, v in params.items():
        if "." in k:
            parts = k.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v
        else:
            cfg[k] = v

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    return out_path
