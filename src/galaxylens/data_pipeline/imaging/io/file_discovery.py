"""
HST 源文件查找
==============
根据透镜ID和波段，在 MAST_download 目录中定位正确的 FITS 文件。

功能:
  1. 按 product_type (DRC/DRZ/FLT/FLC) 自动匹配文件
  2. 优先级: FLC > FLT > DRC > DRZ
  3. 排除 skycell 产品（优先使用目标指向的产品）
  4. FLT/FLC 支持多曝光返回

处理流程:
    输入: lens_id + band + product_type
           │
           ├─ 扫描 raw/{lens}/MAST_download/
           │
           ├─ 按 product_type 过滤:
           │   ├─ DRC → *_drc.fits
           │   ├─ DRZ → *_drz.fits
           │   ├─ FLT → 文件名含波段 且 非drc/drz
           │   └─ FLC → 文件名含波段 且 非drc/drz
           │
           └─ 返回 List[Path] (FLT/FLC可能多个)

使用示例:
  >>> from galaxylens.data_pipeline.imaging.io.file_discovery import find_source_files
  >>> files = find_source_files("J1630+4520", "F814W", "FLT")
  >>> [f.name for f in files]
"""

from __future__ import annotations
import glob
from pathlib import Path
from typing import List, Optional, Literal

from astropy.io import fits

from galaxylens.data_pipeline.imaging.paths import get_raw_root

ProductType = Literal["DRC", "DRZ", "FLT", "FLC"]


# ══════════════════════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════════════════════

def find_source_files(
    lens_id: str,
    band: str,
    product_type: ProductType = "FLC",
    min_size_mb: float = 50.0,
) -> List[Path]:
    """在 MAST_download 中查找指定透镜+波段的源文件。

    Args:
        lens_id: 透镜标识符，如 "J2141-0001"
        band: 波段名，如 "F814W"
        product_type: 数据产品类型 (DRC/DRZ/FLT/FLC)
        min_size_mb: 最小文件大小阈值 (MB)，排除残次下载

    Returns:
        匹配的 FITS 文件路径列表（FLT/FLC可能有多个曝光）

    Raises:
        FileNotFoundError: 若未找到任何匹配文件

    Note:
        - DRC/DRZ 通常返回1个文件
        - FLT/FLC 可能返回多个文件（多次曝光）
        - skycell 产品被自动排除（优先指向目标的）
    """
    mast_dir = get_raw_root() / lens_id / "MAST_download"
    if not mast_dir.is_dir():
        raise FileNotFoundError(
            f"MAST_download not found for {lens_id} at {mast_dir}"
        )

    # 收集所有候选 FITS 文件
    all_fits: List[Path] = sorted(
        p for p in mast_dir.rglob("*.fits")
        if p.stat().st_size > min_size_mb * 1024 * 1024
    )

    if not all_fits:
        raise FileNotFoundError(
            f"No FITS files > {min_size_mb} MB in {mast_dir}"
        )

    # 按产品类型过滤
    band_lower = band.lower()
    matched: List[Path] = []

    if product_type in ("DRC", "DRZ"):
        suffix = f"_{product_type.lower()}.fits"
        matched = [f for f in all_fits
                   if f.name.lower().endswith(suffix)
                   and band_lower in f.name.lower()]
    else:
        # FLT/FLC: 先按波段名匹配，排除 DRC/DRZ/ASN
        matched = [
            f for f in all_fits
            if band_lower in f.name.lower()
            and "_drc" not in f.name.lower()
            and "_drz" not in f.name.lower()
            and "_asn" not in f.name.lower()
        ]
        # 如果没找到，尝试用FITS头FILTER关键词匹配（FLC文件可能不含波段名）
        if not matched:
            for f in all_fits:
                if "_drc" in f.name.lower() or "_drz" in f.name.lower():
                    continue
                if "_asn" in f.name.lower():
                    continue
                try:
                    with fits.open(f) as hdul:
                        for ext in [0, 1]:
                            if ext < len(hdul):
                                hdr = hdul[ext].header
                                # FLC/FLT files use FILTER1/FILTER2, not FILTER
                                f1 = hdr.get("FILTER1", "").strip()
                                f2 = hdr.get("FILTER2", "").strip()
                                filt = hdr.get("FILTER", "").strip()
                                if band.upper() in (f1.upper(), f2.upper(), filt.upper()):
                                    matched.append(f)
                                    break
                except Exception:
                    continue

    # 排除 skycell 产品（优先目标指向的）
    nonsky = [f for f in matched if "skycell" not in f.name.lower()]
    if nonsky:
        matched = nonsky

    # 二次验证：确保文件可读
    valid: List[Path] = []
    for f in matched:
        try:
            with fits.open(f) as hdul:
                # 至少有2个扩展的才是 FLT/DRC
                if len(hdul) >= 2:
                    valid.append(f)
        except Exception:
            continue

    if not valid:
        raise FileNotFoundError(
            f"No valid {product_type} files found for {lens_id} {band} "
            f"in {mast_dir}"
        )

    return sorted(valid)


def find_available_bands(lens_id: str) -> List[str]:
    """扫描 MAST_download，返回实际存在的波段列表。

    通过读取每个 FITS 文件的 FILTER 关键字来确定波段。

    Args:
        lens_id: 透镜标识符

    Returns:
        波段名列表，如 ["F435W", "F814W"]
    """
    mast_dir = get_raw_root() / lens_id / "MAST_download"
    if not mast_dir.is_dir():
        return []

    bands: set[str] = set()
    for fpath in mast_dir.rglob("*.fits"):
        if fpath.stat().st_size < 50_000_000:
            continue
        try:
            with fits.open(fpath) as hdul:
                for ext in [0, 1]:
                    if ext < len(hdul):
                        filt = hdul[ext].header.get("FILTER", "").strip()
                        if filt and filt not in ("", "N/A", "CLEAR"):
                            bands.add(filt)
                            break
        except Exception:
            continue

    return sorted(bands)


def has_files_for_band(lens_id: str, band: str) -> bool:
    """检查指定透镜+波段是否有可用的源文件。

    Args:
        lens_id: 透镜标识符
        band: 波段名

    Returns:
        是否有文件可用
    """
    try:
        files = find_source_files(lens_id, band)
        return len(files) > 0
    except FileNotFoundError:
        return False
