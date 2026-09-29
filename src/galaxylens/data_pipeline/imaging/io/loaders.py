"""
HST 图像加载器 HST image loaders
==============
统一加载 DRC/FLT/FLC/DRZ 产品的 SCI + WHT + WCS。

功能:
  1. 自动检测 extension 结构（DRC: ext 0/1, FLT: ext 1/2）
  2. 提取 SCI, WHT, WCS 三元组
  3. 返回标准化的 RawExposure 数据类

产物类型:
  - DRC: drizzle组合产品，SCI在ext 0或1，WHT在ext 2
  - DRZ: drizzle组合(WFPC2)，类似DRC
  - FLT: 单次曝光，SCI=ext 1, WHT=ext 2, CTX=ext 3
  - FLC: CTE修正FLT，结构同FLT

使用示例:
  >>> from galaxylens.data_pipeline.imaging.io.loaders import load_exposure, RawExposure
  >>> exp = load_exposure(Path("j9c706010.fits"))
  >>> exp.sci.shape, exp.wht.shape
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


@dataclass
class RawExposure:
    """单次 HST 曝光的标准数据结构。

    Attributes:
        path: 源文件路径
        sci: 科学数据 (2D float64, 单位 electrons/s)
        wht: 权重图 (2D float64, 单位 electrons⁻²·s²) 或 None
        header: FITS header (来自 SCI extension)
        wcs: WCS 对象 或 None
        product_type: 产品类型 ("DRC", "DRZ", "FLT", "FLC")
        exptime: 曝光时间 (秒)
        filter_name: 滤光片名
        detector: 探测器名
    """
    path: Path
    sci: np.ndarray
    wht: Optional[np.ndarray]
    header: fits.Header
    wcs: Optional[WCS]
    product_type: str
    exptime: float
    filter_name: str
    detector: str


# ══════════════════════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════════════════════

def load_exposure(filepath: Path, sci_ext: Optional[int] = None) -> RawExposure:
    """加载单个 HST FITS 文件，自动检测产品类型和 extension 结构。

    自动检测逻辑:
      - 若 ext 0 有数据 → DRC/DRZ（数据在 PRIMARY）
      - 若 ext 0 无数据, ext 1 名称为 'SCI' → FLT/FLC
      - 否则尝试 ext 1

    Args:
        filepath: FITS 文件路径
        sci_ext: 显式指定 SCI extension 索引 (FLT/FLC 双 chip 时,
                 目标可能在 chip2 = hdu[4] 而非默认 chip1 = hdu[1])

    Returns:
        RawExposure 对象

    Raises:
        ValueError: 若无法找到科学数据 extension
    """
    with fits.open(filepath) as hdul:
        n_ext = len(hdul)

        # ── 检测产品类型 ──
        # 显式指定 SCI extension (双 chip 时用于加载目标所在 chip)
        if sci_ext is not None:
            if hdul[sci_ext].data is None:
                raise ValueError(f"Extension {sci_ext} has no data in {filepath}")
            data = hdul[sci_ext].data.astype(np.float64)
            header = hdul[sci_ext].header
        # DRC/DRZ: ext 0 通常有数据
        elif hdul[0].data is not None:
            data = hdul[0].data.astype(np.float64)
            header = hdul[0].header
            sci_ext = 0
        elif n_ext > 1 and hdul[1].data is not None:
            data = hdul[1].data.astype(np.float64)
            header = hdul[1].header
            sci_ext = 1
        else:
            raise ValueError(f"Cannot find science data in {filepath}")

        # ── 检测产品类型 ──
        filetype = header.get("FILETYPE", "").strip()
        if "DRC" in filetype or "_drc" in filepath.name.lower():
            product_type = "DRC"
        elif "DRZ" in filetype or "_drz" in filepath.name.lower():
            product_type = "DRZ"
        elif "FLC" in filetype or "_flc" in filepath.name.lower() or "_flc" in str(filepath).lower():
            product_type = "FLC"
        else:
            product_type = "FLT"

        # ── 找 WHT extension ──
        wht: Optional[np.ndarray] = None
        if product_type in ("FLT", "FLC"):
            # FLT/FLC: WHT 在 ext 2
            if n_ext > 2 and hdul[2].data is not None:
                wht = hdul[2].data.astype(np.float64)
        else:
            # DRC/DRZ: WHT 可能在 ext 2 或名称为 'WHT'
            for ext_name in ["WHT", "ERR"]:
                try:
                    wht_ext = hdul[ext_name]
                    if wht_ext.data is not None:
                        wht = wht_ext.data.astype(np.float64)
                        break
                except (KeyError, IndexError):
                    continue

        # ── INHERIT 展开 (FLT/FLC 全局元数据在 PRIMARY) ──
        # FLT/FLC 的全局关键词 (EXPTIME/FILTER*/DETECTOR/INSTRUME/DATE-OBS 等)
        # 存于 PRIMARY header，SCI header 靠 INHERIT=True 引用，但 astropy 不自动展开。
        # 手动把 PRIMARY 中 SCI 缺失/为空的键补进 header，避免读到空值。
        primary_header = hdul[0].header
        for key in primary_header:
            if key in header and header.get(key) not in (None, ""):
                continue
            try:
                header[key] = primary_header[key]
            except Exception:
                continue

        # ── WCS ──
        # SIP 畸变 (A_ORDER/B_ORDER) 需要 CPDis 查找表扩展，
        # 必须传 HDUList 作为 fobj 才能正确解析，否则 WCS(header) 会报
        # "an astropy.io.fits.HDUList is required for Lookup table distortion"。
        try:
            wcs = WCS(header, fobj=hdul)
        except Exception:
            try:
                wcs = WCS(header)
            except Exception:
                wcs = None

        # ── 元数据 ──
        exptime = float(header.get("EXPTIME", 0) or 0.0)
        # FILTER 优先；ACS/WFC 双滤光片轮时 FILTER 常为 CLEAR1L，需取 FILTER2
        filter_name = header.get("FILTER", "").strip()
        if not filter_name or filter_name.upper().startswith("CLEAR"):
            for fk in ("FILTER2", "FILTER1"):
                cand = header.get(fk, "").strip()
                if cand and not cand.upper().startswith("CLEAR"):
                    filter_name = cand
                    break
        detector = header.get("DETECTOR", "").strip()

    return RawExposure(
        path=filepath,
        sci=data,
        wht=wht,
        header=header,
        wcs=wcs,
        product_type=product_type,
        exptime=exptime,
        filter_name=filter_name,
        detector=detector,
    )


def load_exposure_for_target(filepath: Path, ra: float, dec: float) -> RawExposure:
    """加载包含目标天球坐标的 chip (FLT/FLC 双 chip 时自动选择正确 SCI extension)。

    ACS/WFC FLT/FLC 有上下两个 chip (各 2048×4096)，默认 hdu[1] 是北 chip。
    若目标在南 chip (hdu[4])，默认加载会导致 WCS 坐标越界、cutout 裁错位置。
    此函数用目标坐标检测所在 chip 并加载对应 extension。

    Args:
        filepath: FITS 文件路径
        ra, dec: 目标天球坐标 (deg)

    Returns:
        包含目标坐标的 RawExposure (若找不到则返回默认 chip)
    """
    exp = load_exposure(filepath)
    if exp.product_type not in ("FLT", "FLC") or exp.wcs is None:
        return exp
    coord = SkyCoord(ra, dec, unit="deg")
    x, y = exp.wcs.world_to_pixel(coord)
    if 0 <= x < exp.sci.shape[1] and 0 <= y < exp.sci.shape[0]:
        return exp
    # 目标不在默认 chip，遍历其他 SCI extension
    with fits.open(filepath) as hdul:
        for i in range(len(hdul)):
            if i == 1 or hdul[i].data is None:
                continue
            if hdul[i].name != "SCI":
                continue
            try:
                exp2 = load_exposure(filepath, sci_ext=i)
                x2, y2 = exp2.wcs.world_to_pixel(coord)
                if 0 <= x2 < exp2.sci.shape[1] and 0 <= y2 < exp2.sci.shape[0]:
                    return exp2
            except Exception:
                continue
    return exp


# ══════════════════════════════════════════════════════════════════════
# WFPC2 DRZ 芯片定位
# ══════════════════════════════════════════════════════════════════════

_WFPC2_CHIP_INSTRUMENT = {
    1: "WFPC2/PC",
    2: "WFPC2/WF2",
    3: "WFPC2/WF3",
    4: "WFPC2/WF4",
}


def wfpc2_chip_position(
    filepath: Path,
    ra: float,
    dec: float,
    drz_wcs: Optional[WCS] = None,
) -> tuple[Optional[str], Optional[tuple[int, int]]]:
    """WFPC2 DRZ: 用 CTX 图 + HDRTAB 精确定位目标所在芯片及芯片坐标。

    WFPC2 DRZ 是 drizzle 组合整个焦平面 (DETECTOR=MULTIPLE)，目标可能落在
    PC/WF2/WF3/WF4 任一芯片。DRZ 的 WCS 返回焦平面坐标 (约 4588×4573)，
    远超单芯片 0-799 范围，不能直接用于 TinyTim (TinyTim 需芯片内坐标)。

    方法 (全部实测验证):
      1. 用 DRZ WCS 得到目标焦平面像素 (fx, fy)
      2. 读 CTX[fy, fx]，取最低置位 bit → chip 号
         (bit0=PC, bit1=WF2, bit2=WF3, bit3=WF4, 每 4 bit 一个输入曝光)
      3. 读 HDRTAB EXTVER=chip 行的 OCRPIX/OCRVAL/CD(+SIP) 构建芯片 WCS
      4. world_to_pixel 反推芯片坐标 (0-799)

    J2141 验证: CTX=17476→bit2→chip3(WF3)，芯片坐标 (434.7, 422.7)，
    距中心 (34.7, 22.7)px，WFPC2 畸变 <0.1px 可忽略。

    Args:
        filepath: DRZ FITS 文件路径
        ra, dec: 目标天球坐标 (deg)
        drz_wcs: DRZ 焦平面 WCS (可选，用于步骤 1)

    Returns:
        (instrument_full, detector_pos) 或 (None, None)
    """
    coord = SkyCoord(ra, dec, unit="deg")

    # 步骤 1: 焦平面像素
    if drz_wcs is None:
        try:
            drz_wcs = WCS(fits.getheader(filepath, ext=1))
        except Exception:
            return None, None
    try:
        fx, fy = drz_wcs.world_to_pixel(coord)
    except Exception:
        return None, None
    fx, fy = int(round(float(fx))), int(round(float(fy)))

    with fits.open(filepath, memmap=True) as hdul:
        # 步骤 2: CTX 图 → chip 号
        try:
            ctx = hdul["CTX"].data
        except (KeyError, IndexError):
            return None, None
        if not (0 <= fy < ctx.shape[0] and 0 <= fx < ctx.shape[1]):
            return None, None
        ctx_val = int(ctx[fy, fx])
        if ctx_val == 0:
            return None, None
        lowest_bit = (ctx_val & -ctx_val).bit_length() - 1
        chip = (lowest_bit % 4) + 1  # 1=PC, 2=WF2, 3=WF3, 4=WF4

        # 步骤 3-4: HDRTAB 反推芯片坐标
        try:
            ht = hdul["HDRTAB"].data
        except (KeyError, IndexError):
            return None, None
        rows = ht[ht["EXTVER"] == chip]
        if len(rows) == 0:
            return None, None
        row = rows[0]

        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = int(row["ONAXIS1"])
        hdr["NAXIS2"] = int(row["ONAXIS2"])
        # OCTYPE 形如 "RA---TAN"/"DEC--TAN"，加 -SIP 后缀以正确启用 SIP 畸变
        hdr["CTYPE1"] = str(row["OCTYPE1"]).strip() + "-SIP"
        hdr["CTYPE2"] = str(row["OCTYPE2"]).strip() + "-SIP"
        hdr["CRVAL1"] = float(row["OCRVAL1"])
        hdr["CRVAL2"] = float(row["OCRVAL2"])
        hdr["CRPIX1"] = float(row["OCRPIX1"])
        hdr["CRPIX2"] = float(row["OCRPIX2"])
        hdr["CD1_1"] = float(row["CD1_1"])
        hdr["CD1_2"] = float(row["CD1_2"])
        hdr["CD2_1"] = float(row["CD2_1"])
        hdr["CD2_2"] = float(row["CD2_2"])
        # SIP 畸变系数 (A_ORDER/B_ORDER + A_i_j/B_i_j, i+j 从 2 到 order)
        for prefix in ("A", "B"):
            order = int(row[f"{prefix}_ORDER"])
            hdr[f"{prefix}_ORDER"] = order
            for i in range(order + 1):
                for j in range(order + 1):
                    if 2 <= i + j <= order:
                        key = f"{prefix}_{i}_{j}"
                        if key in ht.columns.names:
                            hdr[key] = float(row[key])

        try:
            chip_wcs = WCS(hdr)
            cx, cy = chip_wcs.world_to_pixel(coord)
        except Exception:
            # 回退: 无畸变 (纯 TAN + CD)。目标近中心时畸变 <0.1px，可忽略。
            hdr.remove("A_ORDER", ignore_missing=True)
            hdr.remove("B_ORDER", ignore_missing=True)
            for prefix in ("A", "B"):
                for i in range(4):
                    for j in range(4):
                        hdr.remove(f"{prefix}_{i}_{j}", ignore_missing=True)
            hdr["CTYPE1"] = str(row["OCTYPE1"]).strip()
            hdr["CTYPE2"] = str(row["OCTYPE2"]).strip()
            chip_wcs = WCS(hdr)
            cx, cy = chip_wcs.world_to_pixel(coord)

        return _WFPC2_CHIP_INSTRUMENT[chip], (int(round(float(cx))), int(round(float(cy))))
