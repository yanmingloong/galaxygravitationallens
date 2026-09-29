"""TinyTim PSF：Massey 栅格（ACS）+ 直接编译（WFPC2 等）。

参考 dinos_i pipeline: https://github.com/cytan1/dinos-i
"""
from __future__ import annotations
from typing import Optional, Tuple
from pathlib import Path
import os
import csv
import gzip
import shutil
import subprocess
import tempfile

import numpy as np
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import zoom

from ...config import PSF_STAMP_SIZE
from galaxylens.config.paths import (
    get_tinytim_grid_base,
    get_tinytim_bin,
    get_tinytim_breathing_csv,
)


# ══════════════════════════════════════════════════════════════════════
# TinyTim 配置
# ══════════════════════════════════════════════════════════════════════

# Massey 栅格配置
MASSEY_GRID_TILE_SIZE = 255
MASSEY_GRID_SOURCE_SCALE = 0.03   # arcsec/pixel (TinyTim 原生)
MASSEY_GRID_N_TILES = 30           # 30x30 tiles
# 可用波段
MASSEY_AVAILABLE_BANDS = {"F435W", "F606W", "F775W", "F814W", "F850W"}

# TinyTim 仪器配置
# 注意: detector_range 是【整个焦平面】的像素范围，不是单 chip。
# ACS/WFC 两个 chip (各 2048×4096) 上下拼接成 4096×4096 焦平面，
# world_to_pixel 返回焦平面坐标 (y 可达 4095)。见 PSF_NEEDS.md §6。
# J2141 验证: pixel=(2083,3135) -> tile=(22,15) 需 ymax=4095 (ymax=2047 会得到 tile_row=28, 错)
TINYTIM_CAMERAS = {
    "ACS/WFC":  {"camera_id": 15, "chip_key": "ACS_WFC",  "detector_range": (0, 4095, 0, 4095)},
    "WFPC2/PC": {"camera_id": 6,  "chip_key": "WFPC2_PC", "detector_range": (0, 799, 0, 799)},
    # WFPC2 三个宽场芯片：tiny1 交互 menu 选 5 (WFPC2 WFC) 后需进一步输入 chip 号 (2/3/4)。
    # camera_id 存的是 tiny1 交互 menu 的选择号 (5)，chip 号单独存 wfpc2_chip。
    # tiny1.c case 5: sprintf("WFPC_II_WFC%d", chip); chip = chip-2+WFPC2_WFC_2 → 内部编号 12/13/14。
    "WFPC2/WF2": {"camera_id": 5, "chip_key": "WFPC2_WFC", "detector_range": (0, 799, 0, 799), "wfpc2_chip": 2},
    "WFPC2/WF3": {"camera_id": 5, "chip_key": "WFPC2_WFC", "detector_range": (0, 799, 0, 799), "wfpc2_chip": 3},
    "WFPC2/WF4": {"camera_id": 5, "chip_key": "WFPC2_WFC", "detector_range": (0, 799, 0, 799), "wfpc2_chip": 4},
}

# Z4 焦面偏移 (microns), 参考 dinos_i 模板
TINYTIM_FOCUS_CENTER = {
    "ACS/WFC":  0.028,
    "WFPC2/PC": -0.117,
    # WFPC2 WFC 三个芯片共用 -0.117 (dinos_i psf_templates 模板里 WF3 的值)
    "WFPC2/WF2": -0.117,
    "WFPC2/WF3": -0.117,
    "WFPC2/WF4": -0.117,
}

# Z4 scaling: 次镜 despace -> Z4 的比例
TINYTIM_Z4_SCALING = 0.011  # dinos_i psf_templates 中使用

def _get_hst_focus_mjd(mjd: float) -> float:
    """从 HST 呼吸模型查询给定 MJD 的聚焦偏移 (microns)。

    参考 dinos_i Hubble_dspace.csv。

    Args:
        mjd: Modified Julian Date

    Returns:
        聚焦偏移 (microns), 0 如果模型不可用
    """
    csv_path = get_tinytim_breathing_csv()
    if csv_path is None or not csv_path.exists():
        return 0.0

    try:
        # 二分查找最近的 MJD
        mjd_str = f"{mjd:.5f}"
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            best_mjd = None
            best_focus = 0.0
            for row in reader:
                try:
                    row_mjd = float(row[0])
                    row_focus = float(row[1])
                    if best_mjd is None or abs(row_mjd - mjd) < abs(best_mjd - mjd):
                        best_mjd = row_mjd
                        best_focus = row_focus
                except (ValueError, IndexError):
                    continue
        return best_focus
    except Exception:
        return 0.0


def _compute_focus_offset(
    date_obs: str,
    instrument: str,
) -> float:
    """计算 Z4 聚焦修正量。

    1. 查询 HST 呼吸模型获取 dm (次镜 despace, microns)
    2. Z4_offset = dm * Z4_SCALING

    Args:
        date_obs: 观测日期 "YYYY-MM-DD"
        instrument: "ACS/WFC" or "WFPC2/PC"

    Returns:
        Z4 修正值 (microns)
    """
    try:
        t = Time(date_obs, format='iso')
        mjd = t.mjd
        dm = _get_hst_focus_mjd(mjd)
        z4_offset = dm * TINYTIM_Z4_SCALING
        return z4_offset
    except Exception:
        return 0.0


def _detector_pos_to_tile(
    x_pix: int, y_pix: int,
    detector_range: Tuple[int, int, int, int],
    n_tiles: int = MASSEY_GRID_N_TILES,
) -> Tuple[int, int]:
    """探测器像素坐标 -> Massey 栅格瓦片坐标。

    参考 dinos_i: tile = 1 + (coord/det_max) * (n_tiles-2), clamp 1..(n_tiles-2)

    Args:
        x_pix, y_pix: 探测器像素坐标
        detector_range: (xmin, xmax, ymin, ymax)
        n_tiles: 栅格维度

    Returns:
        (tile_row, tile_col), 0-indexed
    """
    xmin, xmax, ymin, ymax = detector_range
    # 归一化到 0..(n_tiles-2), 留 1 个边缘瓦片
    tile_col = int(1 + (x_pix / xmax) * (n_tiles - 2))
    tile_row = int(1 + (y_pix / ymax) * (n_tiles - 2))
    # clamp
    tile_col = max(1, min(n_tiles - 2, tile_col))
    tile_row = max(1, min(n_tiles - 2, tile_row))
    return tile_row, tile_col


def _find_grid_file(base: Path, band: str, focus_idx: int) -> Optional[Path]:
    """在 Massey 栅格目录中递归搜索聚焦切片文件。

    支持多种目录结构:
      - F814W/TinyTim_models_F814W/TinyTim_f0.fits
      - F435W/435/TinyTim_f0.fits.gz

    Args:
        base: 栅格根目录
        band: 波段名
        focus_idx: 聚焦索引

    Returns:
        文件路径或 None
    """
    band_dir = base / band
    if not band_dir.is_dir():
        return None
    for root, dirs, files in os.walk(str(band_dir)):
        for fn in files:
            if fn == f"TinyTim_f{focus_idx}.fits" or fn == f"TinyTim_f{focus_idx}.fits.gz":
                return Path(root) / fn
    return None


def _load_grid_tile(
    grid_path: Path,
    tile_row: int,
    tile_col: int,
    tile_size: int = MASSEY_GRID_TILE_SIZE,
) -> Optional[np.ndarray]:
    """加载栅格并提取指定瓦片。"""
    try:
        if grid_path.suffix == '.gz':
            with gzip.open(grid_path, 'rb') as gz:
                with fits.open(gz) as hdul:
                    grid_data = hdul[0].data.copy()
        else:
            grid_data = fits.getdata(str(grid_path))
    except Exception:
        return None

    r0 = tile_row * tile_size
    c0 = tile_col * tile_size
    r1 = min(r0 + tile_size, grid_data.shape[0])
    c1 = min(c0 + tile_size, grid_data.shape[1])
    return grid_data[r0:r1, c0:c1]


def _find_best_focus_slice(
    grid_base: Path,
    band: str,
    tile_row: int,
    tile_col: int,
    tile_size: int = MASSEY_GRID_TILE_SIZE,
    focus_range: range = range(-10, 6),
) -> Tuple[Optional[np.ndarray], int]:
    """在多个聚焦切片中找峰值最大的瓦片。"""
    best_peak = -1.0
    best_tile = None
    best_f = 0

    for f_idx in focus_range:
        fpath = _find_grid_file(grid_base, band, f_idx)
        if fpath is None:
            continue
        tile = _load_grid_tile(fpath, tile_row, tile_col, tile_size)
        if tile is not None and tile.max() > best_peak:
            best_peak = tile.max()
            best_tile = tile.copy()
            best_f = f_idx

    return best_tile, best_f

def build_tinytim_massey_psf(
    band: str,
    date_obs: str,
    detector_pos: Optional[Tuple[int, int]] = None,
    pixel_scale: float = 0.05,
    stamp_size: int = PSF_STAMP_SIZE,
    focus_idx: Optional[int] = None,
    grid_base: Optional[Path] = None,
) -> Optional[Tuple[np.ndarray, str]]:
    """从 TinyTim Massey 栅格提取 ACS PSF。

    使用 Richard Massey (Durham) 预计算的 TinyTim v6.3 栅格模型。
    支持探测器位置映射、HST 呼吸聚焦修正、多聚焦切片搜索。

    Args:
        band: 波段 (如 "F814W", "F435W")
        date_obs: 观测日期 "YYYY-MM-DD"
        detector_pos: 探测器像素坐标 (x, y), None 则用中心
        pixel_scale: 目标像元尺度 (arcsec/pixel)
        stamp_size: 输出 PSF 尺寸
        focus_idx: 聚焦切片索引 (None=自动搜索 f-10..f+5)
        grid_base: 栅格根目录, None 用默认

    Returns:
        (psf_normalized, source_label) 或 None
    """
    if band not in MASSEY_AVAILABLE_BANDS:
        return None

    base = grid_base or get_tinytim_grid_base()
    if base is None:
        return None

    # 探测器位置 -> 瓦片
    if detector_pos is not None:
        det_range = TINYTIM_CAMERAS["ACS/WFC"]["detector_range"]
        tile_row, tile_col = _detector_pos_to_tile(
            detector_pos[0], detector_pos[1], det_range
        )
    else:
        # 默认中心瓦片
        tile_row = tile_col = MASSEY_GRID_N_TILES // 2

    # 聚焦搜索或指定
    if focus_idx is not None:
        fpath = _find_grid_file(base, band, focus_idx)
        if fpath is None:
            return None
        tile = _load_grid_tile(fpath, tile_row, tile_col, MASSEY_GRID_TILE_SIZE)
        if tile is None:
            return None
        best_tile = tile
        best_f = focus_idx
    else:
        # 自动搜索最佳聚焦
        result = _find_best_focus_slice(base, band, tile_row, tile_col)
        if result[0] is None:
            return None
        best_tile, best_f = result
        # 如果提供了日期，计算聚焦修正
        if date_obs:
            z4 = _compute_focus_offset(date_obs, "ACS/WFC")
            if abs(z4) > 0.001:
                # Z4 每微米约对应 1 个聚焦切片步长
                f_correction = int(round(z4))
                corrected_f = best_f + f_correction
                corrected_f = max(-10, min(5, corrected_f))
                if corrected_f != best_f:
                    result2 = _find_best_focus_slice(base, band, tile_row, tile_col,
                                                      focus_range=range(corrected_f, corrected_f+1))
                    if result2[0] is not None:
                        best_tile, best_f = result2

    # 从瓦片提取 PSF (rebin 0.03" -> 目标像素尺度, crop center)
    source_scale = MASSEY_GRID_SOURCE_SCALE
    target_scale = pixel_scale

    # 找瓦片峰值位置
    y_peak, x_peak = np.unravel_index(best_tile.argmax(), best_tile.shape)
    margin = stamp_size * int(target_scale / source_scale) + 10
    y0, y1 = max(0, y_peak - margin), min(best_tile.shape[0], y_peak + margin + 1)
    x0, x1 = max(0, x_peak - margin), min(best_tile.shape[1], x_peak + margin + 1)
    region = best_tile[y0:y1, x0:x1]

    # Rebin
    rebinned = zoom(region, source_scale / target_scale, order=1)
    cy, cx = rebinned.shape[0] // 2, rebinned.shape[1] // 2
    half = stamp_size // 2
    psf = rebinned[cy - half:cy + half + 1, cx - half:cx + half + 1]

    psf = np.maximum(psf, 0)
    psf /= psf.sum()

    focus_label = f"f{best_f:+d}"
    pos_label = f"t{tile_row},{tile_col}" if detector_pos else "center"
    label = f"tinytim_massey_{band}_{focus_label}_{pos_label}"

    return psf.astype(np.float32), label

def build_tinytim_direct_psf(
    band: str,
    instrument: str,
    date_obs: str,
    detector_pos: Tuple[int, int],
    spectrum_type: str = "g2v",
    psf_diameter: float = 3.0,
    stamp_size: int = PSF_STAMP_SIZE,
    tinytim_bin: Optional[Path] = None,
    pixel_scale: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, str]]:
    """使用本地 TinyTim v7.5 直接编译生成 PSF。

    tiny1 (参数文件) -> tiny2 (光线追踪) -> 提取 61x61 中心 -> 归一化

    参考 dinos_i 4_Psf_Maker.ipynb 流程。

    Args:
        band: 波段 (如 "F450W", "F606W")
        instrument: "ACS/WFC", "WFPC2/PC", "WFPC2/WF2", "WFPC2/WF3", "WFPC2/WF4"
        date_obs: 观测日期 "YYYY-MM-DD"
        detector_pos: 探测器像素 (x, y)
        spectrum_type: 光谱类型 ("g2v", "blackbody_6000" 等)
        psf_diameter: PSF 直径 (arcsec)
        stamp_size: 输出 stamp 尺寸
        tinytim_bin: TinyTim 可执行目录, None 用默认
        pixel_scale: 目标像元尺度 (arcsec/pixel); 与 TinyTim 原生尺度不同则 rebin

    Returns:
        (psf_normalized, source_label) 或 None
    """
    bin_dir = tinytim_bin or get_tinytim_bin()
    if bin_dir is None:
        print(f"    [WARN] TinyTim bin not set (TINYTIM_BIN env)", flush=True)
        return None
    tiny1 = bin_dir / "tiny1"
    tiny2 = bin_dir / "tiny2"

    if not tiny1.exists() or not tiny2.exists():
        print(f"    [WARN] TinyTim binaries not found at {bin_dir}", flush=True)
        return None

    cam_cfg = TINYTIM_CAMERAS.get(instrument)
    if cam_cfg is None:
        print(f"    [WARN] TinyTim: unknown instrument {instrument}", flush=True)
        return None

    # 焦距修正
    z4_offset = _compute_focus_offset(date_obs, instrument)
    z4 = TINYTIM_FOCUS_CENTER.get(instrument, 0.0) + z4_offset

    # 光谱映射
    spectrum_map = {
        "g2v": "2\n6000",           # blackbody 6000K (近似 G2V)
        "blackbody_6000": "2\n6000",
        "powerlaw_0": "3\n0",
    }
    spec_input = spectrum_map.get(spectrum_type, "2\n6000")

    # 构建 tiny1 输入序列 (参考 dinos_i TinyTim_Interactive_Flow.md)
    cam_id = cam_cfg["camera_id"]
    x, y = detector_pos

    # ACS/WFC 需要选 chip (1 or 2)；WFPC2 WFC 需要选 chip (2/3/4，见 cam_cfg["wfpc2_chip"])
    if instrument == "ACS/WFC":
        chip = 1 if x < 2048 else 2
        tiny1_input = f"{cam_id}\n{chip}\n{x} {y}\n{band.lower()}\n{spec_input}\n{psf_diameter}\nN\n{z4:.3f}\ntinytim_psf\n"
    elif cam_cfg.get("wfpc2_chip") is not None:
        # WFPC2 Wide Field: tiny1 menu 选 5 后需输入 chip 号 (2/3/4)
        chip = cam_cfg["wfpc2_chip"]
        tiny1_input = f"{cam_id}\n{chip}\n{x} {y}\n{band.lower()}\n{spec_input}\n{psf_diameter}\nN\n{z4:.3f}\ntinytim_psf\n"
    elif instrument == "WFPC2/PC":
        tiny1_input = f"{cam_id}\n{x} {y}\n{band.lower()}\n{spec_input}\n{psf_diameter}\nN\n{z4:.3f}\ntinytim_psf\n"
    else:
        tiny1_input = f"{cam_id}\n{x} {y}\n{band.lower()}\n{spec_input}\n{psf_diameter}\nN\n{z4:.3f}\ntinytim_psf\n"

    tmpdir = Path(tempfile.mkdtemp())
    try:
        param_file = tmpdir / "tinytim_psf.par"

        # Step 1: tiny1 (生成参数文件)
        env = os.environ.copy()
        env["TINYTIM"] = str(bin_dir)
        result1 = subprocess.run(
            [str(tiny1), str(param_file)],
            input=tiny1_input,
            capture_output=True, text=True,
            timeout=60, cwd=str(tmpdir),
            env=env,
        )
        if result1.returncode != 0 or not param_file.exists():
            print(f"    [WARN] TinyTim tiny1 failed: rc={result1.returncode}", flush=True)
            return None

        # Step 2: tiny2 (光线追踪)
        result2 = subprocess.run(
            [str(tiny2), str(param_file)],
            capture_output=True, text=True,
            timeout=120, cwd=str(tmpdir),
            env=env,
        )

        # 查找输出 (ACS 产生 *00_psf.fits, WFPC2 产生 *00.fits)
        psf_candidates = list(tmpdir.glob("tinytim_psf00_psf.fits")) + list(tmpdir.glob("tinytim_psf00.fits"))
        if not psf_candidates:
            print(f"    [WARN] TinyTim no PSF output files found in {tmpdir}", flush=True)
            return None

        with fits.open(str(psf_candidates[0])) as hdul:
            psf_full = np.asarray(hdul[0].data, dtype=float)
            native_scale = float(hdul[0].header.get("PIXSCALE", 0.0) or 0.0)

        # Rebin 到目标像元尺度 (TinyTim 输出原生尺度 vs 科学图尺度)。
        # WFPC2 WF 芯片原生 0.0999"/px，但 DRZ 科学图是 0.0455"/px，须放大 ~2.19x。
        if native_scale > 0 and pixel_scale and pixel_scale > 0:
            factor = native_scale / pixel_scale
            if abs(factor - 1.0) > 0.01:
                psf_full = zoom(psf_full, factor, order=1)

        # 以质心为中心裁剪 stamp_size×stamp_size。
        # 旧代码 half_full-30:half_full+31 在 30x30 数组上负索引 wrap，只剩 1/4 PSF。
        yy, xx = np.indices(psf_full.shape)
        tot = psf_full.sum()
        if tot > 0:
            cy = float((yy * psf_full).sum() / tot)
            cx = float((xx * psf_full).sum() / tot)
        else:
            cy, cx = psf_full.shape[0] / 2.0, psf_full.shape[1] / 2.0
        cyi, cxi = int(round(cy)), int(round(cx))

        half = stamp_size // 2
        y0, x0 = cyi - half, cxi - half
        psf_out = np.zeros((stamp_size, stamp_size), dtype=float)
        sy0, sx0 = max(0, y0), max(0, x0)
        sy1, sx1 = min(psf_full.shape[0], y0 + stamp_size), min(psf_full.shape[1], x0 + stamp_size)
        psf_out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = psf_full[sy0:sy1, sx0:sx1]

        psf_out = np.maximum(psf_out, 0)
        tot = psf_out.sum()
        if tot > 0:
            psf_out /= tot

        label = f"tinytim_direct_{band}_{instrument.replace('/','_')}_z4{z4:.3f}"
        return psf_out.astype(np.float32), label

    except Exception as e:
        print(f"    [WARN] TinyTim Direct failed: {e}", flush=True)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

