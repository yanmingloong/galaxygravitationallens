"""
HST 透镜数据预处理总调度
========================
按样本和版本配置，自动化执行完整预处理管线。

处理流程:
  对于每个透镜的每个波段:
    1. file_discovery  → 查找源文件
    2. loaders         → 加载 SCI + WHT + WCS
    3. cutout          → WCS 裁剪到 15"×15"
    4. centroid        → 2D Gaussian 中心微调
    5. background      → 背景估计 + 减除
    6. cosmic_ray      → 宇宙线检测 (FLT/FLC) 或 sigma-clipped stacking
    7. noise           → 噪声图 (根据 product_type 自动选方法)
    8. psf             → PSF 构建 (ePSF → Moffat → Gaussian 回退)
    9. starmask        → 恒星 mask
    10. alignment      → 跨波段对齐 (仅多波段)
    11. output         → FITS 写入 + 诊断图 + 日志 + 质量评估

用法:
  python preprocess_master.py --lens J2141-0001 --version-tag v1_test
  python preprocess_master.py --sample SWELLS --version-tag v2_flc_whtonly
  python preprocess_master.py --all --version-tag v1_test
  python preprocess_master.py --lens J1630+4520 --dry-run
  python preprocess_master.py --lens J1630+4520 --bands F814W

历史:
  2026-06-23: 初始版本，从 preprocess_swells.py / preprocess_elliptical.py 重构
"""

from __future__ import annotations
import sys
import math
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from galaxylens.data_pipeline.imaging.paths import get_processed_root

# 样本族 → 输出目录名映射（与旧数据 processed/{disk,ellip} 层级保持一致）
SAMPLE_DIR = {"SWELLS": "disk", "SLACS": "ellip"}

# ── 导入管线模块 ─────────────────────────────────────────────────
from galaxylens.data_pipeline.imaging.config import (
    get_lens_config, list_lenses, get_bands_for_lens,
    get_pixel_scale, CUTOUT_SIZE_ARCSEC, PSF_STAMP_SIZE,
)
from galaxylens.data_pipeline.imaging.io.file_discovery import find_source_files, find_available_bands
from galaxylens.data_pipeline.imaging.io.loaders import (
    load_exposure, load_exposure_for_target, RawExposure, wfpc2_chip_position,
)
from galaxylens.data_pipeline.imaging.stages.cutout import wcs_cutout_with_wht
from galaxylens.data_pipeline.imaging.stages.centroid import fit_centroid
from galaxylens.data_pipeline.imaging.stages.background import process_background
from galaxylens.data_pipeline.imaging.stages.cosmic_ray import (
    detect_cosmic_rays_laplace,
    detect_cosmic_rays_deepcr,
    sigma_clipped_stack,
    compute_cr_fraction,
)
from galaxylens.data_pipeline.imaging.stages.noise import compute_noise
from galaxylens.data_pipeline.imaging.stages.psf import detect_stars, filter_field_stars, build_psf, psf_ee50
from galaxylens.data_pipeline.imaging.stages.starmask import make_starmask, combine_masks
from galaxylens.data_pipeline.imaging.stages.alignment import compute_alignment, format_alignment_report
from galaxylens.data_pipeline.imaging.output.fits_writer import (
    write_processed_band, write_combined_starmask, make_prefix,
)
from galaxylens.data_pipeline.imaging.output.diagnostics import generate_diagnostic_plot
from galaxylens.data_pipeline.imaging.output.log import ProcessingLog
from galaxylens.data_pipeline.imaging.output.quality import assess_quality


# ══════════════════════════════════════════════════════════════════════
# 单波段处理
# ══════════════════════════════════════════════════════════════════════

def process_band(
    lens_id: str,
    band: str,
    version_tag: str,
    output_base: Path,
    log: ProcessingLog,
    dry_run: bool = False,
    psf_method: str = None,
) -> Optional[dict]:
    """处理单个透镜的单个波段。

    Args:
        lens_id: 透镜标识符
        band: 波段名
        version_tag: 版本标签
        output_base: 输出根目录 (processed/)
        log: 处理日志对象
        dry_run: 仅列出将处理的文件，不实际运行

    Returns:
        处理结果字典或 None
    """
    config = get_lens_config(lens_id)
    output_dir = output_base / SAMPLE_DIR.get(config.sample, config.sample.lower()) / lens_id / version_tag

    print(f"\n  --- [{band}] ---", flush=True)

    # ── 1. 查找源文件（FLC优先，FLT/DRC/DRZ回退）──
    source_files = None
    product_type = None
    for pt in ["FLC", "FLT", "DRC", "DRZ"]:
        try:
            source_files = find_source_files(lens_id, band, product_type=pt)
            product_type = pt
            break
        except FileNotFoundError:
            continue
        except Exception:
            continue

    if source_files is None:
        print(f"    SKIP: No source files found for {band}", flush=True)
        log.add_error(band, "No source files found")
        return None

    print(f"    Found {len(source_files)} file(s):", flush=True)
    for f in source_files:
        size_mb = f.stat().st_size / 1024 ** 2
        print(f"      {f.name} ({size_mb:.0f} MB)", flush=True)

    if dry_run:
        print(f"    [DRY RUN] Would process into {output_dir}", flush=True)
        return {"dry_run": True}

    # ── 2. 加载 ──
    exposures = [load_exposure_for_target(f, config.ra, config.dec) for f in source_files]
    exp = exposures[0]
    n_exp = len(exposures)
    pixel_scale = get_pixel_scale(lens_id, band)
    print(f"    Loaded {n_exp} exposure(s), product={exp.product_type}, "
          f"scale={pixel_scale}\"/pix", flush=True)

    # ── 3. WCS 裁剪 ──
    offset = config.center_offset_arcsec
    sci_cuts = []
    wht_cuts = []
    for e in exposures:
        s, w = wcs_cutout_with_wht(
            e.sci, e.wht, e.wcs,
            config.ra, config.dec,
            CUTOUT_SIZE_ARCSEC, pixel_scale, offset,
        )
        sci_cuts.append(s)
        wht_cuts.append(w)

    # ── 4. 多曝光堆叠 (FLT/FLC only) ──
    if n_exp > 1 and exp.product_type in ("FLT", "FLC"):
        sci_stacked, cr_hitmap = sigma_clipped_stack(sci_cuts)
        wht_stacked = np.mean(np.stack([w for w in wht_cuts if w is not None], axis=0), axis=0) \
                      if any(w is not None for w in wht_cuts) else None
        cr_fraction = compute_cr_fraction(cr_hitmap > 0)
        cr_mask = cr_hitmap > 0
        sci = sci_stacked
        wht = wht_stacked
        cr_method = "stacked"
        print(f"    Stacked {n_exp} exposures, CR-hit: {cr_fraction:.3f}%", flush=True)
    else:
        sci = sci_cuts[0]
        wht = wht_cuts[0]
        if exp.product_type in ("FLT", "FLC"):
            # 优先 deepCR（检测+inpaint 修复），失败回退 LA-cosmic
            instrument = exp.header.get("INSTRUME", "ACS")
            detector = exp.header.get("DETECTOR", "WFC")
            exptime = exp.header.get("EXPTIME", 420.0)
            cr_mask, sci = detect_cosmic_rays_deepcr(
                sci, exptime=exptime,
                instrument=instrument, detector=detector, band=band,
                inpaint=True,
            )
            if cr_mask.sum() > 0:
                cr_fraction = compute_cr_fraction(cr_mask)
                cr_method = "deepCR"
                print(f"    Single exposure, deepCR detected: {cr_fraction:.3f}%", flush=True)
            else:
                cr_mask = detect_cosmic_rays_laplace(sci)
                cr_fraction = compute_cr_fraction(cr_mask)
                cr_method = "LA-cosmic"
                print(f"    Single exposure, LA-cosmic detected: {cr_fraction:.3f}%", flush=True)
        else:
            cr_mask = np.zeros(sci.shape, dtype=bool)
            cr_fraction = 0.0
            cr_method = None

    # ── 5. 中心微调（带迭代修正）──
    cx_init, cy_init = sci.shape[1] // 2, sci.shape[0] // 2
    cx_fit, cy_fit = fit_centroid(sci, cx_init, cy_init)
    print(f"    Center offset: ({cx_fit - cx_init:+.1f}, {cy_fit - cy_init:+.1f}) px", flush=True)

    # 迭代修正：若偏移 > 10 像素，修正 RA/Dec 重新裁剪（参考 dinos_i）
    CENTER_THRESHOLD_PX = 10.0
    if abs(cx_fit - cx_init) > CENTER_THRESHOLD_PX or abs(cy_fit - cy_init) > CENTER_THRESHOLD_PX:
        dx_as = (cx_fit - cx_init) * pixel_scale
        dy_as = (cy_fit - cy_init) * pixel_scale
        cos_dec = math.cos(math.radians(config.dec))
        ra_corr = config.ra - dx_as / 3600.0 / cos_dec
        dec_corr = config.dec - dy_as / 3600.0
        print(f"    Iterative recenter: ({dx_as:.1f}arcsec, {dy_as:.1f}arcsec) -> "
              f"RA={ra_corr:.6f} DEC={dec_corr:.6f}", flush=True)

        # 重新裁剪
        sci_cuts2, wht_cuts2 = [], []
        for e in exposures:
            s, w = wcs_cutout_with_wht(
                e.sci, e.wht, e.wcs,
                ra_corr, dec_corr,
                CUTOUT_SIZE_ARCSEC, pixel_scale, offset,
            )
            sci_cuts2.append(s)
            wht_cuts2.append(w)

        # 重做堆叠/单曝光
        if n_exp > 1 and exp.product_type in ("FLT", "FLC"):
            stack2, _ = sigma_clipped_stack(sci_cuts2)
            sci = stack2
        else:
            sci = sci_cuts2[0]

        cx2, cy2 = sci.shape[1] // 2, sci.shape[0] // 2
        cx_fit, cy_fit = fit_centroid(sci, cx2, cy2)
        print(f"    Recentered offset: ({cx_fit - cx2:+.1f}, {cy_fit - cy2:+.1f}) px", flush=True)

    # ── 6. 背景减除 ──
    sci_bgsub, bg_med, bg_rms = process_background(sci)
    print(f"    BG: median={bg_med:.4f}, rms={bg_rms:.4f}", flush=True)

    # ── 7. 噪声图 ──
    calibrate = (exp.product_type in ("DRC", "DRZ"))
    noise = compute_noise(
        sci_bgsub, wht, exp.product_type,
        bg_rms=bg_rms, n_exposures=n_exp, calibrate=calibrate,
    )
    print(f"    Noise: median={np.median(noise):.5f} (method based on {exp.product_type})", flush=True)

    # ── 8. PSF ──
    psf_fwhm_pix = 2.5
    stars = detect_stars(sci_bgsub, psf_fwhm_pix)
    field_stars = filter_field_stars(stars, center=(cx_fit, cy_fit),
                                     data_shape=sci_bgsub.shape)
    print(f"    Detected {len(stars)} stars, {len(field_stars)} field stars", flush=True)

    # 提取 FITS 元数据供 TinyTim 使用
    date_obs = exp.header.get("DATE-OBS", "")
    # 仪器信息优先从 config 获取（FITS头可能缺失）
    band_cfg = config.bands.get(band)
    instrument_full = (band_cfg.instrument if band_cfg else None)
    if not instrument_full:
        instrument = exp.header.get("INSTRUME", "")
        detector = exp.header.get("DETECTOR", "")
        instrument_full = f"{instrument}/{detector}" if instrument and detector else None

    # 探测器位置：从 WCS 计算透镜在探测器上的像素坐标
    detector_pos = None
    if exp.wcs is not None and instrument_full:
        try:
            coord = SkyCoord(config.ra, config.dec, unit="deg")
            det_x, det_y = exp.wcs.world_to_pixel(coord)
            detector_pos = (int(det_x), int(det_y))
            print(f"    Detector pos: ({detector_pos[0]}, {detector_pos[1]})", flush=True)
        except Exception:
            pass

    # WFPC2 DRZ: 焦平面 WCS 返回的是 drizzle 组合图坐标 (远超芯片 0-799)，
    # 需用 CTX 图 + HDRTAB 精确定位目标所在芯片及芯片内坐标。
    if (exp.product_type == "DRZ"
            and instrument_full and instrument_full.upper().startswith("WFPC2")):
        chip_instr, chip_pos = wfpc2_chip_position(
            exp.path, config.ra, config.dec, drz_wcs=exp.wcs,
        )
        if chip_instr is not None and chip_pos is not None:
            print(f"    WFPC2 chip: {chip_instr}, chip pos {chip_pos}", flush=True)
            instrument_full = chip_instr
            detector_pos = chip_pos
        else:
            print(f"    [WARN] WFPC2 chip 定位失败，回退焦平面坐标", flush=True)

    psf_data, psf_source = build_psf(
        sci_bgsub, field_stars,
        psf_fwhm_pix=psf_fwhm_pix,
        pixel_scale=pixel_scale,
        psf_method=psf_method,
        band=band,
        date_obs=date_obs,
        detector_pos=detector_pos,
        instrument=instrument_full,
    )
    # ── 9. 恒星 mask ──
    starmask = make_starmask(
        sci_bgsub.shape, field_stars, psf_fwhm_pix,
        lens_center=(cx_fit, cy_fit),
        pixel_scale=pixel_scale,
    )
    print(f"    Starmask: {starmask.sum()} px "
          f"({100 * starmask.sum() / starmask.size:.1f}%)", flush=True)

    # ── 10. 写入 FITS ──
    files = write_processed_band(
        output_dir, lens_id, band,
        sci=sci_bgsub, noise=noise, psf=psf_data,
        starmask=starmask, cr_mask=cr_mask,
    )

    # ── 11. 诊断图 ──
    diag_path = generate_diagnostic_plot(
        output_dir, lens_id, band,
        sci=sci_bgsub, noise=noise, psf=psf_data,
        starmask=starmask, bg_rms=bg_rms,
        psf_source=psf_source, cr_mask=cr_mask,
    )

    # ── 12. 质量评估 ──
    quality = assess_quality(
        sci_bgsub, noise, psf_data,
        starmask=starmask, cr_mask=cr_mask,
        bg_rms=bg_rms, pixel_scale=pixel_scale,
        cr_method=cr_method,
    )
    print(f"    Quality: {quality['rating']} "
          f"(SNR peak={quality['metrics']['snr_peak']:.0f})", flush=True)
    if quality["issues"]:
        for issue in quality["issues"]:
            print(f"      ⚠ {issue}", flush=True)

    # ── 13. 记录日志 ──
    log.add_band(band, {
        "source_files": [f.name for f in source_files],
        "product_type": exp.product_type,
        "n_exposures": n_exp,
        "pixel_scale": pixel_scale,
        "cutout_shape": list(sci_bgsub.shape),
        "centroid": [float(cx_fit), float(cy_fit)],
        "bg_median": float(bg_med),
        "bg_rms": float(bg_rms),
        "noise_method": f"auto_{exp.product_type}",
        "psf_source": psf_source,
        "psf_shape": list(psf_data.shape),
        "psf_ee50_px": round(psf_ee50(psf_data)[0], 3),
        "starmask_pixels": int(starmask.sum()),
        "cr_fraction_pct": round(float(cr_fraction), 4),
        "cr_method": cr_method,
        "quality": quality,
    })

    result = {
        "band": band,
        "centroid": (cx_fit, cy_fit),
        "pixel_scale": pixel_scale,
        "starmask": starmask,
        "quality": quality,
    }
    print(f"    ✅ {lens_id} {band} done", flush=True)
    return result


# ══════════════════════════════════════════════════════════════════════
# 单透镜处理
# ══════════════════════════════════════════════════════════════════════

def process_lens(
    lens_id: str,
    version_tag: str,
    output_base: Path,
    bands: Optional[List[str]] = None,
    dry_run: bool = False,
    psf_method: str = None,
) -> Optional[Path]:
    """处理一个透镜的所有可用波段。

    Args:
        lens_id: 透镜标识符
        version_tag: 版本标签
        output_base: 输出根目录
        bands: 指定波段列表 (None = 自动检测)
        dry_run: 仅列出，不运行

    Returns:
        输出目录路径或 None
    """
    config = get_lens_config(lens_id)
    output_dir = output_base / SAMPLE_DIR.get(config.sample, config.sample.lower()) / lens_id / version_tag

    print(f"\n{'=' * 60}")
    print(f"  Processing: {lens_id} ({config.name})")
    print(f"  Sample: {config.sample}, z_l={config.zl}, z_s={config.zs}")
    print(f"  Version: {version_tag}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")

    if bands is None:
        bands = get_bands_for_lens(lens_id)
        if not bands:
            bands = find_available_bands(lens_id)
    print(f"  Bands: {bands}")

    log = ProcessingLog(lens_id, version_tag)
    band_results = {}

    for band in bands:
        try:
            result = process_band(lens_id, band, version_tag, output_base, log, dry_run, psf_method)
            if result is not None:
                band_results[band] = result
        except Exception as e:
            print(f"    ❌ {band} FAILED: {e}", flush=True)
            traceback.print_exc()
            log.add_error(band, str(e))

    if dry_run:
        print(f"\n  [DRY RUN] {lens_id}: {len(bands)} bands listed", flush=True)
        return output_dir

    if not band_results:
        print(f"\n  ❌ {lens_id}: No bands processed successfully", flush=True)
        return None

    # ── 跨波段对齐 ──
    if len(band_results) >= 2:
        centroids = {
            b: (r["centroid"][0], r["centroid"][1], r["pixel_scale"])
            for b, r in band_results.items()
        }
        offsets = compute_alignment(centroids)
        print(f"\n  {format_alignment_report(offsets)}", flush=True)

    # ── 合并 starmask ──
    masks = [r["starmask"] for r in band_results.values()]
    if len(masks) > 0:
        combined = combine_masks(masks)
        write_combined_starmask(output_dir, lens_id, combined)
        print(f"  Combined starmask: {combined.sum()} px "
              f"({100 * combined.sum() / combined.size:.1f}%)", flush=True)

    # ── 版本说明 ──
    write_version_readme(output_dir, lens_id, version_tag, band_results, log)

    # ── 保存日志 ──
    log.save(output_dir)
    print(f"\n  ✅ {lens_id} complete → {output_dir}", flush=True)
    return output_dir


# ══════════════════════════════════════════════════════════════════════
# 版本说明
# ══════════════════════════════════════════════════════════════════════

def write_version_readme(
    output_dir: Path,
    lens_id: str,
    version_tag: str,
    band_results: dict,
    log: ProcessingLog,
) -> None:
    """生成版本 README.md。"""
    config = get_lens_config(lens_id)
    quality_summary = ""
    for b, r in band_results.items():
        if "quality" in r:
            q = r["quality"]
            quality_summary += f"| {b} | {q['rating']} | SNR peak={q['metrics']['snr_peak']:.0f} |\n"

    content = f"""# {lens_id} — {version_tag}

- **输入数据**: `raw/{lens_id}/MAST_download/`
- **处理日期**: {datetime.now().strftime('%Y-%m-%d')}
- **处理脚本**: `galaxylens.data_pipeline.imaging.preprocess_master`
- **样本**: {config.sample}
- **红移**: z_l={config.zl}, z_s={config.zs}

## 方法选择

| 环节 | 方法 | 参数 |
|------|------|------|
| 噪声 | auto (基于产品类型) | — |
| PSF | ePSF → Moffat → Gaussian 回退 | 4×oversampling |
| 宇宙线 | LA-cosmic / sigma-clipped stacking | σ=8.0 |
| 背景 | 边缘 sigma-clipping | σ=3.0 |
| 裁剪 | WCS Cutout2D | {CUTOUT_SIZE_ARCSEC}\"×{CUTOUT_SIZE_ARCSEC}\" |

## 质量评级

| 波段 | 评级 | 备注 |
|------|------|------|
{quality_summary}

## 注意事项

- 原始数据位于 `raw/{lens_id}/MAST_download/`（数据根目录见 GALAXYLENS_DATA_ROOT），请勿修改
- 处理日志见 `processing_log.json`
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(content)


# ══════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HST 透镜数据预处理总调度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_master.py --lens J2141-0001 --version-tag v1_test
  python preprocess_master.py --sample SWELLS --version-tag v2_flc_whtonly
  python preprocess_master.py --all --version-tag v1_test
  python preprocess_master.py --lens J1630+4520 --dry-run
  python preprocess_master.py --lens J1630+4520 --bands F814W
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lens", type=str, help="处理单个透镜")
    group.add_argument("--sample", type=str, choices=["SWELLS", "SLACS"],
                       help="按样本批量处理")
    group.add_argument("--all", action="store_true", help="处理全部透镜")

    parser.add_argument("--version-tag", type=str, default="v1",
                        help="版本标签 (default: v1)")
    parser.add_argument("--bands", type=str, nargs="*",
                        help="指定波段 (e.g., --bands F814W F555W)")
    parser.add_argument("--psf-method", type=str, default=None,
                        choices=["epsf", "moffat", "gaussian", "gaussian_ell",
                                 "tinytim_massey", "tinytim_direct"],
                        help="强制PSF方法 (默认自动: ePSF -> TinyTim -> Moffat)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出将处理的文件，不实际运行")

    args = parser.parse_args()

    # ── 确定透镜列表 ──
    if args.lens:
        lenses = [args.lens]
    elif args.sample:
        lenses = list_lenses(args.sample)
    else:
        lenses = list_lenses()

    print(f"Preprocessing {len(lenses)} lens(es)", flush=True)
    print(f"  Version: {args.version_tag}", flush=True)
    if args.dry_run:
        print(f"  Mode: DRY RUN (no files will be written)", flush=True)

    output_base = get_processed_root()

    for lens in lenses:
        try:
            process_lens(
                lens,
                version_tag=args.version_tag,
                output_base=output_base,
                bands=args.bands,
                dry_run=args.dry_run,
                psf_method=args.psf_method,
            )
        except Exception as e:
            print(f"\n  ❌ {lens} FAILED: {e}", flush=True)
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  All {len(lenses)} lens(es) processed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
