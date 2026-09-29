# Data Pipeline 使用指引

观测数据生产管线：把 HST 原始数据（MAST 下载的 FITS）预处理成 PyAutoLens 就绪的标准化产物（sci / noise / psf / mask）。

## 1. 概述

```
data_pipeline/
├── imaging/        # 透镜图像处理（HST 光学/NIR）
│   ├── config.py   # 16 透镜参数（SWELLS 盘 + SLACS 椭圆）
│   ├── paths.py    # 路径解析（GALAXYLENS_DATA_ROOT 环境变量）
│   ├── io/         # 源文件查找 + FITS 加载
│   ├── stages/     # 处理阶段（cutout/centroid/background/cosmic_ray/noise/psf/starmask/alignment）
│   ├── output/     # 输出（FITS 写入 / 诊断图 / 日志 / 质量评估）
│   └── preprocess_master.py  # 主流程调度 + CLI
├── spectra/        # 光谱管线（待提炼：SDSS / KCWI / DEIMOS）
└── other_obs/      # 其他观测（待提炼：NIRC2 K'-band 等）
```

**设计原则**：数据生产（data_pipeline）与数据消费（lensfit）分离。本模块只做"raw → processed"，产出标准化的 FITS；拟合引擎 lensfit 消费这些产物。因此本模块**只依赖 astropy/photutils**，不依赖 autolens。

## 2. 环境配置

### 2.1 数据根目录（必需）

```bash
export GALAXYLENS_DATA_ROOT=/path/to/DataPipeline
```

指向含 `raw/`（MAST 下载）和 `processed_v2/`（处理产物）的目录。未设置时，`find_source_files` 会抛出 RuntimeError 提示。

### 2.2 SExtractor（可选，背景估计增强）

```bash
export SEXTRACTOR_BIN=/path/to/sex   # 可选
```

不设置则自动用 `shutil.which("sex")` 在 PATH 里找；找不到则回退边缘 sigma-clipping 背景估计（已内置回退，无需担心）。

### 2.3 TinyTim（可选，PSF 物理模型）

```bash
export TINYTIM_GRID_BASE=/path/to/tinytim_psf_models     # Massey 栅格
export TINYTIM_BIN=/path/to/tinytim                      # tiny1/tiny2 可执行目录
export TINYTIM_BREATHING_CSV=/path/to/Hubble_dspace.csv  # HST 呼吸模型
```

三个都可选：未设置时对应功能自动跳过（Massey 栅格 / 直接编译 / 聚焦修正）。

### 2.3 依赖

核心依赖：`numpy`、`scipy`、`astropy`、`photutils`。

```bash
pip install -e ".[data]"
```

## 3. 快速开始

```python
from galaxylens.data_pipeline.imaging.preprocess_master import process_lens
from galaxylens.data_pipeline.imaging.paths import get_processed_root

# 处理单个透镜（输出到 $GALAXYLENS_DATA_ROOT/processed_v2/disk/J2141-0001/v1/）
process_lens("J2141-0001", version_tag="v1", output_base=get_processed_root())
```

## 4. 命令行用法

```bash
# 处理单个透镜
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J2141-0001 --version-tag v1

# 处理整个 SWELLS 样本
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --sample SWELLS --version-tag v1

# 只处理指定波段 + 强制 PSF 方法
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J1630+4520 --bands F814W --psf-method moffat

# 试运行（只列出将处理的文件，不写盘）
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J1630+4520 --dry-run
```

## 5. 处理流程（每个波段）

1. `file_discovery` — 查找源文件（FLC 优先 → FLT/DRC/DRZ 回退）
2. `loaders` — 加载 SCI + WHT + WCS
3. `cutout` — WCS 裁剪到 15″×15″
4. `centroid` — 2D Gaussian 中心微调（偏移 >10px 时迭代修正 RA/Dec）
5. `background` — SExtractor 背景估计（回退边缘 sigma-clipping）
6. `cosmic_ray` — deepCR（单曝光）或 sigma-clipped stacking（多曝光）
7. `noise` — 噪声图（按产品类型自动选方法）
8. `psf` — PSF 构建（ePSF → TinyTim → Moffat 回退，自动对比选优）
9. `starmask` — 恒星 mask
10. `alignment` — 跨波段对齐（多波段时）
11. `output` — FITS 写入 + 诊断图 + 日志 + 质量评估（8 项评级）

## 6. PSF 构建（stages/psf 包）

拆分为子模块：

| 子模块 | 职责 |
|---|---|
| `stars.py` | 场星检测（DAOStarFinder）+ 孤立性筛选 |
| `metrics.py` | EE50（半光半径）+ 二阶矩椭率 |
| `epsf.py` | photutils EPSFBuilder 超采样 ePSF（首选，捕捉真实像差）|
| `analytic.py` | Moffat / Gaussian / 椭圆 Gaussian 解析 PSF |
| `tinytim.py` | TinyTim Massey 栅格（ACS）+ 直接编译（WFPC2）|
| `__init__.py` | `build_psf` 自动选优入口 + re-export |

自动选择策略：ePSF 与 TinyTim 各生成候选 → EE50 ∈ [0.8, 3.0]px 且椭率 < 0.15 硬门槛 → 达标 ePSF 优先 → TinyTim 保底 → Moffat 兜底。

## 7. 教程：处理一个透镜的完整流程

```python
import os
os.environ["GALAXYLENS_DATA_ROOT"] = "/path/to/DataPipeline"

from galaxylens.data_pipeline.imaging.paths import get_raw_root, get_processed_root
from galaxylens.data_pipeline.imaging.io.file_discovery import find_source_files
from galaxylens.data_pipeline.imaging.preprocess_master import process_lens

# 1. 查看某透镜有哪些源文件
files = find_source_files("J2141-0001", "F814W", product_type="FLC")
print(f"找到 {len(files)} 个 FLC 文件")

# 2. 完整预处理
out = process_lens("J2141-0001", version_tag="v1", output_base=get_processed_root())
print(f"产物目录: {out}")
# 产物含 *_sci.fits / *_noise.fits / *_psf.fits / *_starmask.fits / processing_log.json
```

## 8. 环境变量速查

| 环境变量 | 用途 | 必需 |
|---|---|---|
| `GALAXYLENS_DATA_ROOT` | 数据根目录（raw/ + processed_v2/）| 是 |
| `GALAXYLENS_CONFIG_DIR` | configs/ 目录（默认仓库根 configs/）| 否 |
| `SEXTRACTOR_BIN` | SExtractor 二进制 | 否 |
| `TINYTIM_GRID_BASE` | TinyTim Massey 栅格 | 否 |
| `TINYTIM_BIN` | TinyTim 可执行目录 | 否 |
| `TINYTIM_BREATHING_CSV` | HST 呼吸模型 CSV | 否 |

透镜参数已从 `config.py` 硬编码迁移到 `configs/{disk,slacs}/*.yaml`，由 `galaxylens.config.loader.load_lens_config()` 加载为 `LensConfig` dataclass。
