# galaxylens — 引力透镜建模库

**盘星系与椭圆星系的强引力透镜建模。**

核心科学目标：利用星系形成的流体数值模拟结果，研究盘星系的强引力透镜效应，与空间望远镜（HST）观测结果比较，检验数值模拟预言，限制暗物质模型与暗物质粒子性质；椭圆星系作为对比样本。

> **状态**：pre-alpha。仓库骨架已建立，代码正从内部研究项目逐步迁移。

English docs: [README.md](README.md).

## 架构

```
galaxygravitationallens/
├── src/galaxylens/              # Python 包 (src layout)
│   ├── config/                  #   YAML → dataclass 配置加载（共享）
│   ├── data_pipeline/           #   观测数据生产
│   │   ├── imaging/             #     HST 透镜图像（FLC/FLT 预处理）
│   │   ├── spectra/             #     光谱（SDSS / KCWI / DEIMOS）
│   │   └── other_obs/           #     其他观测（NIRC2 K'-band 等）
│   ├── mock_pipeline/           #   模拟数据生产 + 筛选
│   │   ├── selection/           #     TNG 盘透镜类似物筛选
│   │   ├── skirt/               #     SKIRT 辐射转移 → mock 图像
│   │   └── truth/               #     ground-truth 提取（f_disk 等）
│   ├── lensfit/                 #   透镜拟合引擎（观测 + mock 共用）
│   │   ├── data/                #     读 processed → imaging + PSF 对象
│   │   ├── mge/                 #     MGE 拟合 + 去投影
│   │   ├── lens/                #     透镜模型 builder（盘 / 椭圆两套）
│   │   ├── dynamics/            #     jampy（主引擎）/ AGAMA / JAX Jeans（交叉验证）
│   │   ├── fit/                 #     lens-only / dynamics-only / joint 似然
│   │   ├── plotting/            #     统一绘图（论文风格）
│   │   └── validation/          #     ground-truth 对照（mock 验证）
│   └── sample_library/          #   观测样本库访问
├── configs/                     # 各透镜 YAML（disk/ + slacs/）
├── sample_library/              # 观测样本库数据
├── notebooks/                   # 教程
├── tests/                       # 单元测试
└── docs/                        # 文档
```

## 迁移状态

代码正从内部研究项目逐步迁移。

| 模块 | 状态 |
|---|---|
| `data_pipeline/imaging` | ✅ 已迁移（21 文件；psf 拆子包；路径 + SExtractor + TinyTim 环境变量解耦）|
| `config` (YAML + dataclass) | ✅ 已迁移（paths/schema/loader/defaults；configs/ 下 16 个透镜 YAML）|
| `sample_library` | ✅ 已迁移（构建管线 + 访问；5 个 modeling_input CSV）|
| `data_pipeline/spectra` | ⏳ 待完善后迁移 |
| `data_pipeline/other_obs` | ⏳ 待完善后迁移 |
| `lensfit` (data/mge/lens/dynamics/fit/plotting/validation) | ⏳ 待迁移 |
| `mock_pipeline/truth` (kinematics/measures/cosmology) | ✅ 已迁移（纯函数 + 回归测试，348489 f_disk=0.643 复现） |
| `mock_pipeline/selection` (scan/filter/merger/download/verify/catalog) | ✅ 已迁移（按目标配置，筛选判据数据化） |
| `mock_pipeline/skirt` (convert/render/imaging) | ✅ 已迁移（惯性张量对齐 + .ski 生成 + fake HST） |

## 设计原则

- **按能力分，不按论文/星系类型分**——B/D 系列（盘）与 e 系列（椭圆）都调同一 `lensfit` 引擎，差异在透镜模型 builder、动力学引擎选择、配置。
- **数据生产与消费分离**——`data_pipeline` / `mock_pipeline` 生产数据，`lensfit` 消费。mock 拟合 = `lensfit` + ground-truth 对照（`validation/`）。
- **观测与模拟数据本质不同**（有无 ground truth），因此生产层分开，但共享拟合引擎。

## 安装

```bash
pip install -e .            # 核心 core（numpy/scipy/astropy/pyyaml/pandas）
pip install -e ".[lens]"    # + PyAutoLens 拟合栈
pip install -e ".[dynamics]" # + AGAMA / jampy / mgefit
```

## 快速开始

```bash
# 指向含 raw/ 和 processed_v2/ 的目录
export GALAXYLENS_DATA_ROOT=/path/to/DataPipeline

# 预处理一个透镜
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J2141-0001 --version-tag v1
```

完整指引见 [docs/data_pipeline.md](docs/data_pipeline.md)。

## 噪声方法（data_pipeline/imaging）

`preprocess_master.py --noise-method` 为每个透镜系统一次生成三个噪声版本，三个版本共用同一套 science / PSF / starmask，仅噪声图不同：

| 方法 | 噪声图 | 说明 |
|---|---|---|
| `whtonly`  | σ = 1/√WHT | 纯 WHT，物理基准 |
| `rescaled` | 纯 WHT × 全局因子 | 预处理近似（scale = σ_bg / median(noise_raw_bg)）；阶段 1 首选 |
| `bugnoise` | √(1/WHT + SCI) | 信号相关吸收项；备用 |

盘星系中纯 WHT 噪声太小——模型系统误差（≳16 e⁻/s）远大于测量噪声（≈0.35 e⁻/s），χ²/px 高达 ≈2167。Barnabè+2012（SWELLS IV）用全局单一因子把 reduced χ² 拉到 1；`rescaled` 在预处理阶段近似这个因子，而拟合阶段（B4）走 Barnabè 正规两轮（`RESCALE=True`，f = √reduced χ²）。`bugnoise` 是信号相关替代方案，作为备用。

## 许可

MIT
