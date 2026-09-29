# galaxylens 引力透镜建模库

> **Strong gravitational lens modeling for disk and elliptical galaxies.**

盘星系与椭圆星系的强引力透镜建模。
Strong gravitational lens modeling for disk and elliptical galaxies.

核心科学目标：利用星系形成的流体数值模拟结果，研究盘星系的强引力透镜效应，与空间望远镜（HST）观测结果比较，检验数值模拟预言，限制暗物质模型与暗物质粒子性质；椭圆星系作为对比样本。
Core scientific goal: use galaxy-formation hydrodynamical simulations to study strong lensing by disk galaxies, compare with space-telescope observations, test simulation predictions, and constrain dark matter models and the nature of dark matter particles. Elliptical galaxies serve as the comparison sample.

> **状态 Status**：pre-alpha。仓库骨架已建立，代码正从内部研究项目逐步迁移。
> **Status**: pre-alpha. Repository skeleton established; module code is being migrated from the internal research project incrementally.

## 架构 Architecture

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
│   │   ├── dynamics/            #     JAX Jeans / AGAMA / jampy（统一接口）
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

## 迁移状态 Migration status

代码正从内部研究项目逐步迁移。Code is being migrated from the internal research project incrementally.

| 模块 Module | 状态 Status |
|---|---|
| `data_pipeline/imaging` | ✅ 已迁移（21 文件；psf 拆子包；路径 + SExtractor + TinyTim 环境变量解耦）|
| `config` (YAML + dataclass) | ✅ 已迁移（paths/schema/loader/defaults；configs/ 下 16 个透镜 YAML）|
| `sample_library` | ✅ 已迁移（构建管线 + 访问；5 个 modeling_input CSV）|
| `data_pipeline/spectra` | ⏳ 待完善后迁移 |
| `data_pipeline/other_obs` | ⏳ 待完善后迁移 |
| `lensfit` (data/mge/lens/dynamics/fit/plotting/validation) | ⏳ 待迁移 |
| `mock_pipeline` (selection/skirt/truth) | ⏳ 待迁移 |

## 设计原则 Design principles

- **按能力分，不按论文/星系类型分**——B/D 系列（盘）与 e 系列（椭圆）都调同一 `lensfit`，差异在透镜模型 builder、动力学引擎选择、配置。
  **Code is split by capability, not by paper or galaxy type.** B/D series (disk) and e series (elliptical) all call the same `lensfit` engine; the difference lives in lens model builders, dynamics engine choice, and configuration.
- **数据生产与消费分离**——`data_pipeline`/`mock_pipeline` 生产数据，`lensfit` 消费。mock 拟合 = `lensfit` + ground-truth 对照（`validation/`）。
  **Data production vs consumption are separated.** `data_pipeline` / `mock_pipeline` produce data; `lensfit` consumes it. Mock fitting = `lensfit` + ground-truth comparison (`validation/`).
- **观测与模拟数据本质不同**（有无 ground truth），因此生产层分开，但共享拟合引擎。
  **Observational vs simulated data are different in nature** (no ground truth vs ground truth), so they are separated at the production layer but share the fitting engine.

## 安装 Installation

```bash
pip install -e .            # 核心 core（numpy/scipy/astropy/pyyaml/pandas）
pip install -e ".[lens]"    # + PyAutoLens 拟合栈
pip install -e ".[dynamics]" # + AGAMA / jampy / mgefit
```

## 快速开始 Quick start

```bash
# 指向含 raw/ 和 processed_v2/ 的目录
# point to the directory containing raw/ and processed_v2/
export GALAXYLENS_DATA_ROOT=/path/to/DataPipeline

# 预处理一个透镜 preprocess one lens
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J2141-0001 --version-tag v1
```

完整指引见 docs/data_pipeline.md。See docs/data_pipeline.md for the full guide.

## 许可 License

MIT
