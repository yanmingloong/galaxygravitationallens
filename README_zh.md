# galaxylens — 引力透镜建模库

**盘星系与椭圆星系的强引力透镜建模。**

核心科学目标：对盘星系与椭圆星系开展强引力透镜与恒星动力学的联合建模，约束其总质量分布、暗物质晕与恒星质光比，进而限制暗物质模型；并通过 mock 观测，将建模方法与宇宙学流体数值模拟（IllustrisTNG）的预言同空间望远镜（HST）实测对比加以检验。椭圆星系作为对比样本。

本库提供引力透镜 + 动力学的联合拟合管线：引力透镜模型（质量模型 + 光源模型，经光线追踪后与成像数据拟合）由 PyAutoLens 构建，恒星动力学模型由 jampy（Jeans Anisotropic MGE，对多高斯展开的恒星分布求解轴对称 Jeans 方程）构建，二者联合拟合以约束恒星质光比、暗物质晕（gNFW / NFW）与速度各向异性 β_z，进而得到爱因斯坦半径内的暗物质比例 f_DM。方法框架参考 Barnabè et al.（2012，SWELLS IV），实现上采用上述独立软件栈。

> 源自本科大学生创新项目**「盘星系引力透镜研究」**。

> **状态**：pre-alpha。

## 仓库

- Gitee: <https://gitee.com/longym8/galaxygravitationallens>
- GitHub: <https://github.com/yanmingloong/galaxygravitationallens>

两个平台镜像同一仓库，保持同步。

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
│   ├── lensfit/                 #   透镜拟合引擎（可插拔组件 + 配置装配）
│   │   ├── models/              #     可插拔模型组件（dm / stellar / source）
│   │   ├── dynamics/            #     jampy（主引擎）/ AGAMA / JAX Jeans（交叉验证）
│   │   ├── fit/                 #     JointAnalysis + 可插拔透镜约束
│   │   ├── pipeline/            #     配置装配（零物理）
│   │   ├── data/                #     读 processed → imaging + PSF 对象
│   │   ├── mge/                 #     MGE 拟合 + 去投影
│   │   ├── lens/                #     透镜模型 builder（盘 / 椭圆两套）
│   │   ├── plotting/            #     统一绘图（论文风格）
│   │   └── validation/          #     ground-truth 对照（mock 验证）
│   ├── api/                     #   高层 API：gl.fit / gl.analyze / gl.identify / gl.mock / gl.load_config / gl.load_dataset
│   └── sample_library/          #   观测样本库访问
├── configs/                     # 各透镜 YAML（disk/ + slacs/）
├── sample_library/              # 观测样本库数据
├── examples/                    # 集群运行模板（slurm）
├── notebooks/                   # 教程
├── tests/                       # 单元测试
└── docs/                        # 文档
```

## 功能清单

| 模块 | 功能 |
|---|---|
| `data_pipeline/imaging` | HST FLC/FLT 预处理 -> sci/noise/psf/starmask（SExtractor + TinyTim） |
| `config` (YAML + dataclass) | YAML -> dataclass 配置加载（configs/ 下 16 个透镜 YAML） |
| `sample_library` | 观测样本库访问（5 个 modeling_input CSV） |
| `data_pipeline/spectra` | 光谱（SDSS / KCWI / DEIMOS）——规划中 |
| `data_pipeline/other_obs` | 其他观测（NIRC2 K'-band 等）——规划中 |
| `lensfit` | 可插拔拟合引擎：gnfw_virial / nfw / mge_const_ml / mge_radial_ml / single_sersic / pixelized 组件；jampy / jampy_kdc 动力学；LensingConstraint imaging / mein / scale-free；联合拟合 + Trick+2016 逐点拟合 |
| `api` | 高层 API：gl.fit / gl.fit_positions / gl.fit_mein / gl.analyze / gl.mock / gl.load_config / gl.load_dataset |
| `mock_pipeline/truth` | ground-truth 提取：f_disk / f_dm（球 + 柱）/ beta_z / gNFW 晕拟合 |
| `mock_pipeline/lensing` | 真值质量透镜星系 + 爱因斯坦半径 + 弧光线追踪 + dataset 合并 |
| `mock_pipeline/selection` | TNG 盘透镜类似物筛选（J2141 + J1331 内建目标） |
| `mock_pipeline/skirt` | SKIRT 辐射转移 -> mock 图像（惯性张量对齐 + .ski + fake HST + 多波段 dataset） |

## 设计原则

- **按能力分，不按论文/星系类型分**——B/D 系列（盘）与 e 系列（椭圆）都调同一 `lensfit` 引擎，差异在透镜模型 builder、动力学引擎选择、配置。
- **数据生产与消费分离**——`data_pipeline` / `mock_pipeline` 生产数据，`lensfit` 消费。mock 拟合 = `lensfit` + ground-truth 对照（`validation/`）。
- **观测与模拟数据本质不同**（有无 ground truth），因此生产层分开，但共享拟合引擎。

## 安装

### Python 包

```bash
pip install -e .            # 核心 core（numpy/scipy/astropy/pyyaml/pandas）
pip install -e ".[lens]"    # + PyAutoLens 拟合栈
pip install -e ".[dynamics]" # + AGAMA / jampy / mgefit
```

### 外部工具（非 pip 可装）

| 工具 | 用途 | 环境变量 |
|---|---|---|
| PyAutoLens | 透镜成像 / 光线追踪 / 拟合 | pip，`[lens]` extra |
| jampy + mgefit | Jeans 各向异性动力学 | pip，`[dynamics]` extra |
| Nautilus | 嵌套采样 | pip，随 PyAutoLens |
| SKIRT | 蒙特卡洛辐射转移（mock 图像） | `SKIRT_BIN` |
| SExtractor | 点源探测 / 测光 | `SEXTRACTOR_BIN` |
| TinyTim | HST PSF 生成 | `TINYTIM_BIN`、`TINYTIM_GRID_BASE` |

### 环境变量

| 变量 | 指向 | 用于 |
|---|---|---|
| `GALAXYLENS_DATA_ROOT` | 观测数据根（raw/ + processed_v2/） | data_pipeline、lensfit |
| `GALAXYLENS_SAMPLE_LIB` | 样本库根（01_raw ~ 05_exports） | sample_library |
| `GALAXYLENS_OUTPUT_ROOT` | 结果输出根（不进仓库） | facade / 拟合 |
| `GALAXYLENS_CONFIG_DIR` | configs/ 目录（可选，默认仓库 configs/） | 配置加载 |
| `TNG_API_KEY` | IllustrisTNG 下载 API key | mock_pipeline 筛选 |
| `SKIRT_BIN` | SKIRT 可执行文件 | mock_pipeline skirt |
| `SEXTRACTOR_BIN` | SExtractor 二进制 | data_pipeline imaging |
| `TINYTIM_BIN` / `TINYTIM_GRID_BASE` / `TINYTIM_BREATHING_CSV` | TinyTim 路径 | data_pipeline psf |

### 数据

仓库不含任何科学数据（`*.fits` / `*.hdf5` / `*.npz` 已 gitignore），需自行准备：

- **观测数据**：HST FITS（从 MAST 归档下载，或等价来源）。
- **模拟数据**：IllustrisTNG 粒子 HDF5（需 TNG API key）。
- **回归测试数据**（可选）：设 `GALAXYLENS_MF_ROOT` / `GALAXYLENS_TEST_*` 跑数据依赖测试，否则自动跳过。

## 快速开始

```bash
# 指向含 raw/ 和 processed_v2/ 的目录
export GALAXYLENS_DATA_ROOT=/path/to/observations

# 预处理一个透镜
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J2141-0001 --version-tag v1
```

完整指引见 [docs/data_pipeline.md](docs/data_pipeline.md)。

### 拟合（透镜 + 动力学）

```python
import galaxylens as gl

cfg = gl.load_config("configs/fit/j2141_barnabe.yaml")
result = gl.fit(cfg)                    # 装配 -> Nautilus -> 后处理
print(result.beta_z, result.m_star, result.f_dm(2.2 * cfg.pipeline.geometry.rd_arcsec))

gl.analyze(cfg)                         # 一键拟合后分析 + corner 图
```

完整拟合流程见 [docs/usage_guide.md](docs/usage_guide.md) Step 3。

### 识别（透镜类型 → 建模方案）

```python
plan = gl.identify("J2141-0001")     # LensPlan：disk / imaging / jampy_kdc
plan = gl.identify_from_image("j1630_f814w_sci.fits")  # 图像兜底：C+q -> 盘/椭圆
print(plan.lens_type, plan.constraint, plan.template_yaml)
```

`gl.identify` 读样本 catalog，推断透镜类型（盘/椭圆）与约束（弧/四重像），映射到建模模块 + 配方模板。`gl.identify_from_image` 是图像形状兜底——catalog 无形态时直接从 FITS 测光集中度 + 轴比——"识别 → 建模 → 拟合"链的第一环。

### 编排（自动 + 人工）

`gl.Pipeline` / `gl.Stage` 把可插拔组件串成有序运行，支持全自动、部分（`run(only=...)`）与区间（`run(start=..., stop=...)`）三种粒度，每步仍可独立调用。`gl.mock` 是预置的 mock 闭合流水线（select → truth → render → arc → fit → closure）；见 [docs/usage_guide.md](docs/usage_guide.md) Step 4。

### 教程（可跑 notebook）

`notebooks/` 下三个 `# %%` notebook 走通主链路——见 [docs/README_zh.md](docs/README_zh.md#教程notebooks-下的可跑--notebook)：

- `identify_demo.py` — 识别 → 配方（全离线）
- `fit_demo.py` — 配方 → 拟合 → 分析（合成 FITS smoke，离线）
- `mock_pipeline_demo.py` — TNG 选样 → SKIRT → fake HST

### Agent 技能

仓库内置 `galaxylens-pipeline` 技能
（[`.github/skills/galaxylens-pipeline/SKILL.md`](.github/skills/galaxylens-pipeline/SKILL.md)），
让 AI agent（如 Hermes、Claude Code）能够操作本库：门面 API、环境变量、
标准工作流、加权结果读取与操作坑。把 agent 指向该文件即可让它运行
`gl.fit` / `gl.analyze` / `gl.identify` / `gl.mock` 并诊断拟合结果。

## 噪声方法（data_pipeline/imaging）

`preprocess_master.py --noise-method` 为每个透镜系统一次生成三个噪声版本，三个版本共用同一套 science / PSF / starmask，仅噪声图不同：

| 方法 | 噪声图 | 说明 |
|---|---|---|
| `whtonly`  | σ = 1/√WHT | 纯 WHT，物理基准 |
| `rescaled` | 纯 WHT × 全局因子 | 预处理近似（scale = σ_bg / median(noise_raw_bg)）；阶段 1 首选 |
| `bugnoise` | √(1/WHT + SCI) | 信号相关吸收项；备用 |

盘星系中纯 WHT 噪声太小——模型系统误差（≳16 e⁻/s）远大于测量噪声（≈0.35 e⁻/s），χ²/px 高达 ≈2167。Barnabè+2012（SWELLS IV）用全局单一因子把 reduced χ² 拉到 1；`rescaled` 在预处理阶段近似这个因子，而拟合阶段（B4）走 Barnabè 正规两轮（`RESCALE=True`，f = √reduced χ²）。`bugnoise` 是信号相关替代方案，作为备用。

## 跨平台支持

galaxylens 是纯 Python（numpy/scipy/astropy/pyyaml/pandas + 可选 PyAutoLens /
jampy / mgefit / nautilus），Windows、WSL、Linux、macOS（CPU）均可运行。三个诚实边界：

1. **PyAutoLens 以 Linux/macOS 为主**：`autolens`/`autofit` 栈主要在 Linux 和
   macOS 上测试，Windows 上"能用但非一级支持"，galaxylens 的 `[lens]` 额外依赖
   继承此支持度。核心（numpy/scipy/astropy）与纯 Python 的 lensfit 模块三平台一致。
2. **BLAS/LAPACK 数值非确定性 ~1%**：MGE 分解（mgefit）在不同 BLAS 后端会收敛到
   略有差异的高斯集（B6/B7：CALIB 1427 vs 1157），但守恒乘积（CALIB × L_2D_REL）
   一致到 ~1%。跨平台对比应以守恒量为锚，不锚单个分解输出。
3. **slurm 集群调度仅 Linux；GPU/JAX 加速仅 Linux+NVIDIA**：`backend: slurm` 后端
   在 Linux 集群提交 SBATCH 作业；`backend: local` 后端到处可跑。CI 矩阵
   （`.github/workflows/ci.yml`）在 Windows + Linux 跑核心 + 纯 Python 测试，
   完整拟合栈只在 Linux 跑。

## 参考文献

本库所依托的方法与模型。

### 方法与模型

| 文献 | DOI | 参考内容 |
|---|---|---|
| Barnabè, M.; Dutton, A.A.; Marshall, P.J.; et al. 2012, MNRAS, 423, 1073 | [10.1111/j.1365-2966.2012.20934.x](https://doi.org/10.1111/j.1365-2966.2012.20934.x) | 透镜 + 动力学联合拟合方法（galaxy-subtracted） |
| Treu, T.; Dutton, A.A.; Auger, M.W.; et al. 2011, MNRAS, 417, 1601 | [10.1111/j.1365-2966.2011.19378.x](https://doi.org/10.1111/j.1365-2966.2011.19378.x) | SWELLS 巡天（盘透镜样本） |
| Trick, W.H.; van de Ven, G.; Dutton, A.A. 2016, MNRAS, 463, 3151 | [10.1093/mnras/stw2142](https://doi.org/10.1093/mnras/stw2142) | 四重像盘透镜 J1331+3628 |
| Cappellari, M. 2002, MNRAS, 333, 400 | [10.1046/j.1365-8711.2002.05412.x](https://doi.org/10.1046/j.1365-8711.2002.05412.x) | 多高斯展开（MGE） |
| Cappellari, M. 2020, MNRAS, 494, 4819 | [10.1093/mnras/staa1128](https://doi.org/10.1093/mnras/staa1128) | jampy Jeans 各向异性动力学 |
| Wyithe, J.S.B.; Turner, E.L.; Spergel, D.N. 2001, ApJ, 555, 504 | [10.1086/321437](https://doi.org/10.1086/321437) | gNFW 密度轮廓 |
| Navarro, J.F.; Frenk, C.S.; White, S.D.M. 1997, ApJ, 490, 493 | [10.1086/304888](https://doi.org/10.1086/304888) | NFW 密度轮廓 |
| Dutton, A.A.; Macciò, A.V. 2014, MNRAS, 441, 3359 | [10.1093/mnras/stu742](https://doi.org/10.1093/mnras/stu742) | 浓度–质量先验 |
| Wright, C.O.; Brainerd, T.G. 2000, ApJ, 534, 34 | [10.1086/308744](https://doi.org/10.1086/308744) | 解析投影 NFW 质量 |
| Evans, N.W.; Witt, H.J. 2003, MNRAS, 345, 1351 | [10.1046/j.1365-2966.2003.07057.x](https://doi.org/10.1046/j.1365-2966.2003.07057.x) | scale-free 透镜势 |
| Kochanek, C.S. 1991, ApJ, 373, 354 | [10.1086/170057](https://doi.org/10.1086/170057) | 源平面 χ² + 放大加权 |
| Vasiliev, E. 2019, MNRAS, 482, 1525 | [10.1093/mnras/sty2672](https://doi.org/10.1093/mnras/sty2672) | AGAMA 动力学（交叉验证） |
| Nightingale, J.W.; Dye, S.; Massey, R.J. 2018, MNRAS, 478, 4738 | [10.1093/mnras/sty1264](https://doi.org/10.1093/mnras/sty1264) | PyAutoLens 透镜成像 |
| Stetson, P.B. 1987, PASP, 99, 191 | [10.1086/131977](https://doi.org/10.1086/131977) | DAOStarFinder 点源探测 |

### mock 图像管线

| 文献 | DOI | 参考内容 |
|---|---|---|
| Camps, P.; Baes, M. 2020, A&C, 31, 100381 | [10.1016/j.ascom.2020.100381](https://doi.org/10.1016/j.ascom.2020.100381) | SKIRT 辐射转移 |
| Jones, A.P.; Köhler, M.; Ysard, N.; et al. 2017, A&A, 602, A46 | [10.1051/0004-6361/201630225](https://doi.org/10.1051/0004-6361/201630225) | THEMIS 尘埃模型 |
| Bruzual, G.; Charlot, S. 2003, MNRAS, 344, 1000 | [10.1046/j.1365-8711.2003.06897.x](https://doi.org/10.1046/j.1365-8711.2003.06897.x) | 星族合成（BC03） |
| Chabrier, G. 2003, PASP, 115, 763 | [10.1086/376392](https://doi.org/10.1086/376392) | Chabrier IMF |
| Nelson, D.; Pillepich, A.; Springel, V.; et al. 2019, MNRAS, 490, 3234 | [10.1093/mnras/stz2306](https://doi.org/10.1093/mnras/stz2306) | IllustrisTNG 模拟 |
| The Astropy Collaboration; Price-Whelan, A.M.; et al. 2022, ApJ, 935, 167 | [10.3847/1538-4357/ac7c74](https://doi.org/10.3847/1538-4357/ac7c74) | Astropy 软件 |

## AI 使用情况说明

本仓库使用 Hermes Agent 协助整理，大模型：deepseek-v4.1-flash。

## 参与贡献

欢迎在托管平台（Gitee / GitHub）提交 Issue（功能需求、bug 报告）与 Pull Request。参见 [CONTRIBUTING_zh.md](CONTRIBUTING_zh.md)。

## 许可

MIT 许可 — 见 [LICENSE](LICENSE)。
