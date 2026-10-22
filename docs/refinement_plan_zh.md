# galaxylens 提炼与发布方案（蓝图）

> 2026-10-20 定稿。本文档是 galaxylens 从「B 系列脚本搬运」升级为「可插拔组件 + 配置装配 +
> 门面 API」的落地蓝图。后续开发按本文档逐阶段落地，每阶段配回归测试。

*英文版见 [refinement_plan.md](refinement_plan.md).*

## 一、目标与定位

把 galaxylens 打造成像 autolens 一样的代码包：`pip install galaxylens` 后，用户通过
`import galaxylens as gl` 调用 API 完成拟合分析、输出结果、绘图。核心诉求：

1. **可插拔组件**：暗物质模型（gNFW/NFW/SIE…）、恒星质量模型（MGE 常数 M/L / 径向 M/L /
   多组分…）、动力学引擎（jampy/AGAMA/JAX Jeans…）、源模型（单 Sersic / 多 Sersic）各自独立，
   可自由搭配。
2. **配置驱动**：换星系、换方法 = 改 YAML，不改代码。
3. **门面 API**：`gl.fit(cfg)` 一行跑完，`result.beta_z` 直接读物理量，`result.plot_*()` 出图。
4. **跨平台**：Windows / WSL / Linux（CPU）可跑。

**核心原则（两条铁律）**：
- **接口一次设计好（可插拔），组件按需填充（每个组件有真实拟合背书）**——不一次性实现
  "所有模型 × 所有引擎"矩阵，避免过度抽象；D 系列提炼时验证第二个组件。
- **核心层是引擎，门面层是把手**——物理全在组件内，装配器零物理。

## 二、提炼现状回顾

lensfit 现有 17 个 .py（B1–B4 早期口径）：

| 状态 | 模块 | 说明 |
|---|---|---|
| 已提炼 | `psf/stars.py` | B1 场星 Moffat PSF |
| 已提炼 | `mge/fit.py` | B2 MGE 拟合（rbounds 去伪尖核坑已固化 ✓） |
| 已提炼 | `mge/deproject.py` | B3 去投影 |
| 已提炼 | `mge/io.py` | MGE 读写 |
| 已提炼 | `lens/subtract.py` | B4 Step1 galaxy-subtract ✓ |
| 已提炼 | `lens/rescale.py` | B4 噪声 rescale（f=√reduced χ²）✓ |
| **过时** | `lens/models.py` | 用 gNFWMCRLudlow（B4 早期），需更新到 gNFWVirialVelocity（B6/B7） |
| 已提炼 | `plotting/preview_imaging.py` | 成像预览 |
| **空** | `dynamics/` | B5 Jeans 动力学完全没进 |
| **空** | `fit/` | B6/B7 联合拟合完全没进 |
| **空** | `validation/` | 回归 + 简并扫描 |

**待更新到最终口径**：`lens/models.py` 的 gNFWMCRLudlow → gNFWVirialVelocity（v_vir 参数化 +
c_{−2} 自由 + q_h 椭率 + c-M prior）。

## 二·五、mock 管线提炼（上游图像生成，2026-10-20 补充）

mock_pipeline（`truth/` `selection/` `skirt/`）是**上游**：生成带真值的 mock FITS，喂
`api/dataset` → lensfit 拟合闭合。与 lensfit 提炼正交，但有一个**接口契约**必须一并定死：

**接口契约（mock ↔ 拟合）**：mock 产出 = `data/noise/psf` 三 FITS，单位 counts/s，像素尺度与
真实数据一致（如 J1331 0.0455″/px），使 `api/dataset.py` 用**同一套代码**加载 mock 和真实数据。
噪声图为恒定背景 RMS（median=SIGMA、std≈0），勿在 crop/pad 时零填充污染边框。

**本轮提炼 4 项（2026-10-20 J1331 工作成果固化）**：

| # | 函数/能力 | 目标模块 | 锚值 |
|---|---|---|---|
| 1 | `major_axis_angle()`：特征分解测盘长轴，消除 ±90° 简并 | `mock_pipeline/truth/measures.py` | 342537 = −21.32° |
| 2 | `counter_rotation_angle()`：KDC 反旋核夹角硬判据（≈180°） | `mock_pipeline/truth/kinematics.py` | 342537 = 115.7° |
| 3 | 单尺度归一化（颜色比守恒）+ data/noise/psf 分离输出 | `mock_pipeline/skirt/imaging.py` | F814W 锚 0.317 |
| 4 | 4 高斯 Moffat MGE PSF（论文 Table 2 normpsf/sigmapsf） | `lensfit/psf/`（**共享**：mock 卷积 + 拟合去卷积都用） | δ=0.038/0.085/0.169/0.487″ |

**登记 J1331 为第二个目标**：`selection/config.py` 加 `J1331 = TargetConfig(...)`（snap90、
M*∈[1e11,3e11]、σ≥190、incl=70°、obs=J1331_OBS、并合判据**反转**选「有并合」），登记进
`BUILTIN_TARGETS` —— 验证「换目标零代码改动」的接口通用性（J2141 是第一个，J1331 是第二个）。

**已知债（已核实）**：`selection/merger.py` **已同步** mpb.hdf5 主支质量跳变（major>+30%/
minor>+10%，含分支切换 artifact 检测），无需修。早前 migration skill 里「merger.py 没同步」的
note 是过时信息，已随本次更正。

**D 系列（J1331）物理维度补漏**：roadmap 阶段 5 只写了 J1331 组件「`nfw` + `mge_radial_ml`」，
漏了 J1331 区别于 J2141 的标志性物理——**反旋转核 KDC + 切向各向异性 βz=−0.4**。mock 侧
KDC 检测（item 2）与拟合侧 βz 符号是同一件事的两个侧面，提炼时应对应起来（KDC≈180° ⟺ 切向
βz<0），否则 D 系列拟合组件会「复现了 J1331 的质量剖面、却漏了它的动力学标志」。

## 三、两层架构

```
galaxylens/
├── mock_pipeline/           # 上游：mock 图像生成（truth/selection/skirt，喂 api/dataset）
├── lensfit/                 # 核心层：物理引擎（不含用户交互）
│   ├── models/              #   可插拔模型组件
│   │   ├── dm/              #     暗物质：gnfw_virial、gnfw_ludlow、nfw…
│   │   ├── stellar/         #     恒星质量：mge_const_ml、mge_radial_ml…
│   │   └── source/          #     源：single_sersic、multi_sersic…
│   ├── dynamics/            #   动力学引擎：jampy、agama、jax_jeans…
│   ├── fit/                 #   装配器：JointAnalysis
│   ├── pipeline/            #   配置装配：assemble（读 YAML → 组合）
│   ├── mge/ psf/ lens/      #   已有（B1–B4）
│   └── validation/          #   回归 + 简并扫描
├── api/                     # 门面层：用户面向的 API
│   ├── dataset.py           #   load_dataset（FITS → Imaging + mask + PSF + rescale）
│   ├── search.py            #   Nautilus 封装（local / slurm 双后端）
│   ├── result.py            #   Result 对象（直接读物理量，不挖 instance）
│   ├── derived.py           #   派生量：f_dm、M_2D(<R_E)、M*
│   ├── config_loader.py     #   YAML → PipelineConfig
│   └── plotting/            #   corner（加权）、fit、残差、动力学对比
└── __init__.py              #   门面入口：gl.fit / gl.load_config / gl.load_dataset
```

- **核心层**：组件 + 引擎 + 装配 + Analysis，纯物理，可单测，不含"用户怎么调"。
- **门面层**：把核心层包成傻瓜式 API。
- **result.py + derived.py 是相对 autolens 的增值点**：autolens 要用户自己挖
  `instance.galaxies.lens.dark.v_vir`、自己算 βz=1−ratio²、自己算 f_dm；galaxylens 直接给
  `beta_z` / `m_star` / `f_dm()`，把 B 系列踩出的「参数→物理量」换算（CALIB 桥、加权
  percentile、f_dm 反算）固化进去。

## 四、核心层接口设计

### 4.1 基础数据结构（所有接口的公共语言）

```python
@dataclass(frozen=True)
class Geometry:                     # 固定观测量，不是自由参数
    zlens: float
    zsrc: float
    inclination_deg: float          # i=78.5(J2141) / 70(J1331)
    da_mpc: float
    pc_per_arcsec: float
    re_arcsec: float | None = None  # 供 f_dm/M_2D 反算口径
    rd_arcsec: float | None = None

@dataclass(frozen=True)
class KinData:                      # 运动学观测量
    xbin: np.ndarray
    ybin: np.ndarray
    vrms: np.ndarray
    vrms_err: np.ndarray
    vcirc: float | None = None      # 可选 Hα V_circ(2.2Rd)
    vcirc_err: float | None = None

MGE = tuple[np.ndarray, np.ndarray, np.ndarray]   # (surf, sigma, q_obs)

@dataclass(frozen=True)
class Calibration:                  # 固化 B6 的 CALIB 坑：ml_dyn = ml_img * calib
    calib: float
```

### 4.2 MassComponent（DM + 恒星统一）

```python
class MassComponent(ABC):
    name: ClassVar[str]
    def attach(self, lens: af.Model, priors: PriorSet, geom: Geometry) -> None: ...
    def dyn_mge(self, instance, geom: Geometry, calib: Calibration) -> MGE: ...
    @property
    def n_free_params(self) -> int: ...
```

- DM 和恒星统一，因为两者在成像里都是 deflection、动力学里都是 MGE 表面密度，装配器只
  "遍历 components 拼 surf_pot"，换模型 = 换组件，装配器零改动。
- `dyn_mge` 从 instance 读（不是从 model 读），因为拟合后参数值在 instance 里，动力学每次
  likelihood 调用要现算 MGE。

### 4.3 SourceModel / DynamicsEngine

```python
class SourceModel(ABC):             # 源只作 light，不作 mass
    name: ClassVar[str]
    def attach(self, source: af.Model, priors: PriorSet) -> None: ...
    @property
    def n_free_params(self) -> int: ...

class DynamicsEngine(ABC):
    name: ClassVar[str]
    def chi2(self, model_lum: MGE, model_pot: MGE,
             ratio: float, geom: Geometry, kin: KinData) -> tuple[float, dict]: ...
```

- `ratio = sqrt(1 - beta_z)` 作为显式参数传入 engine，**不再用 B6 的零质量 Gaussian 容器 hack**。

### 4.4 先验与配置（组合的菜谱）

```python
@dataclass
class PriorSpec:                    # c-M prior 参数化：None=Uniform，否则 GaussianPrior
    kind: str = "gaussian"
    mean: float = 5.9               # D&M 中心
    sigma: float = 1.49             # 0.11 dex 线性化
    lower: float = 3.0
    upper: float = 20.0

@dataclass
class PriorSet:                     # 所有组件 attach 时从这里取先验
    ml_img: tuple[float, float]
    v_vir: tuple[float, float]
    gamma: tuple[float, float]
    q_h: tuple[float, float]
    ratio: tuple[float, float]
    c_2: PriorSpec | None = None

@dataclass
class PipelineConfig:               # 一份 YAML = 一份复现菜谱
    geometry: Geometry
    kinematics: KinData
    dark: str                       # "gnfw_virial" | "gnfw_ludlow" | "nfw" | ...
    stellar: str                    # "mge_const_ml" | "mge_radial_ml" | ...
    source: str                     # "single_sersic" | ...
    dynamics: str                   # "jampy" | "agama" | ...
    priors: PriorSet
    noise_rescale: bool = True
    subtract_stellar: bool = True
```

### 4.5 装配器 + 联合 Analysis

```python
def assemble(cfg: PipelineConfig) -> tuple[af.Collection, JointAnalysis]:
    """读配置 → 实例化组件 → 拼 Collection + JointAnalysis。不含物理。"""
    ...

class JointAnalysis(al.AnalysisImaging):
    def log_likelihood_function(self, instance):
        chi2_img = self.fit_from(instance).chi_squared
        # 遍历 components 拼 model_pot，调 engine.chi2
        ...
        nu_img = self.n_unmasked - sum(c.n_free_params for c in ...)   # 成像自由度
        nu_dyn = self.kinematics.n_points - sum(c.n_free_params for c in self.components)
        return -0.5 * (chi2_img / nu_img + chi2_dyn / nu_dyn)
```

`JointAnalysis` 完全通用：不知道 DM 是 gNFW 还是 NFW、恒星是单 M/L 还是径向 M/L，只
"遍历拼 MGE → 调 engine"。B 系列和 D 系列的差异全部收敛到 PipelineConfig 的几行 YAML。

### 4.6 关键设计决策（为什么）

1. **DM 和恒星统一成 MassComponent**：装配器不区分两者，换模型 = 换组件，零改动。
2. **ratio 显式化**：它只属于动力学，作为 engine.chi2 独立参数（取值仍从 instance 的
   anisotropy 载体读）。
3. **Calibration 独立成对象**：B6 的 CALIB 坑（成像/动力学 M/L 不同）强制每个恒星组件在
   dyn_mge 里显式过桥。
4. **c-M prior 是 PriorSpec 配置，不是组件硬编码**：同一 gnfw_virial 靠配置在 B6（Uniform）
   和 B7（D&M prior）之间切换。
5. **装配器零物理**：所有物理（zeta 积分、peak 公式、βz 换算）在组件内，assemble 只做
   "读配置→attach→拼 Collection"。

## 五、门面层模块

| 模块 | 职责 | 对应 autolens |
|---|---|---|
| `api/dataset.py` | FITS → Imaging + 圆形/恒星 mask + PSF + rescale 一体化 | `al.Imaging.from_fits()` |
| `api/search.py` | Nautilus 封装（n_live/n_eff/路径/local vs slurm 双后端） | `af.Nautilus()` |
| `api/result.py` | Result 对象：直接读 βz/c/γ/M*/f_dm | `result.instance`（但需挖字段） |
| `api/derived.py` | f_dm(2.2Rd)、M_2D(<R_E)、M* 反算（B6 b6_fdm.py 固化） | 无（autolens 不提供） |
| `api/plotting/` | corner（加权 percentile）、fit、残差、动力学 v_rms 对比 | `result.plot_...` |
| `api/config_loader.py` | YAML → PipelineConfig（复用 config/ 的 YAML+dataclass） | 无 |
| 顶层 `__init__.py` | `gl.fit / gl.load_config / gl.load_dataset` | `import autolens as al` |
| （可选）CLI | `galaxylens fit config.yaml` | autolens CLI |

## 六、使用流程（新星系）

两段：**数据准备（一次性）** + **配置拟合（YAML + gl.fit）**。

数据准备（gl 帮一部分）：成像 FITS（data_pipeline）→ `gl.load_dataset()`；恒星 MGE →
`gl.fit_mge()`（rbounds 坑已固化）；几何填 YAML；运动学数字化（gl 帮不了，手动）。

```python
import galaxylens as gl

cfg = gl.load_config("configs/disk/j1331_trick.yaml")
result = gl.fit(cfg)                        # 装配 → 采样 → 后处理 → 落盘

print(result.beta_z)                        # 直接读物理量
print(result.c200, result.m_star, result.f_dm(2.2 * cfg.rd))
result.plot_corner("j1331_corner.png")
result.plot_fit("j1331_fit.png")
result.plot_dynamics("j1331_vrms.png")
```

**关键点——先验必须进 YAML**：J1331 是切向（βz=−0.4，ratio≈1.18），J2141 是径向
（βz=+0.65，ratio≈0.59），同一 ratio 参数两个星系的物理范围相反。凡是随星系/方法变化的
（几何、先验范围、c-M 中心、数据路径）进 YAML；凡是不变的（zeta 积分、peak 公式、rbounds、
加权 percentile）固化在组件里。

## 七、跨平台策略

**基础**：依赖链全纯 Python（numpy/scipy/astropy/pandas/matplotlib + PyAutoLens + jampy +
mgefit + nautilus），CPU 三平台可跑。路径已用 7 个环境变量解耦（GALAXYLENS_DATA_ROOT 等）。

**三个诚实边界（写进 README 声明）**：
1. PyAutoLens 官方主要测 Linux/macOS，Windows 是"能用但非一级支持"（galaxylens 继承此支持度）。
2. BLAS/LAPACK 数值非确定性 ~1%（B6 CALIB 1427 vs 1157），跨平台对比以同 BLAS 为准。
3. slurm 集群调度仅 Linux；GPU/JAX 加速仅 Linux+NVIDIA。

**落地**：GitHub Actions CI 矩阵（Windows + Linux 双跑 import + 单次 likelihood + MGE 拟合
冒烟）；README 声明区；search.py local/slurm 双后端（YAML `backend: local` 切换）。

## 八、落地路线图（分阶段，每阶段配回归测试）

**阶段 0（上游，mock 管线）**：提炼 4 项（`major_axis_angle` / `counter_rotation_angle` /
单尺度归一化+data/noise/psf 分离 / 4 高斯 Moffat MGE PSF）+ 登记 J1331 为第二 TargetConfig +
merger.py 同步 mpb.hdf5。完成后 mock 侧能端到端出拟合-ready 双波段 mock，供阶段 3 之后做
拟合闭合（MF 系列）。

1. **核心层第一个组件组合**：`gnfw_virial` + `mge_const_ml` + `jampy` + `assemble` + `JointAnalysis`
   —— 用 B7 真实数字（CALIB=1157、βz=0.65、M*=1.437e11、V_circ=290）回归断言复现 likelihood。
2. **门面层**：dataset / search / result / derived / plotting / config_loader / 顶层入口。
3. **第一份完整 YAML**：J2141 Barnabè 复现，`gl.fit()` 全链路跑通 + 回归。
4. **更新过时模块**：`lens/models.py` gNFWMCRLudlow → gNFWVirialVelocity；补 `dynamics/`、`fit/`。
5. **第二个组件（D 系列提炼时）**：`nfw` + `mge_radial_ml`，验证接口通用性。
   ✅ **已完成（2026-10-21）**：提炼了 `nfw`（球对称 v200+c200）+ `mge_radial_ml`（径向 M/L，
   `lum_mge` 发光示踪物 ≠ `dyn_mge` 质量）+ `jampy_kdc`（κ 反旋转核 + βz 直接 + PSF）+
   `MeinConstraint`（单值爱因斯坦质量，修正了 D5 原脚本的 M_star 单位 bug）+ `scale_free.py`
   （四重像逐点重投影，D1）。并抽出 `LensingConstraint` 抽象（ImagingConstraint 像素级 /
   MeinConstraint 单值 / scale-free 四重像），JointAnalysis 成像项可插拔，使管线既支持像素级
   （B 系列弧）也支持逐点坐标级（D 系列四重像）。门面加 `gl.fit_positions` / `gl.fit_mein`。
   回归：tests/verify_lensfit_d1.py + verify_lensfit_d5.py + verify_facade_point.py。
6. **跨平台**：CI 矩阵 + README 声明 + search 双后端。
   ✅ **已完成（2026-10-21）**：search.py 完整落地 slurm 后端（`slurm_script` 渲染 +
   `submit_slurm` 提交，模板对齐已验证的 `examples/j2141_b7.slurm`）；facade `gl.fit`
   接 `backend=slurm`（渲染+写+sbatch），runner 强制 `backend=local`（防重提交）；SearchSpec
   加 `workdir`/`mem`；`.github/workflows/ci.yml`（Windows+Linux 核心+纯 Python，Linux 完整栈）；
   README/README_zh 加跨平台三边界；tests 的硬编码 WSL 路径改可移植 `Path(__file__).parents[1]`。

## 九、验证与发布规范

- **回归测试**：每个提炼模块配断言脚本，用 B 系列真实数字验证封装复现原脚本输出，跑通才算完成。
- **引用标注**：每个模块 docstring 标注引用（Cappellari 2002 MGE / Barnabè 2012 galaxy-subtract /
  Wyithe 2001 gNFW / Ludlow 2016 c-M / Dutton & Maccio 2014 / Cappellari 2020 jampy / Vasiliev 2019 AGAMA）。
- **英文代码**：延续现有 100% 英文化（docstring + 注释 + 字符串字面量）。
- **发布**：pyproject.toml 依赖声明 + MIT + 论文收尾时公开 + Zenodo DOI。
