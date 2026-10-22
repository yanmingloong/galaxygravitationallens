# 变更日志

*英文版见 [CHANGELOG.md](CHANGELOG.md).*

本文件记录项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，本项目遵循[语义化版本 SemVer](https://semver.org/spec/v2.0.0.html)。

## [Unreleased]

### 新增

- **组件化 `lensfit` 核心层 + `api` 门面层**（2026-10-21，蓝图阶段 1-4）：把拟合引擎重构为可插拔模型组件 + 配置驱动装配器。`lensfit.structures`（Geometry/KinData/Calibration/PriorSet/PipelineConfig）；`lensfit.models`（MassComponent/SourceModel/DynamicsEngine 抽象基类；`dm.gnfw_virial` Wyithe-2001 gNFW（v_vir/c_-2/gamma/q_h 参数化）+ `gnfw_virial_profile`、`stellar.mge_const_ml`、`source.single_sersic`）；`lensfit.dynamics.jampy`（Cappellari 2020，各向异性作显式参数 `ratio=√(1−βz)`）；`lensfit.fit.joint`（JointAnalysis，log L = −0.5·(χ²_img/ν_img + χ²_dyn/ν_dyn)）；`lensfit.pipeline.assemble`（组件注册表，零物理）。新增门面 `galaxylens.api`（config_loader / dataset / search(Nautilus) / result / derived / plotting.corner / facade）；`result.py` 直接从 samples.csv 读 βz / c_-2 / γ / v_vir / q_h / M* / f_dm 加权中位数。首份拟合配方 `configs/fit/j2141_barnabe.yaml`；`gl.fit()` 在龙集群复现 B7（job 128011：log Z = −8.04，全部参数在 B7 ±2.5% 内）。`lensfit.lens.models` 从 `gNFWMCRLudlow`（B4）更新为 `GNFWVirialVelocity`。回归：verify_lensfit_joint_b7 / verify_api / verify_lensfit_lens / verify_fit_smoke。
- **第二组件组 + 逐点（坐标级）拟合**（2026-10-21，阶段 5）：`lensfit.models.dm.nfw`（球对称 v200+c200）、`lensfit.models.stellar.mge_radial_ml`（径向 M/L，发光示踪物 ≠ 质量）、`lensfit.dynamics.jampy_kdc`（κ 反旋转核 + 直接 βz + PSF 卷积）、`lensfit.lens.scale_free`（四重像逐点重投影）。抽象出 `lensfit.fit.constraints.LensingConstraint`（ImagingConstraint 像素级 / MeinConstraint 单值爱因斯坦质量 / scale-free 四重像），使管线既支持像素级（B 系列弧）也支持坐标级（D 系列四重像）拟合；新增 `gl.fit_positions` / `gl.fit_mein`。回归：verify_lensfit_d1 / verify_lensfit_d5 / verify_facade_point。
- **跨平台 + slurm 后端 + CI**（2026-10-21/22，阶段 6）：`api.search` slurm 后端（`slurm_script` + `submit_slurm`，模板对齐已验证的 `examples/j2141_b7.slurm`）、`SearchSpec.workdir/mem`、`api.runner` CLI（强制 `backend=local`）、`.github/workflows/ci.yml`（Windows+Linux 核心/纯 Python，完整栈仅 Linux）、测试路径可移植、README 跨平台声明。无头运行时禁用 autofit 自带 anesthetic corner 图（`_disable_autofit_corner_plot`）——仓库自己的加权 corner 读 samples.csv 并用 Agg 后端。

- `mock_pipeline` J1331 支持 + KDC/roll/单尺度（2026-10-20）：`truth.major_axis_angle`（特征分解测盘长轴 roll，消除 0.5*arctan2 公式的 ±90° 简并）、`truth.counter_rotation_angle`（核-外盘角动量夹角 KDC 检验，~180° 硬判据；热核球弱反旋 ≠ 相干反旋）、`skirt.single_scale_normalize` + `skirt.make_mock_dataset`（单光度量标保持 SED 颜色比供多波段联合拟合 + data/noise/psf 拟合-ready FITS；噪声图裁/填后生成，零填充不污染边框）、`lensfit.psf.moffat_mge_psf`（四高斯 Moffat MGE PSF，Trick+2016 Table 2）、`selection.config.J1331`（第二内建目标：snap90、M*∈[1e11,3e11]、σ≥190、倾角 70°、F435W/F814W）。
- `skirt` 观测配置数据化（2026-10-17）：新增 `BandConfig`/`MockObsConfig`（波段+仪器名+像素尺度+噪声+峰值 / 几何），`write_ski` 改收 `BandConfig`+`MockObsConfig`（不再查写死的 BANDS 字典，同滤光片不同仪器如 WFPC2 F814W 可表达），`make_fake_hst` 改收 `BandConfig`（去 F814W 专用默认常量 PX_HST/SIGMA/J_PEAK）。`TargetConfig` 加 `obs` 字段挂观测配置。`snap_to_a` 对未知快照改 raise（不再静默 a=1.0）。
- 翻新 `notebooks/mock_pipeline_demo.py` 为完整教程（8 段：配置/筛选/转换/.ski/SKIRT/fake HST/消光+测量；离线段完整可跑，重活段标注 TNG key / 粒子数据 / SKIRT 二进制前置条件）。
- `read_api_key` 改静默：import 不再打印 "key not found" 提示，改为实际发起下载/请求时按需打印（`_print_key_hint`）。
- `TargetConfig` 加 `incl` 字段 + `read_api_key()`（2026-10-17）：盘倾角进目标配置（J2141=78.5 deg、J1331=70 deg），换目标零代码改动，`skirt.render.write_ski` 从 target.incl 取。TNG API key 改从 env var `TNG_API_KEY` 或 `~/.tng_api_key` 文件读取（仓库不内置 key），未配置时一次性提示指向 docs/mock_pipeline.md 获取指引，`get()`/`download_cutout()` 无 key 快速失败。
- `mock_pipeline.truth` 模块（2026-10-14）：ground-truth 计算，含 kinematics（circularity → f_disk/B_D/盘轴/厚度）、measures（半光半径 / 投影椭圆率协方差口径 / 固有轴比 / 消光）、cosmology（Planck 2015 距离 + SIS θE + 宇宙时间 + TNG 单位换算）。附带回归测试（348489 f_disk=0.643、B/D=0.55 复现）。修正 tng_to_skirt 的 H0_GYR 笔误（0.06774→0.06928）。
- `mock_pipeline.selection` 模块（2026-10-14）：TNG 盘透镜类似物筛选，含 scan（全量扫描，re_star_kpc ×a 修正）、filter（L1 判据数据化）、merger（L2 mpb.hdf5 主支质量跳变）、download（3字段/全字段合并）、verify（L3 disk_kinematics）、catalog（L3→catalog 自动生成，替代手工旧脚本）、run（--target 端到端）。筛选判据进 SelectionCriteria dataclass，换目标零代码改动。

- `mock_pipeline.skirt` 模块（2026-10-14）：SKIRT 辐射转移，含 convert（TNG 粒子 → .dat，惯性张量盘轴对齐，不用 catalog.csv 的旧 L_hat）、render（.ski 生成 + skirt 运行，SKIRT_BIN 环境变量）、imaging（fake HST + 消光图）。修正 convert 的 write_dust a 参数 bug + cosmic_time 向量化。

## [0.1.0] - 2026-09-24

### 新增

- 仓库骨架：包布局（`src/galaxylens/`）、模块目录、pyproject、README、LICENSE、CHANGELOG、.gitignore。
- 架构决策：单仓 monorepo、src 布局、按能力分（不按论文/星系类型）、数据生产与消费分离、mock 拟合复用 `lensfit` + ground-truth 对照。
- `data_pipeline.imaging` 模块迁移（21 文件；psf 拆子包；路径/SExtractor/TinyTim 环境变量解耦）。
- `config` 模块（YAML + dataclass；16 透镜 YAML）。
- `sample_library` 模块迁移（构建管线 + 5 个成品 CSV）。
