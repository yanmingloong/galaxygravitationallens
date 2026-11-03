# 变更日志

*英文版见 [CHANGELOG.md](CHANGELOG.md).*

本文件记录项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，本项目遵循[语义化版本 SemVer](https://semver.org/spec/v2.0.0.html)。

## [Unreleased]

### 新增

- **编排层：`Pipeline` + `Stage`、`gl.analyze` 与 `gl.mock`**（2026-11-03）：在可插拔组件之上加一层薄编排（自动模式），`api.pipeline` 提供 `Stage`（单个命名组件调用）+ `Pipeline`（有序步骤共享 `ctx` 字典），`run(start/stop/only)` 支持全自动/部分/区间运行；每步仍可独立调用（人工模式）。`api.analysis.analyze`（暴露为 `gl.analyze`）是一键拟合后分析：六质量参数加权后验、派生量（βz / M* / median 参数处 f_dm / θ_E / γ_tot）、先验贴边检查（2.5/97.5% 是否落进边界 1% 内）、corner 图。`mock_pipeline.orchestrate.build_mock_pipeline`（暴露为 `gl.mock`）把 mock 闭合链 select → pick → truth → render → make_arc → mock_fit → closure 串成 `Pipeline`，每步薄包装现有 `selection`/`skirt`/`lensing`/`validation` 模块并带清晰前置检查（重步骤需 TNG 数据 + SKIRT 二进制）。回归：verify_pipeline、verify_mock_orchestrate。注意：`analyze` 的 f_dm 用 median 参数算一次，不用 `Result.f_dm` 的 O(N) 逐样本循环（后者每样本重做晕 MGE 拟合，几千样本要几分钟）。

- **透镜识别层：`gl.identify`**（2026-11-03）：`sample_library.identify` 补上缺失的"识别"一环——从 catalog 行推断透镜类型（盘/椭圆，靠 `lens_type_label` + `n_sersic` + `B_D_ratio`）与透镜约束（弧/四重像，靠 `n_images` + 已知四重像名单），再映射到建模模块 + 配方模板（`LensPlan`）。`gl.identify(name)` 是门面入口；`write_recipe(plan, params)` 复制模板 + 覆盖键（支持点号嵌套）产出可跑 YAML。锚点：J2141 → disk/jampy_kdc、J1331 → quad/positions、DEV 行 → elliptical/jampy。回归：verify_identify。

- **椭圆样本库升级**（2026-11-03）：`sample_library.enrich_elliptical` 从 Dinos SLACS 参数（3 个完整：zl/zs/sigma/theta_E/γ/Re/n/q/PA）+ KCWI 运动学表（14 个 sigma_ap/Re/q）组装可建模椭圆表 `modeling_input_elliptical_modeling.csv`（14 个）。`data_complete` 标注 3 个 "full"（有 theta_E/γ）vs 11 个 "sigma_only"（theta_E/γ 仍缺，需从 SLACS 爱因斯坦半径文献补）。`identify._lookup` 现在优先返回 DEV 标签行，J1630/J1402/J2303/J0037/J0330 识别为 elliptical 而非默认盘。回归：verify_identify（断言 14 个 DEV 行）。

- **图像形状识别：`gl.identify_from_image`**（2026-11-03）：`sample_library.morphology` 从 FITS 图像直接测光集中度 C=5·log10(r80/r20) + 轴比 q（纯 numpy/astropy），作为 catalog 形态缺失/不可靠时的图像兜底。真实数据标定：J2141 盘 C=1.58 vs J1630 椭圆 C=4.21（阈值 C<3.1 盘、C>3.4 椭圆）；q 被弧+PSF 拉圆（两者都 ≈0.89），只作辅助。`gl.identify_from_image(fits_path)` 返回 LensPlan。回归：verify_morphology。

- **椭圆透镜孔径弥散（sigma_ap）**（2026-11-01）：`JampyEngine` 现支持单值光加权孔径速度弥散 sigma_ap（椭圆透镜的运动学观测量），经 `jam.axi.proj` 的 `pixsize` + `sigmapsf` 卷积计算。`KinData` 新增 `aperture_radius`/`aperture_vrms`/`aperture_vrms_err`/`kin_psf_sigma`；径向 `v_rms` 剖面可省略。`config_loader` 读取孔径字段；新模板 `configs/fit/elliptical_slacs_example.yaml`（模板，未端到端验证）。回归：verify_ellip_aperture（J2141 B7 模型 3" sigma_ap = 168.8 km/s）。同时新增 `stellar_mge_ellip`（椭圆单 Sérsic 无盘恒星模板，`stellar_template type: ellip`）和首个真实椭圆配方 `configs/fit/j1630_ellip.yaml`（SDSS J1630+4520，SLACS slow rotator：Re 孔径 sigma_ap = 267 km/s）。回归：verify_j1630（配置 + sigma_ap 机制）。bulge_i0 = 232 counts/s/px（I_e=0.1083，来自引擎自己的单 Sérsic 减恒星，I(0)=I_e·exp(b_n)；注意 `al.lp.Sersic.intensity` 是 I_e 不是中心面亮度，n=4 差 exp(b_n)=2141 倍）；先验中心参数下 sigma_ap ~100 km/s vs 267，因参数太轻（M(<Re) ~5.5e10 vs σ_e=267 推的 ~1.3e11 Msun），集群拟合解决。椭圆减恒星用新增 `subtract_stellar_light_sersic`（单 Sérsic n=4、结构固定到 SLACS_light_params、只拟合强度 + 中心；`DataSpec.subtract_method: "sersic_single"`），因为 B4 的 bulge+disk 减恒星硬编码了 J2141 盘结构、不适用于椭圆。端到端 smoke 已验证：load_dataset（sersic_single）→ assemble（13 参数）→ Nautilus（n_live=20）跑通。

- **B9 提炼：像素化源 + AGN 点源减恒星 + sky RMS 噪声**（2026-11-01）：`lensfit.models.source.pixelized`（`PixelizedSource`，PyAutoLens Pixelization + RectangularUniform 网格 + Constant 正则化；`n_source` = 网格像素数 vs `n_free_params` = 正则化系数，故 `nu_img = n_unmasked − n_source` 减的是 225 个网格像素）。`lensfit.lens.subtract.subtract_stellar_light_mge_agn`（文献值 MGE + PSF 形状 AGN 点源，联合最小二乘；B9 配方把减恒星约化 χ² 从 2.58 降到 1.11）。`lensfit.lens.rescale.estimate_sky_noise`（直接 sky RMS，绕开 KAI drizzle 权重图坑）。`SourceModel.n_source` 默认取 `n_free_params`（参数化源不受影响）。回归：verify_lensfit_b9（sky RMS 恢复、减恒星约化 χ² 1.11 复现、像素化源 7 显式参数 / n_source 225）。同时把 B9 路径接进门面：`api.dataset` 新增 `subtract_method="mge_agn"`（sky RMS 噪声 + MGE/AGN 点源减恒星，原生网格、无需独立噪声 FITS），`DataSpec` 新增 `mge_params_path`/`badmask_path`/`psf_sigma_px`/`position_angle_deg`，`PriorSet.ml_img_spec` 新增可选高斯 M/L 先验（B9 K'-band N(0.8, 0.3)）。新配方 `configs/fit/j2141_b9_kband_pixelized.yaml`（像素化源 + mge_agn 减恒星 + K'-band 恒星模板）。同时把 B9 K'-band 恒星模板标定提炼为 `stellar_mge_kband`（两侧桥：`StellarTemplate.calibration` = 动力学 `SURF_FACTOR` 桥（Msun/pc²），`calibration_img` = Σ_crit 归一化成像桥；config loader 里 `stellar_template type: mge_csv`）。回归：verify_b9_recipe。新增 `docs/components_reference.md`（+`_zh`）：可插拔组件 + 可配置参数索引。

- **mock 闭合验证层 + 真值扩展 + 强透镜弧合成**（2026-10-27）：`mock_pipeline.truth` 新增 `particles`（HDF5 粒子加载到物理单位 + 惯性张量 mock 坐标系对齐）、`fdm`（暗物质占比真值，球对称 + edge-on 柱三孔径）、`anisotropy`（速度张量算 `beta_z = 1 - σ_z²/σ_R²`）、`halo`（DM 粒子拟合 gNFW → v_vir/c_-2/γ/q_h，复用 lensfit 的 `zeta`/`halo_cosmo` 参数化）。新增 `mock_pipeline.lensing`（`arc` + `dataset`）：`build_lens_galaxy` / `einstein_radius` / `make_arc`（真值质量模型强透镜弧光线追踪）+ `scale_arc` / `merge_arc`。新增 `lensfit.validation`（mock 闭合）：`fdm`（球/柱口径）、`bias`、`corner`（加权 corner + 真值叠加）、`prior_sensitivity`（M/L 先验中心重加权 + ESS）、`diagnostics`（m=2 Fourier 棒检验）、`closure`（fit samples → f_dm → 真值偏差）。`lensfit.dynamics.jampy` 增加 `beta_z` 直参数化各向异性选项（`anisotropy_param`、`beta_cyl_beta_z`、`beta_z_to_ratio`/`ratio_to_beta_z`），对应 MF 系列约定；`PipelineConfig.dynamics_anisotropy` 选择。回归以 348489 锚定（f_dm 球 0.695 / 柱 0.797、βz +0.40、halo v_vir 218 / c_-2 5.55 / γ 1.53 / q_h 0.74、θE 0.609″），带 `skip-if-missing` 保护。

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
- **椭圆 mock 的 sigma_ap 真值**（2026-11-02）：`mock_pipeline.truth.aperture.sigma_ap`（质量加权弥散 `sqrt(<v_los^2> - <v_los>^2)`、圆形天面孔径、不做 PSF 卷积；是 JampyEngine sigma_ap 可观测量的真值对应）。回归：`tests/test_truth_aperture.py`。另加 mock↔fit 契约回归 `tests/test_mock_api_contract.py`（mock data/noise/psf FITS 经 `api.dataset.load_dataset` 原样加载）。

### 修复

- **联合分析的 multiprocessing pickle**（2026-11-02）：`fit/joint.py` 的 `_JointAnalysis` 是函数内 local class，`number_of_cores > 1` 无法 pickle（`Can't get local object`）。现带稳定模块路径（`__module__`/`__qualname__`）并注册进模块命名空间；Linux（fork）可 pickle。回归：`tests/test_joint_pickle.py`。
- **全量测试收集被 photutils 阻塞**（2026-11-02）：`tests/test_psf_selection.py` 在收集阶段间接 import photutils 而无 skip 守卫。补 `pytest.importorskip("photutils")`。
- **closure 标定口径文档化**（2026-11-02）：`validation.closure.sample_f_dm` docstring 明确两种 calib 口径（F814W `calib=1` + Σ_crit 桥在 `lg_ml`；K′-band `calib=动力学桥` + `lg_ml=log10 M/L`），禁止混用。

## [0.1.0] - 2026-09-24

### 新增

- 仓库骨架：包布局（`src/galaxylens/`）、模块目录、pyproject、README、LICENSE、CHANGELOG、.gitignore。
- 架构决策：单仓 monorepo、src 布局、按能力分（不按论文/星系类型）、数据生产与消费分离、mock 拟合复用 `lensfit` + ground-truth 对照。
- `data_pipeline.imaging` 模块迁移（21 文件；psf 拆子包；路径/SExtractor/TinyTim 环境变量解耦）。
- `config` 模块（YAML + dataclass；16 透镜 YAML）。
- `sample_library` 模块迁移（构建管线 + 5 个成品 CSV）。
