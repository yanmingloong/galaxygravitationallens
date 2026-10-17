# 变更日志 Changelog

本文件记录项目的所有重要变更。All notable changes to this project are documented in this file.

格式基于 Keep a Changelog，遵循语义化版本 SemVer。
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 新增 Added

- `skirt` 观测配置数据化（2026-10-17）：新增 `BandConfig`/`MockObsConfig`（波段+仪器名+像素尺度+噪声+峰值 / 几何），`write_ski` 改收 `BandConfig`+`MockObsConfig`（不再查写死的 BANDS 字典，同滤光片不同仪器如 WFPC2 F814W 可表达），`make_fake_hst` 改收 `BandConfig`（去 F814W 专用默认常量 PX_HST/SIGMA/J_PEAK）。`TargetConfig` 加 `obs` 字段挂观测配置。`snap_to_a` 对未知快照改 raise（不再静默 a=1.0）。
  Data-fied the skirt observation config (2026-10-17): new `BandConfig`/`MockObsConfig` (band+instrument name+pixel scale+noise+peak / geometry); `write_ski` now takes a `BandConfig`+`MockObsConfig` (no more hard-coded BANDS dict — same filter on a different instrument, e.g. WFPC2 F814W, is expressible); `make_fake_hst` takes a `BandConfig` (dropped F814W-specific PX_HST/SIGMA/J_PEAK). `TargetConfig` gains an `obs` field. `snap_to_a` now raises on unknown snapshots instead of silently returning a=1.0.
- 翻新 `notebooks/mock_pipeline_demo.py` 为完整教程（8 段：配置/筛选/转换/.ski/SKIRT/fake HST/消光+测量；离线段完整可跑，重活段标注 TNG key / 粒子数据 / SKIRT 二进制前置条件）。
  Renewed `notebooks/mock_pipeline_demo.py` into a full tutorial (8 sections; offline sections fully runnable, heavy sections gated on TNG key / particle data / SKIRT binary).
- `read_api_key` 改静默：import 不再打印 "key not found" 提示，改为实际发起下载/请求时按需打印（`_print_key_hint`）。
  Made `read_api_key` silent on import; the "key not found" hint now prints on-demand in the download/request paths.
- `TargetConfig` 加 `incl` 字段 + `read_api_key()`（2026-10-17）：盘倾角进目标配置（J2141=78.5 deg、J1331=70 deg），换目标零代码改动，`skirt.render.write_ski` 从 target.incl 取。TNG API key 改从 env var `TNG_API_KEY` 或 `~/.tng_api_key` 文件读取（仓库不内置 key），未配置时一次性提示指向 docs/mock_pipeline.md 获取指引，`get()`/`download_cutout()` 无 key 快速失败。
  Added `incl` field to `TargetConfig` + `read_api_key()` (2026-10-17): disk inclination moved into target config (J2141=78.5 deg, J1331=70 deg); `skirt.render.write_ski` reads it from target.incl. TNG API key now read from env `TNG_API_KEY` or the `~/.tng_api_key` file (no key hard-coded), with a one-time hint pointing to docs/mock_pipeline.md; `get()`/`download_cutout()` fail fast without a key.
- `mock_pipeline.truth` 模块（2026-10-14）：ground-truth 计算，含 kinematics（circularity → f_disk/B_D/盘轴/厚度）、measures（半光半径 / 投影椭圆率协方差口径 / 固有轴比 / 消光）、cosmology（Planck 2015 距离 + SIS θE + 宇宙时间 + TNG 单位换算）。附带回归测试（348489 f_disk=0.643、B/D=0.55 复现）。修正 tng_to_skirt 的 H0_GYR 笔误（0.06774→0.06928）。
  `mock_pipeline.truth` module (2026-10-14): ground-truth computation — kinematics (circularity → f_disk/B_D/disk axis/thickness), measures (half-mass radius / projected-ellipticity covariance convention / intrinsic axis ratios / extinction), cosmology (Planck 2015 distances + SIS theta_E + cosmic time + TNG unit conversions), with regression tests (348489 f_disk=0.643, B/D=0.55 reproduced); fixed tng_to_skirt H0_GYR typo (0.06774→0.06928).
- `mock_pipeline.selection` 模块（2026-10-14）：TNG 盘透镜类似物筛选，含 scan（全量扫描，re_star_kpc ×a 修正）、filter（L1 判据数据化）、merger（L2 mpb.hdf5 主支质量跳变）、download（3字段/全字段合并）、verify（L3 disk_kinematics）、catalog（L3→catalog 自动生成，替代手工旧脚本）、run（--target 端到端）。筛选判据进 SelectionCriteria dataclass，换目标零代码改动。
  `mock_pipeline.selection` module (2026-10-14): TNG disk-lens analog selection — scan / filter (L1 criteria as data) / merger (L2 mpb.hdf5 main-branch mass-jump) / download (3-field & all-field merged) / verify (L3 disk_kinematics) / catalog (L3→catalog auto-assembly) / run (--target end-to-end). Selection criteria live in a SelectionCriteria dataclass; switching target needs no code change.

- `mock_pipeline.skirt` 模块（2026-10-14）：SKIRT 辐射转移，含 convert（TNG 粒子 → .dat，惯性张量盘轴对齐，不用 catalog.csv 的旧 L_hat）、render（.ski 生成 + skirt 运行，SKIRT_BIN 环境变量）、imaging（fake HST + 消光图）。修正 convert 的 write_dust a 参数 bug + cosmic_time 向量化。
  `mock_pipeline.skirt` module (2026-10-14): SKIRT radiative transfer — convert (TNG particles → .dat, inertia-tensor alignment, no legacy L_hat) / render (.ski generation + skirt run, SKIRT_BIN env var) / imaging (fake HST + extinction map). Fixed write_dust ``a``-parameter bug and vectorized cosmic_time.

## [0.1.0] - 2026-09-24

### 新增 Added

- 仓库骨架：包布局（`src/galaxylens/`）、模块目录、pyproject、README、LICENSE、CHANGELOG、.gitignore。
  Repository skeleton: package layout, module directories, pyproject, README, LICENSE, CHANGELOG, .gitignore.
- 架构决策：单仓 monorepo、src 布局、按能力分（不按论文/星系类型）、数据生产与消费分离、mock 拟合复用 `lensfit` + ground-truth 对照。
  Architecture decision: single monorepo, src layout, split by capability (not paper/galaxy type), data-production vs data-consumption separation, mock fitting reuses `lensfit` + ground-truth validation.
- `data_pipeline.imaging` 模块迁移（21 文件；psf 拆子包；路径/SExtractor/TinyTim 环境变量解耦）。
  `data_pipeline.imaging` module migrated (21 files; psf split; paths/SExtractor/TinyTim env-decoupled).
- `config` 模块（YAML + dataclass；16 透镜 YAML）。
  `config` module (YAML + dataclass; 16 lens YAMLs).
- `sample_library` 模块迁移（构建管线 + 5 个成品 CSV）。
  `sample_library` module migrated (build pipeline + 5 product CSVs).
