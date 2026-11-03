# 组件与参数参考

*英文版见 [components_reference.md](components_reference.md).*

这是 `lensfit` 拟合引擎里每个可插拔模型组件与每个可配置参数的权威索引，写新拟合配方（`configs/fit/*.yaml`）时作对照表用。引擎是**组件库 + 配置装配**，不是硬编码流程：一份 YAML 把暗物质模型、恒星质量模型、源模型、动力学引擎与透镜约束组合成一次完整的联合拟合。

目前只有 J2141（盘透镜，延展弧）与 J1331（盘，四重像）两个配方做了端到端验证；其他目标需要备数据 + 写新 YAML（零代码改动）。

## 1. 可插拔组件（`_COMPONENT_REGISTRY`）

| 部分 | 组件（`components.<key>`） | 自由参数 | 背书 |
|---|---|---|---|
| 暗物质（`dark`） | `gnfw_virial` | v_vir, c_-2, gamma, q_h | B 系列（J2141） |
| | `nfw` | v200, c200 | D 系列（J1331） |
| 恒星质量（`stellar`） | `mge_const_ml` | 1（常数 M/L） | B 系列 |
| | `mge_radial_ml` | 逐高斯径向 M/L | D 系列（J1331） |
| 源（`source`） | `single_sersic` | 7（I/Re/n/ell/centre） | B7 |
| | `pixelized` | 1 正则化系数 + mesh 像素 | B8/B9 |
| 动力学（`dynamics`） | `jampy` | ratio 或 beta_z | B 系列 |
| | `jampy_kdc` | 反旋转核 + PSF | D 系列（J1331） |
| 透镜约束（`lens_constraint`） | `imaging`（像素级弧） | — | B 系列 |
| | `mein`（单值爱因斯坦质量） | — | D5 |
| | 四重像 position（走 `gl.fit_positions`） | — | D1 |

新增组件 = 一个实现 `MassComponent` / `SourceModel` / `DynamicsEngine`
接口的类 + 在 `pipeline/assemble.py` 注册表加一条。现有组件无需改动。

## 2. 恒星模板标定（`stellar_template`）

两种，由 `type` 选择：

| `type` | 构建器 | 输入 | 用途 |
|---|---|---|---|
| （默认，F814W） | `stellar_mge_peak` | n_bulge, re_bulge, q_bulge, rd_disk, q_disk, disk_over_bulge, psf_sigma_arcsec, bulge_i0 | B7 / F814W 盘 |
| `ellip`（椭圆） | `stellar_mge_ellip` | n, re_arcsec, q, psf_sigma_arcsec, bulge_i0 | J1630 椭圆 |
| `mge_csv`（K'-band） | `stellar_mge_kband` | mge_params_path, pixel_scale, mstar_ref, ml_ref | B9 / K'-band |

K'-band 构建器是两侧桥：`StellarTemplate.calibration` 是动力学面密度桥
（`surf_s = peak × calib × ml_img`，单位 Msun/pc²），`calibration_img` 是
Σ_crit 归一化成像桥（`g.intensity = peak × calib_img`）。F814W 构建器两侧共用
一个 `calibration`。

## 3. 减恒星与噪声（`data.subtract_method`）

| 方法 | 配方 | 噪声 |
|---|---|---|
| `sersic`（默认） | 自由 Sérsic 核球 + 盘减恒星，走 `al.Tracer`（B4） | 读 `noise_map_path` |
| `mge_agn` | 文献值 MGE + PSF 形状 AGN 点源，联合最小二乘（B9）；需 `mge_params_path`/`badmask_path`/`psf_sigma_px`/`position_angle_deg` | `estimate_sky_noise`（sky RMS，绕开 KAI drizzle 权重图坑）；无需独立噪声 FITS |

## 4. 可配置参数（YAML 各段）

### geometry
`zlens`、`zsrc`、`inclination_deg`、`re_arcsec`、`rd_arcsec`

### kinematics
`v_rms`（`[x_arcsec, vrms, err]` 列表）、`vcirc`、`vcirc_err`；椭圆透镜单值孔径
弥散：`aperture_radius`、`aperture_vrms`、`aperture_vrms_err`、`kin_psf_sigma`
（单值 sigma_ap；`v_rms` 可省略）。

### priors
- `ml_img`：LogUniform 上下界 `[low, high]`（默认）
- `ml_img_spec`：高斯 M/L 先验 `{kind, mean, sigma, lower, upper}`（B9
  K'-band N(0.8, 0.3)）；覆盖 `ml_img`
- `v_vir`、`gamma`、`q_h`、`ratio`：Uniform 上下界 `[low, high]`
- `c_2`：c-M 关系先验 `{kind, mean, sigma, lower, upper}`
- `v200`、`c200`、`beta_z`：D 系列（球对称 NFW / 切向各向异性）

### source_priors（Sérsic 源）
`centre_sigma`、`ell_sigma`、`ell_bound`、`intensity_bounds`、
`effective_radius_bounds`、`sersic_index_bounds`

### source_pixelization（像素化源）
`mesh_shape` `[nx, ny]`、`reg`（`constant`）、`coeff_bounds` `[low, high]`

### stellar_template
见第 2 节。

### data
`data_path`、`noise_map_path`、`psf_path`、`starmask_path`、`pixel_scale`、
`mask_radius`、`oversample`、`subtract_method`、`mge_params_path`、
`badmask_path`、`psf_sigma_px`、`position_angle_deg`

### search
`backend`（`local` | `slurm`）、`name`、`n_live`、`number_of_cores`、
`output_root`；仅 slurm：`partition`、`account`、`conda_env_path`、`workdir`、
`cpus_per_task`、`mem`、`time_hours`

### 开关
`noise_rescale`、`subtract_stellar`

## 5. 诚实边界：已验证与未验证

- 端到端验证过的配方：J2141（Barnabè 2012，`configs/fit/j2141_barnabe.yaml`
  与 `j2141_b9_kband_pixelized.yaml`）、J1331（Trick 2016，
  `configs/fit/j1331_trick.yaml`）和 J1630+4520（SLACS 椭圆，
  `configs/fit/j1630_ellip.yaml`，v2 为权威）。
- 组件是目标无关的：新盘透镜只需备 FITS + v_rms + MGE 和一份新 YAML（零代码
  改动）。椭圆透镜可复用 `jampy` + `gnfw_virial` + `mge_const_ml` + `imaging`
  （jampy 本就是椭圆星系方法）；单值孔径速度弥散入口（`KinData.aperture_*` +
  `JampyEngine` sigma_ap）已在 J1630 上验证，但单值孔径弥散打不破
  mass-anisotropy 简并，椭圆 βz/M* 受先验限制（见
  `EllipticalLenses/AutoLensFit/j1630_joint/README.md`）。

## 6. 编排层（`api.pipeline` + `api.analysis` + `mock_pipeline.orchestrate`）

在可插拔组件之上的一层薄编排（自动模式）；组件仍可独立调用（人工模式）。

- `Stage` — 一个命名组件调用，可声明 `depends`（依赖的其他 stage 名）。
- `Pipeline` — 有序 stage 共享 `ctx` 字典；`run(start/stop/only)` 支持全自动 /
  部分 / 区间运行。
- `gl.analyze(cfg)` — 一键拟合后分析：六质量参数、派生量（βz / M* / median
  参数处 f_dm / θ_E / γ_tot）、先验贴边检查、corner 图。`f_dm` 在 median
  参数处算（`Result.f_dm` 的逐样本循环是 O(N)、慢）。
- `gl.mock(target, ...)` — 预置的 mock 闭合流水线 select → pick → truth →
  render → make_arc → mock_fit → closure。重步骤需 TNG 粒子 HDF5 + SKIRT 二进制。
