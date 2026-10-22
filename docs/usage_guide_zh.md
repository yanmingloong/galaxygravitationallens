# 全流程使用指引：从寻找盘星系到透镜拟合

*英文版见 [usage_guide.md](usage_guide.md).*

本指南走通 `galaxylens` 实现的完整链条：从组装盘星系透镜样本，到建模拟合。仓库按**能力**（而非星系类型或论文）划分为四个子包：

```
sample_library   ->  data_pipeline   ->  mock_pipeline   ->  lensfit
(find disks)         (download+reduce)  (simulate+truth)    (fit models)
```

- `sample_library` — 从多巡天目录聚合 + 去重 + 分类透镜候选。
- `data_pipeline` — 下载（外部）并处理 HST 成像：cutout、噪声、PSF、星掩模、inpaint。
- `mock_pipeline` — 模拟盘透镜系统（TNG + SKIRT）用于 ground-truth 对照。
- `lensfit` — 拟合引擎：可插拔模型组件 + 配置装配，通过 `api` 门面暴露（`gl.fit` / `gl.fit_positions` / `gl.fit_mein`）。

## 环境变量

各包通过 `galaxylens.config.paths` 解析的环境变量与硬编码数据根解耦：

| 变量 | 指向 | 使用者 |
|---|---|---|
| `GALAXYLENS_DATA_ROOT` | 处理后的数据根 | `data_pipeline` |
| `GALAXYLENS_SAMPLE_LIB` | 样本库根（01_raw .. 05_exports） | `sample_library` |
| `GALAXYLENS_CONFIG_DIR` | 逐透镜 YAML 配置 | `config` |

运行任何东西前，先在 shell profile 里设置它们。

---

## Step 1 — 寻找盘星系：`sample_library`

从多个巡天聚合透镜目录，去重、按类型（盘 / 椭圆 / …）分类，并丰富物理参数。

```bash
# 1a. 下载巡天目录（cowls / glue / lenscat / manga / vizier / simard）
python -m galaxylens.sample_library.download.cowls
python -m galaxylens.sample_library.download.lenscat
# ... 每个写入 01_raw/。

# 1b. 运行 B -> C -> D -> E 管线
python -m galaxylens.sample_library.pipeline              # 无网络
python -m galaxylens.sample_library.pipeline --network    # 开启 SDSS/VizieR 查询
python -m galaxylens.sample_library.pipeline --from phase_c   # 断点续跑

# 1c. 加入你自己的候选
python -m galaxylens.sample_library.pipeline --add mine.csv --source paper
```

阶段：**B** 标准化 + 三级去重 → **C** Tier 分类 → **D** 丰富（物理参数）→ **E** 导出建模子集。产出落在 `05_exports/` 的 `modeling_input` CSV（full / sample / elliptical / tier0 / tier01）。盘透镜筛选使用为 B/D 系列制定的盘主导判据。

## Step 2 — 下载 + 处理：`data_pipeline`

从 MAST 下载原始 HST 帧在包外完成（astroquery 或手动）；管线的 `file_discovery` 定位已下载到 raw 根下的 FLC 文件。

```bash
# 单个透镜
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J2141-0001 --version-tag v2_flc_rescaled_tinytim_inpaint

# 整个样本族（SWELLS -> disk/, SLACS -> ellip/）
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --sample SWELLS --version-tag v2_flc_whtonly

# 单波段 / dry run
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J1630+4520 --bands F814W --dry-run
```

11 步流程：`file_discovery → loaders → cutout → centroid → background → cosmic_ray → noise → psf → starmask → alignment → output`。噪声以三种口径生成（`whtonly` / `rescaled` / `bugnoise`），由 `--noise-method` 选择。产出（sci / noise / psf / starmask FITS + 诊断 + 质量）落在 `GALAXYLENS_DATA_ROOT` 的版本 tag 下。

## Step 3 — 拟合：`gl.fit` 门面 + 逐点入口

拟合引擎是可插拔模型组件 + 配置驱动装配器，通过 `galaxylens.api` 门面暴露。两种拟合模式：

```python
import galaxylens as gl

# 3a. 像素级联合拟合（B 系列弧）：一份配方，一次调用
cfg = gl.load_config("configs/fit/j2141_barnabe.yaml")
result = gl.fit(cfg)                          # 装配 -> Nautilus -> 后处理

print(result.beta_z)                          # 加权 beta_z = 1 - ratio²
print(result.c_minus2, result.gamma, result.v_vir, result.q_h)
print(result.c200)                            # c_200 = (2 - gamma) * c_-2
print(result.m_star)                          # M* = ml_img × CALIB × L_2D_REL
print(result.f_dm(2.2 * cfg.pipeline.geometry.rd_arcsec))
result.plot_corner("j2141_corner.png")        # 加权 corner（Agg，无头安全）

# 3b. 逐点四重像拟合（D 系列 J1331）：scale-free 势
sf = gl.fit_positions(theta_obs, err_arcsec=0.05, zlens=0.113, zsrc=0.254)
print(sf.rein_arcsec, sf.mein_1e10, sf.chi2)

# 3c. 逐点联合拟合（D 系列 J1331）：动力学 + 爱因斯坦质量（scipy MAP）
res = gl.fit_mein(geometry, kinematics, lum_mge, ml_array,
                  rein_arcsec=0.91, mein_msun=7.7e10)
print(res["v200"], res["c200"], res["beta_z"], res["chi2_dyn"], res["chi2_lens"])
```

`gl.fit` 端到端跑 Barnabè 式盘透镜流程（galaxy-subtract、噪声 rescale、gNFW + MGE×M/L + Sersic 源、jampy 动力学），用 Nautilus 采样。`gl.fit_positions` / `gl.fit_mein` 是四重像系统（Trick+2016 J1331）的逐点（坐标级）路径——透镜信息是像位置 / 单值爱因斯坦质量，而非像素级弧。完整坑清单见 `lensfit/README.md`，可插拔架构见 `docs/refinement_plan.md`。

## Step 4 — mock 对照：`mock_pipeline`

对同一批盘透镜系统生成 TNG/SKIRT mock 观测，喂回 `lensfit` 与已知真相闭合检验。观测样本没有 ground truth；mock 有。这是验证 `lensfit` 通用性的方式。见 `docs/mock_pipeline.md`。

## 一句话总括

寻找（sample_library）→ 下载（astroquery/MAST）→ 处理（data_pipeline/imaging）→ 拟合（lensfit 可插拔组件 + api 门面 gl.fit）。
