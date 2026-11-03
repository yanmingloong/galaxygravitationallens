# 路线图 / 扩展项

记录待扩展与流程不匹配项，避免丢失。分「规划中」与「流程不匹配/待修」两组。

## mock 拟合（mock 闭合）

### 盘 mock（MF 系列）——已完成
truth（particles/fdm/anisotropy/halo）、validation（closure 等）、弧合成
（`mock_pipeline/lensing`）、dynamics `beta_z` 直参数化均已就位；6/6 候选闭环
（M/L + c-M 先验下 3 改善 / 3 过修正）。

### 盘 mock 下一轮（B9）——暂缓
用 B9 改进重跑 MF mock，先定岔路：
- **像素化源**（`source/pixelized.py`）——直接可用；拆掉源-质量简并（MF 源强度
  0.27 vs 真值 88）。
- **K′-band 减恒星**——需先生成 K′-band mock 图（SKIRT 波段未生成；MF 只有
  F814W / F450W 代理）。

### 椭圆 mock（新系列）——未开始
- `truth/aperture.py`：粒子 → 孔径速度弥散 `sigma_ap`（已补 2026-11-02）。
  口径决策（默认值在括号内）：质量加权 [是]；弥散 sigma =
  sqrt(<v_los^2> - <v_los>^2)，非 v_rms [是]；圆形孔径，标注 jampy 方形
  `pixsize=2*R_ap` 近似差 [是]；v1 不做 PSF 卷积 [是]。与 jampy 的交叉项符号
  待椭圆 mock 实跑时核对。
- `selection`：无椭圆目标登记（`BUILTIN_TARGETS` 只有 j2141/j1331 盘）。
- **J1630 相似体**：SLACS slow rotator，sigma_ap = 267 km/s（Re 孔径），
  theta_E = 1.791″ / gamma = 1.924（Dinos 基准）。匹配形态（Sérsic n~4、低 f_disk /
  slow rotator）、sigma_e ~ 250–270 km/s（M(Re) ~ 1.3e11）、M* ~ 1e11 Msun。

## 分析流程不匹配 / 待修

1. **mock 产出 → api/dataset 口径衔接**——已验证（2026-11-02，
   `tests/test_mock_api_contract.py`）：mock data/noise/psf FITS 经
   `api.dataset.load_dataset` 原样加载。椭圆路径复用同一形态无关 loader。
2. **closure 标定桥**——已记录（2026-11-02）：`sample_f_dm` docstring 明确两种口径
   （F814W calib=1 + 桥在 lg_ml；K′-band calib=动力学桥 + lg_ml=log10 M/L），禁止混用。
3. **椭圆配方未端到端验证**——暂缓（需椭圆 mock 存在后在集群跑完整 Nautilus）。
4. **multiprocessing pickle**——已修（2026-11-02，`tests/test_joint_pickle.py`）：
   `_JointAnalysis` 现在有稳定模块路径 + 命名空间注册；Linux（fork）可 pickle。
   spawn 平台仍单核。

## 数据管线待扩展

- `data_pipeline/spectra`：DEIMOS 归约 + pPXF 提取尚未就位（研究脚本在
  SpectraPipeline；E-MILES 模板在 SpectraPipeline/03_catalogs/miles_templates/）。
- `data_pipeline/other_obs`：空。
