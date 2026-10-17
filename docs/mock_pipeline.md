# mock_pipeline 使用文档 — 模拟数据生产与筛选

> 从 IllustrisTNG 筛选盘星系强透镜类似物，生成 SKIRT mock 图像。
> 本模块生产**带 ground truth 的模拟数据**，与观测样本库 `sample_library`（真实巡天目录，无 ground truth）区分。

## 一、模块结构

```
mock_pipeline/
├── truth/        # ground-truth 计算（selection 与 validation 共用）
│   ├── kinematics.py   # circularity → f_disk / B/D / 盘轴 / 厚度
│   ├── measures.py     # 半光半径 / 投影椭圆率 / 固有轴比 / 消光
│   └── cosmology.py    # Planck 2015 距离 / SIS θE / 宇宙时间 / TNG 单位换算
├── selection/    # TNG 盘透镜类似物筛选
│   ├── scan.py        # 全量扫描（API 翻页 + 详情抓取）
│   ├── filter.py      # L1 筛选（判据数据化）
│   ├── merger.py      # L2 并合史（mpb.hdf5 主支质量跳变）
│   ├── download.py    # 粒子下载（3 字段 / 全字段）
│   ├── verify.py      # L3 验证（下载 + disk_kinematics）
│   ├── catalog.py     # L3 → catalog 自动生成
│   ├── config.py      # TargetConfig / SelectionCriteria（判据）
│   └── run.py         # --target 端到端编排
└── skirt/        # SKIRT 辐射转移 → mock 图像
    ├── convert.py     # TNG 粒子 → .dat（惯性张量盘轴对齐）
    ├── config.py      # BandConfig/MockObsConfig（波段/仪器/几何，数据化）
    ├── render.py      # .ski 生成 + skirt 运行
    └── imaging.py     # fake HST + 消光图
```

## 二、端到端数据流

```
scan（全量扫描，mass__gt 过滤）
  → L1 筛选（场中心 + M* 窗口 + σ 下限，判据来自 TargetConfig）
  → L3 验证（下载 3 字段粒子 → disk_kinematics 算 f_disk/B/D/L_hat）
  → catalog 自动生成（选盘主导子集，L_hat 三列展开）
  → skirt（下载全字段粒子 → convert 惯性张量对齐 → .ski → SKIRT 1e9 → fake HST）
```

L2（并合史）不在主链中，作为独立 `merger.check_merger` 按候选单独调用（与普查流程 scan→L1→L3 一致）。

## 三、快速开始

```bash
# 环境变量
export TNG_API_KEY=...            # TNG API key（或写入 ~/.tng_api_key，见下方指引）
export GALAXYLENS_DATA_ROOT=/path/to/tng_data   # 数据根（catalog/ + stars_l3/）
export SKIRT_BIN=/path/to/skirt   # SKIRT 9 二进制（render 用）

# 一条命令搜 J2141 相似体（内置 target）
python -m galaxylens.mock_pipeline.selection.run \
    --target j2141 --data-root $GALAXYLENS_DATA_ROOT --outdir ./out

# smoke test：限制每快照详情数
python -m galaxylens.mock_pipeline.selection.run \
    --target j2141 --max-detail 5 --l3-workers 1
```

> **TNG API key 获取**（免费，每人可申请）：在 tng-project.org 注册账号 → 登录后在
> [API 文档](https://www.tng-project.org/data/docs/api/) 获取 key → 二选一提供给脚本：
> `export TNG_API_KEY=你的key` 或写入 `~/.tng_api_key`（单行）。仓库不内置任何 key；
> 未配置时脚本会提示「TNG API key not found」并指向本指引。

**换目标 = 加一个 `TargetConfig`**（在 `selection/config.py` 或后续 YAML），流水线代码零改动：

```python
from galaxylens.mock_pipeline.selection.config import TargetConfig, SelectionCriteria
J1331 = TargetConfig(
    name="J1331+3628", z_lens=0.113, z_source=0.254, snap=88,
    incl="70 deg",
    criteria=SelectionCriteria(
        sims=["TNG100-1"], snapshots=[88, 91, 84, 95, 99],
        mass_min=68.0, mstar_range=(5e10, 3e11), sigma_min=140.0,
        f_disk_disk=0.30, f_disk_min=0.60,
    ),
)
```

## 四、关键物理口径（务必区分，勿混用）

| 量 | 口径 | 位置 |
|----|------|------|
| **f_disk** | circularity κ≥0.7 的**质量占比**（κ=jz/(R·vc)，盘粒子判定） | `truth/kinematics.py` |
| **盘轴 L_hat** | 盘粒子角动量轴（迭代一次），**非**总角动量、**非**惯性张量 | 同上 |
| **B/D** | (1−f_disk)/f_disk，无盘时 999.0 哨兵 | 同上 |
| **投影椭圆率** | 1−√(λmin/λmax)，F814W **无阈值**强度加权二阶矩 | `truth/measures.py` |
| **固有轴比** | 对齐粒子 σ_z/σ_x（3D 盘厚度），**非**投影椭圆率 | 同上 |
| **盘轴对齐（SKIRT）** | **惯性张量**三轴（法向→+z、长轴→+x），**不读** catalog.csv 的旧 L_hat | `skirt/convert.py` |
| **宇宙学** | Planck 2015（H0=67.74，Ωm=0.3089，ΩΛ=0.6911），非 Planck18 | `truth/cosmology.py` |
| **kpc/″** | 角直径距离 D_A(z)/206265，随红移变化（J2141 z=0.138 → 2.517） | 同上 |

**历史教训**（易混点）：σ(total) 是**全粒子弥散**（含 DM，冷薄盘落 140–160），σ≥170 会偏核球盘；椭圆率曾有"固有轴比 vs 投影"两口径混用，现统一为投影协方差口径。

## 五、筛选判据数据化

判据在 `SelectionCriteria` dataclass（**数据不是代码**），每目标一份：

| 字段 | 含义 | J2141 默认 |
|------|------|-----------|
| `mstar_range` | L1 恒星质量窗口（Msun） | [5e10, 3e11]（普查口径，放宽；J2141 专用紧窗口 [7e10,1.3e11] 是子集）|
| `sigma_min` | L1 σ(total) 下限（km/s） | 140（放宽；170 偏核球盘）|
| `f_disk_min` | L3 盘主导子集阈值 | 0.6 |
| `f_disk_disk` | L3 "有盘"阈值 | 0.30 |
| `mass_min` | 扫描 pre-filter（总质量，1e10 Msun/h）| 68（已验证对场中心目标不漏）|
| `field_only` | L1 要求场中心（primary_flag=1）| True |

## 六、API 摘要

```python
from galaxylens.mock_pipeline.truth import disk_kinematics, projected_ellipticity, theta_E_SIS, kpc_per_arcsec
from galaxylens.mock_pipeline.selection import run_selection, build_catalog, verify_one
from galaxylens.mock_pipeline.skirt import (
    tng_to_skirt, write_ski, run_skirt, make_fake_hst,
    BandConfig, MockObsConfig, J2141_BANDS, J2141_OBS,
)

# ground truth：粒子 → f_disk/B/D/L_hat/厚度/Re
res = disk_kinematics(pos_kpc, vel, mass, center)   # pos/vel/mass 物理单位

# 端到端筛选
rows = run_selection("j2141", data_root, outdir)    # scan → L1 → L3 → catalog

# SKIRT：全字段粒子 → .dat → .ski → fake HST
tng_to_skirt(stars_h5, gas_h5, "348489", "TNG100-1", 88, outdir)
write_ski("348489", J2141_BANDS[0], outdir, obs=J2141_OBS, roll="0 deg", incl=J2141.incl)
run_skirt(f"{outdir}/348489_F814W.ski", outdir=outdir)
img = make_fake_hst(fits_path, px_mock, J2141_BANDS[0], psf)

# 换仪器（如 J1331 用 WFPC2 F814W、倾角 70°）：新建 BandConfig，零代码改动
wfpc2 = BandConfig("F814W", "HST WFPC2 F814W", 0.10)
write_ski("J1331", wfpc2, outdir, incl="70 deg")
```

## 七、回归测试

```bash
PYTHONPATH=src python -m pytest tests/test_truth.py tests/test_selection.py tests/test_skirt.py -v
```

22 个测试，锚定已知值：348489 f_disk=0.643、B/D=0.55、kpc_per_arcsec(0.138)=2.517、θE(σ=200)=0.89″。
