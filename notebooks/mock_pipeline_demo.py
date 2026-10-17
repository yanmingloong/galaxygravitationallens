# %% [markdown]
# # mock_pipeline 端到端教程
#
# 从 IllustrisTNG 筛选 J2141 盘透镜类似物 → 生成 SKIRT mock 图像 → 合成 fake HST → 测量。
# 这是 `mock_pipeline.selection` + `mock_pipeline.skirt` + `mock_pipeline.truth` 三个子包的全链路演示。
#
# 分两段：
# - **离线段**（§1 配置 / §4 .ski / §6 fake HST / §7 测量）：纯本地、秒级、开箱即跑。
# - **重活段**（§2 筛选 / §3 下载转换 / §5 SKIRT）：需要 TNG key / 本地粒子数据 / SKIRT 二进制，本教程标注前置条件并展示真实产物。
#
# 前置环境变量（缺哪个，对应段会跳过并提示）：
# - `TNG_API_KEY`：TNG API key（§2 筛选、§3 下载粒子用）
# - `GALAXYLENS_DATA_ROOT`：粒子数据根（`catalog/` + `stars_l3/` + `raw_stars/` + `raw_gas/`）
# - `GALAXYLENS_OUTPUT_ROOT`：mock 输出根（.ski / .dat / SKIRT 产物）
# - `SKIRT_BIN`：SKIRT 9 二进制（§5 用，默认取 PATH 上的 `skirt`）

# %%
import os

import numpy as np
from astropy.io import fits

from galaxylens.mock_pipeline.selection import run_selection
from galaxylens.mock_pipeline.selection.config import J2141
from galaxylens.mock_pipeline.skirt import (
    BandConfig, MockObsConfig, J2141_BANDS, J2141_OBS,
    tng_to_skirt, write_ski, run_skirt, make_fake_hst, extinction_map,
)
from galaxylens.mock_pipeline.truth import (
    projected_ellipticity, theta_E_SIS, kpc_per_arcsec,
)

DATA_ROOT = os.environ.get("GALAXYLENS_DATA_ROOT", ".")
OUTPUT_ROOT = os.environ.get("GALAXYLENS_OUTPUT_ROOT", os.path.join(DATA_ROOT, "mock_out"))
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"DATA_ROOT   = {DATA_ROOT}")
print(f"OUTPUT_ROOT = {OUTPUT_ROOT}")

# %% [markdown]
# ## 1. 目标 + 观测配置（全部数据化）
#
# 换目标 = 换一个 `TargetConfig`；换波段/仪器 = 换一个 `BandConfig`。流水线零代码改动。

# %%
print("内置目标 J2141：")
print(J2141)
print()
print("默认观测（J2141_OBS）波段：")
for b in J2141_OBS.bands:
    print(f"  {b.label:6s} -> {b.skirt_name:24s} px={b.pixel_scale:.4f} arcsec/px")

# %% [markdown]
# 换仪器示例：J1331 用 WFPC2 F814W（单波段、无 F160W），倾角 70°。
# 只需新建一个 `BandConfig`，其余代码不动。

# %%
wfpc2_f814w = BandConfig("F814W", "HST WFPC2 F814W", 0.10)  # 0.10"/px 是 WF3 芯片
j1331_obs = MockObsConfig(bands=[wfpc2_f814w])
print("J1331 观测：", [(b.label, b.skirt_name) for b in j1331_obs.bands])

# %% [markdown]
# ## 2. 筛选：scan → L1 → L3 → catalog（需 TNG_API_KEY）
#
# 真实普查走 `python -m galaxylens.mock_pipeline.selection.run --target j2141`。
# notebook 里用 `max_detail=10` 做 smoke（每快照只抓 10 个详情）。
# 结果不进仓库，落在 `OUTPUT_ROOT/catalog.csv`。

# %%
_key = os.environ.get("TNG_API_KEY") or (
    "file" if os.path.exists(os.path.expanduser("~/.tng_api_key")) else ""
)
if not _key:
    print("⚠ 未检测到 TNG API key，跳过筛选（§2）。")
    print("  获取：tng-project.org 注册 → API 文档；存到环境变量 TNG_API_KEY 或 ~/.tng_api_key。")
    rows = []
else:
    rows = run_selection(
        "j2141", DATA_ROOT, OUTPUT_ROOT,
        workers=1, l3_workers=4, max_detail=10,  # smoke test
    )
    print(f"\n盘主导候选 {len(rows)} 个 -> {os.path.join(OUTPUT_ROOT, 'catalog.csv')}")
    for r in rows[:3]:
        print({k: r[k] for k in ("subhalo_id", "f_disk") if k in r})

# %% [markdown]
# ## 3. 下载全字段粒子 + 转 SKIRT .dat（需本地粒子数据）
#
# 这里直接演示已知的 J2141 相似体 **348489**（f_disk 最高的盘主导候选）。
# 若跑过 §2，可改成 `rows` 里的候选。粒子缺时需先 `download_stars` / `download_gas` 下载。

# %%
label, sim, snap = "348489", "TNG100-1", 88  # 演示目标（真实跑可用 §2 的候选）
stars_h5 = os.path.join(DATA_ROOT, "raw_stars", f"{sim}_snap{snap}_id{label}_stars.hdf5")
gas_h5 = os.path.join(DATA_ROOT, "raw_gas", f"{sim}_snap{snap}_id{label}_gas.hdf5")

if os.path.exists(stars_h5) and os.path.exists(gas_h5):
    res = tng_to_skirt(stars_h5, gas_h5, label, sim, snap, OUTPUT_ROOT)
    print(f"{label}: {res['n_stars']} 恒星粒子, {res['n_dust']} 尘埃粒子, "
          f"flat={res['flat']:.3f}, M_star={res['M_star']:.2e} Msun")
else:
    print("⚠ 缺全字段粒子，跳过 §3。")
    print(f"  stars_h5 = {stars_h5}")
    print(f"  gas_h5   = {gas_h5}")
    print("  下载：from galaxylens.mock_pipeline.selection import download_stars, download_gas")

# %% [markdown]
# ## 4. 生成 .ski（完整可跑，纯本地）
#
# 每个波段一份 dust / nodust 双版本。波段名、仪器名、倾角、几何全部来自配置。

# %%
dust_ski, nodust_ski = write_ski(label, J2141_BANDS[0], OUTPUT_ROOT,
                                 obs=J2141_OBS, incl=J2141.incl, roll="0 deg")
print(f"dust   : {dust_ski}")
print(f"nodust : {nodust_ski}")
print("\n--- dust .ski 关键行 ---")
for line in open(dust_ski, encoding="utf-8"):
    s = line.strip()
    if any(k in s for k in ("FrameInstrument", "BroadBand", "BruzualCharlot", "ThemisDustMix")):
        print(s)

# %% [markdown]
# ## 5. 运行 SKIRT（需 SKIRT 二进制 + 数小时）
#
# `run_skirt` 调 SKIRT 二进制（`SKIRT_BIN` env var 或 PATH 上的 `skirt`）。
# 1e9 光子/星系，64 核约 3–4 小时 —— 真实跑在集群，notebook 里注释掉。

# %%
# run_skirt(dust_ski, outdir=OUTPUT_ROOT)
# run_skirt(nodust_ski, outdir=OUTPUT_ROOT)
print("SKIRT 运行已注释（耗时 + 需二进制）。§4 已生成 .ski，集群上执行：")
print(f"  skirt {dust_ski}")
print(f"  skirt {nodust_ski}")

# %% [markdown]
# ## 6. fake HST（完整可跑）
#
# `make_fake_hst`：SKIRT 无噪声 total.fits → 重采样到 HST 像素尺度 → 卷积 PSF →
# 峰值归一 → 加高斯噪声/宇宙线/背景源。需要 SKIRT 已产出 `total.fits`；
# 这里真实产物缺失时用合成图兜底，保证 API 演示始终能跑。

# %%
skirt_dust_fits = os.path.join(OUTPUT_ROOT, f"{label}_1e9", "total.fits")

# PSF：真实 F814W 用 J2141 实测 PSF（j21410001_f814w_psf.fits）；此处 5x5 delta 占位
psf = np.zeros((5, 5)); psf[2, 2] = 1.0

if os.path.exists(skirt_dust_fits):
    img_hst = make_fake_hst(skirt_dust_fits, px_mock=0.20, band_cfg=J2141_BANDS[0], psf=psf)
    print(f"用真实 SKIRT 产物：{skirt_dust_fits}")
else:
    # 离线兜底：合成 50x50 盘状图，演示 API
    y, x = np.mgrid[0:50, 0:50]
    img = np.exp(-((x - 25) ** 2 + ((y - 25) / 0.4) ** 2) / 2 / 8 ** 2) * 4905.0
    tmp = os.path.join(OUTPUT_ROOT, "_demo_total.fits")
    fits.writeto(tmp, img[None], overwrite=True)
    img_hst = make_fake_hst(tmp, px_mock=0.20, band_cfg=J2141_BANDS[0], psf=psf)
    print(f"（未找到真实 SKIRT 产物，用合成图演示。真实路径：{skirt_dust_fits}）")

print("fake HST 图像 shape:", img_hst.shape)

try:
    import matplotlib.pyplot as plt
    plt.imshow(img_hst, origin="lower", cmap="gray_r")
    plt.title(f"fake HST {J2141_BANDS[0].skirt_name}")
    plt.colorbar(label="e-/s")
    plt.show()
except ImportError:
    print("(matplotlib 未安装，跳过绘图)")

# %% [markdown]
# ## 7. 消光图 + 形态测量（完整可跑）
#
# 消光 A = -2.5 log10(dust/nodust) 用 SKIRT 无噪声双版本；形态测量走 `truth` 包。

# %%
skirt_nodust_fits = os.path.join(OUTPUT_ROOT, f"{label}_1e9", "total_nodust.fits")
if os.path.exists(skirt_dust_fits) and os.path.exists(skirt_nodust_fits):
    A = extinction_map(skirt_dust_fits, skirt_nodust_fits, half_px=50)
    c = A.shape[0] // 2
    print("消光图 A 中心 5x5 (mag)：")
    print(np.round(A[c - 2:c + 3, c - 2:c + 3], 2))
    print(f"中心消光 A(center) = {A[c, c]:.2f} mag")
else:
    print("⚠ 缺 SKIRT 无噪声产物，跳过消光图。")

# %% [markdown]
# 形态测量（`truth` 包，纯 NumPy，任何 2D 图像可跑）：

# %%
eps = projected_ellipticity(img_hst)
print(f"projected_ellipticity(fake HST) = {eps:.3f}  (1 - sqrt(l_min/l_max))")

# SIS 爱因斯坦半径 + 尺度换算（用 J2141 的 σ_SIE 与红移）
thE = theta_E_SIS(sigma_kms=197.0, z_lens=J2141.z_lens, z_src=J2141.z_source)
print(f"theta_E_SIS(σ=197, zl={J2141.z_lens}, zs={J2141.z_source}) = {thE:.3f} arcsec")
print(f"kpc_per_arcsec(z={J2141.z_lens}) = {kpc_per_arcsec(J2141.z_lens):.3f} kpc/arcsec")

# %% [markdown]
# 粒子级测量（`half_mass_radius` / `disk_kinematics` / `intrinsic_axis_ratios`）需要
# 对齐后的粒子坐标+质量，跑完 §3 后可用 `tng_to_skirt` 的同款粒子调用，这里从略。
