# %% [markdown]
# # mock_pipeline 端到端示例
#
# 从 IllustrisTNG 筛选 J2141 盘透镜类似物，再生成 SKIRT mock 图像。
# 这是 `selection.run` + `skirt` 三个模块的串联演示。
#
# 前置环境变量：
# - `TNG_API_KEY`：TNG API key（下载粒子用）
# - `GALAXYLENS_DATA_ROOT`：数据根（`catalog/` + `stars_l3/` + `raw_stars/` + `raw_gas/`）
# - `SKIRT_BIN`：SKIRT 9 二进制（`render.run_skirt` 用）

# %%
import os
import numpy as np

from galaxylens.mock_pipeline.selection import run_selection, build_catalog
from galaxylens.mock_pipeline.selection.config import J2141
from galaxylens.mock_pipeline.skirt import tng_to_skirt, write_ski, run_skirt, make_fake_hst

# %% [markdown]
# ## 1. 筛选目标配置（判据数据化）
#
# J2141 是内置目标。换目标 = 加一个 `TargetConfig`，流水线零改动。

# %%
print(J2141)

# %% [markdown]
# ## 2. 端到端筛选：scan → L1 → L3 → catalog
#
# `--max-detail 10` 是 smoke test（限制每快照详情数），真实普查去掉该参数。

# %%
data_root = os.environ.get("GALAXYLENS_DATA_ROOT", ".")
outdir = "./mock_out"

# smoke test: 每快照只抓 10 个详情
rows = run_selection(
    "j2141", data_root, outdir,
    workers=1, l3_workers=4, max_detail=10,
)
print(f"\n盘主导候选 {len(rows)} 个 -> {outdir}/catalog.csv")

# %% [markdown]
# ## 3. 对盘主导候选做 SKIRT mock
#
# 对 catalog 里 f_disk 最高的候选：下载全字段粒子（若未下载）→ convert 惯性张量对齐 → .ski → SKIRT → fake HST。

# %%
if rows:
    top = rows[0]
    label = str(top["subhalo_id"])
    sim = top["simulation"]
    snap = top["snap"]
    print(f"对 {label} (f_disk={top['f_disk']}) 做 mock")

    stars_h5 = os.path.join(data_root, "raw_stars",
                            f"{sim}_snap{snap}_id{label}_stars.hdf5")
    gas_h5 = os.path.join(data_root, "raw_gas",
                          f"{sim}_snap{snap}_id{label}_gas.hdf5")

    if os.path.exists(stars_h5) and os.path.exists(gas_h5):
        # 3a. TNG 粒子 → SKIRT .dat（惯性张量盘轴对齐）
        res = tng_to_skirt(stars_h5, gas_h5, label, sim, snap, outdir)
        print(f"  {res['n_stars']} 恒星粒子, {res['n_dust']} 尘埃粒子, flat={res['flat']:.3f}")

        # 3b. 生成 .ski（dust + nodust 双版）
        dust_ski, nodust_ski = write_ski(label, "F814W", outdir, roll="0 deg")

        # 3c. 运行 SKIRT（1e9 光子，耗时；此处注释，按需取消）
        # run_skirt(dust_ski, outdir=outdir)
        # run_skirt(nodust_ski, outdir=outdir)
        print(f"  .ski 已生成：{dust_ski} / {nodust_ski}（运行 SKIRT 请手动执行 run_skirt）")
    else:
        print(f"  缺全字段粒子（{stars_h5} / {gas_h5}），请先下载")
