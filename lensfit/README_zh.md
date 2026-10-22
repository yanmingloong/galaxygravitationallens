# lensfit — 多方法透镜拟合

`lensfit` 是一个用于拟合**任意形态**（盘星系、椭圆星系……）星系尺度引力透镜的**方法集**。它把第三方引擎——PyAutoLens、`mgefit`、`jampy`、`agama`——组装成可复用、参数化的链条，并把复现文献（B / D / R / e 系列）时踩过的坑固化下来。

它**不是**这些引擎的重新实现：它贡献的是用法约定、坑的防护、以及叠加在其上的参数化。

## 设计原则

1. **多方法**：恒星质量模型（当前 MGE，后续多组分）和动力学引擎（当前 `jampy`，后续 `agama`）在相同接口下可互换。没有任何东西写死绑定某篇论文或某种星系类型。
2. **封装而非重造**：凡是别人的算法，一律做薄封装 + 完整文档。
3. **引用强制**：每个模块的 docstring 都标注其借鉴的论文和/或代码。见下方引用表。

## 子模块

| 子模块 | 用途 | 引擎 | 状态 |
|---|---|---|---|
| `psf` | 场星实测 PSF | photutils | 场星检测完成；Moffat 拟合 TODO |
| `mge` | MGE 拟合 / I/O / 去投影 | mgefit | 完成（B2/B3） |
| `lens` | 透镜模型构建 + galaxy-subtracted 流程 | autolens/autofit | 完成（B4） |
| `dynamics` | Jeans / 动力学引擎 | jampy（后续 agama） | TODO（B5） |
| `fit` | 采样器 / 搜索封装 | Nautilus | TODO |
| `validation` | mock 拟合 ground-truth 对照 | — | TODO |
| `plotting` | 数据/模型预览 | autolens | `preview_imaging` |

## 引用

每个模块都在 docstring 里携带这些引用。权威来源：

- Cappellari 2002, MNRAS 333, 400 — MGE 分解与去投影（`mge`）
- Cappellari 2020, MNRAS 494, 4819 — `jampy`（`dynamics`，未来）
- Barnabè, Dutton, Marshall 等 2012, MNRAS 423, 1073 — SWELLS IV 盘透镜 galaxy-subtracted 建模（`lens`）
- Wyithe, Turner & Spergel 2001, ApJ 555, 504 — gNFW（`lens.models`）
- Ludlow 等 2016, MNRAS 460, 1214 — c–M 关系（`lens.models`）
- Vasiliev 2019, MNRAS 482, 1525 — AGAMA（`dynamics`，未来）
- Stetson 1987, PASP 99, 191 — DAOStarFinder（`psf`）

## 使用

```python
from pathlib import Path
from galaxylens.lensfit import mge, lens, psf

# --- MGE：拟合 Sersic 轮廓（σ_min 受 PSF 限制） ---
import numpy as np
sigma_min = 0.0637          # PSF σ，单位角秒
r = np.geomspace(0.05, 6 * 6.55, 200)
prof = np.exp(-2.095 * ((r / 6.55) ** 0.826 - 1))   # Sersic n=1.21, Re=0.26"
L, sigma = mge.fit_mge_1d(r, prof, ngauss=6, psf_sigma=sigma_min)

# --- MGE：读取 + 去投影（每个分量带 bulge/disk 标签） ---
mge_list = mge.load_mge(Path("mge_params.csv"))
q_intr = mge.deproject(np.array([g.q_obs for g in mge_list]), incl_deg=78.0)

# --- lens：构建 Barnabè 盘透镜模型（gNFW + MGE×M/L + 源） ---
model = lens.build_lens_model(zlens=0.138, zsrc=0.7127, mge=mge_list)
print(model.prior_count)   # 12

# --- lens：galaxy-subtracted 第一步（scipy）+ rescale ---
light_galaxy, chi2 = lens.subtract_stellar_light(dataset, zlens=0.138)
f = lens.noise_rescale_factor(chi2, n_unmasked=31389)
```

## 此处固化的关键坑

- **MGE σ_min ≥ PSF σ**（`mge.fit`）：不传 `rbounds` 时，`fit_1d` 会把 Sersic 平坦核过度分解成一个 `σ < PSF` 的尖峰，向下游注入不物理的致密核心。见 `mge.fit.fit_mge_1d`。
- **减光不减质量**（`lens.models`）：galaxy-subtraction 只减掉恒星的**光**，但恒星**质量**仍然起透镜作用。丢掉它会让晕只承担约 14% 的爱因斯坦质量。
- **噪声 rescale**（`lens.rescale`）：减除残差必须通过把噪声图乘以 `sqrt(reduced chi2)` 来吸收，否则弧拟合的 χ²/px ~ 30 会让后验坍缩、采样器磨好几天。
- **场星峰值**（`psf.stars`）：要在**屏蔽坏像素之后**再定位星系中心（宇宙线会亮过核球）；不要平滑（会拖偏峰值）。

## TODO

- `psf`：对场星候选做 Moffat/PSF 拟合（B1 第二步）。
- `dynamics`：`jampy` 封装 + 手写柱坐标 JAM 交叉检验（B5）。
- `lens`：多组分恒星质量模型（Sersic bulge+disk）作为 MGE 的替代。

---

*英文版见 [README.md](README.md)。*
