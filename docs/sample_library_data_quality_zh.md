# Sample Library 数据质量规范

*英文版见 [sample_library_data_quality.md](sample_library_data_quality.md).*

> 2026-10-14 记录自 J2141 相似体项目样本库数据审计。这些坑曾导致 θE=1.10″、Re=9.16 kpc 等错误值传播进建模输入，务必遵守。

## 1. 硬编码参数字典必须有文献来源注释

`enrich.py` 的 `swell_s_thetas` 字典曾手填 θE，其中 4 个 SWELLS 透镜值错误（无文献校验，J2141 填了 1.10，实为 0.88）。

- 任何「从文献手填」的参数字典，**必须在注释写清来源**：作者、年份、Table 号、列名。
- 字典混入不同巡天的透镜（SWELLS vs SLACS）时，**分组注释**，别混在一个注释下。
- 修正/新增值时，**加日期 + 来源 + 旧值**（格式见 `enrich.py` 的 NOTE 注释）。

## 2. 字段单位/语义在 Phase 传递时必须保持

`Re_bulge_arcsec`/`Re_disk_arcsec` 曾是 **kpc 值误标成 arcsec**——Phase B/C 把 SWELLS 原始 Excel 的 `Re/kpc` 直接复制进 arcsec 字段，未做 kpc→arcsec 换算。

- 每个字段的**单位必须在 schema 显式声明**，Phase 间传递时做单位换算，不能「同名直拷」。
- 原始数据源字段名不可信：SWELLS Excel 的 `Re/kpc` 对 disk 实际存的是 **Rd（标长）**，不是 Re（半光半径 R50）。**字段语义要对照原始论文核验**，不能只看字段名。

## 3. kpc→arcsec 换算必须用各自红移

不同透镜红移不同，kpc/″ 因子不同（Planck 2015）：

| z_lens | kpc/″ |
|--------|-------|
| 0.104 | 1.972 |
| 0.131 | 2.408 |
| 0.138 | 2.517 |
| 0.158 | 2.818 |

**不能用一个固定的 2.517 套所有透镜。**

## 4. 副本同步铁律

`DiskLenses/SampleLibrary/05_exports/` 的 modeling_input CSV 有副本在 `galaxygravitationallens/sample_library/`（进 git）。**改源文件必须同步副本**，否则仓库数据悄悄落后（曾发生：早期 θE patch 只改了 DiskLenses 源，副本停在旧值，直到几周后才被发现）。

## 5. 数据修正要留记录

任何数据修正（改值、改单位、补缺失）都要在 `sample_library/README.md` 的「数据修正记录」小节登记：改了什么、旧值→新值、来源、日期。否则后来者不知道某个值为何是这个数。

## 6. 已知遗留（已文档标注，未修）

- 19 个 SLACS 透镜的 θE 未核对 SLACS 文献（Auger+2009）。
- `Re_disk` 字段语义（Rd vs Re）与 bulge 的 Re 定义待核——原始数据源口径，仅做单位换算。
- J1037+3517 在 SWELLS III Table 1 也是省略号，无权威 θE 值。
