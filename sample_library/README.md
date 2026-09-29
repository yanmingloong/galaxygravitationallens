# 观测样本库 Observational Sample Library

盘星系 + 椭圆星系引力透镜观测样本库。数据为**真实巡天透镜目录（无 ground truth）**，与模拟筛选库（`mock_pipeline` 的 selection 阶段，有 ground truth）区分。
Observational lens sample library for disk + elliptical galaxies. Data are **real survey lens catalogs (no ground truth)**, distinct from the mock screening library (`mock_pipeline` selection stage, which has ground truth).

## 结构 Structure

- **代码 Code**：`src/galaxylens/sample_library/`（构建管线 + 访问 / build pipeline + access）
- **数据 Data**：本目录（成品 modeling_input CSV / product modeling_input CSVs）

## 数据（成品） Data (products)

5 个建模输入 CSV，来自构建管线 Phase E 导出。5 modeling input CSVs exported by Phase E of the build pipeline.

| 文件 File | 内容 Content | 大小 Size |
|---|---|---|
| modeling_input_full.csv | 全样本（tier 0/1/2/3/X） full sample | 2.6M |
| modeling_input_sample.csv | 建模抽样 modeling sample | 79K |
| modeling_input_elliptical.csv | 椭圆透镜 elliptical lenses | 99K |
| modeling_input_tier0.csv | Tier 0（确认盘） confirmed disk | 1.4K |
| modeling_input_tier01.csv | Tier 0+1 | 14K |

**权威数据源 Authoritative source**：`GALAXYLENS_SAMPLE_LIB` 指向的研究项目 SampleLibrary 目录（含 `01_raw`~`05_exports` 全量数据 + 原始巡天下载）。本目录只放"成品"，原始下载（505M）和中间产物（可重新生成）留在本地不进 git。
The authoritative source is the research-project SampleLibrary dir pointed to by `GALAXYLENS_SAMPLE_LIB` (full `01_raw`~`05_exports` + raw survey downloads). This dir holds only products; raw downloads (505M) and regenerable intermediates stay local (not in git).

## 重跑构建 Rebuild

```bash
# 指向样本库根目录（含 01_raw ~ 05_exports）
# point to the sample library root (containing 01_raw ~ 05_exports)
export GALAXYLENS_SAMPLE_LIB=/path/to/SampleLibrary

# 完整重跑 B→C→D→E（无网络查询） full rebuild (no network)
python -m galaxylens.sample_library.pipeline

# 断点续跑 resume from a phase
python -m galaxylens.sample_library.pipeline --from phase_d

# 含网络查询（Phase D 在线补充） with network queries
python -m galaxylens.sample_library.pipeline --network

# 重新下载某巡天星表（Phase A） re-download one survey
python -m galaxylens.sample_library.download.lenscat
```

## 构建管线 Build pipeline（5 阶段 / 5 phases）

| Phase | 模块 Module | 产出 Output |
|---|---|---|
| A 下载 download | `download/`（6 巡天）| 01_raw |
| B 标准化 normalize | `build.py`（48 字段 Schema + 三级去重）| 02_merged/master.csv |
| C 分类 classify | `classify.py`（Tier 0/1/2/3/X）| 03_identified |
| D 丰富 enrich | `enrich.py`（SDSS/Simard/Auger）| 04_enriched/tier012_vN.csv |
| E 导出 export | `export.py` | 05_exports |

**加新数据库 Add a new survey**：写一个 `download/xxx.py` 下载脚本 + 在 `config.py` 的 `RAW_SOURCES` 加一项，Phase B 会自动合并去重。
Write a `download/xxx.py` fetcher + add an entry in `config.py` `RAW_SOURCES`; Phase B merges and deduplicates automatically.
