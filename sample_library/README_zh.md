# 观测样本库

*英文版见 [README.md](README.md).*

盘星系 + 椭圆星系引力透镜观测样本库。数据为**真实巡天透镜目录（无 ground truth）**，与模拟筛选库（`mock_pipeline` 的 selection 阶段，有 ground truth）区分。

## 结构

- **代码**：`src/galaxylens/sample_library/`（构建管线 + 访问）
- **数据**：本目录（成品 modeling_input CSV）

## 数据（成品）

5 个建模输入 CSV，来自构建管线 Phase E 导出。

| 文件 | 内容 | 大小 |
|---|---|---|
| modeling_input_full.csv | 全样本（tier 0/1/2/3/X） | 2.6M |
| modeling_input_sample.csv | 建模抽样 | 79K |
| modeling_input_elliptical.csv | 椭圆透镜 | 99K |
| modeling_input_tier0.csv | Tier 0（确认盘） | 1.4K |
| modeling_input_tier01.csv | Tier 0+1 | 14K |

**权威数据源**：`GALAXYLENS_SAMPLE_LIB` 指向的研究项目 SampleLibrary 目录（含 `01_raw`~`05_exports` 全量数据 + 原始巡天下载）。本目录只放「成品」，原始下载（505M）和中间产物（可重新生成）留在本地不进 git。

## 重跑构建

```bash
# 指向样本库根目录（含 01_raw ~ 05_exports）
export GALAXYLENS_SAMPLE_LIB=/path/to/SampleLibrary

# 完整重跑 B→C→D→E（无网络查询）
python -m galaxylens.sample_library.pipeline

# 断点续跑
python -m galaxylens.sample_library.pipeline --from phase_d

# 含网络查询（Phase D 在线补充）
python -m galaxylens.sample_library.pipeline --network

# 重新下载某巡天星表（Phase A）
python -m galaxylens.sample_library.download.lenscat
```

## 构建管线（5 阶段）

| Phase | 模块 | 产出 |
|---|---|---|
| A 下载 | `download/`（6 巡天） | 01_raw |
| B 标准化 | `build.py`（48 字段 Schema + 三级去重） | 02_merged/master.csv |
| C 分类 | `classify.py`（Tier 0/1/2/3/X） | 03_identified |
| D 丰富 | `enrich.py`（SDSS/Simard/Auger） | 04_enriched/tier012_vN.csv |
| E 导出 | `export.py` | 05_exports |

**加新数据库**：写一个 `download/xxx.py` 下载脚本 + 在 `config.py` 的 `RAW_SOURCES` 加一项，Phase B 会自动合并去重。
