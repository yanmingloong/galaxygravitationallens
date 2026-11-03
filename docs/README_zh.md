# galaxylens 文档索引

*英文版见 [README.md](README.md).*

| 文档 | 内容 |
|---|---|
| [data_pipeline_zh.md](data_pipeline_zh.md) | 观测数据生产管线使用指引 + 教程 |
| [sample_library_data_quality_zh.md](sample_library_data_quality_zh.md) | 样本库数据质量规范（单位/语义/副本同步/修正记录） |
| [mock_pipeline_zh.md](mock_pipeline_zh.md) | 模拟数据生产与筛选（TNG 搜寻 → SKIRT mock 图像） |
| [usage_guide_zh.md](usage_guide_zh.md) | 全流程使用：寻找盘星系 → 下载 → 处理 → 分析拟合 |
| [components_reference_zh.md](components_reference_zh.md) | 拟合引擎：可插拔组件 + 可配置参数索引 |
| [refinement_plan_zh.md](refinement_plan_zh.md) | 可插拔架构蓝图（组件 + 门面 + 路线图） |
| [roadmap_zh.md](roadmap_zh.md) | 待提炼 + 流程不匹配 / 扩展项 |
| [../lensfit/README_zh.md](../lensfit/README_zh.md) | lensfit 拟合引擎指引（PSF/MGE/透镜，引用表 + 坑清单） |
| [README_zh.md](../README_zh.md) | 项目总览 + 架构 |

## 教程（`notebooks/` 下的可跑 `# %%` notebook）

| Notebook | 内容 | 是否离线 |
|---|---|---|
| [identify_demo.py](../notebooks/identify_demo.py) | 透镜识别：名字 -> LensPlan -> 配方 -> 图像形状兜底 | 全离线 |
| [fit_demo.py](../notebooks/fit_demo.py) | 配方 -> 拟合 -> 分析（合成 FITS smoke + 重活段标注） | 离线 smoke + 重活段标注 |
| [mock_pipeline_demo.py](../notebooks/mock_pipeline_demo.py) | mock 端到端：TNG 选样 -> SKIRT -> fake HST | 离线段 + 重活段标注 |

每个 notebook 是 `# %%`（jupytext-light）Python：可直接当脚本跑，也可在 VS Code
里当交互 notebook 打开。离线段开箱即跑；重活段（TNG 下载、SKIRT 渲染、Nautilus
拟合）标注前置条件、缺时优雅跳过。

## 环境配置速查

```bash
# 数据根目录（必需）
export GALAXYLENS_DATA_ROOT=/path/to/DataPipeline

# 结果输出根（结果不进仓库，统一放这里）
export GALAXYLENS_OUTPUT_ROOT=/path/to/results

# SExtractor（可选）
export SEXTRACTOR_BIN=/path/to/sex

# 样本库根（sample_library 模块）
export GALAXYLENS_SAMPLE_LIB=/path/to/SampleLibrary

# TNG API key（mock_pipeline.selection/skirt 下载用）
export TNG_API_KEY=...

# SKIRT 9 二进制（mock_pipeline.skirt.render 用）
export SKIRT_BIN=/path/to/skirt
```
