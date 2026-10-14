# galaxylens 文档索引 Documentation index

| 文档 Doc | 内容 Content |
|---|---|
| [data_pipeline.md](data_pipeline.md) | 观测数据生产管线使用指引 + 教程 / observational data pipeline guide + tutorial |
| [sample_library_data_quality.md](sample_library_data_quality.md) | 样本库数据质量规范（单位/语义/副本同步/修正记录）/ sample library data-quality rules |
| [mock_pipeline.md](mock_pipeline.md) | 模拟数据生产与筛选（TNG 搜寻 → SKIRT mock 图像）/ mock data production + selection |
| [README.md](../README.md) | 项目总览 + 架构 / project overview + architecture |

## 环境配置速查 Environment quick reference

```bash
# 数据根目录（必需） data root (required)
export GALAXYLENS_DATA_ROOT=/path/to/DataPipeline

# 结果输出根（结果不进仓库，统一放这里） results output root (keep out of the repo)
export GALAXYLENS_OUTPUT_ROOT=/path/to/results

# SExtractor（可选） SExtractor (optional)
export SEXTRACTOR_BIN=/path/to/sex

# 样本库根（sample_library 模块） sample library root
export GALAXYLENS_SAMPLE_LIB=/path/to/SampleLibrary

# TNG API key（mock_pipeline.selection/skirt 下载用） TNG API key
export TNG_API_KEY=...

# SKIRT 9 二进制（mock_pipeline.skirt.render 用） SKIRT binary
export SKIRT_BIN=/path/to/skirt
```
