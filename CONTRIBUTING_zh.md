# 贡献指南

*英文版见 [CONTRIBUTING.md](CONTRIBUTING.md).*

维护 `galaxygravitationallens` 仓库的约定。

## 语言政策

- **代码注释（docstring + 行内注释）：只用英文。** 这保证源码干净、国际可读。
- **文档（.md）：中英双语，以两份独立文件提供。**
  - `README.md` — 英文（默认入口，GitHub 自动渲染）。
  - `README_zh.md` — 中文。
  - 每份文件在头部链接到另一份。

## 从研究管线同步

仓库（`src/galaxylens/`）是研究管线（位于 `DataPipeline/` 下）的版本化形态。当研究管线发生变化时，把增量同步进仓库：

1. 找出 `DataPipeline/pipeline/` 中变更的文件。
2. 映射到 `src/galaxylens/data_pipeline/imaging/`（或对应的模块）。
3. 应用功能性变更（按语言政策，用英文 docstring）。
4. 若变更影响用法或设计，更新 `README.md` 和 `README_zh.md`。
5. 用描述变更的提交信息提交；推送到两个远端（`git push origin` → GitHub + Gitee）。

研究项目本身（`GalaxyGravLens/`）**不**纳入 git；只有本仓库纳入。

## 仓库布局

- `src/galaxylens/` — 包（config / data_pipeline / mock_pipeline / lensfit / sample_library）。
- `configs/` — 逐透镜 YAML 配置。
- `sample_library/` — 精选样本数据（git 跟踪的子集）。
- `examples/` — 集群/运行模板（slurm），非包代码。
- `tests/`、`notebooks/`、`docs/`。

## 仓库组织

让这里保持一个**代码库**（而非数据/结果存储）的约定。

### 输出位置

- **结果与中间数据绝不放在仓库内** —— 它们体积大、不可 diff、且依赖环境。
- 通过环境变量定位它们，绝不硬编码路径：
  - `GALAXYLENS_DATA_ROOT` — 观测数据根（data_pipeline）。
  - `GALAXYLENS_OUTPUT_ROOT` — 结果根（不进仓库）。
  - `GALAXYLENS_SAMPLE_LIB` — 样本库根（01_raw~05_exports）。
  - `TNG_API_KEY`、`SKIRT_BIN`、`SEXTRACTOR_BIN`、`TINYTIM_*` — 外部工具/密钥。
- Notebook 和脚本从这些变量读取输出（见 `notebooks/mock_pipeline_demo.py`）。

### 文档位置

- 使用文档**集中在 `docs/` 下，并带索引**（`docs/README.md`）—— 一个索引找到一切，而不是散落的模块 README。
- 模块细节属于代码 docstring；只有模块专属的**数据清单**（如 `sample_library/README.md` 的 git 跟踪 CSV 列表）可以放在模块目录里。

### 本地 vs 集群

- **单一代码版本，不分本地/集群 fork。** 核心逻辑与环境无关。
- 环境差异通过函数参数（如 `workers`）和 `examples/` 下的外部调度模板（`run_selection_cluster.slurm`）来表达。
- 绝不在包内添加按集群分支的逻辑。
