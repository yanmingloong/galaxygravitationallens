# 变更日志 Changelog

本文件记录项目的所有重要变更。All notable changes to this project are documented in this file.

格式基于 Keep a Changelog，遵循语义化版本 SemVer。
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-24

### 新增 Added

- 仓库骨架：包布局（`src/galaxylens/`）、模块目录、pyproject、README、LICENSE、CHANGELOG、.gitignore。
  Repository skeleton: package layout, module directories, pyproject, README, LICENSE, CHANGELOG, .gitignore.
- 架构决策：单仓 monorepo、src 布局、按能力分（不按论文/星系类型）、数据生产与消费分离、mock 拟合复用 `lensfit` + ground-truth 对照。
  Architecture decision: single monorepo, src layout, split by capability (not paper/galaxy type), data-production vs data-consumption separation, mock fitting reuses `lensfit` + ground-truth validation.
- `data_pipeline.imaging` 模块迁移（21 文件；psf 拆子包；路径/SExtractor/TinyTim 环境变量解耦）。
  `data_pipeline.imaging` module migrated (21 files; psf split; paths/SExtractor/TinyTim env-decoupled).
- `config` 模块（YAML + dataclass；16 透镜 YAML）。
  `config` module (YAML + dataclass; 16 lens YAMLs).
- `sample_library` 模块迁移（构建管线 + 5 个成品 CSV）。
  `sample_library` module migrated (build pipeline + 5 product CSVs).
