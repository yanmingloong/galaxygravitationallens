# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-24

### Added
- Repository skeleton: package layout (`src/galaxylens/`), module directories
  (config / data_pipeline / mock_pipeline / lensfit / sample_library), pyproject,
  README, LICENSE, CHANGELOG, .gitignore.
- Architecture decision recorded: single monorepo, src layout, split by
  capability (not by paper or galaxy type), data-production vs data-consumption
  separation, mock fitting reuses `lensfit` + ground-truth validation.
