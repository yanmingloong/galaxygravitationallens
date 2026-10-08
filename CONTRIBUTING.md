# Contributing

Conventions for maintaining the `galaxygravitationallens` repository.

## Language policy

- **Code comments (docstrings + inline comments): English only.** This keeps the
  source clean and internationally readable.
- **Documents (.md): bilingual, as two separate files.**
  - `README.md` — English (default entry point, auto-rendered by GitHub).
  - `README_zh.md` — Chinese.
  - Each file links to the other in its header.

## Extraction workflow (DataPipeline → repo)

The repo (`src/galaxylens/`) is the distilled, versioned form of the research
pipeline under `DataPipeline/`. When the research pipeline changes, distill the
delta into the repo:

1. Identify the changed files in `DataPipeline/pipeline/`.
2. Map them to `src/galaxylens/data_pipeline/imaging/` (or the matching module).
3. Apply the functional changes (English docstrings, per the language policy).
4. Update `README.md` and `README_zh.md` if the change affects usage or design.
5. Commit with a message describing the distilled change; push to both remotes
   (`git push origin` → GitHub + Gitee).

The research project itself (`GalaxyGravLens/`) is NOT under git; only this repo is.

## Repository layout

- `src/galaxylens/` — package (config / data_pipeline / mock_pipeline / lensfit / sample_library).
- `configs/` — per-lens YAML configs.
- `sample_library/` — curated sample data (git-tracked subset).
- `tests/`, `notebooks/`, `docs/`.
