# coint4 (coint2)

This repository contains a cointegration pairs trading framework and related tooling.

## Repo layout

- `coint4/`: main application workspace (full src, tests, configs, UI, Docker files)
- `legacy/`: archived root-level code/tests/configs/scripts and legacy tooling
- `docs/`: architecture, data, and testing docs (paths assume `cd coint4`)
- `data/`: local datasets (ignored)
- `outputs/`, `results/`, `artifacts/` (including `artifacts/live/logs/`): generated artifacts (ignored)

Notes:
- The active Poetry project lives in `coint4/pyproject.toml`. Legacy Poetry and requirements files are in `legacy/`.
- The primary CLI entrypoint is `coint2`.

## Data and outputs

Large datasets and generated outputs are intentionally excluded from Git. Use external storage or Git LFS for any large files you must version.
See `docs/data_storage.md` for details.

## Quickstart

See `docs/quickstart.md` and `docs/testing_guide.md` (paths assume `cd coint4`).
