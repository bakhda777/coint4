# coint4 (coint2)

This repository contains a cointegration pairs trading framework and related tooling.

## Repo layout

- `coint4/`: main application workspace (full src, tests, configs, UI, Docker files)
- `coint4/data_downloaded/`: canonical dataset location (ignored, large files)
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
For production runs, use `docs/production_checklist.md`. Overlays and configs:
- `coint4/configs/data_quality_strict.yaml` (strict QA)
- `coint4/configs/data_window_clean.yaml` (clean window + symbol exclusions overlay)
- `coint4/configs/main_2024_wfa_balanced.yaml` (balanced WFA)
Note: WFA supports optional fixed universe via `--pairs-file` and custom output via `--results-dir` (see `docs/quickstart.md`).

Optimization plan: `docs/optimization_plan_20260114.md`.
Selection grid runs (filters): `docs/optimization_runs_20260115.md` with configs in `coint4/configs/selection_grid_20260115/` and strict p-value grid in `coint4/configs/selection_grid_20260115_strictpv/` (parallel run notes in docs).
Sharpe target runs (strict signals): `docs/optimization_runs_20260115.md` with configs in `coint4/configs/sharpe_target_20260115/` and агрегатором `coint4/artifacts/wfa/aggregate/20260115_sharpe_target/`.
Quality universe (data-driven exclusions): `docs/optimization_runs_20260115.md` with configs in `coint4/configs/quality_runs_20260115/` (including corr0.45 alignment runs, signal_strict, tradeability/hl0p05-45, z0p9/z1p0 exit0p1 variants, and denylist `coint4/configs/quality_runs_20260115/denylist_symbols_20260115.yaml`), universe artifacts in `coint4/artifacts/universe/quality_universe_20260115/`, `coint4/artifacts/universe/quality_universe_20260115_250k/`, and `coint4/artifacts/universe/quality_universe_20260115_200k/`, WFA aggregates in `coint4/artifacts/wfa/aggregate/20260115_quality_universe_500k/`, `coint4/artifacts/wfa/aggregate/20260115_quality_universe/`, and `coint4/artifacts/wfa/aggregate/20260115_quality_universe_200k/`.
