> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Scripts Migration Guide

## 📋 Overview

We've reorganized the scripts directory for better clarity and reduced duplication. Most scripts have been consolidated into unified entrypoints with command-line flags.

## 🔄 Migration Table

### Optimization Scripts

| Old Script | New Command |
|------------|-------------|
| `run_optuna_real.py` | `core/optimize.py --real-data` |
| `run_optuna_with_resume.py` | `core/optimize.py --resume` |
| `run_optuna_with_traces.py` | `core/optimize.py --traces` |
| `run_large_scale_optuna.py` | `core/optimize.py --mode large --n-trials 1000` |
| `optimization/run_strict_optimization.py` | `core/optimize.py --mode strict` |
| `optimization/run_relaxed_optimization.py` | `core/optimize.py --mode relaxed` |
| `optimization/run_fixed_optimization.py` | `core/optimize.py --mode balanced` |
| `optimization/02_parallel_optimization.py` | `core/optimize.py --parallel --n-jobs 8` |
| `optimization/03_iterative_optimization.py` | `core/optimize.py --mode iterative` |

### Walk-Forward Scripts

| Old Script | New Command |
|------------|-------------|
| `run_walk_forward.py` | `core/walk_forward.py` |
| `run_walk_forward_with_resume.py` | `core/walk_forward.py --resume` |
| `run_wfa_with_portfolio.py` | `core/walk_forward.py --with-portfolio` |
| `run_scale_wfa.py` | `core/walk_forward.py --scale large` |
| `run_walk_forward_stub.py` | `core/walk_forward.py --stub` |

### Backtest Scripts

| Old Script | New Command |
|------------|-------------|
| `run_optuna_backtest.py` | `core/backtest.py --mode portfolio` |
| `run_paper_week.py` | `core/backtest.py --paper --window week` |
| `run_paper_canary.py` | `core/backtest.py --canary` |

### Universe Scripts

| Old Script | New Command |
|------------|-------------|
| `scan_data_inventory.py` | `universe/scan_data.py` |
| `generate_universe_windows.py` | `universe/generate_windows.py` |
| `data/build_universe.py` | `universe/build.py` |
| `aggregate_universe.py` | `universe/aggregate.py` |
| `validate_yaml.py` | `universe/validate.py` |

### Test Scripts

| Old Script | New Command |
|------------|-------------|
| `test_smoke.sh` | `test.sh` |
| `test_fast.sh` | `pytest -m "fast and not slow"` |
| `test_super_fast.sh` | `test.sh` |
| `test_ci.sh` | `pytest -m ci` |
| `test_dev.sh` | `pytest -m "not slow"` |
| `test_unit_fast.sh` | `pytest -m "unit and not slow"` |
| `test_full.sh` | `test_full.sh` (unchanged) |

### Analysis Scripts

| Old Script | New Command |
|------------|-------------|
| `analysis/analyze_optimization_results.py` | `analysis/optuna_report.py analyze` |
| `analysis/analyze_negative_sharpe.py` | `analysis/optuna_report.py negative` |
| `analysis/monitor_optimization.py` | `analysis/optuna_report.py monitor` |
| `analysis/analyze_all_studies.py` | `analysis/optuna_report.py compare` |

### Data Quality Scripts

| Old Script | New Command |
|------------|-------------|
| `data_utils/parquet_duplicates_checker.py` | `data/quality.py check-duplicates` |
| `data_utils/remove_duplicates.py` | `data/quality.py remove-duplicates` |
| `run_data_quality.py` | `data/quality.py stats` |
| `utils/clean_data_structure.py` | `data/quality.py validate-schema` |

## 📁 New Structure

```
scripts/
├── core/              # Main entrypoints (3 files)
│   ├── optimize.py    # All optimization modes
│   ├── walk_forward.py # Walk-forward analysis
│   └── backtest.py    # Backtesting
├── universe/          # Pair selection (5 files)
│   ├── scan_data.py
│   ├── generate_windows.py
│   ├── build.py
│   ├── aggregate.py
│   └── validate.py
├── data/              # Data management (3 files)
│   ├── quality.py     # Duplicates, gaps, validation
│   ├── build_portfolio.py
│   └── scan_data_range.py
├── analysis/          # Reporting (2 files)
│   └── optuna_report.py # Unified Optuna analysis
├── live/              # Production trading (6 files)
│   ├── run.py
│   ├── extract_snapshot.py
│   ├── risk_manager.py
│   ├── audit_execution_model.py
│   ├── calibrate_execution_costs.py
│   └── run_fee_scenarios.py
├── ops/               # Operations (9 files)
│   ├── env_lock.py
│   ├── data_lock.py
│   ├── dashboard.sh
│   ├── monitor_drift.py
│   ├── detect_vol_regimes.py
│   ├── rotate_portfolio_by_regime.py
│   ├── freeze_best_params.py
│   ├── promote_params.py
│   └── rollback_params.py
├── ci/                # CI/CD (5 files)
│   ├── ci_gates.py
│   ├── ci_smoke.py
│   ├── validate_ci.py
│   ├── build_artifact_index.py
│   └── build_results_manifest.py
├── devtools/          # Development only (36+ files)
│   ├── debug/         # Debugging tools
│   ├── maintenance/   # Fix utilities
│   └── experimental/  # Experiments
├── optimize.py        # Main optimization script
├── test.sh           # Quick smoke tests
└── test_full.sh      # Complete test suite
```

**Total production scripts: ~33 files** (down from 113)

## 🚀 Quick Examples

### Run optimization with different modes:
```bash
# Quick test
python scripts/core/optimize.py --mode fast --n-trials 25

# Strict optimization with resume
python scripts/core/optimize.py --mode strict --n-trials 200 --resume

# Large parallel optimization
python scripts/core/optimize.py --mode large --n-trials 1000 --parallel --n-jobs 8
```

### Run walk-forward analysis:
```bash
# Standard walk-forward
python scripts/core/walk_forward.py --config configs/main.yaml

# Resume with portfolio
python scripts/core/walk_forward.py --resume --with-portfolio

# Large scale
python scripts/core/walk_forward.py --scale large --n-windows 12
```

### Run backtests:
```bash
# Single pair
python scripts/core/backtest.py --mode single --pair BTCUSDT/ETHUSDT

# Portfolio
python scripts/core/backtest.py --mode portfolio --pairs-file benchmarks/pairs_universe.yaml

# Paper trading
python scripts/core/backtest.py --paper --window week
```

## ⚠️ Deprecation Notices

The following scripts have compatibility wrappers that will be **removed in 2 weeks**:
- `run_optuna_real.py`
- `run_optuna_with_resume.py`
- `run_walk_forward_with_resume.py`

These wrappers will show a deprecation warning and redirect to the new commands.

## 🔧 Environment Variables

All scripts now respect these environment variables:
- `QUICK_TEST=true` - Use smaller datasets for testing
- `DEBUG=true` - Enable debug output
- `CONFIG_PATH` - Override default config path

## 📝 Notes

1. **Backup created**: A full backup was created at `scripts_backup_YYYYMMDD_HHMMSS.tar.gz`
2. **DevTools**: Debug and experimental scripts moved to `devtools/` - not for production use
3. **Test simplification**: Only 2 test scripts remain: `test.sh` (smoke) and `test_full.sh` (complete)
4. **Unified CLIs**: All major scripts now use consistent argument patterns

---
Migration completed: 2025-08-11