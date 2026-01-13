> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Scripts Directory Structure

## 🎯 Overview

Optimized scripts structure with **42 production scripts** organized by function, separated from development tools.

## 📁 Directory Structure

```
scripts/
├── core/           # Main entrypoints (3 files)
├── universe/       # Pair selection pipeline (5 files)
├── data/           # Data management (3 files)  
├── analysis/       # Analytics & reporting (7 files)
├── live/           # Production trading (7 files)
├── ops/            # Operations & monitoring (11 files)
├── ci/             # CI/CD automation (8 files)
├── devtools/       # Development only (not in production)
├── test.sh         # Quick smoke tests
└── test_full.sh    # Complete test suite
```

## 🚀 Quick Start

### Main Workflows

```bash
# Run optimization
python scripts/core/optimize.py --mode balanced --n-trials 100

# Walk-forward analysis  
python scripts/core/walk_forward.py --config configs/main.yaml

# Run backtest
python scripts/core/backtest.py --mode single --pair BTCUSDT/ETHUSDT

# Quick tests
./scripts/test.sh

# Full test suite
./scripts/test_full.sh
```

## 📊 Production Scripts (42 files)

### Core (3)
- `optimize.py` - Unified hyperparameter optimization
- `walk_forward.py` - Walk-forward validation
- `backtest.py` - Backtesting engine

### Universe (5)
- `scan_data.py` - Scan available data
- `generate_windows.py` - Generate time windows
- `build.py` - Build pair universe
- `aggregate.py` - Aggregate pair results
- `validate.py` - Validate universe configuration

### Data (3)
- `quality.py` - Data quality checks & cleaning
- `build_portfolio.py` - Portfolio construction
- `scan_data_range.py` - Data range analysis

### Analysis (7)
- `optuna_report.py` - Unified Optuna analysis
- `pnl_attribution.py` - P&L attribution
- `sensitivity.py` - Sensitivity analysis
- `uncertainty.py` - Uncertainty analysis
- `optimization_summary.py` - Optimization summaries
- `filter_reasons.py` - Pair filter analysis
- `analyze_optuna_issues.py` - Debug Optuna issues

### Live (7)
- `run.py` - Live trading runner
- `extract_snapshot.py` - Extract live snapshots
- `risk_manager.py` - Risk management
- `audit_execution_model.py` - Execution audit
- `calibrate_execution_costs.py` - Cost calibration
- `run_fee_scenarios.py` - Fee scenario testing
- `tune_execution.py` - Execution tuning

### Ops (11)
- `scheduler.py` - Job scheduler
- `dashboard.sh` - Optuna dashboard
- `monitor_drift.py` - Monitor strategy drift
- `detect_vol_regimes.py` - Volatility regime detection
- `rotate_portfolio_by_regime.py` - Regime-based rotation
- `rebalance_study.py` - Rebalancing analysis
- `freeze_best_params.py` - Freeze parameters
- `promote_params.py` - Promote to production
- `rollback_params.py` - Rollback parameters
- `env_lock.py` - Environment locking
- `data_lock.py` - Data version locking
- `setup_postgresql.sh` - PostgreSQL setup

### CI (8)
- `ci_gates.py` - CI quality gates
- `ci_smoke.py` - Smoke tests
- `validate_ci.py` - CI validation
- `validate.py` - General validation
- `preflight.py` - Pre-flight checks
- `build_artifact_index.py` - Artifact indexing
- `build_results_manifest.py` - Results manifest
- `test_ci.sh` - CI test runner

## 🔧 Development Tools (in devtools/)

**Not for production use:**
- `debug/` - Debugging and profiling tools
- `experimental/` - Experimental scripts
- `maintenance/` - Maintenance utilities
- `diagnostics/` - Diagnostic tools
- `archive/` - Old scripts and wrappers

## 📝 Migration Guide

See [MIGRATION.md](MIGRATION.md) for migrating from old script structure.

## 🎯 Design Principles

1. **Minimal root** - Only essential files in scripts/ root
2. **Clear separation** - Production vs development
3. **Unified entrypoints** - Single script with modes instead of many scripts
4. **Logical grouping** - Scripts grouped by function
5. **No duplication** - Consolidated similar scripts

## 📊 Statistics

- **Production scripts**: 42 Python files
- **Development scripts**: 61+ in devtools/
- **Reduction**: 113 → 42 production scripts (63% reduction)
- **Organization**: 7 production directories + devtools

## 🚦 Testing

```bash
# Quick smoke tests (< 30 seconds)
./scripts/test.sh

# Full test suite
./scripts/test_full.sh

# Specific test markers
pytest -m "unit and not slow"
pytest -m "integration"
```

## 🔒 Production Safety

All development and experimental scripts are isolated in `devtools/` and excluded from production deployments via `.gitignore`.

---

Last updated: 2025-08-11