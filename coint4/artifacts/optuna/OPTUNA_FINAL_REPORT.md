> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Optuna Optimization Report

## Study: canary_real

### Summary
- **Best PSR**: -1.000
- **Total Trials**: 25
- **Completed Trials**: 25

### Data Configuration
- **Period**: 2024-01-01 to 2024-03-31
- **Timeframe**: 15T
- **K-folds**: 3
- **Pairs File**: bench/pairs_canary.yaml

### Best Parameters
```yaml
zscore_threshold: 2.249080237694725
zscore_exit: 0.45071430640991617
rolling_window: 118
max_holding_days: 64
```

### Performance Metrics
- **OOS Sharpe**: -1
- **OOS PSR**: -1.0
- **Total Trades**: 0

### Reproduction Command
```bash
python scripts/run_optuna_real.py \
  --study-name canary_real \
  --n-trials 25 \
  --data-root ./data_downloaded \
  --period-start 2024-01-01 \
  --period-end 2024-03-31 \
  --timeframe 15T \
  --pairs-file bench/pairs_canary.yaml \
  --k-folds 3 \
  --resume
```

Generated: 2025-08-11T21:38:07.065464
