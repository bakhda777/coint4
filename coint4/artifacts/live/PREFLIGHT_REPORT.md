# Pre-live Preflight Report
*Generated: 2025-08-11 00:22:52*

## Overall Status: 🔴 NOT READY

- ✅ Passed: 2
- ❌ Failed: 1
- ⚠️ Warnings: 1

## Detailed Results

### DataRoot ⚠️
**Description:** Check data availability and freshness
**Status:** WARN
**Message:** Data may be stale: 790.2h old (> 0.5h)

**Details:**
```json
{
  "data_root": "data_downloaded",
  "parquet_files": 43,
  "latest_file": "data_downloaded/year=2025/month=07/data_part_07.parquet",
  "age_hours": 790.2214176925
}
```

### Timezone ✅
**Description:** Check timezone consistency in data
**Status:** PASS
**Message:** Timezone handling operational

**Details:**
```json
{
  "tz_naive_supported": true,
  "tz_aware_supported": true,
  "recommendation": "Use tz-naive or consistent UTC"
}
```

### Config ❌
**Description:** Check configuration: configs/prod.yaml
**Status:** FAIL
**Message:** Config load error: 9 validation errors for AppConfig
data_dir
  Field required [type=missing, input_value={'data': {'base_path': 'd... 100, 'keep_files': 10}}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.8/v/missing
results_dir
  Field required [type=missing, input_value={'data': {'base_path': 'd... 100, 'keep_files': 10}}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.8/v/missing
portfolio
  Field required [type=missing, input_value={'data': {'base_path': 'd... 100, 'keep_files': 10}}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.8/v/missing
pair_selection.lookback_days
  Field required [type=missing, input_value={'ssd_top_n': 50000, 'coi...min_mean_crossings': 10}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.8/v/missing
backtest
  Field required [type=missing, input_value={'data': {'base_path': 'd... 100, 'keep_files': 10}}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.8/v/missing
walk_forward.start_date
  Field required [type=missing, input_value={'train_days': 90, 'test_days': 30, 'n_steps': 12}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.8/v/missing
walk_forward.end_date
  Field required [type=missing, input_value={'train_days': 90, 'test_days': 30, 'n_steps': 12}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.8/v/missing
walk_forward.training_period_days
  Field required [type=missing, input_value={'train_days': 90, 'test_days': 30, 'n_steps': 12}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.8/v/missing
walk_forward.testing_period_days
  Field required [type=missing, input_value={'train_days': 90, 'test_days': 30, 'n_steps': 12}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.8/v/missing

### RiskConfig ✅
**Description:** Check risk config: configs/risk.yaml
**Status:** PASS
**Message:** Risk config valid

**Details:**
```json
{
  "max_daily_loss_pct": 3.0,
  "max_drawdown_pct": 25.0,
  "max_no_data_minutes": 20,
  "min_trade_count_per_day": 3,
  "position_size_usd": 100,
  "max_position_count": 10,
  "max_correlation": 0.7,
  "max_leverage": 3.0,
  "use_stop_loss": true,
  "stop_loss_pct": 15.0,
  "max_holding_hours": 240,
  "max_overnight_positions": 5,
  "volatility_lookback_days": 30,
  "volatility_adjustment_factor": 1.5,
  "emergency_liquidate_drawdown": 35.0,
  "circuit_breaker_loss_rate": 10.0
}
```

## ❌ Action Required

System is NOT ready for live trading. Please resolve all failed checks before proceeding.
