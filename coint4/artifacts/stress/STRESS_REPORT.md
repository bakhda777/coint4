> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Stress Test Report
*Generated: 2025-08-10 20:30:00*

## Summary

- Total scenarios: 12
- Passed: 10
- Failed: 2 (expected failures)
- Success rate: 83.3%

## Baseline Performance

| Metric | Value |
|--------|-------|
| Total PnL | $2500 |
| Win Rate | 65.0% |
| Trade Count | 150 |
| Sharpe Ratio | 1.80 |
| Max Drawdown | 3.0% |

## Detailed Results

### FeesStress

**fees_1.0x** ✅
- Config: commission_pct=0.001, slippage_pct=0.0005
- PnL Change: 0.0%
- Trade Count Change: 0.0%

**fees_2.0x** ✅
- Config: commission_pct=0.002, slippage_pct=0.001
- PnL Change: -12.5%
- Trade Count Change: 0.0%

**fees_3.0x** ✅
- Config: commission_pct=0.003, slippage_pct=0.0015
- PnL Change: -25.0%
- Trade Count Change: 0.0%

### MissingData

**missing_data_5.0%** ✅
- Config: drop_rate=0.05
- PnL Change: -2.5%
- Trade Count Change: -2.5%

**missing_data_10.0%** ✅
- Config: drop_rate=0.1
- PnL Change: -5.0%
- Trade Count Change: -5.0%

**missing_data_20.0%** ✅
- Config: drop_rate=0.2
- PnL Change: -10.0%
- Trade Count Change: -10.0%

### RollingWindow

**window_0.5x** ✅
- Config: rolling_window=30
- PnL Change: -10.0%
- Trade Count Change: +50.0%

**window_1.0x** ✅
- Config: rolling_window=60
- PnL Change: 0.0%
- Trade Count Change: 0.0%

**window_2.0x** ✅
- Config: rolling_window=120
- PnL Change: +10.0%
- Trade Count Change: -30.0%

### InvalidGap

**gap_30min** ✅
- Config: gap_minutes=30
- Error: Gap 30 minutes not supported for 15T timeframe

**gap_60min** ✅
- Config: gap_minutes=60
- Error: Gap 60 minutes not supported for 15T timeframe

**gap_15min** ✅
- Config: gap_minutes=15
- Status: PASSED

## Robustness Assessment

**Overall Assessment:** 🟢 ROBUST - System handles most stress conditions well
**Robustness Score:** 83.3%

### Key Findings

- Baseline performance maintained under normal stress
- System degrades gracefully under extreme conditions
- Validation correctly rejects invalid configurations
- Cost increases have proportional impact on profitability
- Missing data reduces opportunities but maintains stability
- Rolling window size affects trade frequency vs signal quality