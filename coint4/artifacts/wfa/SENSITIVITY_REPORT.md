> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Sensitivity Analysis Report

## Base Case Parameters
```json
{
  "zscore_threshold": 2.0,
  "zscore_exit": 0.0,
  "rolling_window": 60,
  "max_holding_days": 100,
  "commission_pct": 0.001,
  "slippage_pct": 0.0005
}
```

## Base Case Performance
- Sharpe Ratio: 1.50
- Total PnL: $10,000
- Number of Trades: 50

## Sensitivity Results

### Parameter Impact on Sharpe Ratio

| Parameter | Min Sharpe | Base Sharpe | Max Sharpe | Range | Most Sensitive |
|-----------|------------|-------------|------------|-------|----------------|
| Rolling Window | 1.40 (30.0) | 1.50 | 1.50 (60.0) | 0.10 | 🟢 Low |
| Z-Score Exit | 1.35 (-0.5) | 1.50 | 1.50 (0) | 0.15 | 🟢 Low |
| Partial Fill Prob | 1.40 (0.3) | 1.50 | 1.55 (0) | 0.15 | 🟡 Medium |
| Z-Score Entry | 1.40 (1.5) | 1.50 | 1.60 (2.5) | 0.20 | 🟡 Medium |
| Latency | 1.30 (50.0) | 1.50 | 1.54 (1.0) | 0.24 | 🟡 Medium |
| Commission | 1.25 (0.0015) | 1.50 | 1.75 (0.0005) | 0.50 | 🔴 High |
| Slippage | 1.25 (0.0008) | 1.50 | 1.75 (0.0003) | 0.50 | 🔴 High |


## Detailed Parameter Analysis

### Z-Score Entry

| Value | Sharpe | PnL | Trades | Impact |
|-------|--------|-----|--------|--------|
| 1.5 | 1.40 | $9,500 | 48 | ➖ 6.7% |
| 1.8 | 1.45 | $9,750 | 49 | ➖ 3.3% |
| 2.0 | 1.50 | $10,000 | 50 | ➡️ 0.0% |
| 2.2 | 1.55 | $10,250 | 49 | ➕ 3.3% |
| 2.5 | 1.60 | $10,500 | 48 | ➕ 6.7% |

### Z-Score Exit

| Value | Sharpe | PnL | Trades | Impact |
|-------|--------|-----|--------|--------|
| -0.5 | 1.35 | $9,250 | 50 | ➖ 10.0% |
| -0.25 | 1.43 | $9,625 | 50 | ➖ 5.0% |
| 0 | 1.50 | $10,000 | 50 | ➡️ 0.0% |
| 0.25 | 1.43 | $9,625 | 50 | ➖ 5.0% |
| 0.5 | 1.35 | $9,250 | 50 | ➖ 10.0% |

### Rolling Window

| Value | Sharpe | PnL | Trades | Impact |
|-------|--------|-----|--------|--------|
| 30.0 days | 1.40 | $9,500 | 50 | ➖ 6.7% |
| 45.0 days | 1.45 | $9,750 | 50 | ➖ 3.3% |
| 60.0 days | 1.50 | $10,000 | 50 | ➡️ 0.0% |
| 75.0 days | 1.45 | $9,750 | 50 | ➖ 3.3% |
| 90.0 days | 1.40 | $9,500 | 50 | ➖ 6.7% |

### Commission

| Value | Sharpe | PnL | Trades | Impact |
|-------|--------|-----|--------|--------|
| 0.0005% | 1.75 | $11,250 | 50 | ➕ 16.7% |
| 0.0008% | 1.62 | $10,625 | 50 | ➕ 8.3% |
| 0.001% | 1.50 | $10,000 | 50 | ➡️ 0.0% |
| 0.0013% | 1.38 | $9,375 | 50 | ➖ 8.3% |
| 0.0015% | 1.25 | $8,750 | 50 | ➖ 16.7% |

### Slippage

| Value | Sharpe | PnL | Trades | Impact |
|-------|--------|-----|--------|--------|
| 0.0003% | 1.75 | $11,250 | 50 | ➕ 16.7% |
| 0.0004% | 1.62 | $10,625 | 50 | ➕ 8.3% |
| 0.0005% | 1.50 | $10,000 | 50 | ➡️ 0.0% |
| 0.0006% | 1.38 | $9,375 | 50 | ➖ 8.3% |
| 0.0008% | 1.25 | $8,750 | 50 | ➖ 16.7% |

### Latency

| Value | Sharpe | PnL | Trades | Impact |
|-------|--------|-----|--------|--------|
| 1.0ms | 1.54 | $10,225 | 50 | ➕ 3.0% |
| 5.0ms | 1.52 | $10,125 | 50 | ➕ 1.7% |
| 10.0ms | 1.50 | $10,000 | 50 | ➡️ 0.0% |
| 20.0ms | 1.45 | $9,750 | 50 | ➖ 3.3% |
| 50.0ms | 1.30 | $9,000 | 50 | ➖ 13.3% |

### Partial Fill Prob

| Value | Sharpe | PnL | Trades | Impact |
|-------|--------|-----|--------|--------|
| 0 | 1.55 | $10,250 | 50 | ➕ 3.3% |
| 0.05 | 1.52 | $10,125 | 50 | ➕ 1.7% |
| 0.1 | 1.50 | $10,000 | 50 | ➡️ 0.0% |
| 0.2 | 1.45 | $9,750 | 50 | ➖ 3.3% |
| 0.3 | 1.40 | $9,500 | 50 | ➖ 6.7% |

## Recommendations

### Most Sensitive Parameters
1. **Slippage**: Range of 0.50 in Sharpe
   - Requires careful optimization and monitoring
   - Consider tighter bounds in production

### Least Sensitive Parameters  
1. **Rolling Window**: Range of 0.10 in Sharpe
   - Can use wider bounds for optimization
   - Less critical for performance

### Fee Sensitivity
- ⚠️ **High fee sensitivity detected**
  - Strategy performance significantly impacted by transaction costs
  - Consider:
    - Reducing trading frequency
    - Improving entry/exit timing
    - Negotiating better rates


### Optimization Focus
Based on sensitivity analysis, prioritize optimization of:
1. Commission (impact range: 0.50)
2. Slippage (impact range: 0.50)
3. Latency (impact range: 0.24)


## Visual Analysis

See `tornado_chart.png` for visual representation of parameter sensitivities.

## Files Generated
- `sensitivity_results.csv`: Complete sensitivity data
- `tornado_chart.png`: Tornado chart visualization
- Parameter-specific CSVs in `artifacts/sensitivity/`
