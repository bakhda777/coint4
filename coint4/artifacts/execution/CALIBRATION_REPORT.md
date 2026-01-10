# Execution Cost Calibration Report

*Generated: 2025-08-10T22:20:32.185475*

## Summary

- Pairs analyzed: 3
- Data window: 30 days
- Samples: 52,071

## Aggregate Model

```
slippage = 0.000500
         + -0.000000 * ATR_pct
         + 0.002500 * spread_proxy
         + 0.005000 * volatility

R² Score: 1.0000
```

## Piecewise Models by Volatility Regime

### Low Vol
- Intercept: 0.000500
- ATR coefficient: -0.000000
- Spread coefficient: 0.002500
- R² Score: 1.0000

### Mid Vol
- Intercept: 0.000500
- ATR coefficient: -0.000000
- Spread coefficient: 0.002500
- R² Score: 1.0000

### High Vol
- Intercept: 0.000500
- ATR coefficient: 0.000000
- Spread coefficient: 0.002500
- R² Score: 1.0000

## Market Statistics

- Average ATR %: 108.7868%
- Average spread proxy: 56.1557%
- Average volatility: 0.6009%

## Recommendations

Based on calibration results:

- ⚠️ High base slippage detected. Consider:
  - Using limit orders instead of market orders
  - Trading during more liquid hours
  - Reducing position sizes

## Configuration Update

Suggested `configs/execution.yaml` parameters:

```yaml
execution:
  base_slippage: 0.000500
  atr_multiplier: -0.000000
  vol_multiplier: 0.005000
  regime_aware: true
```
