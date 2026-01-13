> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Optuna Quick Demo Report

**Date**: 2025-08-10T19:21:06.120588
**Trials**: 20
**Best Sharpe**: 4.640

## Best Parameters

```json
{
  "rolling_window": 98,
  "z_enter": 2.023125887208523,
  "z_exit": -0.24457266548665352,
  "max_holding_period": 120
}
```

## Top 5 Trials

|    |   trial |   sharpe |   rolling_window |   z_enter |     z_exit |   max_holding_period |
|---:|--------:|---------:|-----------------:|----------:|-----------:|---------------------:|
| 12 |      12 |  4.64043 |               98 |   2.02313 | -0.244573  |                  120 |
| 14 |      14 |  4.52332 |              100 |   2.01915 | -0.493699  |                  121 |
| 11 |      11 |  4.3608  |               96 |   1.95085 | -0.485574  |                  122 |
| 15 |      15 |  3.92702 |               84 |   2.29167 | -0.0443648 |                  115 |
| 13 |      13 |  3.84386 |               87 |   2.05799 | -0.287655  |                  118 |