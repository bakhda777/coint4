> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Confidence Intervals Report
## Bootstrap Configuration
- Block Size: 20
- Bootstrap Samples: 100
- Confidence Levels: [0.05, 0.5, 0.95]
- Generated: 2025-08-11 00:35

## Portfolio Overview

### Portfolio Confidence Intervals

| Metric | P05 | P50 (Median) | P95 | Observed |
|--------|-----|--------------|-----|----------|
| SHARPE | 0.000 | 0.000 | 0.000 | 0.000 |
| PSR | 0.500 | 0.500 | 0.500 | 0.500 |
| DSR | 0.500 | 0.500 | 0.500 | 0.500 |

## Individual Pair Analysis

| Pair | Metric | P05 | P50 | P95 | Observed |
|------|--------|-----|-----|-----|----------|
| BTC/USDT | SHARPE | 1.688 | 4.823 | 8.625 | 4.823 |
| BTC/USDT | PSR | 0.739 | 0.990 | 1.000 | 0.990 |
| BTC/USDT | DSR | 0.510 | 0.610 | 0.657 | 0.610 |
| ETH/USDT | SHARPE | -6.626 | -2.376 | 1.261 | -2.266 |
| ETH/USDT | PSR | 0.002 | 0.135 | 0.905 | 0.135 |
| ETH/USDT | DSR | 0.000 | 0.051 | 0.473 | 0.072 |
| ADA/USDT | SHARPE | -0.163 | 1.834 | 4.075 | 1.861 |
| ADA/USDT | PSR | 0.481 | 0.841 | 0.990 | 0.818 |
| ADA/USDT | DSR | 0.283 | 0.512 | 0.608 | 0.493 |
| SOL/USDT | SHARPE | -0.096 | 1.406 | 3.397 | 1.406 |
| SOL/USDT | PSR | 0.446 | 0.754 | 0.934 | 0.754 |
| SOL/USDT | DSR | 0.171 | 0.421 | 0.560 | 0.422 |

## Interpretation
- **P05**: 5th percentile (lower bound of 90% confidence interval)
- **P50**: Median (robust point estimate)
- **P95**: 95th percentile (upper bound of 90% confidence interval)
- **Observed**: Point estimate from original data

## Risk Assessment
- 🚨 **HIGH RISK**: Portfolio P05 Sharpe < 0.3
- 🚨 **LOW CONFIDENCE**: Portfolio P05 PSR < 0.70
