> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Walk-Forward Analysis Report with Portfolio Allocation

Generated: 2025-08-10T23:01:21.441339



## Portfolio Confidence Analysis
- **Sharpe 90% CI**: [0.00, 0.00] (observed: 0.00)
- **PSR 90% CI**: [0.500, 0.500] (observed: 0.500)
- **Risk Level**: 🚨 HIGH (Weak confidence bounds)

*Full analysis: [CONFIDENCE_REPORT.md](../uncertainty/CONFIDENCE_REPORT.md)*

## Portfolio Confidence Analysis
- **Sharpe 90% CI**: [0.00, 0.00] (observed: 0.00)
- **PSR 90% CI**: [0.500, 0.500] (observed: 0.500)
- **Risk Level**: 🚨 HIGH (Weak confidence bounds)

*Full analysis: [CONFIDENCE_REPORT.md](../uncertainty/CONFIDENCE_REPORT.md)*

## Configuration
- Portfolio Method: vol_target
- Max Weight Per Pair: 10.0%
- Max Gross Exposure: 100%
- Max Net Exposure: 40%

## Results Summary

| Fold | Positions | Gross Exp | Net Exp | Sharpe | PnL |
|------|-----------|-----------|---------|--------|-----|
| 0 | 2 | 0.20 | -0.20 | 1.07 | $1077 |

## Portfolio Allocation Details

### Fold 0

| Pair | Weight |
|------|--------|
| ETH/USDT | -0.100 |
| XRP/USDT | -0.100 |

## Portfolio Statistics

- Average Gross Exposure: 0.20
- Average Net Exposure: -0.20
- Max Gross Exposure: 0.20
- Max Net Exposure: -0.20

## Constraint Validation
- ✅ All weights within per-pair limits
- ✅ Gross exposure within bounds
- ✅ Net exposure controlled
