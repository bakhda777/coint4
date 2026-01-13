> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Drift Monitoring Dashboard
Generated: 2025-08-11 00:35:59

## Status Overview
- **Overall Status**: 🚨 **FAIL**
- **Degradation Level**: 3/3
- **Actions Taken**: 2 action(s)

## Performance Drift Analysis

### Recent vs Baseline Comparison
| Metric | Short Window | Long Window | Change | Drop |
|--------|--------------|-------------|--------|------|
| Sharpe | 1.674 | 1.674 | 0.0% | 0.0% |
| PSR | 0.000 | 0.000 | 0.000 | 0.000 |

### Confidence Bounds Analysis

| Metric | P05 | P50 | P95 | Observed | Status |
|--------|-----|-----|-----|----------|--------|
| SHARPE | 0.000 | 0.000 | 0.000 | 0.000 | 🚨 |
| PSR | 0.500 | 0.500 | 0.500 | 0.500 | 🚨 |
| DSR | 0.500 | 0.500 | 0.500 | 0.500 | ➖ |

## Actions Taken
1. Derisk: Scaled positions to 25%
2. Portfolio: FAILED to trigger rebuild

## Threshold Reference
- **Sharpe P05 Min**: 0.6
- **PSR P05 Min**: 0.9
- **Max Sharpe Drop**: 35%
- **Max PSR Drop**: 20%

## Data Quality
- Short window observations: 5
- Long window observations: 5
- Window sizes: 14d / 60d

## Next Steps
- 🚨 **IMMEDIATE ACTION REQUIRED**: Review degradation causes
- 🔧 **Consider**: Manual portfolio review and parameter adjustment
- 📊 **Monitor**: Portfolio rebuild in progress
