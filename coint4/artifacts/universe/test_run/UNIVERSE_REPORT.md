# Universe Selection Report

## Summary
- **Period**: 2024-01-01 to 2024-01-15
- **Total pairs tested**: 15
- **Pairs passed criteria**: 1
- **Pairs selected**: 1
- **Selection rate**: 6.7%

## Criteria Used
```yaml
beta_drift_max: 0.2
coint_pvalue_max: 0.1
hl_max: 300
hl_min: 5
hurst_max: 0.65
hurst_min: 0.15
min_cross: 8

```

## Distributions

### P-value Distribution
- Min: 0.0006
- Median: 0.0691
- Max: 0.6000
- Passed (<0.1): 10

### Half-life Distribution
- Min: 12.7
- Median: 282.7
- Max: 693.4
- In range (5-300): 8

## Top 20 Selected Pairs

| Pair | Score | P-value | Half-life | Crossings | Beta Drift |
|------|-------|---------|-----------|-----------|------------|
| ADAEUR/ADAUSDC | 1.633 | 0.0697 | 12.7 | 708 | 0.034 |


---
Generated: 2025-08-17T19:54:02.834667+00:00


## Rejection Breakdown

**Tested**: 15 pairs  
**Passed**: 1 pairs

### Top Rejection Reasons:
- **beta_drift**: 7 pairs (46.7%)
- **pvalue**: 5 pairs (33.3%)
- **half_life**: 2 pairs (13.3%)
