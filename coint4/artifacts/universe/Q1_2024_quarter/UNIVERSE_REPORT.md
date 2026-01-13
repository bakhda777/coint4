> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Universe Selection Report

## Summary
- **Period**: 2024-01-01 to 2024-03-31
- **Total pairs tested**: 56280
- **Pairs passed criteria**: 0
- **Pairs selected**: 0
- **Selection rate**: 0.0%

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
- Min: 0.0000
- Median: 0.0740
- Max: 1.0000
- Passed (<0.1): 30841

### Half-life Distribution
- Min: 1.0
- Median: 311.1
- Max: inf
- In range (5-300): 26752

### Hurst Exponent Distribution
- Min: 0.535
- Median: 0.999
- Max: 1.000
- In range (0.15-0.65): 10

## Top 20 Selected Pairs

| Pair | Score | P-value | Half-life | Hurst | Crossings | Beta Drift |
|------|-------|---------|-----------|-------|-----------|------------|


---
Generated: 2025-08-14T17:43:29.912748+00:00


## Rejection Breakdown

**Tested**: 56280 pairs  
**Passed**: 0 pairs

### Top Rejection Reasons:
- **pvalue**: 25439 pairs (45.2%)
- **hurst**: 24453 pairs (43.4%)
- **half_life**: 6388 pairs (11.4%)
