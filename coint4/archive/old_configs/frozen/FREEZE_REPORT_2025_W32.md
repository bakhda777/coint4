> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Parameter Freeze Report

## Freeze Information
- **Version**: 2025_W32_3798d6a6
- **Week**: 2025_W32
- **Frozen At**: 2025-08-10T21:37:35.545391
- **Valid Period**: 2025-08-10 to 2025-08-17
- **Source**: manual
- **Hash**: 3798d6a6

## Frozen Parameters
```yaml
max_holding_days: 100
rolling_window: 60
zscore_exit: 0.0
zscore_threshold: 2.0

```

## Deployment Status
- **Approved**: ❌
- **Deployed**: ❌

## Files Generated
- JSON: `configs/frozen/params_2025_W32.json`
- YAML: `configs/frozen/params_2025_W32.yaml`
- Current: `configs/frozen/current.yaml`

## Next Steps
1. Review parameters for production suitability
2. Run validation: `python scripts/validate.py --config configs/frozen/params_2025_W32.yaml`
3. Deploy to production: Update live config to use frozen params
4. Monitor performance during the week
