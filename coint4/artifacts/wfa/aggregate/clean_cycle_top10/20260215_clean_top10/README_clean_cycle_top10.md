# Clean cycle TOP-10 (clean_cycle_top10)

- DATESTAMP: 20260215
- cycle: 20260215_clean_top10

## Paths (relative to app-root `coint4/`)
- CLEAN_ROOT: artifacts/wfa/runs_clean/20260215_clean_top10
- BASELINE_DIR: artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10
- OPT_DIR: artifacts/wfa/runs_clean/20260215_clean_top10/opt_sweeps
- CLEAN_AGG_DIR: artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10

## FIXED_WINDOWS.walk_forward
All baseline runs and clean sweeps must use exactly these walk_forward params.

```json
{
  "end_date": "2024-12-31",
  "gap_minutes": 15,
  "max_steps": 5,
  "refit_frequency": "weekly",
  "start_date": "2024-05-01",
  "step_size_days": 30,
  "testing_period_days": 30,
  "training_period_days": 90
}
```

## Overwrite rules
- This init script is idempotent: it only creates missing directories.
- README is created once; existing README is not overwritten unless `--overwrite` is used.
