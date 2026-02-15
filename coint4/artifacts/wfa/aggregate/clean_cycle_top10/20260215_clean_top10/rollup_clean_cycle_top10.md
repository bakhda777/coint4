# rollup_clean_cycle_top10

- sort_mode: score
- score_lambda_dd: 0.02
- filters: status==completed, canonical_metrics_present==true
- fixed_windows_fingerprint: eef1a68cab0c0af9dbb9bedbdbbe410066a5a650f4dbdc27e4568d12a7d51fa0
- rows: 13

## Metrics source-of-truth

This rollup reads ONLY `canonical_metrics.json` from each `results_dir`:

- `metrics.canonical_sharpe`
- `metrics.canonical_pnl_abs`
- `metrics.canonical_max_drawdown_abs`

`canonical_metrics.json` is expected to be produced from `equity_curve.csv` via `scripts/optimization/recompute_canonical_metrics.py` (see `coint2.core.canonical_metrics`).

## Canonical Sharpe definition (`canonical_sharpe`)

As implemented in `coint2.core.sharpe.annualized_sharpe_ratio_from_equity`:

- returns: `(equity_t - equity_{t-1}) / equity_{t-1}`
- sharpe: `sqrt(periods_per_year) * mean(excess_returns) / std(excess_returns)`
- std: sample stdev (ddof=1); `risk_free_rate` is per-period (canonical_metrics uses 0.0)

Annualization parameters (`periods_per_year` / `bar_minutes`) are recorded in `canonical_metrics.json` under `meta.annualization`.

## Score definition (for sorting)

`score = canonical_sharpe - lambda_dd * abs(canonical_max_drawdown_abs)`

## FIXED_WINDOWS.walk_forward (normalized)

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

## Top-20

| rank | phase | run_name | canonical_sharpe | canonical_max_drawdown_abs | canonical_pnl_abs | score | results_dir |
| ---: | :---: | :------- | ---------------: | ------------------------: | ----------------: | ----: | :---------- |
| 1 | baseline | b07_20260125_realcost_churngrid_holdout_relaxed8_nokpss_20260125_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2 | 8.791508142746425 | -60.671365362853976 | 1151.4024041534722 | 7.578080835489345 | artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10/b07_20260125_realcost_churngrid_holdout_relaxed8_nokpss_20260125_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2 |
| 2 | baseline | b08_20260128_realcost_zgrid_top30_holdout_relaxed8_nokpss_20260128_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2 | 8.791508142746425 | -60.671365362853976 | 1151.4024041534722 | 7.578080835489345 | artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10/b08_20260128_realcost_zgrid_top30_holdout_relaxed8_nokpss_20260128_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2 |
| 3 | baseline | b09_20260129_realcost_riskgrid_top30_holdout_relaxed8_nokpss_20260129_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2_r0p015_maxpos15 | 8.791508142746425 | -60.671365362853976 | 1151.4024041534722 | 7.578080835489345 | artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10/b09_20260129_realcost_riskgrid_top30_holdout_relaxed8_nokpss_20260129_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2_r0p015_maxpos15 |
| 4 | baseline | b10_20260129_realcost_riskgrid_top30_holdout_relaxed8_nokpss_20260129_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2_r0p01_maxpos15 | 8.791508142746425 | -60.671365362853976 | 1151.4024041534722 | 7.578080835489345 | artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10/b10_20260129_realcost_riskgrid_top30_holdout_relaxed8_nokpss_20260129_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2_r0p01_maxpos15 |
| 5 | baseline | b01_20260120_realcost_shortlist_holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2 | 9.090691822685168 | -82.22392571321689 | 1135.2603312664796 | 7.44621330842083 | artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10/b01_20260120_realcost_shortlist_holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2 |
| 6 | baseline | b02_20260123_relaxed8_nokpss_u250_churnfix_top50_churngrid_holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2 | 9.090691822685168 | -82.22392571321689 | 1135.2603312664796 | 7.44621330842083 | artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10/b02_20260123_relaxed8_nokpss_u250_churnfix_top50_churngrid_holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2 |
| 7 | baseline | b03_20260120_realcost_shortlist_holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1 | 9.014018513951408 | -82.13105578476097 | 1114.7130215086654 | 7.371397398256188 | artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10/b03_20260120_realcost_shortlist_holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1 |
| 8 | baseline | b04_20260122_relaxed8_nokpss_u250_churnfix_top50_sens_holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1 | 9.014018513951408 | -82.13105578476097 | 1114.7130215086654 | 7.371397398256188 | artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10/b04_20260122_relaxed8_nokpss_u250_churnfix_top50_sens_holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1 |
| 9 | baseline | b05_20260120_realcost_shortlist_holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p15 | 8.949843760534021 | -80.32265747000929 | 1107.5211531952737 | 7.343390611133835 | artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10/b05_20260120_realcost_shortlist_holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p15 |
| 10 | baseline | b06_20260123_relaxed8_nokpss_u250_churnfix_top50_churngrid_holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p15 | 8.949843760534021 | -80.32265747000929 | 1107.5211531952737 | 7.343390611133835 | artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10/b06_20260123_relaxed8_nokpss_u250_churnfix_top50_churngrid_holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p15 |
| 11 | sweep | s001_b07_ms0p1 | 0.0 | 0.0 | 0.0 | 0.0 | artifacts/wfa/runs_clean/20260215_clean_top10/opt_sweeps/s001_b07_ms0p1 |
| 12 | sweep | s002_b07_ms0p15 | 0.0 | 0.0 | 0.0 | 0.0 | artifacts/wfa/runs_clean/20260215_clean_top10/opt_sweeps/s002_b07_ms0p15 |
| 13 | sweep | s003_b07_ms0p2 | 0.0 | 0.0 | 0.0 | 0.0 | artifacts/wfa/runs_clean/20260215_clean_top10/opt_sweeps/s003_b07_ms0p2 |

