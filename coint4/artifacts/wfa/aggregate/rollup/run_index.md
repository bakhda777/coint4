# WFA run index

Generated at: 2026-02-24 10:44:32Z

Notes:
- `sharpe_ratio_abs` is recomputed from `equity_curve.csv` with inferred bar frequency (periods/year = 365 * periods/day).
- `sharpe_ratio_abs_raw` is the value stored in `strategy_metrics.csv` (legacy runs may be under-annualized).
- `psr` / `dsr` are computed from run returns (`equity_curve.csv` fallback: `daily_pnl.csv`); `dsr_trials` is inferred from run_group size.
- `tail_loss_*` fields are derived from `trade_statistics.csv` and show net tail-loss concentration by pair and WF period.

## Top Sharpe
- sharpe_sweep_20260214_corr0p6_cpv0p08_ptn1000_exit0p12 | sharpe=9.3848 pnl=63865.70 dd=-12107.20 | artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p6_cpv0p08_ptn1000_exit0p12
- sharpe_sweep_20260214_corr0p6_cpv0p08_ptn1500_exit0p12 | sharpe=9.3848 pnl=63865.70 dd=-12107.20 | artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p6_cpv0p08_ptn1500_exit0p12
- sharpe_sweep_20260214_corr0p55_cpv0p08_ptn1000_exit0p12 | sharpe=9.2709 pnl=62500.52 dd=-11848.45 | artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p55_cpv0p08_ptn1000_exit0p12
- sharpe_sweep_20260214_corr0p55_cpv0p08_ptn1500_exit0p12 | sharpe=9.2709 pnl=62500.52 dd=-11848.45 | artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p55_cpv0p08_ptn1500_exit0p12
- holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2 | sharpe=9.0907 pnl=1135.26 dd=-82.22 | artifacts/wfa/runs/20260120_realcost_shortlist/holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2
- holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2 | sharpe=9.0907 pnl=1135.26 dd=-82.22 | artifacts/wfa/runs/20260123_relaxed8_nokpss_u250_churnfix_top50_churngrid/holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2
- risk_sweep_20260116_cooldown4 | sharpe=9.0347 pnl=61992.33 dd=-6923.66 | artifacts/wfa/runs/20260116_risk_sweep/risk_sweep_20260116_cooldown4
- holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1 | sharpe=9.0140 pnl=1114.71 dd=-82.13 | artifacts/wfa/runs/20260120_realcost_shortlist/holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1
- holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1 | sharpe=9.0140 pnl=1114.71 dd=-82.13 | artifacts/wfa/runs/20260122_relaxed8_nokpss_u250_churnfix_top50_sens/holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1
- holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p15 | sharpe=8.9498 pnl=1107.52 dd=-80.32 | artifacts/wfa/runs/20260120_realcost_shortlist/holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p15

## Top PnL
- sharpe_sweep_20260214_corr0p55_cpv0p05_ptn1000_exit0p12 | sharpe=8.5208 pnl=73813.99 dd=-10603.76 | artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p55_cpv0p05_ptn1000_exit0p12
- sharpe_sweep_20260214_corr0p55_cpv0p05_ptn1500_exit0p12 | sharpe=8.5208 pnl=73813.99 dd=-10603.76 | artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p55_cpv0p05_ptn1500_exit0p12
- sharpe_sweep_20260214_corr0p6_cpv0p05_ptn1000_exit0p12 | sharpe=8.5208 pnl=73813.99 dd=-10603.76 | artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p6_cpv0p05_ptn1000_exit0p12
- sharpe_sweep_20260214_corr0p6_cpv0p05_ptn1500_exit0p12 | sharpe=8.5208 pnl=73813.99 dd=-10603.76 | artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p6_cpv0p05_ptn1500_exit0p12
- sharpe_sweep_20260214_corr0p6_cpv0p08_ptn1000_exit0p12 | sharpe=9.3848 pnl=63865.70 dd=-12107.20 | artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p6_cpv0p08_ptn1000_exit0p12
- sharpe_sweep_20260214_corr0p6_cpv0p08_ptn1500_exit0p12 | sharpe=9.3848 pnl=63865.70 dd=-12107.20 | artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p6_cpv0p08_ptn1500_exit0p12
- sharpe_sweep_20260214_corr0p55_cpv0p08_ptn1000_exit0p12 | sharpe=9.2709 pnl=62500.52 dd=-11848.45 | artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p55_cpv0p08_ptn1000_exit0p12
- sharpe_sweep_20260214_corr0p55_cpv0p08_ptn1500_exit0p12 | sharpe=9.2709 pnl=62500.52 dd=-11848.45 | artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p55_cpv0p08_ptn1500_exit0p12
- risk_sweep_20260116_cooldown4 | sharpe=9.0347 pnl=61992.33 dd=-6923.66 | artifacts/wfa/runs/20260116_risk_sweep/risk_sweep_20260116_cooldown4
- ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd40000 | sharpe=6.6158 pnl=58548.58 dd=-17209.17 | artifacts/wfa/runs/20260115_ssd_topn_sweep_3vals/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd40000

## Tail-Loss Offenders
- 20260114_071935_main_2024_wfa | pair=ANKRUSDT-MVUSDT pnl=-19.21 share=9.5% | period=05/30-08/28 pnl=-37.40 share=100.0% | artifacts/wfa/runs/20260114_071935_main_2024_wfa
- 20260114_075638_smoke_fast20_wfa | pair=BTCDAI-ETHDAI pnl=-19.78 share=16.7% | period=12/12-01/11 pnl=-36.69 share=100.0% | artifacts/wfa/runs/20260114_075638_smoke_fast20_wfa
- 20260114_075758_smoke_fast20_wfa_repeat | pair=BTCDAI-ETHDAI pnl=-19.78 share=16.7% | period=12/12-01/11 pnl=-36.69 share=100.0% | artifacts/wfa/runs/20260114_075758_smoke_fast20_wfa_repeat
- 20260114_081835_main_2024_wfa_refresh | pair=ANKRUSDT-MVUSDT pnl=-19.21 share=9.5% | period=05/30-08/28 pnl=-37.40 share=100.0% | artifacts/wfa/runs/20260114_081835_main_2024_wfa_refresh
- 20260114_084102_smoke_time_stop_clamp | pair=BTCDAI-ETHDAI pnl=-19.78 share=16.7% | period=12/12-01/11 pnl=-36.69 share=100.0% | artifacts/wfa/runs/20260114_084102_smoke_time_stop_clamp
- 20260114_093244_main_2024_wfa_step5 | pair=ANKRUSDT-MVUSDT pnl=-19.21 share=6.5% | period=05/30-08/28 pnl=-74.42 share=100.0% | artifacts/wfa/runs/20260114_093244_main_2024_wfa_step5
- 20260114_095105_main_2024_wfa_step5_repeat | pair=ANKRUSDT-MVUSDT pnl=-19.21 share=6.5% | period=05/30-08/28 pnl=-74.42 share=100.0% | artifacts/wfa/runs/20260114_095105_main_2024_wfa_step5_repeat
- 20260114_145500_optimize_q4_sanity_wfa_zscore_0p8 | pair=CELOUSDT-IDUSDT pnl=-12.68 share=5.8% | period=09/01-11/30 pnl=-10.97 share=100.0% | artifacts/wfa/runs/20260114_145500_optimize_q4_sanity_wfa_zscore_0p8
- 20260114_153500_optimize_q4_sanity_wfa_zscore_0p7 | pair=OMGUSDT-QTUMUSDT pnl=-20.82 share=5.4% | period=09/01-11/30 pnl=-56.58 share=100.0% | artifacts/wfa/runs/20260114_153500_optimize_q4_sanity_wfa_zscore_0p7
- 20260114_154800_optimize_q4_sanity_wfa_zscore_0p8_cd1 | pair=CELOUSDT-IDUSDT pnl=-12.68 share=5.8% | period=09/01-11/30 pnl=-10.97 share=100.0% | artifacts/wfa/runs/20260114_154800_optimize_q4_sanity_wfa_zscore_0p8_cd1
