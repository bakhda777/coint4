# WFA run index

Generated at: 2026-03-05 21:50:12Z

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
- selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p60_c0p35_hl0p001-100 | pair=OMGUSDT-ONEUSDT pnl=-39.11 share=9.8% | period=09/01-11/30 pnl=-84.44 share=100.0% | artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p60_c0p35_hl0p001-100
- selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p65_c0p40_hl0p001-100 | pair=OMGUSDT-ONEUSDT pnl=-39.11 share=8.7% | period=09/01-11/30 pnl=-86.30 share=100.0% | artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p65_c0p40_hl0p001-100
- selgrid_20260115_exit0p06_pv0p40_kpss0p05_h0p65_c0p40_hl0p001-100 | pair=CELOUSDT-IDUSDT pnl=-12.68 share=5.8% | period=09/01-11/30 pnl=-12.83 share=100.0% | artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p40_kpss0p05_h0p65_c0p40_hl0p001-100
- selgrid_20260115_strictpv_exit0p06_pv0p05_kpss0p05_h0p55_c0p45_hl0p01-60 | pair=AXSUSDT-UNIUSDT pnl=-12.00 share=24.8% | period=09/01-11/30 pnl=-4.02 share=100.0% | artifacts/wfa/runs/20260115_selgrid_strictpv/selgrid_20260115_strictpv_exit0p06_pv0p05_kpss0p05_h0p55_c0p45_hl0p01-60
- selgrid_20260115_strictpv_exit0p06_pv0p05_kpss0p05_h0p55_c0p45_hl0p05-60 | pair=AXSUSDT-UNIUSDT pnl=-12.00 share=26.0% | period=09/01-11/30 pnl=-2.76 share=100.0% | artifacts/wfa/runs/20260115_selgrid_strictpv/selgrid_20260115_strictpv_exit0p06_pv0p05_kpss0p05_h0p55_c0p45_hl0p05-60
- selgrid_20260115_strictpv_exit0p06_pv0p05_kpss0p05_h0p55_c0p50_hl0p01-60 | pair=AXSUSDT-UNIUSDT pnl=-12.00 share=24.8% | period=09/01-11/30 pnl=-4.02 share=100.0% | artifacts/wfa/runs/20260115_selgrid_strictpv/selgrid_20260115_strictpv_exit0p06_pv0p05_kpss0p05_h0p55_c0p50_hl0p01-60
- ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd5000 | pair=CELOUSDT-IDUSDT pnl=-12.68 share=5.8% | period=09/01-11/30 pnl=-12.83 share=100.0% | artifacts/wfa/runs/20260115_ssd_topn_sweep/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd5000
- ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd5000 | pair=CELOUSDT-IDUSDT pnl=-12.68 share=5.8% | period=09/01-11/30 pnl=-12.83 share=100.0% | artifacts/wfa/runs/20260115_ssd_topn_sweep_4vals/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd5000
- baseline_20260118_z0p85_exit0p12_corr0p65_ssd25000 | pair=ENSUSDT-NYMUSDT pnl=-15.83 share=4.8% | period=11/30-02/28 pnl=-15.56 share=100.0% | artifacts/wfa/runs/20260118_baseline/baseline_20260118_z0p85_exit0p12_corr0p65_ssd25000
- shortlist_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000 | pair=ENSUSDT-NYMUSDT pnl=-15.83 share=4.8% | period=11/30-02/28 pnl=-15.56 share=100.0% | artifacts/wfa/runs/20260118_shortlist/shortlist_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000
