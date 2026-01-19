# WFA run index

Generated at: 2026-01-19 16:04:55Z

Notes:
- `sharpe_ratio_abs` is recomputed from `equity_curve.csv` with inferred bar frequency (periods/year = 365 * periods/day).
- `sharpe_ratio_abs_raw` is the value stored in `strategy_metrics.csv` (legacy runs may be under-annualized).

## Top Sharpe
- pair_sweep_20260117_corr0p55_z0p85_exit0p12_ssd25000 | sharpe=7.3013 pnl=666.32 dd=-95.32 | artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_corr0p55_z0p85_exit0p12_ssd25000
- pair_sweep_20260117_corr0p6_z0p85_exit0p12_ssd25000 | sharpe=7.3013 pnl=666.32 dd=-95.32 | artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_corr0p6_z0p85_exit0p12_ssd25000
- risk_sweep_20260118_risk0p01_pos10_margin0p4_kelly0p15_z0p85_exit0p12_corr0p65_ssd25000 | sharpe=7.2187 pnl=653.63 dd=-88.17 | artifacts/wfa/runs/20260118_risk_sweep/risk_sweep_20260118_risk0p01_pos10_margin0p4_kelly0p15_z0p85_exit0p12_corr0p65_ssd25000
- risk_sweep_20260118_risk0p008_pos8_margin0p35_kelly0p15_z0p85_exit0p12_corr0p65_ssd25000 | sharpe=7.1928 pnl=650.27 dd=-88.17 | artifacts/wfa/runs/20260118_risk_sweep/risk_sweep_20260118_risk0p008_pos8_margin0p35_kelly0p15_z0p85_exit0p12_corr0p65_ssd25000
- risk_sweep_20260118_risk0p012_pos12_margin0p45_kelly0p2_z0p85_exit0p12_corr0p65_ssd25000 | sharpe=7.1791 pnl=650.51 dd=-88.17 | artifacts/wfa/runs/20260118_risk_sweep/risk_sweep_20260118_risk0p012_pos12_margin0p45_kelly0p2_z0p85_exit0p12_corr0p65_ssd25000
- corr_ab_20260118_corr0p65_thr0p65_z0p85_exit0p12_ssd25000 | sharpe=7.1746 pnl=651.02 dd=-88.17 | artifacts/wfa/runs/20260118_corr_ab/corr_ab_20260118_corr0p65_thr0p65_z0p85_exit0p12_ssd25000
- quality_sweep_20260118_corr0p65_z0p85_exit0p12_ssd25000 | sharpe=7.1746 pnl=651.02 dd=-88.17 | artifacts/wfa/runs/20260118_quality_sweep/quality_sweep_20260118_corr0p65_z0p85_exit0p12_ssd25000
- pair_sweep_20260117_corr0p6_pv0p03_top800_z0p85_exit0p12_ssd25000 | sharpe=7.0452 pnl=621.19 dd=-91.63 | artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_corr0p6_pv0p03_top800_z0p85_exit0p12_ssd25000
- quality_sweep_20260118_corr0p7_z0p85_exit0p12_ssd25000 | sharpe=6.8143 pnl=545.01 dd=-88.17 | artifacts/wfa/runs/20260118_quality_sweep/quality_sweep_20260118_corr0p7_z0p85_exit0p12_ssd25000
- pair_sweep_20260117_hl0p1_30_corr0p6_z0p85_exit0p12_ssd25000 | sharpe=6.6920 pnl=547.88 dd=-95.32 | artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_hl0p1_30_corr0p6_z0p85_exit0p12_ssd25000

## Top PnL
- leader_validate_20260116_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000 | sharpe=5.1489 pnl=1388.71 dd=-199.31 | artifacts/wfa/runs/20260116_leader_validation/leader_validate_20260116_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000
- ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000 | sharpe=6.3707 pnl=1205.94 dd=-199.31 | artifacts/wfa/runs/20260115_ssd_topn_sweep/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000
- ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000 | sharpe=6.3707 pnl=1205.94 dd=-199.31 | artifacts/wfa/runs/20260115_ssd_topn_sweep_4vals/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000
- ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd30000 | sharpe=5.2403 pnl=1154.73 dd=-201.94 | artifacts/wfa/runs/20260115_ssd_topn_sweep_3vals/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd30000
- leader_validate_20260118_z0p85_exit0p12_ssd25000 | sharpe=5.6285 pnl=1007.18 dd=-95.32 | artifacts/wfa/runs/20260118_leader_validation/leader_validate_20260118_z0p85_exit0p12_ssd25000
- stability_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2 | sharpe=5.8116 pnl=958.72 dd=-188.98 | artifacts/wfa/runs/20260119_stability_relaxed5/stability_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2
- pair_sweep_20260117_pv0p03_top800_kpss0p03_z0p85_exit0p12_ssd25000 | sharpe=5.2749 pnl=946.25 dd=-193.91 | artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_pv0p03_top800_kpss0p03_z0p85_exit0p12_ssd25000
- leader_validate_20260118_z0p8_exit0p12_ssd25000 | sharpe=5.1320 pnl=941.02 dd=-110.02 | artifacts/wfa/runs/20260118_leader_validation/leader_validate_20260118_z0p8_exit0p12_ssd25000
- stability_relaxed7_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2_t90 | sharpe=5.2468 pnl=914.61 dd=-201.41 | artifacts/wfa/runs/20260119_stability_relaxed7/stability_relaxed7_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2_t90
- stability_relaxed6_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2 | sharpe=5.3588 pnl=894.35 dd=-123.46 | artifacts/wfa/runs/20260119_stability_relaxed6/stability_relaxed6_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2
