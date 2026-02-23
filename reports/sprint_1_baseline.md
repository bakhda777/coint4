# Sprint 1 — Baseline snapshot (top-10 rollup)

- Дата: 2026-02-23
- Источник: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` (rollup generated at: 2026-02-23 12:26:39Z)
- Сортировка: `Sharpe desc`, затем `|DD| asc`, затем `Trades desc` (best-effort как в `tools/sprint_manager.py`)

| Sharpe | \|DD\| | Trades | config_path | run_group |
|---:|---:|---:|---|---|
| 9.385 | 0.236 | 15933 | configs/_tmp_sharpe_sweep_20260214/sharpe_sweep_20260214_corr0p6_cpv0p08_ptn1000_exit0p12.yaml | 20260214_sharpe_sweep |
| 9.385 | 0.236 | 15933 | configs/_tmp_sharpe_sweep_20260214/sharpe_sweep_20260214_corr0p6_cpv0p08_ptn1500_exit0p12.yaml | 20260214_sharpe_sweep |
| 9.271 | 0.252 | 15992 | configs/_tmp_sharpe_sweep_20260214/sharpe_sweep_20260214_corr0p55_cpv0p08_ptn1000_exit0p12.yaml | 20260214_sharpe_sweep |
| 9.271 | 0.252 | 15992 | configs/_tmp_sharpe_sweep_20260214/sharpe_sweep_20260214_corr0p55_cpv0p08_ptn1500_exit0p12.yaml | 20260214_sharpe_sweep |
| 9.091 | 0.008 | 11384 | configs/holdout_20260123_relaxed8_nokpss_u250_churnfix_top50_churngrid/holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2.yaml | 20260120_realcost_shortlist |
| 9.091 | 0.008 | 11384 | configs/holdout_20260123_relaxed8_nokpss_u250_churnfix_top50_churngrid/holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2.yaml | 20260123_relaxed8_nokpss_u250_churnfix_top50_churngrid |
| 9.035 | 0.212 | 10776 | configs/risk_sweep_20260116/risk_sweep_20260116_cooldown4.yaml | 20260116_risk_sweep |
| 9.014 | 0.008 | 11414 | configs/holdout_20260122_relaxed8_nokpss_u250_churnfix_top50_sens/holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1.yaml | 20260120_realcost_shortlist |
| 9.014 | 0.008 | 11414 | configs/holdout_20260122_relaxed8_nokpss_u250_churnfix_top50_sens/holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1.yaml | 20260122_relaxed8_nokpss_u250_churnfix_top50_sens |
| 8.950 | 0.008 | 11400 | configs/holdout_20260123_relaxed8_nokpss_u250_churnfix_top50_churngrid/holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p15.yaml | 20260120_realcost_shortlist |

## Краткий вывод (что подозрительно / что проверить дальше)
- Sharpe ~9+ выглядит экстремально высоким — стоит перепроверить корректность annualization/частоты баров (см. `run_index.md` про пересчёт `sharpe_ratio_abs`).
- `sharpe_sweep_*` даёт высокий Sharpe при заметном DD (~24–25%) и очень большом числе трейдов — проверить модель комиссий/скольжения и концентрацию tail-loss.
- Есть дубли с одинаковыми метриками (например, `ptn1000` vs `ptn1500`, а также одинаковый holdout-конфиг в разных `run_group`) — при отборе кандидатов полезна дедупликация/robust-критерий.
