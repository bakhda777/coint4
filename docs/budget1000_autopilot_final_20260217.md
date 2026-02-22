# Budget $1000 autopilot: итог

Generated at (UTC): 2026-02-17T23:25:11Z
stop_reason: `max_rounds_reached: max_rounds=5`
best_score: `4.3748255681123185`
no_improvement_streak: `3`

## Лучший найденный кандидат (stop-condition достигнут)

- run_group: `20260217_budget1000_bl11_r02_corr`
- variant_id: `prod_final_budget1000_risk0p006_slusd1p91_risk0p006_vm1p0075_corr0p34_pv0p39_corr0p28_min_beta0p0005_vm1p0095_slusd2p16_corr0p26_vm1p0035_vm1p0065_max_pairs21p0_corr0p335`
- score: `4.3748255681123185`
- worst-window robust Sharpe: `4.3748255681123185`
- worst-window DD pct: `0.07652976963450397`
- sample config: `configs/budget1000_autopilot/20260217_budget1000_bl11_r02_corr/holdout_prod_final_budget1000_risk0p006_slusd1p91_risk0p006_vm1p0075_corr0p34_pv0p39_corr0p28_min_beta0p0005_vm1p0095_slusd2p16_corr0p26_vm1p0035_vm1p0065_max_pairs21p0_oos20220601_20230430_corr0p335.yaml`

## История шагов (суммарно)

- 20260217_budget1000_bl11_r01_max_pairs | knob=pair_selection.max_pairs | score=3.323 | improved=True
- 20260217_budget1000_bl11_r02_corr | knob=pair_selection.min_correlation | score=4.375 | improved=True
- 20260217_budget1000_bl11_r03_min_beta | knob=filter_params.min_beta | score=3.247 | improved=False
- 20260217_budget1000_bl11_r04_max_hurst_exponent | knob=filter_params.max_hurst_exponent | score=4.111 | improved=False
- 20260217_budget1000_bl11_r05_risk | knob=portfolio.risk_per_position_pct | score=4.386 | improved=False

## Примечания

- База: multi-window worst-case robust Sharpe = min_window(min(holdout, stress)).
- Score (если включён dd_penalty): score = worst_robust_sharpe - dd_penalty * max(0, worst_dd_pct - dd_target_pct).
- Heavy execution выполняется только на VPS через run_server_job.sh (STOP_AFTER=1).
- Heavy артефакты `coint4/artifacts/wfa/runs_clean/**` не коммитить.
