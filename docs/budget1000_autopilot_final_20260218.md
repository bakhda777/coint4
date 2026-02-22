# Budget $1000 autopilot: итог

Generated at (UTC): 2026-02-18T03:06:41Z
stop_reason: `max_rounds_reached: max_rounds=5`
best_score: `3.472401402269669`
no_improvement_streak: `4`

## Лучший найденный кандидат (stop-condition достигнут)

- run_group: `20260218_budget1000_bl13_r01_max_pairs`
- variant_id: `prod_final_budget1000_risk0p006_slusd1p91_risk0p006_vm1p0075_corr0p34_pv0p39_corr0p28_min_beta0p0005_vm1p0095_slusd2p16_corr0p26_vm1p0035_vm1p0065_max_pairs21p0_corr0p335_risk0p007_risk0p0065_vm1p0095_max_pairs18p0`
- score: `3.472401402269669`
- worst-window robust Sharpe: `3.472401402269669`
- worst-window DD pct: `0.07630420619363759`
- sample config: `configs/budget1000_autopilot/20260218_budget1000_bl13_r01_max_pairs/holdout_prod_final_budget1000_risk0p006_slusd1p91_risk0p006_vm1p0075_corr0p34_pv0p39_corr0p28_min_beta0p0005_vm1p0095_slusd2p16_corr0p26_vm1p0035_vm1p0065_max_pairs21p0_corr0p335_risk0p007_risk0p0065_vm1p0095_oos20220601_20230430_max_pairs18p0.yaml`

## История шагов (суммарно)

- 20260218_budget1000_bl13_r01_max_pairs | knob=pair_selection.max_pairs | score=3.472 | improved=True
- 20260218_budget1000_bl13_r02_corr | knob=pair_selection.min_correlation | score=0.000 | improved=False
- 20260218_budget1000_bl13_r03_pv | knob=pair_selection.coint_pvalue_threshold | score=0.000 | improved=False
- 20260218_budget1000_bl13_r04_min_beta | knob=filter_params.min_beta | score=0.000 | improved=False
- 20260218_budget1000_bl13_r05_max_hurst_exponent | knob=filter_params.max_hurst_exponent | score=0.000 | improved=False

## Примечания

- База: multi-window worst-case robust Sharpe = min_window(min(holdout, stress)).
- Score (если включён dd_penalty): score = worst_robust_sharpe - dd_penalty * max(0, worst_dd_pct - dd_target_pct).
- Heavy execution выполняется только на VPS через run_server_job.sh (STOP_AFTER=1).
- Heavy артефакты `coint4/artifacts/wfa/runs_clean/**` не коммитить.
