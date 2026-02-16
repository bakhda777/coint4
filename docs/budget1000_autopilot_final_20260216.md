# Budget $1000 autopilot: итог

Generated at (UTC): 2026-02-16T08:06:56Z
stop_reason: `no_improvement_streak_reached: streak=1, rounds=1, min_improvement=0.02`
best_score: `2.8563555378997316`
no_improvement_streak: `1`

## Лучший найденный кандидат (stop-condition достигнут)

- run_group: `20260216_budget1000_cl_r02_risk`
- variant_id: `prod_final_budget1000_risk0p019_slusd6p5_slusd4p5_vm1p0035_risk0p015_slusd2p5_risk0p011_risk0p009`
- score: `2.8563555378997316`
- worst-window robust Sharpe: `3.269437519079036`
- worst-window DD pct: `0.20163524764741306`
- sample config: `configs/budget1000_autopilot/20260216_budget1000_cl_r02_risk/holdout_prod_final_budget1000_risk0p019_slusd6p5_slusd4p5_vm1p0035_risk0p015_slusd2p5_risk0p011_oos20220601_20230430_risk0p009.yaml`

## История шагов (суммарно)

- 20260216_budget1000_cl_r01_risk | knob=portfolio.risk_per_position_pct | score=2.606 | improved=True
- 20260216_budget1000_cl_r02_risk | knob=portfolio.risk_per_position_pct | score=2.856 | improved=True
- 20260216_budget1000_cl_r03_risk | knob=portfolio.risk_per_position_pct | score=2.064 | improved=False

## Примечания

- База: multi-window worst-case robust Sharpe = min_window(min(holdout, stress)).
- Score (если включён dd_penalty): score = worst_robust_sharpe - dd_penalty * max(0, worst_dd_pct - dd_target_pct).
- Heavy execution выполняется только на VPS через run_server_job.sh (STOP_AFTER=1).
- Heavy артефакты `coint4/artifacts/wfa/runs_clean/**` не коммитить.
