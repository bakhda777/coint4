# Budget $1000 autopilot: итог

Generated at (UTC): 2026-02-16T02:02:14Z

## Лучший найденный кандидат (max_rounds достигнут)

- run_group: `20260215_budget1000_ap_r03_risk`
- variant_id: `prod_final_budget1000_risk0p019_slusd6p5_slusd4p5_vm1p0035_risk0p015`
- score: `2.3571884529063536`
- worst-window robust Sharpe: `3.3102227982544865`
- worst-window DD pct: `0.34060686906962656`
- knobs (из sample config): `risk_per_position_pct=0.015`, `pair_stop_loss_usd=4.5`, `max_var_multiplier=1.0035`
- sample config: `configs/budget1000_autopilot/20260215_budget1000_ap_r03_risk/holdout_prod_final_budget1000_oos20220601_20230430_risk0p019_oos20220601_20230430_slusd6p5_oos20220601_20230430_slusd4p5_oos20220601_20230430_vm1p0035_oos20220601_20230430_risk0p015.yaml`

## История шагов (суммарно)

- 20260215_budget1000_ap_r01_risk | knob=portfolio.risk_per_position_pct | score=0.811 | improved=True
- 20260215_budget1000_ap_r01_slusd | knob=backtest.pair_stop_loss_usd | score=1.907 | improved=True
- 20260215_budget1000_ap_r01_vm | knob=backtest.max_var_multiplier | score=1.907 | improved=False
- 20260215_budget1000_ap_r02_risk | knob=portfolio.risk_per_position_pct | score=1.907 | improved=False
- 20260215_budget1000_ap_r02_slusd | knob=backtest.pair_stop_loss_usd | score=2.112 | improved=True
- 20260215_budget1000_ap_r02_vm | knob=backtest.max_var_multiplier | score=2.156 | improved=True
- 20260215_budget1000_ap_r03_risk | knob=portfolio.risk_per_position_pct | score=2.357 | improved=True
- 20260215_budget1000_ap_r03_slusd | knob=backtest.pair_stop_loss_usd | score=2.357 | improved=False
- 20260215_budget1000_ap_r03_vm | knob=backtest.max_var_multiplier | score=2.357 | improved=False

## Примечания

- База: multi-window worst-case robust Sharpe = min_window(min(holdout, stress)).
- Score (если включён dd_penalty): score = worst_robust_sharpe - dd_penalty * max(0, worst_dd_pct - dd_target_pct).
- Heavy execution выполняется только на VPS через run_server_job.sh (STOP_AFTER=1).
- Heavy артефакты `coint4/artifacts/wfa/runs_clean/**` не коммитить.
