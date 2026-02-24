# Optimization runs — 2026-02-24 (S3: remote tailguard holdout queue)

## S3: remote tailguard holdout (top-6, max_steps<=5) — план

**run_group:** `20260224_s3_tailguard_holdout_ms5_top6` (12 задач; 6 кандидатов × holdout+stress)

Что запускаем:
- Baseline: `tailguard_r04_v07_h2_quality_mild` на holdout-окне `2024-05-01 → 2025-06-30`.
- Top-3 из `20260223_tailguard_r07_fullspan_confirm_top3` (v01/v02/v03) на том же окне.
- Top-2 из `20260223_tailguard_r05b_ddfocus_fixend` (v14/v11) на том же окне.
- Во всех конфигах: `walk_forward.max_steps: 5` (queue-guardrail).

Артефакты/очередь:
- `coint4/artifacts/wfa/aggregate/20260224_s3_tailguard_holdout_ms5_top6/run_queue.csv`
- результаты: `coint4/artifacts/wfa/runs/20260224_s3_tailguard_holdout_ms5_top6/<run_id>/`

Критерии успеха (после выполнения):
- Holdout: `coverage_ratio>=0.95`, `min_pairs>=20`, `min_trades>=200`, `pnl>0`, и `worst_dd_pct<=0.20`.
- Stress: `pnl>0` и `worst_dd_pct<=0.25` (при прочих равных).
- Победитель: максимальный `score(worst_robust_sharpe)` среди вариантов, прошедших gates.

Команда исполнения (скелет, запускать с рабочей машины, не здесь):
- `STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'bash scripts/batch/run_heavy_queue.sh --queue artifacts/wfa/aggregate/20260224_s3_tailguard_holdout_ms5_top6/run_queue.csv'`
