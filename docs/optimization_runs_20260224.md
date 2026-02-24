# Optimization runs — 2026-02-24 (S3–S7: remote tailguard holdout queues)

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

## Postprocess: sync queue + rollup (2026-02-24)

- Best-effort синхронизация статусов очередей:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/sync_queue_status.py --queue-dir artifacts/wfa/aggregate --from-statuses planned`
  - Изменений не потребовалось (planned остаётся там, где нет `strategy_metrics.csv`).
- Пересобран rollup индекс:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
  - `run_index` обновлён (entries=8321).

## S4: remote tailguard holdout (top-10, max_steps<=5) — план

**run_group:** `20260224_s4_tailguard_holdout_ms5_top10` (20 задач; 10 кандидатов × holdout+stress)

Что запускаем (кандидаты):
- Из S3 (top-6): `r04_v07`, `r07_v01..v03`, `r05b_v14`, `r05b_v11` на holdout-окне `2024-05-01 → 2025-06-30`.
- Добивка до top-10: `r05b_v03`, `r05b_v05`, `r04_v08`, `r04_v11` на том же окне.
- Во всех конфигах: `walk_forward.max_steps: 5` (queue-guardrail).

Артефакты/очередь:
- `coint4/artifacts/wfa/aggregate/20260224_s4_tailguard_holdout_ms5_top10/run_queue.csv`
- результаты: `coint4/artifacts/wfa/runs/20260224_s4_tailguard_holdout_ms5_top10/<run_id>/`

Критерии успеха (после выполнения):
- Holdout: `coverage_ratio>=0.95`, `min_pairs>=20`, `min_trades>=200`, `pnl>0`, и `worst_dd_pct<=0.20`.
- Stress: `pnl>0` и `worst_dd_pct<=0.25` (при прочих равных).
- Победитель: максимальный `score(worst_robust_sharpe)` среди вариантов, прошедших gates.

Команда исполнения (скелет, запускать с рабочей машины, не здесь):
- `STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'bash scripts/batch/run_heavy_queue.sh --queue artifacts/wfa/aggregate/20260224_s4_tailguard_holdout_ms5_top10/run_queue.csv'`

## Postprocess: sync queue + rollup (2026-02-24, refresh)

- Best-effort синхронизация статусов очередей (S1–S4 группы):
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/20260223_s1_remote_ms5_queue/run_queue.csv --queue artifacts/wfa/aggregate/20260223_s2_tailguard_holdout_ms5_top3/run_queue.csv --queue artifacts/wfa/aggregate/20260223_s3_tailguard_holdout_ms5_top6/run_queue.csv --queue artifacts/wfa/aggregate/20260224_s3_tailguard_holdout_ms5_top6/run_queue.csv --queue artifacts/wfa/aggregate/20260224_s4_tailguard_holdout_ms5_top10/run_queue.csv`
  - Изменений не потребовалось (нет `strategy_metrics.csv` в `results_dir`).
- Пересобран rollup индекс:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/build_run_index.py --no-auto-sync-status --output-dir artifacts/wfa/aggregate/rollup`
  - `run_index` обновлён (entries=8341).

## S5: remote tailguard holdout (top-6, OOS 2025-01-01 → 2025-06-30, max_steps<=5) — план

**run_group:** `20260224_s5_tailguard_holdout_oos20250101_20250630_ms5_top6` (12 задач; 6 кандидатов × holdout+stress)

Что запускаем (кандидаты):
- Baseline: `tailguard_r04_v07_h2_quality_mild`.
- Top-3 из `20260223_tailguard_r07_fullspan_confirm_top3` (v01/v02/v03).
- Top-2 из `20260223_tailguard_r05b_ddfocus_fixend` (v14/v11).
- Во всех конфигах: `walk_forward.start_date: 2025-01-01`, `walk_forward.end_date: 2025-06-30`, `walk_forward.max_steps: 5` (queue-guardrail).

Артефакты/очередь:
- `coint4/artifacts/wfa/aggregate/20260224_s5_tailguard_holdout_oos20250101_20250630_ms5_top6/run_queue.csv`
- результаты: `coint4/artifacts/wfa/runs/20260224_s5_tailguard_holdout_oos20250101_20250630_ms5_top6/<run_id>/`

Критерии успеха (после выполнения):
- Holdout: `coverage_ratio>=0.95`, `min_pairs>=20`, `min_trades>=200`, `pnl>0`, и `worst_dd_pct<=0.20`.
- Stress: `pnl>0` и `worst_dd_pct<=0.25` (при прочих равных).
- Победитель: максимальный `score(worst_robust_sharpe)` среди вариантов, прошедших gates.

Команда исполнения (скелет, запускать с рабочей машины, не здесь):
- `STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'bash scripts/batch/run_heavy_queue.sh --queue artifacts/wfa/aggregate/20260224_s5_tailguard_holdout_oos20250101_20250630_ms5_top6/run_queue.csv'`

## Postprocess: sync queue + rollup (2026-02-24, ralph-tui-2133ac36)

- Best-effort синхронизация статусов очередей:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/sync_queue_status.py --queue-dir artifacts/wfa/aggregate`
  - Изменений не потребовалось.
- Пересобран rollup индекс:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
  - `run_index` обновлён (entries=8353).

## S6: remote tailguard holdout (top-10, OOS 2024-01-01 → 2024-06-30, max_steps<=5) — план

**run_group:** `20260224_s6_tailguard_holdout_oos20240101_20240630_ms5_top10` (20 задач; 10 кандидатов × holdout+stress)

Что запускаем (кандидаты):
- Baseline + добивка (из `20260223_tailguard_r04`): `r04_v07`, `r04_v08`, `r04_v11`.
- Top-3 (из `20260223_tailguard_r07_fullspan_confirm_top3`): `r07_v01..v03`.
- Top-4 (из `20260223_tailguard_r05b_ddfocus_fixend`): `r05b_v14`, `r05b_v11`, `r05b_v03`, `r05b_v05`.
- Во всех конфигах: `walk_forward.start_date: 2024-01-01`, `walk_forward.end_date: 2024-06-30`, `walk_forward.max_steps: 5` (queue-guardrail).

Артефакты/очередь:
- `coint4/artifacts/wfa/aggregate/20260224_s6_tailguard_holdout_oos20240101_20240630_ms5_top10/run_queue.csv`
- результаты: `coint4/artifacts/wfa/runs/20260224_s6_tailguard_holdout_oos20240101_20240630_ms5_top10/<run_id>/`

Критерии успеха (после выполнения):
- Holdout: `coverage_ratio>=0.95`, `min_pairs>=20`, `min_trades>=200`, `pnl>0`, и `worst_dd_pct<=0.20`.
- Stress: `pnl>0` и `worst_dd_pct<=0.25` (при прочих равных).
- Победитель: максимальный `score(worst_robust_sharpe)` среди вариантов, прошедших gates.

Команда исполнения (скелет, запускать с рабочей машины, не здесь):
- `STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'bash scripts/batch/run_heavy_queue.sh --queue artifacts/wfa/aggregate/20260224_s6_tailguard_holdout_oos20240101_20240630_ms5_top10/run_queue.csv'`

## Postprocess: sync queue + rollup (2026-02-24, ralph-tui-22ea02a4)

- Best-effort синхронизация статусов очередей (точечно по группам): `20260220_confirm_top10_bl11`, `20260222_tailguard_r01`, `20260222_tailguard_r02`, `20260222_tailguard_r03`, `20260223_s1_remote_ms5_queue`, `20260223_s2_tailguard_holdout_ms5_top3`, `20260223_s3_tailguard_holdout_ms5_top6`, `20260223_tailguard_r04`, `20260223_tailguard_r05_ddfocus`, `20260223_tailguard_r05b_ddfocus_fixend`, `20260223_tailguard_r06_ddfocus_wideuniverse`, `20260223_tailguard_r07_fullspan_confirm_top3`, `20260224_s3_tailguard_holdout_ms5_top6`, `20260224_s4_tailguard_holdout_ms5_top10`, `20260224_s5_tailguard_holdout_oos20250101_20250630_ms5_top6`, `20260224_s6_tailguard_holdout_oos20240101_20240630_ms5_top10`.
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv`
  - Изменений не потребовалось.
- Пересобран rollup индекс:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
  - `run_index` обновлён (entries=8373).

## S7: remote tailguard holdout (top-6, OOS 2023-07-01 → 2023-12-31, max_steps<=5) — план

**run_group:** `20260224_s7_tailguard_holdout_oos20230701_20231231_ms5_top6` (12 задач; 6 кандидатов × holdout+stress)

Что запускаем (кандидаты):
- Baseline: `tailguard_r04_v07_h2_quality_mild`.
- Top-3 из `20260223_tailguard_r07_fullspan_confirm_top3` (v01/v02/v03).
- Top-2 из `20260223_tailguard_r05b_ddfocus_fixend` (v14/v11).
- Во всех конфигах: `walk_forward.start_date: 2023-07-01`, `walk_forward.end_date: 2023-12-31`, `walk_forward.max_steps: 5` (queue-guardrail).

Артефакты/очередь:
- `coint4/artifacts/wfa/aggregate/20260224_s7_tailguard_holdout_oos20230701_20231231_ms5_top6/run_queue.csv`
- результаты: `coint4/artifacts/wfa/runs/20260224_s7_tailguard_holdout_oos20230701_20231231_ms5_top6/<run_id>/`

Критерии успеха (после выполнения):
- Holdout: `coverage_ratio>=0.95`, `min_pairs>=20`, `min_trades>=200`, `pnl>0`, и `worst_dd_pct<=0.20`.
- Stress: `pnl>0` и `worst_dd_pct<=0.25` (при прочих равных).
- Победитель: максимальный `score(worst_robust_sharpe)` среди вариантов, прошедших gates.

Команда исполнения (скелет, запускать с рабочей машины, не здесь):
- `STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'bash scripts/batch/run_heavy_queue.sh --queue artifacts/wfa/aggregate/20260224_s7_tailguard_holdout_oos20230701_20231231_ms5_top6/run_queue.csv'`

## Postprocess: sync queue + rollup (2026-02-24, ralph-tui-18f44b61)

- Best-effort синхронизация статусов очередей (точечно по группам): `20260223_s1_remote_ms5_queue`, `20260223_s2_tailguard_holdout_ms5_top3`, `20260223_s3_tailguard_holdout_ms5_top6`, `20260224_s3_tailguard_holdout_ms5_top6`, `20260224_s4_tailguard_holdout_ms5_top10`, `20260224_s5_tailguard_holdout_oos20250101_20250630_ms5_top6`, `20260224_s6_tailguard_holdout_oos20240101_20240630_ms5_top10`, `20260224_s7_tailguard_holdout_oos20230701_20231231_ms5_top6`.
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv`
  - Изменений не потребовалось.
- Пересобран rollup индекс:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
  - `run_index` обновлён (entries=8385).

## S8: remote tailguard holdout (top-10, OOS 2024-07-01 → 2024-12-31, max_steps<=5) — план

**run_group:** `20260224_s8_tailguard_holdout_oos20240701_20241231_ms5_top10` (20 задач; 10 кандидатов × holdout+stress)

Что запускаем (кандидаты):
- Baseline + добивка (из `20260223_tailguard_r04`): `r04_v07`, `r04_v08`, `r04_v11`.
- Top-3 (из `20260223_tailguard_r07_fullspan_confirm_top3`): `r07_v01..v03`.
- Top-4 (из `20260223_tailguard_r05b_ddfocus_fixend`): `r05b_v14`, `r05b_v11`, `r05b_v03`, `r05b_v05`.
- Во всех конфигах: `walk_forward.start_date: 2024-07-01`, `walk_forward.end_date: 2024-12-31`, `walk_forward.max_steps: 5` (queue-guardrail).

Артефакты/очередь:
- `coint4/artifacts/wfa/aggregate/20260224_s8_tailguard_holdout_oos20240701_20241231_ms5_top10/run_queue.csv`
- результаты: `coint4/artifacts/wfa/runs/20260224_s8_tailguard_holdout_oos20240701_20241231_ms5_top10/<run_id>/`

Критерии успеха (после выполнения):
- Holdout: `coverage_ratio>=0.95`, `min_pairs>=20`, `min_trades>=200`, `pnl>0`, и `worst_dd_pct<=0.20`.
- Stress: `pnl>0` и `worst_dd_pct<=0.25` (при прочих равных).
- Победитель: максимальный `score(worst_robust_sharpe)` среди вариантов, прошедших gates.

Команда исполнения (скелет, запускать с рабочей машины, не здесь):
- `STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'bash scripts/batch/run_heavy_queue.sh --queue artifacts/wfa/aggregate/20260224_s8_tailguard_holdout_oos20240701_20241231_ms5_top10/run_queue.csv'`

## Postprocess: sync queue + rollup (2026-02-24, ralph-tui-4247d117)

- Best-effort синхронизация статусов очередей (группы): `20260222_tailguard_r01`, `20260222_tailguard_r02`, `20260222_tailguard_r03`, `20260223_tailguard_r04`, `20260223_tailguard_r05_ddfocus`, `20260223_tailguard_r05b_ddfocus_fixend`, `20260223_tailguard_r06_ddfocus_wideuniverse`, `20260224_s5_tailguard_holdout_oos20250101_20250630_ms5_top6`, `20260224_s6_tailguard_holdout_oos20240101_20240630_ms5_top10`, `20260224_s7_tailguard_holdout_oos20230701_20231231_ms5_top6`.
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv`
  - Изменений не потребовалось.
- Пересобран rollup индекс:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
  - `run_index` обновлён (entries=8405).

## S9: remote tailguard holdout (top-6, OOS 2023-01-01 → 2023-06-30, max_steps<=5) — план

**run_group:** `20260224_s9_tailguard_holdout_oos20230101_20230630_ms5_top6` (12 задач; 6 кандидатов × holdout+stress)

Что запускаем (кандидаты):
- Baseline: `tailguard_r04_v07_h2_quality_mild`.
- Top-3 (из `20260223_tailguard_r07_fullspan_confirm_top3`): `r07_v01..v03`.
- Top-2 (из `20260223_tailguard_r05b_ddfocus_fixend`): `r05b_v14`, `r05b_v11`.
- Во всех конфигах: `walk_forward.start_date: 2023-01-01`, `walk_forward.end_date: 2023-06-30`, `walk_forward.max_steps: 5` (queue-guardrail).

Артефакты/очередь:
- `coint4/artifacts/wfa/aggregate/20260224_s9_tailguard_holdout_oos20230101_20230630_ms5_top6/run_queue.csv`
- результаты: `coint4/artifacts/wfa/runs/20260224_s9_tailguard_holdout_oos20230101_20230630_ms5_top6/<run_id>/`

Критерии успеха (после выполнения):
- Holdout: `coverage_ratio>=0.95`, `min_pairs>=20`, `min_trades>=200`, `pnl>0`, и `worst_dd_pct<=0.20`.
- Stress: `pnl>0` и `worst_dd_pct<=0.25` (при прочих равных).
- Победитель: максимальный `score(worst_robust_sharpe)` среди вариантов, прошедших gates.

Команда исполнения (скелет, запускать с рабочей машины, не здесь):
- `STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'bash scripts/batch/run_heavy_queue.sh --queue artifacts/wfa/aggregate/20260224_s9_tailguard_holdout_oos20230101_20230630_ms5_top6/run_queue.csv'`

## Postprocess: sync queue + rollup (2026-02-24, ralph-tui-dc43a702)

- Best-effort синхронизация статусов очередей (запуск по `artifacts/wfa/aggregate`).
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/sync_queue_status.py --queue-dir artifacts/wfa/aggregate`
  - Очереди для `20260223_tailguard_r07_fullspan_confirm_top3` и `20260224_s9_tailguard_holdout_oos20230101_20230630_ms5_top6`: статус `completed`.
- Пересобран rollup индекс:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
  - `run_index` обновлён (entries=8417).

## S10: выводы по `20260223_tailguard_r07_fullspan_confirm_top3` и следующий шаг к Sharpe>3

Источник метрик: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` (holdout+stress; coverage≈`0.988`).

### Top-3 (по robust-sharpe = `min(sh_holdout, sh_stress)`)

1) `v02_from_r06v03_trade_balanced_B` — **top-1**
   - holdout: `sh=0.709`, `pnl=325.560`, `dd=-215.031`, `tail_pair_total=181.373`
   - stress: `sh=0.643`, `pnl=287.493`, `dd=-210.620`, `tail_pair_total=183.813`
   - robust: `min_sh=0.643` (лучше по `min_sh`, DD и tail относительно v01)

2) `v03_from_r06v05_bidask_only`
   - holdout: `sh=0.709`, `pnl=325.560`, `dd=-215.031`, `tail_pair_total=181.373`
   - stress: `sh=0.643`, `pnl=287.493`, `dd=-210.620`, `tail_pair_total=183.813`
   - robust: `min_sh=0.643` (по rollup метрикам полностью совпал с v02 → считаем tie)

3) `v01_from_r06v02_trade_mild_A`
   - holdout: `sh=0.646`, `pnl=289.902`, `dd=-247.615`, `tail_pair_total=204.878`
   - stress: `sh=0.602`, `pnl=265.025`, `dd=-236.945`, `tail_pair_total=206.806`
   - robust: `min_sh=0.602`

### Почему top-1

- Все 3 варианта проходят базовые gates ширины/покрытия: `pairs_traded>=20`, `total_trades≈5.3k`, `coverage≈0.988`, `pnl>0` в holdout+stress.
- `v02` и `v03` дают лучший robust-sharpe (tie) и меньший DD/tail, чем `v01`.
- Tie-breaker `v02 > v03`: `v02` более “консервативный” tradeability-gate (liquidity/funding/tick/min_days_live), меньше риск live-drift при равных backtest-метриках.

### Следующий run_group (к Sharpe>3)

Выбор: **tradeability/quality sweep** (не fullspan confirm).

- Следующий run_group: `20260224_tailguard_r08_tradeability_quality_sweep`
- Цель: поднять `robust-sharpe` (и следом `score(worst_robust_sharpe)`) за счёт агрессивного tradeability/quality отбора вокруг `v02` (без изменения `risk/stop/z/dstop/maxpos`).
