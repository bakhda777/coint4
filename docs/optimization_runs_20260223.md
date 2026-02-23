# Optimization runs — 2026-02-23 (S1 hypotheses + next remote run_group)

Контекст:
- Канонический rollup: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` (generated at `2026-02-23 12:26:39Z`, см. `run_index.md`).
- Тяжёлое исполнение — только на VPS `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh` / `coint4/scripts/batch/run_heavy_queue.sh`.

## Baseline (что считаем “базой” прямо сейчас)

**Baseline для live/cutover (LOCK, fail-closed):**
- `run_group=20260213_budget1000_dd_sprint08_stoplossusd_micro`
- `run_id=holdout_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91`
- `score(worst_robust_sharpe)=3.530254`, `worst_dd_pct=0.132205`

**Baseline для следующей оптимизационной итерации FS-009 (после tailguard r04):**
- `run_group=20260223_tailguard_r04`
- `variant_id=tailguard_r04_v07_h2_quality_mild` (единственный strict-pass в r04 по `fullspan_v1` при `min_pairs>=20`, `min_trades>=200`, `min_pnl>=0`)
- Метрики: `worst_robust_sh=0.608`, `worst_dd_pct=0.190`, `worst_pnl=261.94`, `worst_step_pnl=-30.39`, `q20_step_pnl=-2.16`

## Дешёвые гипотезы (самые информативные за минимум времени/денег)

1) **“Декоративный Sharpe” может приходить от пропусков нулевых окон / разреженного покрытия тест-дней.**
   - Симптом: высокий Sharpe при узкой ширине (пары/сделки) или при неполном покрытии тест-периода.
   - Статус: окна без отобранных пар должны писаться как `0-PnL`; для ранжирования есть coverage-gate (`--min-coverage-ratio`, fail-closed на missing).
   - Дешёвая проверка: перед любым сравнением “до/после” убедиться, что rollup пересобран текущим `build_run_index.py` и содержит `coverage_ratio` (иначе ранкер корректно отбросит все варианты как missing).

2) **В `sharpe_sweep` есть “мертвый” параметр/дубликаты (ptn), и это ломает интуицию отбора по rollup.**
   - Факт: у `...ptn1000_exit0p12` и `...ptn1500_exit0p12` идентичны `equity_curve.csv` (одинаковый SHA256) и метрики в rollup.
   - Дешёвая проверка: дедупликация кандидатов по `equity_curve.csv` (hash) перед сравнением, и отдельная проверка в коде/генерации конфигов что `ptn` реально влияет на стратегию (а не override/неиспользуется).

3) **Основной рычаг ускорения после r04 — dd-focus вокруг worst-DD диапазона для `tailguard_r04_v07_h2_quality_mild`.**
   - Наблюдение: max-DD по robust daily PnL пришёлся на `2023-09-27 → 2024-05-28`; рекомендованный dd-focus диапазон WFA: `2023-06-29 → 2024-06-27`.
   - Гипотеза: локальный поиск (WFA только на dd-focus диапазоне) быстрее даст улучшение `worst_dd_pct`/tail без ухода в “узкий режим”, чем очередной fullspan sweep.

## Следующий remote run_group (план)

**run_group:** `20260223_tailguard_r05_ddfocus`

Почему:
- `r04` нашёл первый “честный” сдвиг в сторону положительного stress при сохранении ширины (v07), но DD ещё высокий.
- Самый дешёвый следующий шаг — ускоренный WFA на dd-focus диапазоне с вариациями вокруг v07 по осям tradeability+quality (без трогания `risk/stop/z/dstop/maxpos`), а уже победителей подтверждать отдельным fullspan-блоком.

Артефакты/очередь (канон):
- `coint4/artifacts/wfa/aggregate/20260223_tailguard_r05_ddfocus/run_queue.csv`
- результаты: `coint4/artifacts/wfa/runs/20260223_tailguard_r05_ddfocus/<run_id>/`

Команда исполнения (скелет, запускать с рабочей машины, не здесь):
- `STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'bash scripts/batch/run_heavy_queue.sh --queue artifacts/wfa/aggregate/20260223_tailguard_r05_ddfocus/run_queue.csv'`

## S1: remote run_queue.csv (max_steps<=5)

**run_group:** `20260223_s1_remote_ms5_queue` (8 задач; во всех конфигах `walk_forward.max_steps: 5`)

Артефакты/очередь:
- `coint4/artifacts/wfa/aggregate/20260223_s1_remote_ms5_queue/run_queue.csv`
- результаты: `coint4/artifacts/wfa/runs/20260223_s1_remote_ms5_queue/<run_id>/`

Команда исполнения (скелет, запускать с рабочей машины, не здесь):
- `STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'bash scripts/batch/run_heavy_queue.sh --queue artifacts/wfa/aggregate/20260223_s1_remote_ms5_queue/run_queue.csv'`

## Ops: sync queue status + rollup (best-effort)

- `sync_queue_status.py` прогнан по всем `coint4/artifacts/wfa/aggregate/**/run_queue.csv` (best-effort) — изменений не потребовалось.
- Rollup пересобран: `PYTHONPATH=src ./.venv/bin/python3 scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup` (entries: 8297).

## S2: audit “слишком высокий Sharpe” (top-3 из run_index)

Запущен `tools/audit_sharpe.py` (пересчёт Sharpe из `equity_curve.csv` по rollup-формуле) для 3 топовых прогонов:

- `artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p6_cpv0p08_ptn1000_exit0p12`:
  - `stored=9.38478`, `computed_full=9.38478`, `abs_diff=1.1e-14`, `period_sec=900`, `n=8640`, `computed_daily_only=0.95783`
- `artifacts/wfa/runs/20260214_sharpe_sweep/sharpe_sweep_20260214_corr0p55_cpv0p08_ptn1000_exit0p12`:
  - `stored=9.27090`, `computed_full=9.27090`, `abs_diff=1.6e-14`, `period_sec=900`, `n=8640`, `computed_daily_only=0.94621`
- `artifacts/wfa/runs/20260116_risk_sweep/risk_sweep_20260116_cooldown4`:
  - `stored=9.03471`, `computed_full=9.03471`, `abs_diff=4.6e-14`, `period_sec=900`, `n=8640`, `computed_daily_only=0.92210`

Вывод: метрика Sharpe в этих топ-3 консистентна (stored ≈ recompute из `equity_curve.csv`, частота корректно инферится как 15m: `period_sec=900`). Признаков артефактов annualization/таймстемпов для “слишком высокого Sharpe” не найдено.

## S2: remote tailguard holdout (top-3, max_steps<=5) — план

**run_group:** `20260223_s2_tailguard_holdout_ms5_top3` (6 задач; 3 кандидата × holdout+stress)

Что запускаем:
- Top-3 из `20260223_tailguard_r07_fullspan_confirm_top3` (v01/v02/v03), но на holdout-окне `2024-05-01 → 2025-06-30`.
- Во всех конфигах: `walk_forward.max_steps: 5` (queue-guardrail).

Артефакты/очередь:
- `coint4/artifacts/wfa/aggregate/20260223_s2_tailguard_holdout_ms5_top3/run_queue.csv`
- результаты: `coint4/artifacts/wfa/runs/20260223_s2_tailguard_holdout_ms5_top3/<run_id>/`

Критерии успеха (после выполнения):
- Holdout: `coverage_ratio>=0.95`, `min_pairs>=20`, `min_trades>=200`, `pnl>0`, и `worst_dd_pct<=0.20`.
- Stress: `pnl>0` и `worst_dd_pct<=0.25` (при прочих равных).
- Победитель: максимальный `score(worst_robust_sharpe)` среди вариантов, прошедших gates.

Команда исполнения (скелет, запускать с рабочей машины, не здесь):
- `STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'bash scripts/batch/run_heavy_queue.sh --queue artifacts/wfa/aggregate/20260223_s2_tailguard_holdout_ms5_top3/run_queue.csv'`

## S3: remote tailguard holdout (top-6, max_steps<=5) — план

**run_group:** `20260223_s3_tailguard_holdout_ms5_top6` (12 задач; 6 кандидатов × holdout+stress)

Что запускаем:
- Baseline: `tailguard_r04_v07_h2_quality_mild` на holdout-окне `2024-05-01 → 2025-06-30`.
- Top-3 из `20260223_tailguard_r07_fullspan_confirm_top3` (v01/v02/v03) на том же окне.
- Top-2 из `20260223_tailguard_r05b_ddfocus_fixend` (v14/v11) на том же окне.
- Во всех конфигах: `walk_forward.max_steps: 5` (queue-guardrail).

Артефакты/очередь:
- `coint4/artifacts/wfa/aggregate/20260223_s3_tailguard_holdout_ms5_top6/run_queue.csv`
- результаты: `coint4/artifacts/wfa/runs/20260223_s3_tailguard_holdout_ms5_top6/<run_id>/`

Критерии успеха (после выполнения):
- Holdout: `coverage_ratio>=0.95`, `min_pairs>=20`, `min_trades>=200`, `pnl>0`, и `worst_dd_pct<=0.20`.
- Stress: `pnl>0` и `worst_dd_pct<=0.25` (при прочих равных).
- Победитель: максимальный `score(worst_robust_sharpe)` среди вариантов, прошедших gates.

Команда исполнения (скелет, запускать с рабочей машины, не здесь):
- `STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'bash scripts/batch/run_heavy_queue.sh --queue artifacts/wfa/aggregate/20260223_s3_tailguard_holdout_ms5_top6/run_queue.csv'`
