# План дальнейших действий (FS-009: r04 итоги -> fast-loop по просадке)

Дата среза: 2026-02-23

## Текущее состояние

- Канонический rollup: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` (`8167` entries; `8168` строк с header).
- Очередь `r01`: `coint4/artifacts/wfa/aggregate/20260222_tailguard_r01/run_queue.csv` -> `completed=10/10`.
- Очередь `r02`: `coint4/artifacts/wfa/aggregate/20260222_tailguard_r02/run_queue.csv` -> `completed=32/32`.
- Очередь `r03`: `coint4/artifacts/wfa/aggregate/20260222_tailguard_r03/run_queue.csv` -> `completed=48/48`.
- Очередь `r04`: `coint4/artifacts/wfa/aggregate/20260223_tailguard_r04/run_queue.csv` -> `completed=26/26` (heavy run на VPS, результаты синкнуты обратно; VPS выключен).
- `r03` strict/diagnostic (`tail_worst_gate_pct=0.20/0.21`) дали одинаковый top-1:
  - `score=1.043`, `worst_robust_sh=1.142`, `worst_dd_pct=0.291`, `worst_pnl=1147.93`, `worst_step_pnl=-198.90`.
- Глобальная цель не достигнута: `gap=+1.858` до `worst_robust_sh=3.0`.
- Вердикт `r03`: completed-without-improvement (плато относительно `r01/r02`).

## Статус выполнения плана r03

| # | Задача | Статус | Параметр завершения (факт) |
|---:|---|---|---|
| 1 | Зафиксировать контракт `r03` и baseline. | done | Блок `FS-009 / r03 contract (start)` добавлен в `docs/optimization_state.md`. |
| 2 | Сформировать гипотезы выхода из плато. | done | `search_space.md`: `H1/H2/H3` + baseline control. |
| 3 | Собрать search-space `r03` и убрать дубли. | done | `search_space.csv`: `24` вариантов, дубликатов по `variant_id` нет. |
| 4 | Сгенерировать конфиги/очередь `r03`. | done | `48` YAML (holdout+stress), queue создана. |
| 5 | Выполнить preflight/sanity перед heavy-run. | done | Dry-run queue прошёл без ошибок валидации. |
| 6 | Запустить heavy `r03` на VPS через API helper. | done | `completed=48/48`, `SYNC_BACK=1`, VPS выключен (SSH timeout). |
| 7 | Канонический postprocess `r03`. | done | `sync_queue_status.py` + `build_run_index.py --no-auto-sync-status`; `run_index` содержит `48` строк `r03`. |
| 8 | Снять strict/diagnostic ranking `r03`. | done | Выполнены оба ranking; `strict_pass=2`, `diagnostic_pass=2`. |
| 9 | Сравнить `r03` vs best-known (`r01/r02`) и пересчитать gap. | done | Top-метрики совпали; gap к Sharpe-цели остался `+1.858`. |
| 10 | Обновить документацию и закрыть цикл handoff. | done | Обновлены `plan.md`, `docs/optimization_state.md`, `docs/optimization_runs_20260222.md`. |

## Итоги r04 (tradeability/quality/stability; fullspan single-window)

Ключевые наблюдения (на факте `20260223_tailguard_r04`):
- H1 tradeability (v01–v06) почти не сдвинул stress: guardrail прижимает мягкие пороги, а выбитые пары часто не торговали в stress.
- H2 quality mild (v07) дал первый “честный” рычаг: `holdout_sh≈0.654`, `stress_sh≈0.608`, `min_pairs=21`, `min_trades≈5305`, `worst_step_pnl≈-30.39` (робастно).
- v08/v10 (med/kpss-only) выглядят “красиво” по Sharpe, но проваливают гейты ширины (`pairs=2`, `trades=64`) и не пригодны как winner.
- H3 stability (v11–v13) ухудшает: режет торговлю и даёт отрицательный robust.

Технический долг, выявленный r04:
- В WFA метрики были смещены вверх для “разреженных” стратегий (окна без отобранных пар не попадали в `daily_pnl.csv`/`equity_curve.csv`). Исправление сделано в коде (включаем нулевой PnL на тест-окнах без пар).

## Критерий успеха r04

- Минимальный: `worst_robust_sh > 1.142` при соблюдении hard-gates.
- Целевой: `worst_robust_sh >= 3.0` при `worst_pnl > 0`, `worst_dd_pct <= 0.35`, `worst_step_pnl >= -200`.

## План r05 (fast-loop: фокус на периоде максимальной просадки)

Цель: ускорить итерации, “бить” по хвостам/просадке, но принимать решение только после fullspan подтверждения.

Выбранный dd-focus (baseline: `20260223_tailguard_r04` / `v07`):
- Max-DD на robust daily pnl (min(holdout, stress)): peak `2023-09-27` → trough `2024-05-28`.
- WFA dd-focus диапазон (peak-90d, trough+30d): `2023-06-29` → `2024-06-27`.
- Важно: `walk_forward.start_date` в коде = начало *тестового* периода; для первого шага данные будут загружены с `start_date - training_period_days` (сейчас `training_period_days=90`).

| # | Задача | Параметр завершения (DoD) |
|---:|---|---|
| 1 | Выбрать baseline для fast-loop (текущий честный кандидат: `r04 v07`). | Зафиксирован run_dir (holdout+stress) и вычислен период max-DD + worst-rolling-loss (даты). |
| 2 | Определить dd-focus диапазон дат для WFA. | Диапазон задан явно: `2023-06-29` → `2024-06-27` (учтён `training_period_days=90`). |
| 3 | Сгенерировать `r05_ddfocus` queue/config (комбинации вокруг v07). | Есть `search_space.csv/md`, `run_queue.csv`, dry-run проходит. |
| 4 | Прогнать heavy `r05_ddfocus` на VPS. | `completed=100%`, `SYNC_BACK=1`, VPS выключен. |
| 5 | Отранжировать `r05_ddfocus` по robust (гейты ширины + tail). | Ранжирование выполнено (`--min-windows 1 --min-pairs 20 --min-trades 200` + fullspan_v1 tail), есть top-3. |
| 6 | Promoting: top-1/top-3 прогнать fullspan (весь период) для проверки. | Fullspan прогоны завершены, сравнение vs baseline v07 оформлено. |
| 7 | Решение: двигаться дальше (новые quality метрики: ECM/beta stability/break) или закреплять пороги. | Решение зафиксировано (и почему), обновлён gap к цели `Sharpe>3`. |
