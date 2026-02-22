# Optimization runs — 2026-02-20 (baseline refresh before next loop)

Контекст: перед новым loop нужен свежий baseline на актуальном rollup и отдельный снимок clean-кандидатов.

## Rebuild rollup (including `runs_clean`)

Команда (из `coint4/`):

`PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`

Результат:
- `Run index entries: 1800`.
- Покрытие `runs_clean` в `run_index.csv`: `20` строк, `20/20 status=completed`, `20/20 metrics_present=true`.

## Baseline snapshot: top robust candidates (global multi-window)

Команда (из `coint4/`):

`PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_multiwindow_robust_runs.py --top 10 --max-dd-pct 0.14 --min-windows 3 --min-trades 200 --min-pairs 20 --contains budget1000 --include-noncompleted`

Результат (top-10):

| rank | worst_robust_sh | avg_robust_sh | worst_dd_pct | avg_dd_pct | windows | run_group | variant_id |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 3.530 | 4.388 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p91 |
| 2 | 3.530 | 4.388 | 0.132 | 0.089 | 3 | 20260214_budget1000_dd_sprint09_hurst_slusd1p91 | prod_final_budget1000_risk0p006_slusd1p91_max_hurst_exponent0p8 |
| 3 | 3.470 | 4.357 | 0.128 | 0.088 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p88 |
| 4 | 3.448 | 4.306 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p85_risk0p006 |
| 5 | 3.448 | 4.306 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p85 |
| 6 | 3.323 | 4.254 | 0.137 | 0.087 | 3 | 20260213_budget1000_dd_sprint05_stoplossusd_refine | pruned168_slusd1p75_risk0p006 |
| 7 | 3.323 | 4.254 | 0.137 | 0.087 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p75_risk0p006 |
| 8 | 3.299 | 4.165 | 0.138 | 0.091 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p8_risk0p006 |
| 9 | 3.179 | 4.200 | 0.137 | 0.086 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p65_risk0p006 |
| 10 | 2.864 | 4.115 | 0.108 | 0.073 | 3 | 20260213_budget1000_dd_sprint05_stoplossusd_refine | pruned168_slusd1p5_risk0p006 |

Baseline freeze для следующего loop:
- Baseline score (worst robust Sharpe): `3.530254`.
- Baseline worst_dd_pct: `0.132205`.
- Baseline run_group: `20260213_budget1000_dd_sprint08_stoplossusd_micro`.
- Baseline run_id (worst-window holdout): `holdout_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91`.
- Baseline paired stress run_id (worst-window): `stress_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91`.
- Baseline variant_id: `prod_final_budget1000_risk0p006_slusd1p91`.
- Gate check: `worst_dd_pct=0.132205 <= 0.14`, `windows=3`, `min_trades>=200`, `min_pairs>=20`.
- Tie-break decision: у `20260214_budget1000_dd_sprint09_hurst_slusd1p91` тот же `worst_robust_sharpe`; для continuity baseline оставлен на `20260213...sprint08`.

## Snapshot: top clean candidates (`runs_clean`)

Команда (из `coint4/`):

`PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_robust_runs.py --top 10 --contains runs_clean --contains confirm_shortlist`

Результат (top-5):

| rank | robust_sharpe | run_group | run_id | holdout_pnl | stress_pnl | note |
|---:|---:|---|---|---:|---:|---|
| 1 | 0.000 | 20260215_confirm_shortlist | b01_20260120_realcost_shortlist_..._ms0p2 | 0.00 | 0.00 | no-op |
| 2 | 0.000 | 20260215_confirm_shortlist | b02_20260123_relaxed8_nokpss_u250_churnfix_..._ms0p2 | 0.00 | 0.00 | no-op |
| 3 | 0.000 | 20260215_confirm_shortlist | b03_20260120_realcost_shortlist_..._ms0p1 | 0.00 | 0.00 | no-op |
| 4 | 0.000 | 20260215_confirm_shortlist | b04_20260122_relaxed8_nokpss_u250_churnfix_..._ms0p1 | 0.00 | 0.00 | no-op |
| 5 | 0.000 | 20260215_confirm_shortlist | b05_20260120_realcost_shortlist_..._ms0p15 | 0.00 | 0.00 | no-op |

Вывод по clean snapshot:
- `10/10` paired holdout/stress в `runs_clean` имеют `robust_sharpe=0`, `PnL=0`, `DD=0`.
- До выяснения причины no-op clean-shortlist не используется как baseline для запуска нового loop.

## VPS confirm replay: top-10 BL11 (20260220_confirm_top10_bl11)

Цель: сделать confirm-replay для актуального top-10 параметров (multi-window robust ranking) на VPS `85.198.90.128`.

Подготовка:
- Локально собрана очередь: `coint4/artifacts/wfa/aggregate/20260220_confirm_top10_bl11/run_queue.csv` (`60` записей = `10` вариантов × `3` окна × `holdout+stress`).
- Зафиксирован shortlist с метриками отбора: `coint4/artifacts/wfa/aggregate/20260220_confirm_top10_bl11/top10_variants.csv`.
- Перед запуском подтянуты отсутствующие локально YAML-конфиги `20260217_budget1000_bl11_r{02,03,05}_*` с VPS через `coint4/scripts/remote/run_server_job.sh` + `SYNC_BACK`.

Запуск на VPS (из root repo):

`SYNC_UP=1 UPDATE_CODE=0 STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'echo RUN_HOST=$(hostname); bash scripts/optimization/watch_wfa_queue.sh --queue artifacts/wfa/aggregate/20260220_confirm_top10_bl11/run_queue.csv --parallel 12'`

Результат:
- `RUN_HOST=coint` (подтверждение remote execution).
- `run_queue.csv`: `60/60 completed`.
- `sync_queue_status.py`: `no changes (metrics_present=60, missing=0, skipped=0)`.
- VPS после sync_back выключен (`STOP_AFTER=1`, `ssh ... "echo"` -> no response).

Постпроцесс:
- Пересобран rollup: `PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`.
- Новый размер индекса: `Run index entries: 8071`.
- Проверка подтверждённой группы:
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_multiwindow_robust_runs.py --contains 20260220_confirm_top10_bl11 --top 15 --min-windows 3 --min-trades 200 --min-pairs 20 --max-dd-pct 0.14`
  - Top-10 и метрики совпали с исходным shortlist `top10_variants.csv` (воспроизведение 1:1).

Top-3 confirm (worst-window robust):

| rank | worst_robust_sh | worst_dd_pct | worst_pnl | variant |
|---:|---:|---:|---:|---|
| 1 | 4.505 | 0.085 | 252.89 | `..._slusd1p81` |
| 2 | 4.366 | 0.112 | 203.77 | `..._max_pairs24p0` |
| 3 | 4.311 | 0.090 | 200.46 | `..._pv0p365` |

## US-LOOP-003: запуск closed-loop (attempt в sandbox, 2026-02-20)

Команда запуска (из `coint4/`):

`PYTHONPATH=src ./.venv/bin/python scripts/optimization/autonomous_optimize.py --until-done --use-codex-exec`

Фактический результат:
- Запуск выполнен повторно в `2026-02-20T01:27:38Z` (retry после предыдущего wait-цикла).
- `codex exec` вернул `RC=1` с reconnect/stream disconnect к `https://chatgpt.com/backend-api/codex/responses` (network sandbox).
- Оркестратор перешёл в data-collection fallback и попытался выполнить powered queue.

Подтверждение remote markers (из iteration log):
- Вызов `scripts/optimization/run_wfa_queue_powered.py`.
- Параметры: `--compute-host 85.198.90.128`, `--poweroff true`, `--wait-completion true`.
- Для текущего run_group `20260219_budget1000_bl11_r09_pairgate02_micro24` powered runner завершился `RC=4`, `FAIL reason=ServerspaceError`.
- Логи retry: `coint4/artifacts/optimization_state/decisions/codex_exec_20260220_012738.jsonl`, `coint4/artifacts/wfa/aggregate/20260219_budget1000_bl11_r09_pairgate02_micro24/logs/powered_20260220_012804.log`.

Первопричина блокировки:
- DNS/сеть в sandbox: ошибка резолва `https://api.serverspace.ru` (`Temporary failure in name resolution`).
- Проверка обхода через `--skip-power true` тоже не проходит: SSH к `85.198.90.128:22` завершается `socket: Operation not permitted`.

Fail-closed решение для этого окружения:
- Не подменять LLM-decision вручную и не форсировать локальные heavy-прогоны.
- Состояние оставлено в ожидании удалённого шага: `last_error=POWERED_WAIT:SERVERSPACEERROR`, `last_iteration_phase=waiting_powered`.
- Для продолжения цикла до `next_action=stop` нужен запуск в среде с разрешённым outbound доступом к `chatgpt.com`, `api.serverspace.ru` и SSH к `85.198.90.128`.

## VPS fullspan replay: top-3 BL11 (20260220_top3_fullspan_wfa)

Цель: прогнать top-3 BL11 на максимально доступном fullspan-периоде WFA (без ограничения `max_steps`), чтобы проверить устойчивость лидеров confirm-top10.

Данные и период:
- Локальный диапазон parquet (`coint4/data_downloaded/**/*.parquet`): `2021-12-31T21:15:00Z` -> `2025-06-30T21:00:00Z`.
- Выбран fullspan OOS-период для запуска: `2022-03-01` -> `2025-06-30`.
- В fullspan-конфигах выставлено: `walk_forward.start_date/end_date`, `data_filters.clean_window.start_date/end_date`, `walk_forward.max_steps: null`.

Подготовка:
- Конфиги: `coint4/configs/budget1000_autopilot/20260220_top3_fullspan_wfa/` (`6` yaml = top-3 × `holdout+stress`).
- Очередь: `coint4/artifacts/wfa/aggregate/20260220_top3_fullspan_wfa/run_queue.csv`.
- Manifest: `coint4/artifacts/wfa/aggregate/20260220_top3_fullspan_wfa/manifest.csv`.

Запуск на VPS (из root repo):

`SYNC_UP=1 UPDATE_CODE=0 STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'echo RUN_HOST=$(hostname); PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py --queue artifacts/wfa/aggregate/20260220_top3_fullspan_wfa/run_queue.csv --statuses planned,stalled --parallel 6'`

Результат:
- `RUN_HOST=coint` (подтверждён remote execution на `85.198.90.128`).
- Очередь: `6/6 completed`.
- Проверка целостности: `Sharpe consistency OK (6 run(s))`.
- Типичная длительность одного fullspan run по `run.log`: ~`19` минут.
- VPS после sync_back выключен (`STOP_AFTER=1`, SSH недоступен).

Постпроцесс:
- Пересобран rollup:
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
  - `Run index entries: 8077`.
- Сводка fullspan результатов (top-3 variants): `coint4/artifacts/wfa/aggregate/20260220_top3_fullspan_wfa/fullspan_summary.csv`.

Итоги top-3 (fullspan robust = `min(Sharpe_holdout, Sharpe_stress)`):

| rank | variant (short) | holdout_sh | stress_sh | robust_sh | holdout_pnl | stress_pnl | robust_pnl | worst_dd_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `..._slusd1p81` | 1.463 | 1.088 | 1.088 | 2541.80 | 1500.25 | 1500.25 | 0.393 |
| 2 | `..._pv0p365` | 1.367 | -0.342 | -0.342 | 2330.14 | -210.92 | -210.92 | 0.456 |
| 3 | `..._max_pairs24p0` | 1.478 | -0.346 | -0.346 | 3698.41 | -215.28 | -215.28 | 0.423 |

Вывод:
- На fullspan only `..._slusd1p81` остаётся устойчивым в stress (положительные Sharpe/PnL и в holdout, и в stress).
- `..._pv0p365` и `..._max_pairs24p0` показывают хороший holdout, но ломаются в stress (отрицательные stress Sharpe/PnL), поэтому не проходят базовый robust gate по `worst_pnl >= 0`.
