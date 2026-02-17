# Optimization runs — 2026-02-17 (BL9 executor pass)

Контекст: выполнен исполнительский цикл по bridge09 (`20260216_budget1000_bl9`) и возвращено управление аналитику.

## Команда запуска

Из `coint4/`:

`PYTHONPATH=src ./.venv/bin/python scripts/optimization/autopilot_budget1000.py --config configs/autopilot/budget1000_batch_loop_bridge09_20260216.yaml --reset`

## Исполнение (remote/VPS)

- Heavy execution шёл только через `coint4/scripts/remote/run_server_job.sh`.
- Целевой VPS: `85.198.90.128`.
- `STOP_AFTER=1` подтверждён: shutdown выполнялся после каждого раунда.

Исполненные run_group:
- `20260216_budget1000_bl9_r01_vm` (`72/72 completed`)
- `20260216_budget1000_bl9_r02_max_pairs` (`72/72 completed`)
- `20260216_budget1000_bl9_r03_corr` (`72/72 completed`)
- `20260216_budget1000_bl9_r04_pv` (`72/72 completed`)

## Пост-обработка (локально)

Команды (из `coint4/`):

`PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/20260216_budget1000_bl9_r01_vm/run_queue.csv --queue artifacts/wfa/aggregate/20260216_budget1000_bl9_r02_max_pairs/run_queue.csv --queue artifacts/wfa/aggregate/20260216_budget1000_bl9_r02_risk/run_queue.csv --queue artifacts/wfa/aggregate/20260216_budget1000_bl9_r02_slusd/run_queue.csv --queue artifacts/wfa/aggregate/20260216_budget1000_bl9_r03_corr/run_queue.csv --queue artifacts/wfa/aggregate/20260216_budget1000_bl9_r04_pv/run_queue.csv --queue artifacts/wfa/aggregate/20260216_budget1000_bl9_r05_max_hurst_exponent/run_queue.csv --queue artifacts/wfa/aggregate/20260216_budget1000_bl9_r05_min_beta/run_queue.csv --queue artifacts/wfa/aggregate/20260216_budget1000_bl9_r05_vm/run_queue.csv`

`PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`

Итог:
- Prefix totals (`20260216_budget1000_bl9_r*`): `completed=288`, `stalled=0`, `planned=228`, `total=516`.
- `run_index` после пересборки: `5415` записей.

## Результат цикла

- Stop reason: `search_space_exhausted: round=5, reason=Queue generation produced 0 entries for run_group=20260216_budget1000_bl9_r05_pv`.
- Best candidate (BL9): `run_group=20260216_budget1000_bl9_r01_vm`, `score=4.0621396020`, `worst_robust_sharpe=4.3656349732`, `worst_dd_pct=0.1118608540`.
- Controller state: `coint4/artifacts/wfa/aggregate/20260216_budget1000_bl9_autopilot/state.json`.
- Final report: `docs/budget1000_autopilot_final_20260217.md`.

## Примечание по безопасному default

`BL-ANL.completionNotes` в `.ralph-tui/prd.json` содержал placeholder `Completed by agent`, поэтому для исполнения использованы:
- `metadata.latestAnalystReview.recommendedNextConfig`
- команда из предыдущего progress log аналитика.
