# Clean Cycle: Check-лист воспроизводимости (clean-цикл)

Цель: цикл должен повторяться одинаково на любом чистом клоне репозитория:
`SYNC_UP=1` (tracked-only) -> `VPS run` -> `sync_back`, и Git не должен содержать тяжёлых/генерённых артефактов.

## 0) Локально: чистота и гигиена репозитория

1. `git status --porcelain` пустой (перед VPS-run).
1. `make hygiene` проходит (нет трекаемых `outputs/`, `coint4/outputs/`, `coint4/data_downloaded/*` и больших blob'ов).
1. (опционально) `make ci` проходит локально.

## 1) Локально: в Git есть всё необходимое для SYNC_UP=1

`SYNC_UP=1` синкает на VPS только **tracked** файлы, поэтому:

1. Если добавлены новые конфиги/очереди/скрипты: они должны быть `git add <files>` и закоммичены до запуска (иначе не попадут на VPS).
1. Для clean-cycle TOP-10: проверить tracked inputs (если актуально):
   - `bash coint4/scripts/optimization/clean_cycle_top10/verify_tracked_inputs.sh coint4/configs/clean_cycle_top10/tracked_inputs_20260215_clean_top10.txt`

## 2) VPS run: baseline/sweeps через remote helper

Рекомендуемый запуск (из `coint4/`):

```bash
SYNC_UP=1 STOP_AFTER=1 bash scripts/remote/run_server_job.sh \
  bash scripts/optimization/watch_wfa_queue.sh --queue artifacts/wfa/aggregate/<group>/run_queue.csv
```

Инварианты:

1. `SYNC_UP=1` автоматически форсит `UPDATE_CODE=0` (избегаем конфликтов `git pull` на dirty worktree на VPS).
1. По умолчанию `STOP_AFTER=1` выключает VPS после job. Не оставлять включённым без явной причины.

## 3) После sync_back: локальный пост-процессинг

Минимум:

1. Пересобрать rollup индекс:
   - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
1. Если прогоны запускались вручную (не через watcher/queue-runner), синхронизировать статусы очереди:
   - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv`
1. Обновить дневник/состояние:
   - `docs/optimization_state.md`
   - `docs/optimization_runs_YYYYMMDD.md`

