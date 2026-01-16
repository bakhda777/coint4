# Журнал прогонов оптимизации (2026-01-16)

Назначение: продолжение после ночного отключения сервера и возобновление очередей.

## Статусы
- `active` — идет выполнение.
- `candidate` — выбран для валидации.
- `rejected` — отклонен по результатам валидации.
- `aborted` — прерван вручную/по ошибке.
- `legacy/archived` — устаревший или остановленный прогон.

## Состояние после отключения
- Все запуски остановлены в середине WFA; очередь возобновления оформлена через `run_queue.csv` и списки конфигов.
- Возобновление фиксируем в этом журнале, с указанием фильтрации пар по шагам (stdout/run.log).
- Лимит WFA: максимум 5 шагов без отдельного согласования.

## Очереди на возобновление
- SSD top-N sweep (6 значений): `coint4/artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv`.
- SSD top-N sweep (3 значения): `coint4/artifacts/wfa/aggregate/20260115_ssd_topn_sweep_3vals/run_queue.csv`.
- Strict PV grid: `coint4/artifacts/wfa/aggregate/20260115_selgrid_strictpv/strictpv_configs.txt` (часть в очереди).
- Selection grid: `coint4/artifacts/wfa/aggregate/20260115_selgrid/selected_runs.csv` (оставшиеся конфиги).

## Команды возобновления (из `coint4/`)

WFA очереди (stalled/planned):
```bash
PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py \
  --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv

PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py \
  --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep_3vals/run_queue.csv
```

Strict PV grid (по одному прогону, `n_jobs: -1` внутри конфигов):
```bash
cat artifacts/wfa/aggregate/20260115_selgrid_strictpv/strictpv_configs.txt | \
  xargs -P 1 -I {} bash -lc 'cfg="$1"; run_id=$(basename "$cfg" .yaml); ./run_wfa_fullcpu.sh "$cfg" "artifacts/wfa/runs/20260115_selgrid_strictpv/$run_id"' _ {}
```

Selection grid (очередь выбранных конфигов):
```bash
cat artifacts/wfa/aggregate/20260115_selgrid/selected_runs.csv | \
  tail -n +2 | cut -d, -f1 | \
  xargs -P 1 -I {} bash -lc 'cfg="$1"; run_id=$(basename "$cfg" .yaml); ./run_wfa_fullcpu.sh "$cfg" "artifacts/wfa/runs/20260115_selgrid/$run_id"' _ {}
```

## Rollup индекс прогонов
```bash
PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py \
  --output-dir artifacts/wfa/aggregate/rollup
```

## Ссылки
- План дальнейшей оптимизации: `docs/optimization_plan_20260116.md`.
