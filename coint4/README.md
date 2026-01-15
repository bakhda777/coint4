# coint4 workspace

Короткая шпаргалка для работы в директории `coint4/`.

## Быстрый старт

```bash
# запуск CLI
./.venv/bin/coint2 --help
```

## Типовые сценарии

Политика фильтрации данных закреплена в `data_filters` (clean window + список исключенных символов).
По умолчанию список исключений пустой, чистое окно — `2022-03-01` → `2025-06-30`.

Сканирование пар:
```bash
./.venv/bin/coint2 scan \
  --config configs/criteria_relaxed.yaml \
  --base-config configs/main_2024.yaml \
  --output-dir bench
```

Быстрая итерация (smoke, ограниченные символы):
```bash
./.venv/bin/coint2 scan \
  --config configs/criteria_relaxed.yaml \
  --base-config configs/main_2024_smoke.yaml \
  --symbols-file configs/symbols_fast20.yaml \
  --train-days 30 \
  --valid-days 10 \
  --end-date 2025-06-30 \
  --top-n 50 \
  --output-dir bench/fast_iter
```

Расширенная быстрая итерация (50 символов, top-N=100):
```bash
./.venv/bin/coint2 scan \
  --config configs/criteria_relaxed.yaml \
  --base-config configs/main_2024_smoke.yaml \
  --symbols-file configs/symbols_fast50.yaml \
  --train-days 45 \
  --valid-days 15 \
  --end-date 2025-06-30 \
  --top-n 100 \
  --output-dir bench/fast_iter_top100
```

Сканирование пар на shortlist (top‑100 символов):
```bash
./.venv/bin/coint2 scan \
  --config configs/criteria_relaxed.yaml \
  --base-config configs/main_2024.yaml \
  --symbols-file bench/symbols_top_100.yaml \
  --end-date 2025-06-30 \
  --output-dir bench/top100 \
  --top-n 200
```

Фиксированный бэктест:
```bash
./.venv/bin/coint2 backtest \
  --config configs/main_2024.yaml \
  --pairs-file bench/pairs_universe.yaml \
  --period-start 2023-06-01 \
  --period-end 2023-08-31 \
  --out-dir outputs/fixed_run
```

Фиксированный бэктест (строгий режим качества данных):
```bash
./.venv/bin/coint2 backtest \
  --config configs/main_2024.yaml \
  --config-delta configs/data_quality_strict.yaml \
  --pairs-file bench/pairs_smoke.yaml \
  --period-start 2024-01-01 \
  --period-end 2024-03-31 \
  --out-dir outputs/fixed_run_strict
```

Walk-forward:
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024.yaml
```

`walk_forward.max_steps` ограничивает количество шагов WFA (по умолчанию 5).

Walk-forward с фиксированным universe (для сравнения с fixed backtest):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024.yaml \
  --pairs-file bench/clean_window_20260114_top200_step3/pairs_universe.yaml \
  --results-dir artifacts/wfa/runs/20260114_110612_main_2024_wfa_fixed_top200_step5
```

Walk-forward (сбалансированный, чистое окно):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_wfa_balanced.yaml
```

Полный пайплайн:
```bash
bash scripts/run_pipeline.sh
```

## UI (Streamlit)

```bash
PYTHONPATH=src ./.venv/bin/python scripts/ui_preflight.py \
  --config configs/main_2024.yaml \
  --search-space configs/search_spaces/web_ui.yaml

PYTHONPATH=src ./.venv/bin/streamlit run ui/app.py
```

## Проверка данных

```bash
PYTHONPATH=src ./.venv/bin/python scripts/validate_data_dump.py \
  --data-root data_downloaded \
  --mode monthly \
  --config configs/main_2024.yaml
```

## Тесты

```bash
./.venv/bin/pytest -q
```

## Оптимизация параметров

См. `docs/optimization_plan_20260114.md` для актуального плана, критериев и команд.
Selection grid по фильтрам (2026-01-15): `docs/optimization_runs_20260115.md`, конфиги в `configs/selection_grid_20260115/`, агрегатор в `artifacts/wfa/aggregate/20260115_selgrid/`.

## Чек-лист запуска

См. `docs/production_checklist.md`.

## Полезные директории

- `configs/` — основные конфиги и search spaces
- `data_downloaded/` — помесячные parquet-данные (канон); legacy `symbol=...` тоже поддерживается (игнорируется)
- `outputs/`, `results/`, `bench/` — артефакты прогонов
- `ui/` — Streamlit интерфейс
- `tests/` — тесты
