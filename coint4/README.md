# coint4 workspace

Короткая шпаргалка для работы в директории `coint4/`.

## Быстрый старт

```bash
# запуск CLI
./.venv/bin/coint2 --help
```

## Типовые сценарии

Сканирование пар:
```bash
./.venv/bin/coint2 scan \
  --config configs/criteria_relaxed.yaml \
  --base-config configs/main_2024.yaml \
  --output-dir bench
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

Walk-forward:
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024.yaml
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
  --mode raw \
  --config configs/main_2024.yaml
```

## Тесты

```bash
./.venv/bin/pytest -q
```

## Полезные директории

- `configs/` — основные конфиги и search spaces
- `data_downloaded/` — данные (игнорируются)
- `outputs/`, `results/`, `bench/` — артефакты прогонов
- `ui/` — Streamlit интерфейс
- `tests/` — тесты
