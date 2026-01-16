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

См. `docs/optimization_plan_20260116.md` для актуального плана, критериев и команд (`docs/optimization_plan_20260114.md` помечен как legacy/archived).
Selection grid по фильтрам (2026-01-15): `docs/optimization_runs_20260115.md`, базовая сетка в `configs/selection_grid_20260115/`, строгая сетка p-value в `configs/selection_grid_20260115_strictpv/` (параллельный запуск описан в docs), агрегаторы в `artifacts/wfa/aggregate/20260115_selgrid/` и `artifacts/wfa/aggregate/20260115_selgrid_strictpv/`.
SSD top-N sweep (dynamic selection): `docs/optimization_runs_20260115.md`, конфиги в `configs/ssd_topn_sweep_20260115/`, агрегатор `artifacts/wfa/aggregate/20260115_ssd_topn_sweep/` (конфиги обновлены под `n_jobs: -1`, запуск по одному).
SSD top-N sweep (subset 4 values): `docs/optimization_runs_20260115.md`, конфиги в `configs/ssd_topn_sweep_20260115_4vals/`, агрегатор `artifacts/wfa/aggregate/20260115_ssd_topn_sweep_4vals/`.
SSD top-N sweep (subset 3 values, 30k/40k/50k): `docs/optimization_runs_20260115.md`, конфиги в `configs/ssd_topn_sweep_20260115_3vals/`, агрегатор `artifacts/wfa/aggregate/20260115_ssd_topn_sweep_3vals/`.
Sharpe target (строгие сигналы): `docs/optimization_runs_20260115.md`, конфиги в `configs/sharpe_target_20260115/`, агрегатор `artifacts/wfa/aggregate/20260115_sharpe_target/`.
Quality universe (исключение мусорных символов): `docs/optimization_runs_20260115.md`, конфиги `configs/quality_runs_20260115/` (включая corr0.45 alignment, signal_strict, tradeability/hl0p05-45, z0p9/z1p0 exit0p1 и denylist `configs/quality_runs_20260115/denylist_symbols_20260115.yaml`), артефакты `artifacts/universe/quality_universe_20260115/`, `artifacts/universe/quality_universe_20260115_250k/` и `artifacts/universe/quality_universe_20260115_200k/`, агрегаторы `artifacts/wfa/aggregate/20260115_quality_universe_500k/`, `artifacts/wfa/aggregate/20260115_quality_universe/` и `artifacts/wfa/aggregate/20260115_quality_universe_200k/`.
Журнал возобновления (после остановки): `docs/optimization_runs_20260116.md`.
Rollup индекс прогонов: `artifacts/wfa/aggregate/rollup/` (генерация `scripts/optimization/build_run_index.py`).
SSD refine/signal/risk sweeps (2026-01-16): конфиги `configs/ssd_topn_refine_20260116/`, `configs/signal_sweep_20260116/`, `configs/signal_grid_20260116/`, `configs/risk_sweep_20260116/`, агрегаторы `artifacts/wfa/aggregate/20260116_ssd_topn_refine/`, `artifacts/wfa/aggregate/20260116_signal_sweep/`, `artifacts/wfa/aggregate/20260116_signal_grid/`, `artifacts/wfa/aggregate/20260116_risk_sweep/`.
Piogoga grid (leader filters, zscore sweep): `docs/optimization_runs_20260116.md`, конфиги `configs/piogoga_grid_20260116/`, агрегатор `artifacts/wfa/aggregate/20260116_piogoga_grid/`.
Leader validation (post-analysis, SSD leader): `docs/optimization_runs_20260116.md`, конфиги `configs/leader_validation_20260116/`, агрегатор `artifacts/wfa/aggregate/20260116_leader_validation/`.
Очереди WFA с CPU‑heartbeat (без зависимости от логов): `scripts/optimization/watch_wfa_queue.sh`.
Состояние оптимизации: `docs/optimization_state.md` (обновлять после каждого блока прогонов).
Шаблон prompt для headless Codex: `scripts/optimization/on_done_codex_prompt.txt` (ключевая строка: "Прогон завершён, продолжай выполнение плана", + headless‑инструкция и запись причины в `docs/optimization_state.md` при ошибке).

Пример запуска watcher (с heartbeat и проверкой max_steps<=5):
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv \
  --parallel 1
```

Пример запуска watcher с on-done (headless Codex + лог):
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv \
  --parallel 1 \
  --on-done-prompt-file scripts/optimization/on_done_codex_prompt.txt \
  --on-done-log artifacts/wfa/aggregate/20260115_ssd_topn_sweep/codex_on_done.log
```

Сборка quality universe (пример):
```bash
./.venv/bin/python scripts/universe/build_quality_universe.py \
  --data-root data_downloaded \
  --period-start 2023-01-01 \
  --period-end 2023-09-30 \
  --bar-minutes 15 \
  --min-history-days 180 \
  --min-coverage-ratio 0.9 \
  --min-avg-daily-turnover-usd 250000 \
  --max-days-since-last 14 \
  --out-dir artifacts/universe/quality_universe_20260115_250k
```

## Чек-лист запуска

См. `docs/production_checklist.md`.

## Полезные директории

- `configs/` — основные конфиги и search spaces
- `data_downloaded/` — помесячные parquet-данные (канон); legacy `symbol=...` тоже поддерживается (игнорируется)
- `outputs/`, `results/`, `bench/` — артефакты прогонов
- `ui/` — Streamlit интерфейс
- `tests/` — тесты
