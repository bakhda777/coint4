# Clean Cycle TOP-10: схема оптимизации (score и фильтры)

Этот документ фиксирует **каноническую** схему ранжирования и фильтрации для clean-цикла (baseline + sweeps).

## Принципы

- Старую партию (`coint4/artifacts/wfa/runs/**`, `coint4/artifacts/wfa/aggregate/rollup/run_index.*`) используем **только** чтобы выбрать initial TOP-10 и сформировать `baseline_manifest`.
- Дальше оптимизация смотрит **только** на clean-контур: baseline + sweeps, агрегированные через clean rollup.
- Метрики для score берём **только** из `canonical_metrics.json` (поля `canonical_*`), пересчитанных из `equity_curve.csv`.
- Все прогоны внутри цикла должны совпадать по `FIXED_WINDOWS.walk_forward` (см. `coint4/scripts/optimization/clean_cycle_top10/definitions.py`).

## Score

Скалярная метрика (default):

`score = canonical_sharpe - lambda_dd * abs(canonical_max_drawdown_abs)`

- `canonical_max_drawdown_abs` измеряется в единицах equity (например, USD для budget1000) и обычно `<= 0`, поэтому берём `abs(...)`.
- `lambda_dd` параметризуется через CLI (default: `0.02`).

Источник истины по формуле: `coint4/scripts/optimization/clean_cycle_top10/scoring.py`.

## Сортировка

`coint4/scripts/optimization/clean_cycle_top10/build_clean_rollup.py` поддерживает 2 режима:

- `--sort-mode score` (default):
  - `score` desc
  - `canonical_sharpe` desc
  - `abs(canonical_max_drawdown_abs)` asc
  - `canonical_pnl_abs` desc
  - далее детерминированные tie-breakers (phase/run_name/ids/paths)
- `--sort-mode multi`:
  - `canonical_sharpe` desc
  - `abs(canonical_max_drawdown_abs)` asc
  - `canonical_pnl_abs` desc
  - далее детерминированные tie-breakers (phase/run_name/ids/paths)

## Фильтры (default)

Rollup строится только по:

- `status == completed`
- `canonical_metrics.json` присутствует (`canonical_metrics_present == true`)

Флаги для ослабления фильтров (только для дебага):

- `--include-noncompleted`
- `--include-missing-canonical`

## Команда (пример)

Запускать из `coint4/`:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/build_clean_rollup.py \
  --baseline-manifest artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_manifest.json \
  --sweeps-manifest artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/sweeps_manifest.json \
  --output-csv artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/rollup_clean_cycle_top10.csv \
  --output-md artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/rollup_clean_cycle_top10.md \
  --lambda-dd 0.02 \
  --sort-mode score \
  --overwrite
```

Примечание: `lambda_dd` зависит от масштаба equity. Если меняется стартовый капитал (например, не $1000), шкала `canonical_max_drawdown_abs` меняется, поэтому `lambda_dd` нужно пересмотреть.

