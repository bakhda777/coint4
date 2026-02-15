# Sharpe Audit (Phase A)

Цель: доказать, что в текущем пайплайне WFA Sharpe (`sharpe_ratio_abs`) считается **годовым (annualized)** и **сопоставимо** между старыми и новыми прогонами (даже если в старых `strategy_metrics.csv` Sharpe был посчитан иначе).

## 1) Где в коде считается `sharpe_ratio_abs`

### 1.1 Walk-forward orchestrator (генерация `strategy_metrics.csv`)

Файл: `coint4/src/coint2/pipeline/walk_forward_orchestrator.py`

- Эквити-доходности берутся как `pct_change` от equity curve (проценты, не $PnL).
- Annualization учитывает частоту баров: `annualizing_factor * (24*60/bar_minutes)`.

Ключевой фрагмент (строки): `coint4/src/coint2/pipeline/walk_forward_orchestrator.py:2002`:

- `equity_returns = equity_series.ffill().pct_change(fill_method=None).dropna()`
- `periods_per_year = float(cfg.backtest.annualizing_factor)`
- `periods_per_year *= 24 * 60 / bar_minutes` (если есть `bar_minutes`)
- `sharpe_abs = performance.sharpe_ratio(equity_returns, periods_per_year)`

### 1.2 Формула Sharpe

Файл: `coint4/src/coint2/core/performance.py`

`coint4/src/coint2/core/performance.py:37`:

```
Sharpe = sqrt(annualizing_factor) * mean(returns) / std(returns)
```

Где `std(returns)` это стандартное отклонение pandas по умолчанию (sample std, `ddof=1`).

### 1.3 Rollup/индексация прогонов (канонизация Sharpe)

Файл: `coint4/src/coint2/ops/run_index.py`

Rollup **пере-считает** Sharpe из `equity_curve.csv` и кладёт:

- `sharpe_ratio_abs_raw` = как в `strategy_metrics.csv`
- `sharpe_ratio_abs` = computed (если удалось), иначе raw

См. `coint4/src/coint2/ops/run_index.py:265` и `coint4/src/coint2/ops/run_index.py:268`.

Пере-счёт Sharpe из `equity_curve.csv` реализован в `coint4/src/coint2/ops/run_index.py:78`:

- returns: `(equity_t - equity_{t-1}) / equity_{t-1}`
- `period_seconds = median(delta_ts_seconds)`
- `periods_per_year = 365 * (86400 / period_seconds)`
- `Sharpe = sqrt(periods_per_year) * mean(returns) / std(returns)` (sample std, `ddof=1`)

## 2) Annualization factor (для 15m)

Для 15-минутных баров:

- `bar_minutes = 15`
- баров в сутки = `24*60/15 = 96`
- периодов в год = `365 * 96 = 35040`

Т.е. годовой Sharpe должен масштабироваться на `sqrt(35040)`.

## 3) Аудит артефактов: пересчёт Sharpe из `equity_curve.csv` vs `strategy_metrics.csv`

Скрипт: `tools/audit_sharpe.py`

Он:

1. Находит все `coint4/artifacts/wfa/runs/**/strategy_metrics.csv`
2. Берёт `stored_sharpe_ratio_abs` из `strategy_metrics.csv`
3. Пересчитывает Sharpe из `equity_curve.csv` **точно как rollup** (с инференсом частоты по timestamp deltas)
4. Дополнительно считает:
   - `computed_sharpe_daily_only`: Sharpe с annualization `sqrt(365)` (без учёта 15m частоты) для детекта legacy-bug
   - `computed_sharpe_from_daily_pnl`: Sharpe по daily equity (восстановленной из `daily_pnl.csv`) **только как диагностическую метрику**
     - почему diagnostic-only: `sharpe_ratio_abs` считается на **баровых** equity returns (`equity_curve.csv`, 15m), а `daily_pnl.csv` даёт 1 значение PnL в день (другая частота/дисперсия); из-за этого mean/std returns и итоговый Sharpe могут заметно отличаться даже при одинаковом total PnL.
5. В CSV также пишет диагностику по `daily_pnl.csv`:
   - `daily_pnl_cols` (заголовок/колонки; обычно `<index>|PnL`)
   - `daily_pnl_rows_read` (сколько строк PnL реально прочитано/распарсено)
   - `daily_pnl_initial_equity` (какой initial equity использовали для реконструкции daily equity)

Запуск (из корня репо):

```bash
coint4/.venv/bin/python tools/audit_sharpe.py \
  --runs-glob 'coint4/artifacts/wfa/runs/**/strategy_metrics.csv' \
  --out-csv outputs/sharpe_audit_rows.csv
```

Результат (сводка из stdout):

- `runs_scanned: 1426`
- `runs_compared (stored+equity): 1422`
- `missing_equity_curve_or_unparseable: 4` (equity_curve слишком короткий, <2 точек)
- `max_abs_diff_full: 6.5561074942243724`
- `p95_abs_diff_full: 3.274682433293491`
- `likely_underannualized_count: 151`
- подробные строки: `outputs/sharpe_audit_rows.csv`

Ключевое разбиение:

- **1271 run**: `abs_diff_full < 1e-9` (совпадение stored vs recompute до машинного эпсилона)
  - p95 по этой группе: `2.93e-13`
- **151 run**: `stored_sharpe_ratio_abs` совпадает с `computed_sharpe_daily_only` и отличается от bar-aware Sharpe примерно в `sqrt(96)=9.79796` раз
  - это выглядит как legacy-bug: annualization делали как будто данные дневные (365), забыв домножить на баровую частоту (96/день)

Также важно:

- Sharpe, посчитанный из `daily_pnl.csv` (daily equity), **в среднем заметно отличается** от `sharpe_ratio_abs`:
  - median `abs_diff_daily_pnl ≈ 0.231`
  - p95 `abs_diff_daily_pnl ≈ 3.419`
  - это ожидаемо, т.к. `sharpe_ratio_abs` в коде считается на **баровых** equity returns, а не на дневных.

## 4) Два репрезентативных примера

### 4.1 "Новый" прогон: stored == recompute (bar-aware)

Run dir:

- `coint4/artifacts/wfa/runs/20260213_budget1000_dd_sprint08_stoplossusd_micro/holdout_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91/`

Строка из `outputs/sharpe_audit_rows.csv` (after-fix; с округлением):

- `stored_sharpe_ratio_abs = 3.73091729412658`
- `computed_sharpe_full     = 3.7309172941265625` (diff ~ `1.7e-14`)
- `computed_sharpe_from_daily_pnl = 3.3280672380302683` (diff ~ `0.403`; diagnostic-only)
- `daily_pnl_cols = <index>|PnL`, `daily_pnl_rows_read = 76`, `daily_pnl_initial_equity = 1000.0`

Вывод: `strategy_metrics.csv` уже содержит **годовой bar-aware Sharpe**, совпадающий с rollup-формулой.

### 4.2 "Старый" прогон: stored underannualized, rollup исправляет

Run dir:

- `coint4/artifacts/wfa/runs/20260114_071935_main_2024_wfa/`

Строка из `outputs/sharpe_audit_rows.csv` (с округлением):

- `stored_sharpe_ratio_abs        = 0.16955909705274833`
- `computed_sharpe_full           = 1.661333076104956`
- `computed_sharpe_daily_only     = 0.16955909705272978` (совпадает со stored до `~1e-14`)
- `ratio computed_full/stored     = 9.79796` (= `sqrt(96)` для 15m)

Вывод: в этом run `strategy_metrics.csv` хранит Sharpe с annualization как будто **без учёта** 15m частоты, но rollup пере-считывает Sharpe из `equity_curve.csv` и получает bar-aware годовой Sharpe.

## 5) Каноническое определение Sharpe для ранжирования

Для целей WFA-ранжирования и дальнейших фаз (top-5 audit, sweep-ы) **каноническим** считаем:

- `run_index.csv:sharpe_ratio_abs` (т.е. Sharpe, пересчитанный из `equity_curve.csv` по rollup-формуле)

А не:

- `strategy_metrics.csv:sharpe_ratio_abs` (может быть legacy-underannualized для части старых прогонов)

Это уже реализовано в rollup (см. `coint4/src/coint2/ops/run_index.py:265-271`).

## 6) Итог по Фазе A (A1–A4)

- A1–A2: места расчёта и формула/annualization зафиксированы (см. разделы 1–2).
- A3: добавлен аудит-скрипт `tools/audit_sharpe.py`, результаты сохранены в `outputs/sharpe_audit_rows.csv`.
- A4: значимые расхождения объясняются legacy-underannualization (151 run) и уже компенсируются rollup-канонизацией (`sharpe_ratio_abs` пересчитывается из `equity_curve.csv`). Правок live-кода/торгового движка не требуется.
