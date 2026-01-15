# План оптимизации параметров (2026-01-14)

Журнал прогонов: `docs/optimization_runs_20260114.md`.
Дополнение (2026-01-15): `docs/optimization_runs_20260115.md` (selection grid по фильтрам, запуск частично завершен).

## Цель и критерии отбора
- Основная метрика: `sharpe_ratio_abs` по WFA (5 шагов).
- Ограничения для выбора кандидата:
  - `total_trades >= 200`
  - `total_pairs_traded >= 50`
  - `max_drawdown_abs >= -200`

## Данные и издержки
- Clean window: `2022-03-01` → `2025-06-30`.
- `exclude_symbols`: пусто.
- Модель издержек: агрегированная (`commission_pct` + `slippage_pct`), `enable_realistic_costs: false`.

## Фиксация допущений (не менять в рамках текущего цикла)
- Лимит WFA: максимум 5 шагов (для большего количества требуется согласование).
- Seed оптимизации: `42` (default в `scripts/core/optimize.py`).
- Clean window / exclude_symbols / модель издержек — как указано выше.
- `pair_selection.min_correlation` применяется в WFA через `walk_forward_orchestrator.py`.

## Схема оценки
- Оптимизация: WFA с `walk_forward.max_steps = 3`, динамическим отбором (`ssd_top_n=5000`) и окном Q4 2023 (без пересечения с holdout).
- Валидация: WFA (5 шагов) с динамическим отбором пар на конфиге кандидата.
- Holdout: fixed backtest `2024-01-01` → `2024-06-30` на top-200 universe.
- Детерминизм: повтор WFA и fixed backtest для выбранного кандидата.

## Universe
- Оптимизация: динамический отбор с `ssd_top_n=5000` (см. конфиг оптимизации).
- Валидация WFA: динамический отбор пар.
- Fixed backtest: `bench/clean_window_20260114_top200_step3/pairs_universe.yaml`.
Примечание: оптимизация поддерживает `walk_forward.pairs_file` (fast objective), но фиксированный universe используется только для справки.

## Search space (signals/risk only)
Используем `configs/search_spaces/optimize_signals.yaml`:
- `zscore_threshold`: 0.8–2.0
- `zscore_exit`: -0.2–0.2
- `rolling_window`: 48–144 (step 12)
- `stop_loss_multiplier`: 2.0–3.0
- `time_stop_multiplier`: 3.0–4.0 (гарантирует `time_stop_multiplier >= stop_loss_multiplier`)
- `cooldown_hours`: 1–6
Базовый trial формируется из `configs/main_2024.yaml` и автоматически клипуется в диапазоны search space.

## Анализ фильтрации пар (последние прогоны)
- WFA candidate (`ssd_top_n=50000`): прошедшие пары 21–85 на шаг, сделки 94–560 на шаг; доминируют отсева `low_correlation`, `beta`, `pvalue`, `hurst`, `kpss`.
- WFA sanity Q4 2023 (`ssd_top_n=2000`): прошедшие пары 7–33 на шаг, сделки 46–294 на шаг; доминируют отсева `beta`, `pvalue`, `hurst`, `kpss`.
- WFA sanity Q4 2023 (`ssd_top_n=1000`): прошедшие пары 4–29 на шаг, сделки 27–276 на шаг; доминируют отсева `pvalue`, `hurst`, `kpss`, `beta`.
- Optuna v4 (`ssd_top_n=1000`): прошедшие пары 4–12 на шаг (по stdout), сильно меньше целевого минимума.
- WFA sanity Q4 2023 (`ssd_top_n=5000`, relaxed v3): прошедшие пары 21–75 на шаг, сделки 105–661 на шаг; шаг 2 по сделкам ниже минимума.
- WFA sanity Q4 2023 (`ssd_top_n=5000`, `zscore=0.8`): прошедшие пары 21–75 на шаг, сделки 273–1841 на шаг; положительный PnL, Sharpe ~0.26.
- WFA sanity Q4 2023 (`ssd_top_n=5000`, `zscore=1.2`): прошедшие пары 21–75 на шаг, сделки 56–321 на шаг; отрицательный PnL, шаг 2 ниже минимума по сделкам.

Целевые минимумы для стабильной статистики:
- Минимум: ≥20 пар и ≥200 сделок на шаг WFA.
- Цель: ≥30 пар и ≥400 сделок на шаг WFA.

Текущая итерация фильтров (sanity):
- `ssd_top_n=5000`, `min_correlation=0.4`, `max_hurst_exponent=0.65`, `kpss_pvalue_threshold=0.05`, `coint_pvalue_threshold=0.35`.

### Selection grid (2026-01-15)
- Сгенерировано 108 конфигов (полный факторный набор) для `coint_pvalue_threshold`, `kpss_pvalue_threshold`, `max_hurst_exponent`, `min_correlation`, `half_life`.
- Очередь запусков и частичные результаты: `docs/optimization_runs_20260115.md`.
- Продолжить: `coint4/artifacts/wfa/aggregate/20260115_selgrid/selected_runs.csv` или `screening_runs.csv` (см. журнал).

Микро-чувствительность сигналов (sanity):
- `zscore=0.8` дает лучший PnL и более плотные сделки по сравнению с `zscore=1.2`; кандидатом для следующей проверки остается нижний порог (0.8–1.0).
- `zscore_exit=0.1` улучшает Sharpe до `0.2927` и PnL до `181.40` при сохранении объема сделок; текущий лидер среди ручных проверок.
- `zscore_exit=0.15` дает Sharpe `0.2750` и PnL `169.82`, хуже чем `zscore_exit=0.1`.
- `stop_loss=2.5` / `time_stop=3.5` не дают улучшения относительно базового `zscore=0.8`.
- Строгие фильтры (`min_correlation=0.5`, `coint_pvalue=0.25`, `max_hurst=0.6`) снижают Sharpe (`0.2311`) и уменьшают число пар/сделок.
- `min_spread_move_sigma=0.2` не влияет на метрики (совпадает с базовым `zscore=0.8`).
- `cooldown_hours=4` не влияет на метрики (совпадает с базовым `zscore=0.8`).
- Ослабление KPSS до `0.03` на Q4 sanity снижает Sharpe до `0.2010` при PnL `189.90` → отклонено (без WFA5).
- Ослабление KPSS до `0.03` на 5 шагах увеличивает пары/PNL (Sharpe `0.2406`, PnL `355.80`, max DD `-125.69`), но Sharpe остается около 0.24.

## Команды (из `coint4/`)
Полный CPU (WFA helper):
```bash
./run_wfa_fullcpu.sh configs/main_2024_smoke.yaml \
  artifacts/wfa/runs/$(date +%Y%m%d_%H%M%S)_smoke_fullcpu
```

Coarse-оптимизация (динамический отбор, 3 шага, Q4 2023):
```bash
PYTHONPATH=src ./.venv/bin/python scripts/core/optimize.py \
  --mode balanced \
  --n-trials 4 \
  --config configs/main_2024_optimize_dynamic.yaml \
  --search-space configs/search_spaces/optimize_signals.yaml \
  --study-name opt_signals_dynamic_coarse_20260114_v4
```

## Текущий кандидат (на проверку)
- Источник: `opt_signals_dynamic_coarse_20260114_v2` (Q4 2023, `ssd_top_n=2000`).
- Метрики best trial: Sharpe `-14.8938`, `total_trades=29851`, `win_rate=0.0000`, `max_drawdown_abs≈0.0032`.
- Конфиг: `configs/main_2024_optuna_candidate.yaml`.
- Holdout (2024-01-01 → 2024-06-30, top-200): `total_pnl=148.37`, `sharpe_ratio=0.0165`, `max_drawdown=-280.11`, `num_trades=191505`.
- WFA candidate (5 шагов): `total_pnl=-228.49`, `sharpe_ratio_abs=-0.0939`, `max_drawdown_abs=-379.04`, `total_trades=1816`.
- Статус: кандидат отклонен (убыточен на WFA).

## Sanity-кандидат (zscore=0.8)
- Конфиг: `configs/main_2024_optimize_dynamic_zscore_0p8.yaml`.
- WFA sanity Q4 2023 (3 шага): `total_pnl=158.54`, `sharpe_ratio_abs=0.2569`, `max_drawdown_abs=-82.19`, `total_trades=3687`.
- WFA validation Aug-Dec 2023 (5 шагов): `total_pnl=235.03`, `sharpe_ratio_abs=0.2157`, `max_drawdown_abs=-148.28`, `total_trades=5838`, `total_pairs_traded=248` (шаг 4: 18 пар, ниже целевого минимума).
- Holdout fixed (2024-01-01 → 2024-06-30, top-200): `total_pnl=1461.36`, `sharpe_ratio=0.2710`, `max_drawdown=-329.99`, `num_trades=74366`.
- Детерминизм: повтор fixed backtest дал идентичные метрики.
- Статус: базовый ориентир; WFA5 Sharpe `0.2157` и шаг 4 < 20 пар → нужна дополнительная устойчивость/улучшение. Лидерство смещается к варианту zscore_exit=0.06 + pvalue=0.4 (см. ниже).

## Sanity-кандидат (zscore=0.8, zscore_exit=0.1)
- Конфиг: `configs/main_2024_optimize_dynamic_zscore_0p8_exit_0p1.yaml`.
- WFA sanity Q4 2023 (3 шага): `total_pnl=181.40`, `sharpe_ratio_abs=0.2927`, `max_drawdown_abs=-81.79`, `total_trades=3690`.
- WFA validation Aug-Dec 2023 (5 шагов): `total_pnl=260.99`, `sharpe_ratio_abs=0.2394`, `max_drawdown_abs=-137.99`, `total_trades=5843`.
- WFA validation Aug-Dec 2023 (5 шагов, kpss=0.03): `total_pnl=355.80`, `sharpe_ratio_abs=0.2406`, `max_drawdown_abs=-125.69`, `total_trades=8749`.
- Holdout fixed (zscore_exit=0.1, 2024-01-01 → 2024-06-30, top-200): `total_pnl=51.13`, `sharpe_ratio=0.0080`, `max_drawdown=-436.40`, `num_trades=125689`.
- Статус: holdout показал слабую устойчивость; вариант `zscore_exit=0.1` отклонен, продолжаем поиск/валидацию.

## Sanity-кандидат (zscore=0.8, zscore_exit=0.05)
- Конфиг: `configs/main_2024_optimize_dynamic_zscore_0p8_exit0p05.yaml`.
- WFA sanity Q4 2023 (3 шага): `total_pnl=171.03`, `sharpe_ratio_abs=0.2764`, `max_drawdown_abs=-82.22`, `total_trades=3688`.
- Статус: хороший результат, но уступает комбинации exit0p05 + pvalue0p4; держим как альтернативный кандидат.

## Sanity-кандидат (zscore=0.8, zscore_exit=0.05, pvalue=0.4)
- Конфиг: `configs/main_2024_optimize_dynamic_zscore_0p8_exit0p05_pvalue0p4.yaml`.
- WFA sanity Q4 2023 (3 шага): `total_pnl=179.11`, `sharpe_ratio_abs=0.2855`, `max_drawdown_abs=-82.22`, `total_trades=3768`.
- Статус: сильный кандидат, но уступает exit0p06 + pvalue0p4 по Sharpe/PnL.

## Sanity-кандидат (zscore=0.8, zscore_exit=0.06, pvalue=0.4)
- Конфиг: `configs/main_2024_optimize_dynamic_zscore_0p8_exit0p06_pvalue0p4.yaml`.
- WFA sanity Q4 2023 (3 шага): `total_pnl=182.31`, `sharpe_ratio_abs=0.2905`, `max_drawdown_abs=-81.50`, `total_trades=3768`.
- Статус: текущий лидер на Q4 sanity; требуется WFA5 и holdout перед финализацией.

## Sanity-кандидат (zscore=0.8, zscore_exit=0.05, hurst=0.7)
- Конфиг: `configs/main_2024_optimize_dynamic_zscore_0p8_exit0p05_hurst0p7.yaml`.
- WFA sanity Q4 2023 (3 шага): `total_pnl=193.25`, `sharpe_ratio_abs=0.2831`, `max_drawdown_abs=-98.87`, `total_trades=4112`.
- Статус: альтернативный кандидат (PnL выше, но Sharpe и просадка хуже лидера).

Последний coarse-прогон:
- `opt_signals_dynamic_coarse_20260114_v4` (Q4 2023, `ssd_top_n=1000`): Sharpe `-11.6984`, `total_trades=11056`, `win_rate=0.0110`, `max_drawdown=0.0016`.
- Конфиг: `configs/best_config__opt_signals_dynamic_coarse_20260114_v4__20260114_135848.yaml`.
- Статус: отклонен (отрицательный Sharpe).

Holdout fixed backtest для кандидата:
```bash
./.venv/bin/coint2 backtest \
  --config configs/main_2024_optuna_candidate.yaml \
  --pairs-file bench/clean_window_20260114_top200_step3/pairs_universe.yaml \
  --period-start 2024-01-01 \
  --period-end 2024-06-30 \
  --out-dir outputs/fixed_run_optuna_candidate_holdout
```

Holdout fixed backtest (sanity-кандидат zscore=0.8):
```bash
./.venv/bin/coint2 backtest \
  --config configs/main_2024_optimize_dynamic_zscore_0p8.yaml \
  --pairs-file bench/clean_window_20260114_top200_step3/pairs_universe.yaml \
  --period-start 2024-01-01 \
  --period-end 2024-06-30 \
  --out-dir outputs/fixed_run_zscore_0p8_holdout_20260114
```

WFA для кандидата:
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optuna_candidate.yaml
```

## Legacy/archived
- `opt_signals_dynamic_coarse_20260114_v3` (динамический universe, 3 шага) остановлен из-за чрезмерного времени выполнения; результаты не использовать.
- `opt_signals_fixed_coarse_20260114_v1` остановлен до добавления поддержки `pairs_file`; результаты не использовать.
- `opt_signals_fixed_coarse_20260114_v2` завершен на top-200 universe; фиксировать только для справки, не использовать для выбора кандидата.
- `opt_signals_fixed_coarse_20260114_v3` завершен на top-500 universe с отрицательным Sharpe; использовать только как справку.
- `opt_signals_dynamic_coarse_20260114_v1` (ранний 2022, `ssd_top_n=2000`) дал отрицательный Sharpe; использовать только как справку.
