# Итоговый отчет (2026-01-14)

## Политики
- `data_filters.clean_window`: `2022-03-01` → `2025-06-30`, `exclude_symbols`: пусто.
- Модель издержек: агрегированная (`commission_pct` + `slippage_pct`), `enable_realistic_costs: false`.
- Ограничение WFA: `walk_forward.max_steps: 5` (для большего количества шагов требуется согласование).

## Обновление universe (clean window, top-200)
Команда:
```bash
./.venv/bin/coint2 scan \
  --config configs/criteria_relaxed.yaml \
  --base-config configs/main_2024.yaml \
  --end-date 2025-06-30 \
  --top-n 200 \
  --output-dir bench/clean_window_20260114_top200_step3
```

Результаты:
- протестировано пар: 5253
- прошли фильтры: 598
- отобрано: 200

Артефакты: `bench/clean_window_20260114_top200_step3/` (`pairs_universe.yaml`, `universe_full.csv`, `UNIVERSE_REPORT.md`, старый `bench/clean_window_20260114_top200/` — archived).

## Fixed backtest на расширенном наборе (top-200)
Команда:
```bash
./.venv/bin/coint2 backtest \
  --config configs/main_2024.yaml \
  --pairs-file bench/clean_window_20260114_top200_step3/pairs_universe.yaml \
  --period-start 2023-06-01 \
  --period-end 2023-08-31 \
  --out-dir outputs/fixed_run_clean_window_top200_20260114_step3
```

Метрики:
- total_pnl: -243.99
- sharpe_ratio: -0.1758
- max_drawdown: -251.82
- num_trades: 37558
- win_rate: 0.3841
- avg_bars_held: 44.06

Повтор для детерминизма:
```bash
./.venv/bin/coint2 backtest \
  --config configs/main_2024.yaml \
  --pairs-file bench/clean_window_20260114_top200_step3/pairs_universe.yaml \
  --period-start 2023-06-01 \
  --period-end 2023-08-31 \
  --out-dir outputs/fixed_run_clean_window_top200_20260114_step3_repeat
```
Результат: метрики полностью совпали (diff = 0). Артефакты: `outputs/fixed_run_clean_window_top200_20260114_step3/`, `outputs/fixed_run_clean_window_top200_20260114_step3_repeat/` (старые `outputs/fixed_run_clean_window_top200_20260114/` и `outputs/fixed_run_clean_window_top200_20260114_repeat/` — archived).

## WFA сравнение (5 шагов)
Артефакты WFA:
- main: `coint4/artifacts/wfa/runs/20260114_093244_main_2024_wfa_step5/`
- main repeat: `coint4/artifacts/wfa/runs/20260114_095105_main_2024_wfa_step5_repeat/`
- balanced: `coint4/artifacts/wfa/runs/20260114_072317_balanced_2024_wfa/`
- balanced repeat: `coint4/artifacts/wfa/runs/20260114_073405_balanced_2024_wfa_repeat/`
- main refresh (legacy): `coint4/artifacts/wfa/runs/20260114_081835_main_2024_wfa_refresh/`
- baseline prev: `coint4/results/strategy_metrics_baseline_prev.csv`

Метрики:
| Run | total_pnl | sharpe_ratio_abs | max_drawdown_abs | total_trades | total_pairs_traded |
| --- | --- | --- | --- | --- | --- |
| main_2024_wfa_5steps | 311.91 | 0.2618 | -126.23 | 2694 | 290 |
| balanced_2024_wfa_5steps | 1.30 | 0.0140 | -20.38 | 265 | 29 |
| baseline_prev | -7.85 | -0.1001 | -14.38 | 113 | 14 |

Разница:
- main - balanced: +310.61 PnL, +0.2478 Sharpe, более глубокая просадка (-105.85).
- balanced - baseline: +9.15 PnL, +0.1141 Sharpe, просадка хуже на 6.00.
- main - baseline: +319.76 PnL, +0.3619 Sharpe, просадка хуже на 111.85.

Повтор WFA (balanced и main): метрики полностью совпали (детерминизм подтвержден).

## Быстрая итерация (smoke)
Артефакты:
- scan (top-50): `bench/fast_iter_20260114_top50/`
- scan (top-100): `bench/fast_iter_20260114_top100/`
- fixed (1 месяц): `outputs/fixed_run_fast_iter_20260114_top50/`
- fixed (1 месяц repeat): `outputs/fixed_run_fast_iter_20260114_top50_repeat/`
- fixed (Q3, top-50): `outputs/fixed_run_fast_iter_20260114_top50_q3/`
- fixed (Q3, top-100): `outputs/fixed_run_fast_iter_20260114_top100/`
- WFA smoke: `coint4/artifacts/wfa/runs/20260114_075638_smoke_fast20_wfa/`
- WFA smoke repeat: `coint4/artifacts/wfa/runs/20260114_075758_smoke_fast20_wfa_repeat/`
- WFA smoke после фикса time_stop_limit: `coint4/artifacts/wfa/runs/20260114_084102_smoke_time_stop_clamp/`

Fixed метрики:
- 1 месяц, top-50: total_pnl 7.61, sharpe_ratio 0.8280, max_drawdown -2.28, trades 1777
- 1 месяц repeat: diff = 0
- Q3, top-50: total_pnl -5.99, sharpe_ratio -0.1782, max_drawdown -11.72
- Q3, top-100: total_pnl -7785.79, sharpe_ratio -0.2466, max_drawdown -8775.97

WFA smoke (1 шаг): total_pnl -52.40, sharpe_ratio_abs -0.6205, max_drawdown_abs -66.69, trades 394, repeat diff = 0.
WFA smoke после фикса time_stop_limit (1 шаг): total_pnl -52.40, sharpe_ratio_abs -0.6205, max_drawdown_abs -66.69, trades 394, ошибок `time_stop_limit` нет.

## Команда production-пайплайна (для согласования)
```bash
# Scan → fixed backtest → WFA (5 шагов)
./.venv/bin/coint2 scan \
  --config configs/criteria_relaxed.yaml \
  --base-config configs/main_2024.yaml \
  --end-date 2025-06-30 \
  --top-n 200 \
  --output-dir bench/clean_window_20260114_top200_step3

./.venv/bin/coint2 backtest \
  --config configs/main_2024.yaml \
  --pairs-file bench/clean_window_20260114_top200_step3/pairs_universe.yaml \
  --period-start 2023-06-01 \
  --period-end 2023-08-31 \
  --out-dir outputs/fixed_run_clean_window_top200_20260114_step3

./.venv/bin/coint2 walk-forward \
  --config configs/main_2024.yaml
```

## Замечания
- WFA использует gap в 1 бар (15 минут), поэтому предупреждает о буфере 0 дней между train/test. Если нужен дневной буфер - требуется отдельная настройка.
- WFA всегда отбирает пары динамически по `pair_selection` на каждом шаге; фиксированный top‑200 применяется только для fixed backtest. Для WFA по фиксированному universe нужен отдельный режим/CLI.
- В WFA main refresh были ошибки по части пар: `time_stop_limit` < 1 (до фикса, сигнал о слишком коротком лимите времени удержания).
- Теперь `time_stop_limit` < 1 автоматически поднимается до 1 с предупреждением; smoke WFA после фикса ошибок не показал.
