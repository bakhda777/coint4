# Журнал прогонов оптимизации (2026-01-15)

Назначение: фиксировать параметры запусков, артефакты и метрики для повторов и сравнения кандидатов.

## Статусы
- `active` — идет выполнение.
- `candidate` — выбран для валидации.
- `rejected` — отклонен по результатам валидации.
- `aborted` — прерван вручную/по ошибке.
- `legacy/archived` — устаревший или остановленный прогон.

## Selection grid (dynamic selection, Q4 2023, 3 шага)

Базовый конфиг для генерации: `coint4/configs/main_2024_optimize_dynamic_zscore_0p8_exit0p06_pvalue0p4.yaml`.

Диапазоны (108 конфигов, полный факторный набор):
- `coint_pvalue_threshold`: 0.30, 0.40, 0.50
- `kpss_pvalue_threshold`: 0.03, 0.05, 0.07
- `max_hurst_exponent`: 0.60, 0.65, 0.70
- `min_correlation`: 0.35, 0.40
- `half_life`: (min=0.001, max=100) и (min=0.01, max=60)

Артефакты и агрегация:
- Сгенерированные конфиги: `coint4/configs/selection_grid_20260115/`
- Manifest: `coint4/configs/selection_grid_20260115/manifest.csv`
- Агрегатор запусков: `coint4/artifacts/wfa/aggregate/20260115_selgrid/`
  - `run_log.txt`, `selected_run_log.txt`, `screening_run_log.txt`
  - `metrics_partial.csv` (частичные метрики)
  - `selected_runs.csv`, `screening_runs.csv` (очереди)

Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/selection_grid_20260115/selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p60_c0p35_hl0p001-100.yaml \
  artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p60_c0p35_hl0p001-100
```

### Выполненные прогоны (Q4 2023, 3 шага)

#### selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p60_c0p35_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `128.36`, sharpe_ratio_abs `0.1612`, max_drawdown_abs `-114.11`, total_trades `4695`, total_pairs_traded `192`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p60_c0p35_hl0p001-100/`.
- Статус: `candidate` (позитивный PnL, Sharpe ниже текущего лидера Q4).

Сводка фильтрации пар (из stdout, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 83
remaining_after_stage:
  after_low_correlation: 4970
  after_beta: 3512
  after_mean_crossings: 3512
  after_half_life: 3470
  after_pvalue: 2280
  after_hurst: 1312
  after_kpss: 83
  after_market_microstructure: 83
---
step: 2
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 25
remaining_after_stage:
  after_low_correlation: 4447
  after_beta: 3206
  after_mean_crossings: 3206
  after_half_life: 3190
  after_pvalue: 1984
  after_hurst: 1480
  after_kpss: 25
  after_market_microstructure: 25
---
step: 3
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 93
remaining_after_stage:
  after_low_correlation: 4895
  after_beta: 3564
  after_mean_crossings: 3564
  after_half_life: 3564
  after_pvalue: 2166
  after_hurst: 1151
  after_kpss: 93
  after_market_microstructure: 93
```

#### selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p65_c0p40_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `203.73`, sharpe_ratio_abs `0.2194`, max_drawdown_abs `-136.78`, total_trades `5414`, total_pairs_traded `226`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p65_c0p40_hl0p001-100/`.
- Статус: `candidate` (лучше по Sharpe/PnL, просадка хуже).

Сводка фильтрации пар (из stdout, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 96
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2279
  after_hurst: 1494
  after_kpss: 96
  after_market_microstructure: 96
---
step: 2
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 30
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 1955
  after_hurst: 1538
  after_kpss: 30
  after_market_microstructure: 30
---
step: 3
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 114
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2166
  after_hurst: 1323
  after_kpss: 114
  after_market_microstructure: 114
```

### Прогоны, прерванные вручную
- `selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p60_c0p35_hl0p01-60`: `aborted` (ручная остановка в шаге 2).
- `selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p70_c0p35_hl0p001-100`: `aborted` (ручная остановка в шаге 2).
- `selgrid_20260115_exit0p06_pv0p30_kpss0p05_h0p60_c0p40_hl0p001-100`: `aborted` (ручная остановка в шаге 1).

### Очереди для продолжения
- `selected_runs.csv` (27 конфигов, широкая выборка по pvalue/kpss/hurst + корреляция/half-life).
- `screening_runs.csv` (12 конфигов, минимальный скрининг по pvalue/hurst + sensitivity по kpss/corr).

## Итоги на текущий момент
- Лучший из завершенных: `pv=0.30, kpss=0.03, hurst=0.65, corr=0.40` (Sharpe `0.2194`, PnL `203.73`).
- Полный факторный прогон не завершен; требуется продолжить по очередям `selected_runs.csv`/`screening_runs.csv`.
