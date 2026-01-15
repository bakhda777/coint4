# Журнал прогонов оптимизации (2026-01-14)

Назначение: фиксировать параметры запусков, артефакты и метрики для повторов и сравнения кандидатов.

## Статусы
- `active` — идет выполнение.
- `candidate` — выбран для валидации.
- `rejected` — отклонен по результатам валидации.
- `legacy/archived` — устаревший или остановленный прогон.

## Шаблон записи (для читаемости Codex)
```yaml
run_id: opt_signals_dynamic_coarse_YYYYMMDD_vX
stage: optimization | wfa | holdout
config: configs/main_2024_optimize_dynamic.yaml
search_space: configs/search_spaces/optimize_signals.yaml
data_window:
  train_days: 60
  test_days: 30
  start_date: 2023-10-01
  end_date: 2023-12-31
pair_filter_summary:
  - step: 1
    period: MM/DD-MM/DD
    candidates_total: 0
    passed_pairs: 0
    dropped_by_reason:
      low_correlation: 0
      beta: 0
      mean_crossings: 0
      half_life: 0
      pvalue: 0
      hurst: 0
      kpss: 0
      market_microstructure: 0
    dropped_other: 0
    remaining_after_stage:
      after_low_correlation: 0
      after_beta: 0
      after_mean_crossings: 0
      after_half_life: 0
      after_pvalue: 0
      after_hurst: 0
      after_kpss: 0
      after_market_microstructure: 0
```

## Оптимизация (Optuna)

### opt_signals_dynamic_coarse_20260114_v4
Команда (из `coint4/`):
```bash
PYTHONPATH=src ./.venv/bin/python scripts/core/optimize.py \
  --mode balanced \
  --n-trials 4 \
  --config configs/main_2024_optimize_dynamic.yaml \
  --search-space configs/search_spaces/optimize_signals.yaml \
  --study-name opt_signals_dynamic_coarse_20260114_v4
```
- Тип: Optuna (balanced).
- Период WFA: тест `2023-10-01` → `2023-12-30`, 3 шага, обучение 60 дней.
- Universe: динамический, `ssd_top_n=1000`.
- Лучшие метрики (лог Optuna): sharpe_ratio `-11.6984`, total_trades `11056`, win_rate `0.0110`, max_drawdown `0.0016`.
- Лучшие параметры: `zscore_threshold=1.798931168960506`, `zscore_exit=-0.11506435572868955`, `rolling_window=60`, `stop_loss_multiplier=2.1834045098534336`, `time_stop_multiplier=3.3042422429595377`, `cooldown_hours=4`.
- Артефакты: `outputs/studies/opt_signals_dynamic_coarse_20260114_v4.db`, `configs/best_config__opt_signals_dynamic_coarse_20260114_v4__20260114_135848.yaml`.
- Статус: `rejected` (отрицательный Sharpe).

#### Сводка фильтрации пар (best trial #3)
Источник: stdout оптимизации (trial #3, `rolling_window=60`).

```yaml
step: 1
period: 10/01-10/31
candidates_total: 1000
passed_pairs: 4
dropped_by_reason:
  low_correlation: 1
  beta: 272
  mean_crossings: 0
  half_life: 3
  pvalue: 280
  hurst: 260
  kpss: 180
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 999
  after_beta: 727
  after_mean_crossings: 727
  after_half_life: 724
  after_pvalue: 444
  after_hurst: 184
  after_kpss: 4
  after_market_microstructure: 4
---
step: 2
period: 10/31-11/30
candidates_total: 1000
passed_pairs: 4
dropped_by_reason:
  low_correlation: 142
  beta: 187
  mean_crossings: 0
  half_life: 2
  pvalue: 267
  hurst: 165
  kpss: 233
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 858
  after_beta: 671
  after_mean_crossings: 671
  after_half_life: 669
  after_pvalue: 402
  after_hurst: 237
  after_kpss: 4
  after_market_microstructure: 4
---
step: 3
period: 11/30-12/30
candidates_total: 1000
passed_pairs: 12
dropped_by_reason:
  low_correlation: 18
  beta: 196
  mean_crossings: 0
  half_life: 0
  pvalue: 270
  hurst: 271
  kpss: 233
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 982
  after_beta: 786
  after_mean_crossings: 786
  after_half_life: 786
  after_pvalue: 516
  after_hurst: 245
  after_kpss: 12
  after_market_microstructure: 12
```

### opt_signals_dynamic_coarse_20260114_v2
Команда (из `coint4/`):
```bash
PYTHONPATH=src ./.venv/bin/python scripts/core/optimize.py \
  --mode balanced \
  --n-trials 12 \
  --config configs/main_2024_optimize_dynamic.yaml \
  --search-space configs/search_spaces/optimize_signals.yaml \
  --study-name opt_signals_dynamic_coarse_20260114_v2
```
- Тип: Optuna (balanced).
- Период WFA: тест `2023-10-01` → `2023-12-30`, 3 шага, обучение 60 дней.
- Universe: динамический, `ssd_top_n=2000`.
- Лучшие метрики (лог Optuna): sharpe_ratio `-14.8938`, total_trades `29851`, win_rate `0.0000`, max_drawdown `0.0032`.
- Лучшие параметры: `zscore_threshold=1.2093748675144944`, `zscore_exit=-0.015260574604334757`, `rolling_window=48`, `stop_loss_multiplier=2.385850643358193`, `time_stop_multiplier=2.8561156882189986`, `cooldown_hours=1`.
- Артефакты: `outputs/studies/opt_signals_dynamic_coarse_20260114_v2.db`, `configs/best_config__opt_signals_dynamic_coarse_20260114_v2__20260114_122737.yaml`.
- Статус: `rejected` (отрицательный Sharpe).

### opt_signals_dynamic_coarse_20260114_v3
Команда (из `coint4/`):
```bash
PYTHONPATH=src ./.venv/bin/python scripts/core/optimize.py \
  --mode balanced \
  --n-trials 12 \
  --config configs/main_2024_optimize_dynamic.yaml \
  --search-space configs/search_spaces/optimize_signals.yaml \
  --study-name opt_signals_dynamic_coarse_20260114_v3
```
- Тип: Optuna (balanced).
- Период WFA: тест `2023-10-01` → `2023-12-30`, 3 шага, обучение 60 дней.
- Universe: динамический, `ssd_top_n=2000`.
- Статус: `legacy/archived` (прерван вручную из-за слишком долгих trial; результаты не использовать).

### opt_signals_dynamic_coarse_20260114_v1
- Тип: Optuna (balanced).
- Период WFA: ранний 2022, 3 шага, обучение 60 дней.
- Universe: динамический, `ssd_top_n=2000`.
- Лучшие метрики (лог Optuna): sharpe_ratio `-25.9446`, total_trades `52520`, win_rate `0.0220`, max_drawdown `0.0055`.
- Артефакты: `outputs/studies/opt_signals_dynamic_coarse_20260114_v1.db`, `configs/best_config__opt_signals_dynamic_coarse_20260114_v1__20260114_121626.yaml`.
- Статус: `legacy/archived` (отрицательный Sharpe).

### opt_signals_fixed_coarse_20260114_v3
- Тип: Optuna (balanced).
- Период WFA: ранний 2022, 3 шага, обучение 60 дней.
- Universe: фиксированный top-500 (`bench/clean_window_20260114_top500_step3/pairs_universe.yaml`).
- Лучшие метрики (лог Optuna): sharpe_ratio `-7.3729`, total_trades `9830`, win_rate `0.0330`, max_drawdown `0.0008`.
- Артефакты: `outputs/studies/opt_signals_fixed_coarse_20260114_v3.db`, `configs/best_config__opt_signals_fixed_coarse_20260114_v3__20260114_120400.yaml`.
- Статус: `legacy/archived` (отрицательный Sharpe).

### opt_signals_fixed_coarse_20260114_v2
- Тип: Optuna (balanced).
- Период WFA: ранний 2022, 3 шага, обучение 60 дней.
- Universe: фиксированный top-200 (`bench/clean_window_20260114_top200_step3/pairs_universe.yaml`).
- Лучшие метрики (лог Optuna): sharpe_ratio `-5.6486`, total_trades `3596`, win_rate `0.0659`, max_drawdown `0.0020`.
- Артефакты: `outputs/studies/opt_signals_fixed_coarse_20260114_v2.db`, `configs/best_config__opt_signals_fixed_coarse_20260114_v2__20260114_115721.yaml`.
- Статус: `legacy/archived` (отрицательный Sharpe).

### opt_signals_fixed_coarse_20260114_v1
- Тип: Optuna (balanced).
- Причина остановки: отсутствие поддержки `walk_forward.pairs_file` в fast objective на момент запуска.
- Статус: `legacy/archived`.

## Валидация кандидата (configs/main_2024_optuna_candidate.yaml)

### Holdout fixed backtest (top-200, 2024-01-01 → 2024-06-30)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 backtest \
  --config configs/main_2024_optuna_candidate.yaml \
  --pairs-file bench/clean_window_20260114_top200_step3/pairs_universe.yaml \
  --period-start 2024-01-01 \
  --period-end 2024-06-30 \
  --out-dir outputs/fixed_run_optuna_candidate_holdout
```
- Метрики: total_pnl `148.37`, sharpe_ratio `0.0165`, max_drawdown `-280.11`, num_trades `191505`, win_rate `0.3342`.
- Артефакты: `outputs/fixed_run_optuna_candidate_holdout/` (metrics.yaml, equity.csv, trades.csv).
- Детерминизм: повтор не запускался.
- Статус: `rejected` (кандидат не прошел WFA).

### WFA candidate (5 шагов)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optuna_candidate.yaml
```
- Метрики: total_pnl `-228.49`, sharpe_ratio_abs `-0.0939`, max_drawdown_abs `-379.04`, total_trades `1816`, total_pairs_traded `290`.
- Артефакты: `artifacts/wfa/runs/20260114_124742_optuna_candidate_wfa/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Детерминизм: повтор не запускался.
- Статус: `rejected`.

#### Сводка фильтрации пар (WFA candidate)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_124742_optuna_candidate_wfa/`.

```yaml
step: 1
period: 03/01-05/30
candidates_total: 7875
passed_pairs: 21
dropped_by_reason:
  low_correlation: 3001
  beta: 1706
  mean_crossings: 0
  half_life: 0
  pvalue: 965
  hurst: 990
  kpss: 1192
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4874
  after_beta: 3168
  after_mean_crossings: 3168
  after_half_life: 3168
  after_pvalue: 2203
  after_hurst: 1213
  after_kpss: 21
  after_market_microstructure: 21
---
step: 2
period: 03/31-06/29
candidates_total: 10440
passed_pairs: 85
dropped_by_reason:
  low_correlation: 650
  beta: 3626
  mean_crossings: 0
  half_life: 43
  pvalue: 2621
  hurst: 1789
  kpss: 1626
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 9790
  after_beta: 6164
  after_mean_crossings: 6164
  after_half_life: 6121
  after_pvalue: 3500
  after_hurst: 1711
  after_kpss: 85
  after_market_microstructure: 85
---
step: 3
period: 04/30-07/29
candidates_total: 11476
passed_pairs: 80
dropped_by_reason:
  low_correlation: 1605
  beta: 3266
  mean_crossings: 0
  half_life: 0
  pvalue: 2355
  hurst: 2244
  kpss: 1926
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 9871
  after_beta: 6605
  after_mean_crossings: 6605
  after_half_life: 6605
  after_pvalue: 4250
  after_hurst: 2006
  after_kpss: 80
  after_market_microstructure: 80
---
step: 4
period: 05/30-08/28
candidates_total: 12561
passed_pairs: 56
dropped_by_reason:
  low_correlation: 4686
  beta: 2498
  mean_crossings: 0
  half_life: 10
  pvalue: 1977
  hurst: 1705
  kpss: 1629
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 7875
  after_beta: 5377
  after_mean_crossings: 5377
  after_half_life: 5367
  after_pvalue: 3390
  after_hurst: 1685
  after_kpss: 56
  after_market_microstructure: 56
---
step: 5
period: 06/29-09/27
candidates_total: 13041
passed_pairs: 52
dropped_by_reason:
  low_correlation: 5588
  beta: 2497
  mean_crossings: 0
  half_life: 5
  pvalue: 1565
  hurst: 1438
  kpss: 1896
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 7453
  after_beta: 4956
  after_mean_crossings: 4956
  after_half_life: 4951
  after_pvalue: 3386
  after_hurst: 1948
  after_kpss: 52
  after_market_microstructure: 52
```

### WFA sanity (Q4 2023, dynamic selection)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic.yaml \
  --results-dir artifacts/wfa/runs/20260114_125200_optimize_q4_sanity_wfa
```
- Метрики: total_pnl `-20.33`, sharpe_ratio_abs `-0.0903`, max_drawdown_abs `-68.38`, total_trades `473`, total_pairs_traded `51`.
- Артефакты: `artifacts/wfa/runs/20260114_125200_optimize_q4_sanity_wfa/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `legacy/archived` (sanity run).

#### Сводка фильтрации пар (WFA sanity)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_125200_optimize_q4_sanity_wfa/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 2000
passed_pairs: 14
dropped_by_reason:
  low_correlation: 2
  beta: 572
  mean_crossings: 0
  half_life: 3
  pvalue: 564
  hurst: 496
  kpss: 349
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 1998
  after_beta: 1426
  after_mean_crossings: 1426
  after_half_life: 1423
  after_pvalue: 859
  after_hurst: 363
  after_kpss: 14
  after_market_microstructure: 14
---
step: 2
period: 09/01-11/30
candidates_total: 2000
passed_pairs: 7
dropped_by_reason:
  low_correlation: 221
  beta: 485
  mean_crossings: 0
  half_life: 4
  pvalue: 507
  hurst: 324
  kpss: 452
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 1779
  after_beta: 1294
  after_mean_crossings: 1294
  after_half_life: 1290
  after_pvalue: 783
  after_hurst: 459
  after_kpss: 7
  after_market_microstructure: 7
---
step: 3
period: 10/01-12/30
candidates_total: 2000
passed_pairs: 33
dropped_by_reason:
  low_correlation: 31
  beta: 409
  mean_crossings: 0
  half_life: 0
  pvalue: 587
  hurst: 515
  kpss: 425
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 1969
  after_beta: 1560
  after_mean_crossings: 1560
  after_half_life: 1560
  after_pvalue: 973
  after_hurst: 458
  after_kpss: 33
  after_market_microstructure: 33
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=1000)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic.yaml \
  --results-dir artifacts/wfa/runs/20260114_141327_optimize_q4_sanity_wfa_top1000
```
- Метрики: total_pnl `-21.10`, sharpe_ratio_abs `-0.1000`, max_drawdown_abs `-62.65`, total_trades `418`, total_pairs_traded `42`.
- Артефакты: `artifacts/wfa/runs/20260114_141327_optimize_q4_sanity_wfa_top1000/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `legacy/archived` (sanity run).

#### Сводка фильтрации пар (WFA sanity, ssd_top_n=1000)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_141327_optimize_q4_sanity_wfa_top1000/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 1000
passed_pairs: 12
dropped_by_reason:
  low_correlation: 0
  beta: 262
  mean_crossings: 0
  half_life: 2
  pvalue: 320
  hurst: 238
  kpss: 166
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 1000
  after_beta: 738
  after_mean_crossings: 738
  after_half_life: 736
  after_pvalue: 416
  after_hurst: 178
  after_kpss: 12
  after_market_microstructure: 12
---
step: 2
period: 09/01-11/30
candidates_total: 1000
passed_pairs: 4
dropped_by_reason:
  low_correlation: 62
  beta: 229
  mean_crossings: 0
  half_life: 4
  pvalue: 270
  hurst: 192
  kpss: 239
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 938
  after_beta: 709
  after_mean_crossings: 709
  after_half_life: 705
  after_pvalue: 435
  after_hurst: 243
  after_kpss: 4
  after_market_microstructure: 4
---
step: 3
period: 10/01-12/30
candidates_total: 1000
passed_pairs: 29
dropped_by_reason:
  low_correlation: 6
  beta: 160
  mean_crossings: 0
  half_life: 0
  pvalue: 278
  hurst: 284
  kpss: 243
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 994
  after_beta: 834
  after_mean_crossings: 834
  after_half_life: 834
  after_pvalue: 556
  after_hurst: 272
  after_kpss: 29
  after_market_microstructure: 29
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic.yaml \
  --results-dir artifacts/wfa/runs/20260114_141715_optimize_q4_sanity_wfa_top5000
```
- Метрики: total_pnl `-54.83`, sharpe_ratio_abs `-0.1702`, max_drawdown_abs `-99.12`, total_trades `809`, total_pairs_traded `92`.
- Артефакты: `artifacts/wfa/runs/20260114_141715_optimize_q4_sanity_wfa_top5000/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `legacy/archived` (sanity run).

#### Сводка фильтрации пар (WFA sanity, ssd_top_n=5000)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_141715_optimize_q4_sanity_wfa_top5000/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 35
dropped_by_reason:
  low_correlation: 79
  beta: 1452
  mean_crossings: 0
  half_life: 34
  pvalue: 1349
  hurst: 1113
  kpss: 938
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4921
  after_beta: 3469
  after_mean_crossings: 3469
  after_half_life: 3435
  after_pvalue: 2086
  after_hurst: 973
  after_kpss: 35
  after_market_microstructure: 35
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 13
dropped_by_reason:
  low_correlation: 994
  beta: 1138
  mean_crossings: 0
  half_life: 5
  pvalue: 1132
  hurst: 668
  kpss: 1050
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4006
  after_beta: 2868
  after_mean_crossings: 2868
  after_half_life: 2863
  after_pvalue: 1731
  after_hurst: 1063
  after_kpss: 13
  after_market_microstructure: 13
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 47
dropped_by_reason:
  low_correlation: 150
  beta: 1317
  mean_crossings: 0
  half_life: 0
  pvalue: 1551
  hurst: 1160
  kpss: 775
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4850
  after_beta: 3533
  after_mean_crossings: 3533
  after_half_life: 3533
  after_pvalue: 1982
  after_hurst: 822
  after_kpss: 47
  after_market_microstructure: 47
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, relaxed v1)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic.yaml \
  --results-dir artifacts/wfa/runs/20260114_142519_optimize_q4_sanity_wfa_top5000_relaxed
```
- Метрики: total_pnl `-30.18`, sharpe_ratio_abs `-0.1787`, max_drawdown_abs `-59.88`, total_trades `475`, total_pairs_traded `56`.
- Артефакты: `artifacts/wfa/runs/20260114_142519_optimize_q4_sanity_wfa_top5000_relaxed/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `legacy/archived` (sanity run).

#### Сводка фильтрации пар (WFA sanity, relaxed v1)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_142519_optimize_q4_sanity_wfa_top5000_relaxed/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 18
dropped_by_reason:
  low_correlation: 79
  beta: 1452
  mean_crossings: 0
  half_life: 34
  pvalue: 1349
  hurst: 887
  kpss: 1181
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4921
  after_beta: 3469
  after_mean_crossings: 3469
  after_half_life: 3435
  after_pvalue: 2086
  after_hurst: 1199
  after_kpss: 18
  after_market_microstructure: 18
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 8
dropped_by_reason:
  low_correlation: 994
  beta: 1138
  mean_crossings: 0
  half_life: 5
  pvalue: 1132
  hurst: 454
  kpss: 1269
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4006
  after_beta: 2868
  after_mean_crossings: 2868
  after_half_life: 2863
  after_pvalue: 1731
  after_hurst: 1277
  after_kpss: 8
  after_market_microstructure: 8
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 33
dropped_by_reason:
  low_correlation: 150
  beta: 1317
  mean_crossings: 0
  half_life: 0
  pvalue: 1551
  hurst: 947
  kpss: 1002
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4850
  after_beta: 3533
  after_mean_crossings: 3533
  after_half_life: 3533
  after_pvalue: 1982
  after_hurst: 1035
  after_kpss: 33
  after_market_microstructure: 33
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, relaxed v1 + min_correlation)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic.yaml \
  --results-dir artifacts/wfa/runs/20260114_143215_optimize_q4_sanity_wfa_top5000_relaxed_corr
```
- Метрики: total_pnl `-30.18`, sharpe_ratio_abs `-0.1787`, max_drawdown_abs `-59.88`, total_trades `475`, total_pairs_traded `56`.
- Артефакты: `artifacts/wfa/runs/20260114_143215_optimize_q4_sanity_wfa_top5000_relaxed_corr/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `legacy/archived` (sanity run).

#### Сводка фильтрации пар (WFA sanity, relaxed v1 + min_correlation)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_143215_optimize_q4_sanity_wfa_top5000_relaxed_corr/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 18
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1370
  hurst: 887
  kpss: 1185
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2090
  after_hurst: 1203
  after_kpss: 18
  after_market_microstructure: 18
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 8
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1290
  hurst: 462
  kpss: 1336
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 1806
  after_hurst: 1344
  after_kpss: 8
  after_market_microstructure: 8
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 33
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1577
  hurst: 947
  kpss: 1002
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 1982
  after_hurst: 1035
  after_kpss: 33
  after_market_microstructure: 33
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, relaxed v2)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic.yaml \
  --results-dir artifacts/wfa/runs/20260114_143955_optimize_q4_sanity_wfa_top5000_relaxed2
```
- Метрики: не сформированы (прерван вручную в шаге 1 из-за слишком большого числа пар и времени выполнения).
- Артефакты: `artifacts/wfa/runs/20260114_143955_optimize_q4_sanity_wfa_top5000_relaxed2/` (filter_reasons_20260114_144221.csv).
- Статус: `legacy/archived` (aborted).

#### Сводка фильтрации пар (WFA sanity, relaxed v2, шаг 1)
Источник: `filter_reasons_20260114_144221.csv`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 1590
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 0
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 1590
  after_market_microstructure: 1590
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, relaxed v3)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic.yaml \
  --results-dir artifacts/wfa/runs/20260114_144344_optimize_q4_sanity_wfa_top5000_relaxed3
```
- Метрики: total_pnl `-76.61`, sharpe_ratio_abs `-0.1514`, max_drawdown_abs `-178.04`, total_trades `1396`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_144344_optimize_q4_sanity_wfa_top5000_relaxed3/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `legacy/archived` (sanity run).

#### Сводка фильтрации пар (WFA sanity, relaxed v3)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_144344_optimize_q4_sanity_wfa_top5000_relaxed3/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### Fixed backtest (holdout 2024-01-01 → 2024-06-30, top-200, zscore=0.8)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 backtest \
  --config configs/main_2024_optimize_dynamic_zscore_0p8.yaml \
  --pairs-file bench/clean_window_20260114_top200_step3/pairs_universe.yaml \
  --period-start 2024-01-01 \
  --period-end 2024-06-30 \
  --out-dir outputs/fixed_run_zscore_0p8_holdout_20260114
```
- Метрики (metrics.yaml): total_pnl `1461.36`, sharpe_ratio `0.2710`, max_drawdown `-329.99`, num_trades `74366`, win_rate `0.3701`.
- Артефакты: `outputs/fixed_run_zscore_0p8_holdout_20260114/` (metrics.yaml, equity.csv, trades.csv).
- Статус: holdout run.

### Fixed backtest (holdout repeat, determinism check)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 backtest \
  --config configs/main_2024_optimize_dynamic_zscore_0p8.yaml \
  --pairs-file bench/clean_window_20260114_top200_step3/pairs_universe.yaml \
  --period-start 2024-01-01 \
  --period-end 2024-06-30 \
  --out-dir outputs/fixed_run_zscore_0p8_holdout_20260114_repeat
```
- Метрики (metrics.yaml): total_pnl `1461.36`, sharpe_ratio `0.2710`, max_drawdown `-329.99`, num_trades `74366`, win_rate `0.3701`.
- Артефакты: `outputs/fixed_run_zscore_0p8_holdout_20260114_repeat/` (metrics.yaml, equity.csv, trades.csv).
- Статус: determinism repeat (идентичные метрики).

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8.yaml \
  --results-dir artifacts/wfa/runs/20260114_145500_optimize_q4_sanity_wfa_zscore_0p8
```
- Метрики: total_pnl `158.54`, sharpe_ratio_abs `0.2569`, max_drawdown_abs `-82.19`, total_trades `3687`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_145500_optimize_q4_sanity_wfa_zscore_0p8/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run.

#### Сводка фильтрации пар (WFA sanity, zscore=0.8)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_145500_optimize_q4_sanity_wfa_zscore_0p8/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=1.2)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_1p2.yaml \
  --results-dir artifacts/wfa/runs/20260114_150400_optimize_q4_sanity_wfa_zscore_1p2
```
- Метрики: total_pnl `-25.24`, sharpe_ratio_abs `-0.0618`, max_drawdown_abs `-120.52`, total_trades `670`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_150400_optimize_q4_sanity_wfa_zscore_1p2/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run.

#### Сводка фильтрации пар (WFA sanity, zscore=1.2)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_150400_optimize_q4_sanity_wfa_zscore_1p2/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.7)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p7.yaml \
  --results-dir artifacts/wfa/runs/20260114_153500_optimize_q4_sanity_wfa_zscore_0p7
```
- Метрики: total_pnl `156.85`, sharpe_ratio_abs `0.2040`, max_drawdown_abs `-113.28`, total_trades `6566`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_153500_optimize_q4_sanity_wfa_zscore_0p7/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run.

#### Сводка фильтрации пар (WFA sanity, zscore=0.7)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_153500_optimize_q4_sanity_wfa_zscore_0p7/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.9)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p9.yaml \
  --results-dir artifacts/wfa/runs/20260114_154200_optimize_q4_sanity_wfa_zscore_0p9
```
- Метрики: total_pnl `-24.67`, sharpe_ratio_abs `-0.0421`, max_drawdown_abs `-114.92`, total_trades `2212`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_154200_optimize_q4_sanity_wfa_zscore_0p9/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run.

#### Сводка фильтрации пар (WFA sanity, zscore=0.9)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_154200_optimize_q4_sanity_wfa_zscore_0p9/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, cooldown=1h)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_cd1.yaml \
  --results-dir artifacts/wfa/runs/20260114_154800_optimize_q4_sanity_wfa_zscore_0p8_cd1
```
- Метрики: total_pnl `158.54`, sharpe_ratio_abs `0.2569`, max_drawdown_abs `-82.19`, total_trades `3687`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_154800_optimize_q4_sanity_wfa_zscore_0p8_cd1/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run (метрики совпали с zscore=0.8).

#### Сводка фильтрации пар (WFA sanity, zscore=0.8, cooldown=1h)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_154800_optimize_q4_sanity_wfa_zscore_0p8_cd1/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, cooldown=4h)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_cd4.yaml \
  --results-dir artifacts/wfa/runs/20260114_175849_optimize_q4_sanity_wfa_zscore_0p8_cd4
```
- Метрики: total_pnl `158.54`, sharpe_ratio_abs `0.2569`, max_drawdown_abs `-82.19`, total_trades `3687`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_175849_optimize_q4_sanity_wfa_zscore_0p8_cd4/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run (метрики совпали с zscore=0.8).

#### Сводка фильтрации пар (WFA sanity, zscore=0.8, cooldown=4h)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_175849_optimize_q4_sanity_wfa_zscore_0p8_cd4/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, rolling_window=72)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_rw72.yaml \
  --results-dir artifacts/wfa/runs/20260114_155700_optimize_q4_sanity_wfa_zscore_0p8_rw72
```
- Метрики: total_pnl `96.04`, sharpe_ratio_abs `0.1346`, max_drawdown_abs `-108.61`, total_trades `3816`, total_pairs_traded `144`.
- Артефакты: `artifacts/wfa/runs/20260114_155700_optimize_q4_sanity_wfa_zscore_0p8_rw72/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run.

#### Сводка фильтрации пар (WFA sanity, zscore=0.8, rolling_window=72)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_155700_optimize_q4_sanity_wfa_zscore_0p8_rw72/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 61
dropped_by_reason:
  low_correlation: 36
  beta: 1510
  mean_crossings: 0
  half_life: 38
  pvalue: 1057
  hurst: 834
  kpss: 1464
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4964
  after_beta: 3454
  after_mean_crossings: 3454
  after_half_life: 3416
  after_pvalue: 2359
  after_hurst: 1525
  after_kpss: 61
  after_market_microstructure: 61
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 20
dropped_by_reason:
  low_correlation: 712
  beta: 1254
  mean_crossings: 0
  half_life: 14
  pvalue: 1023
  hurst: 422
  kpss: 1555
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4288
  after_beta: 3034
  after_mean_crossings: 3034
  after_half_life: 3020
  after_pvalue: 1997
  after_hurst: 1575
  after_kpss: 20
  after_market_microstructure: 20
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 71
dropped_by_reason:
  low_correlation: 121
  beta: 1292
  mean_crossings: 0
  half_life: 1
  pvalue: 1292
  hurst: 884
  kpss: 1339
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4879
  after_beta: 3587
  after_mean_crossings: 3587
  after_half_life: 3586
  after_pvalue: 2294
  after_hurst: 1410
  after_kpss: 71
  after_market_microstructure: 71
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, rolling_window=120)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_rw120.yaml \
  --results-dir artifacts/wfa/runs/20260114_160200_optimize_q4_sanity_wfa_zscore_0p8_rw120
```
- Метрики: total_pnl `-4.41`, sharpe_ratio_abs `-0.0064`, max_drawdown_abs `-160.55`, total_trades `3594`, total_pairs_traded `156`.
- Артефакты: `artifacts/wfa/runs/20260114_160200_optimize_q4_sanity_wfa_zscore_0p8_rw120/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run.

#### Сводка фильтрации пар (WFA sanity, zscore=0.8, rolling_window=120)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_160200_optimize_q4_sanity_wfa_zscore_0p8_rw120/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 68
dropped_by_reason:
  low_correlation: 53
  beta: 1447
  mean_crossings: 0
  half_life: 30
  pvalue: 1023
  hurst: 836
  kpss: 1543
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4947
  after_beta: 3500
  after_mean_crossings: 3500
  after_half_life: 3470
  after_pvalue: 2447
  after_hurst: 1611
  after_kpss: 68
  after_market_microstructure: 68
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 649
  beta: 1204
  mean_crossings: 0
  half_life: 11
  pvalue: 1029
  hurst: 444
  kpss: 1642
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4351
  after_beta: 3147
  after_mean_crossings: 3147
  after_half_life: 3136
  after_pvalue: 2107
  after_hurst: 1663
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 103
  beta: 1349
  mean_crossings: 0
  half_life: 1
  pvalue: 1250
  hurst: 870
  kpss: 1352
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4897
  after_beta: 3548
  after_mean_crossings: 3548
  after_half_life: 3547
  after_pvalue: 2297
  after_hurst: 1427
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, zscore_exit=0.1)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_exit_0p1.yaml \
  --results-dir artifacts/wfa/runs/20260114_161240_optimize_q4_sanity_wfa_zscore_0p8_exit_0p1
```
- Метрики: total_pnl `181.40`, sharpe_ratio_abs `0.2927`, max_drawdown_abs `-81.79`, total_trades `3690`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_161240_optimize_q4_sanity_wfa_zscore_0p8_exit_0p1/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run.

#### Сводка фильтрации пар (WFA sanity, zscore=0.8, zscore_exit=0.1)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_161240_optimize_q4_sanity_wfa_zscore_0p8_exit_0p1/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, stop_loss=2.5, time_stop=3.5)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_sl2p5_ts3p5.yaml \
  --results-dir artifacts/wfa/runs/20260114_162005_optimize_q4_sanity_wfa_zscore_0p8_sl2p5_ts3p5
```
- Метрики: total_pnl `158.54`, sharpe_ratio_abs `0.2569`, max_drawdown_abs `-82.19`, total_trades `3687`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_162005_optimize_q4_sanity_wfa_zscore_0p8_sl2p5_ts3p5/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run.

#### Сводка фильтрации пар (WFA sanity, zscore=0.8, stop_loss=2.5, time_stop=3.5)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_162005_optimize_q4_sanity_wfa_zscore_0p8_sl2p5_ts3p5/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, filters strict)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_filters_strict.yaml \
  --results-dir artifacts/wfa/runs/20260114_162724_optimize_q4_sanity_wfa_zscore_0p8_filters_strict
```
- Метрики: total_pnl `108.93`, sharpe_ratio_abs `0.2311`, max_drawdown_abs `-80.50`, total_trades `2770`, total_pairs_traded `115`.
- Артефакты: `artifacts/wfa/runs/20260114_162724_optimize_q4_sanity_wfa_zscore_0p8_filters_strict/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run.

#### Сводка фильтрации пар (WFA sanity, zscore=0.8, filters strict)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_162724_optimize_q4_sanity_wfa_zscore_0p8_filters_strict/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 47
dropped_by_reason:
  low_correlation: 79
  beta: 1452
  mean_crossings: 0
  half_life: 34
  pvalue: 1349
  hurst: 887
  kpss: 1152
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4921
  after_beta: 3469
  after_mean_crossings: 3469
  after_half_life: 3435
  after_pvalue: 2086
  after_hurst: 1199
  after_kpss: 47
  after_market_microstructure: 47
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 17
dropped_by_reason:
  low_correlation: 994
  beta: 1138
  mean_crossings: 0
  half_life: 5
  pvalue: 1132
  hurst: 454
  kpss: 1260
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4006
  after_beta: 2868
  after_mean_crossings: 2868
  after_half_life: 2863
  after_pvalue: 1731
  after_hurst: 1277
  after_kpss: 17
  after_market_microstructure: 17
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 57
dropped_by_reason:
  low_correlation: 150
  beta: 1317
  mean_crossings: 0
  half_life: 0
  pvalue: 1551
  hurst: 947
  kpss: 978
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4850
  after_beta: 3533
  after_mean_crossings: 3533
  after_half_life: 3533
  after_pvalue: 1982
  after_hurst: 1035
  after_kpss: 57
  after_market_microstructure: 57
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, zscore_exit=0.15)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_exit_0p15.yaml \
  --results-dir artifacts/wfa/runs/20260114_163803_optimize_q4_sanity_wfa_zscore_0p8_exit_0p15
```
- Метрики: total_pnl `169.82`, sharpe_ratio_abs `0.2750`, max_drawdown_abs `-82.23`, total_trades `3694`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_163803_optimize_q4_sanity_wfa_zscore_0p8_exit_0p15/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run.

#### Сводка фильтрации пар (WFA sanity, zscore=0.8, zscore_exit=0.15)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_163803_optimize_q4_sanity_wfa_zscore_0p8_exit_0p15/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, min_spread_move_sigma=0.2)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_sigma_0p2.yaml \
  --results-dir artifacts/wfa/runs/20260114_172208_optimize_q4_sanity_wfa_zscore_0p8_sigma_0p2
```
- Метрики: total_pnl `158.54`, sharpe_ratio_abs `0.2569`, max_drawdown_abs `-82.19`, total_trades `3687`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_172208_optimize_q4_sanity_wfa_zscore_0p8_sigma_0p2/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run (метрики совпали с базовым zscore=0.8).

#### Сводка фильтрации пар (WFA sanity, zscore=0.8, min_spread_move_sigma=0.2)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_172208_optimize_q4_sanity_wfa_zscore_0p8_sigma_0p2/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, kpss=0.03)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_kpss0p03.yaml \
  --results-dir artifacts/wfa/runs/20260114_175017_optimize_q4_sanity_wfa_zscore_0p8_kpss0p03
```
- Метрики: total_pnl `189.90`, sharpe_ratio_abs `0.2010`, max_drawdown_abs `-134.26`, total_trades `5653`, total_pairs_traded `235`.
- Артефакты: `artifacts/wfa/runs/20260114_175017_optimize_q4_sanity_wfa_zscore_0p8_kpss0p03/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run (хуже базового kpss=0.05).

#### Сводка фильтрации пар (WFA sanity, zscore=0.8, kpss=0.03)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_175017_optimize_q4_sanity_wfa_zscore_0p8_kpss0p03/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 102
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1488
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 102
  after_market_microstructure: 102
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 30
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1613
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 30
  after_market_microstructure: 30
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 117
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1302
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 117
  after_market_microstructure: 117
```

### WFA validation (Aug-Dec 2023, 5 steps, dynamic selection, ssd_top_n=5000, zscore=0.8)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_wfa5.yaml \
  --results-dir artifacts/wfa/runs/20260114_173620_optimize_wfa5_zscore_0p8
```
- Метрики: total_pnl `235.03`, sharpe_ratio_abs `0.2157`, max_drawdown_abs `-148.28`, total_trades `5838`, total_pairs_traded `248`.
- Артефакты: `artifacts/wfa/runs/20260114_173620_optimize_wfa5_zscore_0p8/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: WFA validation run (baseline zscore=0.8).

#### Сводка фильтрации пар (WFA validation, zscore=0.8)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_173620_optimize_wfa5_zscore_0p8/`.

```yaml
step: 1
period: 06/02-08/31
candidates_total: 5000
passed_pairs: 31
dropped_by_reason:
  low_correlation: 701
  beta: 1158
  mean_crossings: 0
  half_life: 2
  pvalue: 465
  hurst: 857
  kpss: 1786
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4299
  after_beta: 3141
  after_mean_crossings: 3141
  after_half_life: 3139
  after_pvalue: 2674
  after_hurst: 1817
  after_kpss: 31
  after_market_microstructure: 31
---
step: 2
period: 07/02-09/30
candidates_total: 5000
passed_pairs: 84
dropped_by_reason:
  low_correlation: 217
  beta: 1531
  mean_crossings: 0
  half_life: 0
  pvalue: 556
  hurst: 838
  kpss: 1774
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4783
  after_beta: 3252
  after_mean_crossings: 3252
  after_half_life: 3252
  after_pvalue: 2696
  after_hurst: 1858
  after_kpss: 84
  after_market_microstructure: 84
---
step: 3
period: 08/01-10/30
candidates_total: 5000
passed_pairs: 69
dropped_by_reason:
  low_correlation: 29
  beta: 1467
  mean_crossings: 0
  half_life: 51
  pvalue: 975
  hurst: 771
  kpss: 1638
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4971
  after_beta: 3504
  after_mean_crossings: 3504
  after_half_life: 3453
  after_pvalue: 2478
  after_hurst: 1707
  after_kpss: 69
  after_market_microstructure: 69
---
step: 4
period: 08/31-11/29
candidates_total: 5000
passed_pairs: 18
dropped_by_reason:
  low_correlation: 787
  beta: 1172
  mean_crossings: 0
  half_life: 32
  pvalue: 973
  hurst: 430
  kpss: 1588
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4213
  after_beta: 3041
  after_mean_crossings: 3041
  after_half_life: 3009
  after_pvalue: 2036
  after_hurst: 1606
  after_kpss: 18
  after_market_microstructure: 18
---
step: 5
period: 09/30-12/29
candidates_total: 5000
passed_pairs: 73
dropped_by_reason:
  low_correlation: 116
  beta: 1306
  mean_crossings: 0
  half_life: 17
  pvalue: 1342
  hurst: 891
  kpss: 1255
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4884
  after_beta: 3578
  after_mean_crossings: 3578
  after_half_life: 3561
  after_pvalue: 2219
  after_hurst: 1328
  after_kpss: 73
  after_market_microstructure: 73
```

### WFA validation (Aug-Dec 2023, 5 steps, dynamic selection, ssd_top_n=5000, zscore=0.8, zscore_exit=0.1)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_exit_0p1_wfa5.yaml \
  --results-dir artifacts/wfa/runs/20260114_164814_optimize_wfa5_zscore_0p8_exit_0p1
```
- Метрики: total_pnl `260.99`, sharpe_ratio_abs `0.2394`, max_drawdown_abs `-137.99`, total_trades `5843`, total_pairs_traded `248`.
- Артефакты: `artifacts/wfa/runs/20260114_164814_optimize_wfa5_zscore_0p8_exit_0p1/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: validation run (5 шагов).

#### Сводка фильтрации пар (WFA validation, zscore=0.8, zscore_exit=0.1, 5 шагов)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_164814_optimize_wfa5_zscore_0p8_exit_0p1/`.

```yaml
step: 1
period: 06/02-08/31
candidates_total: 5000
passed_pairs: 31
dropped_by_reason:
  low_correlation: 701
  beta: 1158
  mean_crossings: 0
  half_life: 2
  pvalue: 465
  hurst: 857
  kpss: 1786
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4299
  after_beta: 3141
  after_mean_crossings: 3141
  after_half_life: 3139
  after_pvalue: 2674
  after_hurst: 1817
  after_kpss: 31
  after_market_microstructure: 31
---
step: 2
period: 07/02-09/30
candidates_total: 5000
passed_pairs: 84
dropped_by_reason:
  low_correlation: 217
  beta: 1531
  mean_crossings: 0
  half_life: 0
  pvalue: 556
  hurst: 838
  kpss: 1774
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4783
  after_beta: 3252
  after_mean_crossings: 3252
  after_half_life: 3252
  after_pvalue: 2696
  after_hurst: 1858
  after_kpss: 84
  after_market_microstructure: 84
---
step: 3
period: 08/01-10/30
candidates_total: 5000
passed_pairs: 69
dropped_by_reason:
  low_correlation: 29
  beta: 1467
  mean_crossings: 0
  half_life: 51
  pvalue: 975
  hurst: 771
  kpss: 1638
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4971
  after_beta: 3504
  after_mean_crossings: 3504
  after_half_life: 3453
  after_pvalue: 2478
  after_hurst: 1707
  after_kpss: 69
  after_market_microstructure: 69
---
step: 4
period: 08/31-11/29
candidates_total: 5000
passed_pairs: 18
dropped_by_reason:
  low_correlation: 787
  beta: 1172
  mean_crossings: 0
  half_life: 32
  pvalue: 973
  hurst: 430
  kpss: 1588
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4213
  after_beta: 3041
  after_mean_crossings: 3041
  after_half_life: 3009
  after_pvalue: 2036
  after_hurst: 1606
  after_kpss: 18
  after_market_microstructure: 18
---
step: 5
period: 09/30-12/29
candidates_total: 5000
passed_pairs: 73
dropped_by_reason:
  low_correlation: 116
  beta: 1306
  mean_crossings: 0
  half_life: 17
  pvalue: 1342
  hurst: 891
  kpss: 1255
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4884
  after_beta: 3578
  after_mean_crossings: 3578
  after_half_life: 3561
  after_pvalue: 2219
  after_hurst: 1328
  after_kpss: 73
  after_market_microstructure: 73
```

### WFA validation (Aug-Dec 2023, 5 steps, dynamic selection, ssd_top_n=5000, zscore=0.8, zscore_exit=0.1, kpss=0.03)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_exit_0p1_wfa5_kpss0p03.yaml \
  --results-dir artifacts/wfa/runs/20260114_170231_optimize_wfa5_zscore_0p8_exit_0p1_kpss0p03
```
- Метрики: total_pnl `355.80`, sharpe_ratio_abs `0.2406`, max_drawdown_abs `-125.69`, total_trades `8749`, total_pairs_traded `382`.
- Артефакты: `artifacts/wfa/runs/20260114_170231_optimize_wfa5_zscore_0p8_exit_0p1_kpss0p03/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: validation run (5 шагов).

#### Сводка фильтрации пар (WFA validation, zscore=0.8, zscore_exit=0.1, kpss=0.03, 5 шагов)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_170231_optimize_wfa5_zscore_0p8_exit_0p1_kpss0p03/`.

```yaml
step: 1
period: 06/02-08/31
candidates_total: 5000
passed_pairs: 47
dropped_by_reason:
  low_correlation: 701
  beta: 1158
  mean_crossings: 0
  half_life: 2
  pvalue: 465
  hurst: 857
  kpss: 1770
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4299
  after_beta: 3141
  after_mean_crossings: 3141
  after_half_life: 3139
  after_pvalue: 2674
  after_hurst: 1817
  after_kpss: 47
  after_market_microstructure: 47
---
step: 2
period: 07/02-09/30
candidates_total: 5000
passed_pairs: 121
dropped_by_reason:
  low_correlation: 217
  beta: 1531
  mean_crossings: 0
  half_life: 0
  pvalue: 556
  hurst: 838
  kpss: 1737
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4783
  after_beta: 3252
  after_mean_crossings: 3252
  after_half_life: 3252
  after_pvalue: 2696
  after_hurst: 1858
  after_kpss: 121
  after_market_microstructure: 121
---
step: 3
period: 08/01-10/30
candidates_total: 5000
passed_pairs: 119
dropped_by_reason:
  low_correlation: 29
  beta: 1467
  mean_crossings: 0
  half_life: 51
  pvalue: 975
  hurst: 771
  kpss: 1588
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4971
  after_beta: 3504
  after_mean_crossings: 3504
  after_half_life: 3453
  after_pvalue: 2478
  after_hurst: 1707
  after_kpss: 119
  after_market_microstructure: 119
---
step: 4
period: 08/31-11/29
candidates_total: 5000
passed_pairs: 27
dropped_by_reason:
  low_correlation: 787
  beta: 1172
  mean_crossings: 0
  half_life: 32
  pvalue: 973
  hurst: 430
  kpss: 1579
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4213
  after_beta: 3041
  after_mean_crossings: 3041
  after_half_life: 3009
  after_pvalue: 2036
  after_hurst: 1606
  after_kpss: 27
  after_market_microstructure: 27
---
step: 5
period: 09/30-12/29
candidates_total: 5000
passed_pairs: 106
dropped_by_reason:
  low_correlation: 116
  beta: 1306
  mean_crossings: 0
  half_life: 17
  pvalue: 1342
  hurst: 891
  kpss: 1222
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4884
  after_beta: 3578
  after_mean_crossings: 3578
  after_half_life: 3561
  after_pvalue: 2219
  after_hurst: 1328
  after_kpss: 106
  after_market_microstructure: 106
```

### Holdout fixed backtest (2024-01-01 → 2024-06-30, top-200, zscore=0.8, zscore_exit=0.1, kpss=0.03)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 backtest \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_exit_0p1_wfa5_kpss0p03.yaml \
  --pairs-file bench/clean_window_20260114_top200_step3/pairs_universe.yaml \
  --period-start 2024-01-01 \
  --period-end 2024-06-30 \
  --out-dir outputs/fixed_run_zscore_0p8_exit_0p1_kpss0p03_holdout_20260114
```
- Метрики: total_pnl `51.13`, sharpe_ratio `0.0080`, max_drawdown `-436.40`, num_trades `125689`, win_rate `0.5301`.
- Артефакты: `outputs/fixed_run_zscore_0p8_exit_0p1_kpss0p03_holdout_20260114/` (metrics.yaml, equity.csv, trades.csv).
- Статус: holdout run (слабая метрика Sharpe).

### Holdout fixed backtest (2024-01-01 → 2024-06-30, top-200, zscore=0.8, zscore_exit=0.1)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 backtest \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_exit_0p1.yaml \
  --pairs-file bench/clean_window_20260114_top200_step3/pairs_universe.yaml \
  --period-start 2024-01-01 \
  --period-end 2024-06-30 \
  --out-dir outputs/fixed_run_zscore_0p8_exit_0p1_holdout_20260114
```
- Метрики: total_pnl `51.13`, sharpe_ratio `0.0080`, max_drawdown `-436.40`, num_trades `125689`, win_rate `0.5301`.
- Артефакты: `outputs/fixed_run_zscore_0p8_exit_0p1_holdout_20260114/` (metrics.yaml, equity.csv, trades.csv).
- Статус: holdout run (Sharpe близок к 0).

## Дополнительные WFA sanity прогоны (параметрические, Q4 2023)

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, corr>=0.35)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_corr0p35.yaml \
  --results-dir artifacts/wfa/runs/20260114_181839_optimize_q4_sanity_wfa_zscore_0p8_corr0p35
```
- Метрики: total_pnl `158.54`, sharpe_ratio_abs `0.2569`, max_drawdown_abs `-82.19`, total_trades `3687`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_181839_optimize_q4_sanity_wfa_zscore_0p8_corr0p35/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run (без улучшения относительно базового).

#### Сводка фильтрации пар (WFA sanity, corr>=0.35)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_181839_optimize_q4_sanity_wfa_zscore_0p8_corr0p35/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 30
  beta: 1458
  mean_crossings: 0
  half_life: 42
  pvalue: 1048
  hurst: 831
  kpss: 1526
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4970
  after_beta: 3512
  after_mean_crossings: 3512
  after_half_life: 3470
  after_pvalue: 2422
  after_hurst: 1591
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 553
  beta: 1241
  mean_crossings: 0
  half_life: 16
  pvalue: 1080
  hurst: 436
  kpss: 1653
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4447
  after_beta: 3206
  after_mean_crossings: 3206
  after_half_life: 3190
  after_pvalue: 2110
  after_hurst: 1674
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 105
  beta: 1331
  mean_crossings: 0
  half_life: 0
  pvalue: 1267
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4895
  after_beta: 3564
  after_mean_crossings: 3564
  after_half_life: 3564
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, hurst<=0.7)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_hurst0p7.yaml \
  --results-dir artifacts/wfa/runs/20260114_182548_optimize_q4_sanity_wfa_zscore_0p8_hurst0p7
```
- Метрики: total_pnl `177.30`, sharpe_ratio_abs `0.2602`, max_drawdown_abs `-98.84`, total_trades `4110`, total_pairs_traded `166`.
- Артефакты: `artifacts/wfa/runs/20260114_182548_optimize_q4_sanity_wfa_zscore_0p8_hurst0p7/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run (Sharpe чуть выше базового, но хуже по просадке).

#### Сводка фильтрации пар (WFA sanity, hurst<=0.7)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_182548_optimize_q4_sanity_wfa_zscore_0p8_hurst0p7/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 72
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 665
  kpss: 1684
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1756
  after_kpss: 72
  after_market_microstructure: 72
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 22
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 348
  kpss: 1709
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1731
  after_kpss: 22
  after_market_microstructure: 22
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 84
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 765
  kpss: 1448
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1532
  after_kpss: 84
  after_market_microstructure: 84
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, kpss=0.07)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_kpss0p07.yaml \
  --results-dir artifacts/wfa/runs/20260114_183705_optimize_q4_sanity_wfa_zscore_0p8_kpss0p07
```
- Метрики: total_pnl `102.99`, sharpe_ratio_abs `0.1878`, max_drawdown_abs `-77.16`, total_trades `2934`, total_pairs_traded `119`.
- Артефакты: `artifacts/wfa/runs/20260114_183705_optimize_q4_sanity_wfa_zscore_0p8_kpss0p07/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `rejected` (хуже базового).

#### Сводка фильтрации пар (WFA sanity, kpss=0.07)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_183705_optimize_q4_sanity_wfa_zscore_0p8_kpss0p07/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 49
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1541
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 49
  after_market_microstructure: 49
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 14
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1629
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 14
  after_market_microstructure: 14
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 62
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1357
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 62
  after_market_microstructure: 62
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, pvalue=0.3)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_pvalue0p3.yaml \
  --results-dir artifacts/wfa/runs/20260114_184657_optimize_q4_sanity_wfa_zscore_0p8_pvalue0p3
```
- Метрики: total_pnl `157.00`, sharpe_ratio_abs `0.2548`, max_drawdown_abs `-82.19`, total_trades `3628`, total_pairs_traded `151`.
- Артефакты: `artifacts/wfa/runs/20260114_184657_optimize_q4_sanity_wfa_zscore_0p8_pvalue0p3/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run (без улучшения относительно базового).

#### Сводка фильтрации пар (WFA sanity, pvalue=0.3)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_184657_optimize_q4_sanity_wfa_zscore_0p8_pvalue0p3/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 63
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1181
  hurst: 785
  kpss: 1431
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2279
  after_hurst: 1494
  after_kpss: 63
  after_market_microstructure: 63
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1141
  hurst: 417
  kpss: 1517
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 1955
  after_hurst: 1538
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1393
  hurst: 843
  kpss: 1248
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2166
  after_hurst: 1323
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, pvalue=0.4)
Команда (из `coint4/`):
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_optimize_dynamic_zscore_0p8_pvalue0p4.yaml \
  --results-dir artifacts/wfa/runs/20260114_185355_optimize_q4_sanity_wfa_zscore_0p8_pvalue0p4
```
- Метрики: total_pnl `166.62`, sharpe_ratio_abs `0.2662`, max_drawdown_abs `-82.19`, total_trades `3767`, total_pairs_traded `157`.
- Артефакты: `artifacts/wfa/runs/20260114_185355_optimize_q4_sanity_wfa_zscore_0p8_pvalue0p4/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run (лучше базового по Sharpe/PnL, но без подтверждения WFA5).

#### Сводка фильтрации пар (WFA sanity, pvalue=0.4)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_185355_optimize_q4_sanity_wfa_zscore_0p8_pvalue0p4/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 69
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 913
  hurst: 864
  kpss: 1614
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2547
  after_hurst: 1683
  after_kpss: 69
  after_market_microstructure: 69
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 916
  hurst: 455
  kpss: 1704
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2180
  after_hurst: 1725
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1103
  hurst: 913
  kpss: 1468
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2456
  after_hurst: 1543
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.8, fullcpu повтор)
Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/main_2024_optimize_dynamic_zscore_0p8.yaml \
  "artifacts/wfa/runs/20260114_192502_fullcpu_zscore0p8_sanity"
```
- Метрики: total_pnl `158.54`, sharpe_ratio_abs `0.2569`, max_drawdown_abs `-82.19`, total_trades `3687`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_192502_fullcpu_zscore0p8_sanity/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: sanity run (повтор базового; детерминизм подтвержден).

#### Сводка фильтрации пар (WFA sanity, fullcpu повтор)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_192502_fullcpu_zscore0p8_sanity/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA smoke (short window, fullcpu)
Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/main_2024_smoke.yaml \
  "artifacts/wfa/runs/20260114_192408_smoke_fullcpu"
```
- Метрики: total_pnl `-52.40`, sharpe_ratio_abs `-0.6205`, max_drawdown_abs `-66.69`, total_trades `394`, total_pairs_traded `108`.
- Артефакты: `artifacts/wfa/runs/20260114_192408_smoke_fullcpu/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `legacy/archived` (smoke run; метрики отрицательные).
- Примечание: в каталоге есть лишние `filter_reasons_*.csv` из других запусков; для smoke учитывать только `filter_reasons_20260114_192423.csv`.

#### Сводка фильтрации пар (WFA smoke, 1 шаг)
Источник: `filter_reasons_20260114_192423.csv` из `artifacts/wfa/runs/20260114_192408_smoke_fullcpu/`.

```yaml
step: 1
period: 01/01-02/01
candidates_total: 5000
passed_pairs: 3108
dropped_by_reason:
  low_correlation: 698
  beta: 382
  mean_crossings: 0
  half_life: 493
  pvalue: 265
  hurst: 54
  kpss: 0
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4302
  after_beta: 3920
  after_mean_crossings: 3920
  after_half_life: 3427
  after_pvalue: 3162
  after_hurst: 3108
  after_kpss: 3108
  after_market_microstructure: 3108
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.75)
Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/main_2024_optimize_dynamic_zscore_0p8_zscore0p75.yaml \
  "artifacts/wfa/runs/20260114_194852_optimize_q4_sanity_wfa_zscore_0p8_zscore0p75"
```
- Метрики: total_pnl `89.92`, sharpe_ratio_abs `0.1326`, max_drawdown_abs `-138.49`, total_trades `4845`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_194852_optimize_q4_sanity_wfa_zscore_0p8_zscore0p75/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `rejected` (хуже базового).

#### Сводка фильтрации пар (WFA sanity, zscore=0.75)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_194852_optimize_q4_sanity_wfa_zscore_0p8_zscore0p75/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.85)
Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/main_2024_optimize_dynamic_zscore_0p8_zscore0p85.yaml \
  "artifacts/wfa/runs/20260114_195745_optimize_q4_sanity_wfa_zscore_0p8_zscore0p85"
```
- Метрики: total_pnl `15.44`, sharpe_ratio_abs `0.0266`, max_drawdown_abs `-106.95`, total_trades `2864`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_195745_optimize_q4_sanity_wfa_zscore_0p8_zscore0p85/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `rejected` (хуже базового).

#### Сводка фильтрации пар (WFA sanity, zscore=0.85)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_195745_optimize_q4_sanity_wfa_zscore_0p8_zscore0p85/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore_exit=0.05)
Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/main_2024_optimize_dynamic_zscore_0p8_exit0p05.yaml \
  "artifacts/wfa/runs/20260114_200445_optimize_q4_sanity_wfa_zscore_0p8_exit0p05"
```
- Метрики: total_pnl `171.03`, sharpe_ratio_abs `0.2764`, max_drawdown_abs `-82.22`, total_trades `3688`, total_pairs_traded `153`.
- Артефакты: `artifacts/wfa/runs/20260114_200445_optimize_q4_sanity_wfa_zscore_0p8_exit0p05/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `candidate` (хороший результат, но уступает комбинации exit0p05 + pvalue0p4).

#### Сводка фильтрации пар (WFA sanity, zscore_exit=0.05)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_200445_optimize_q4_sanity_wfa_zscore_0p8_exit0p05/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 65
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 831
  kpss: 1525
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1590
  after_kpss: 65
  after_market_microstructure: 65
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 436
  kpss: 1622
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1643
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 878
  kpss: 1344
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1419
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore_exit=0.05, pvalue=0.4)
Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/main_2024_optimize_dynamic_zscore_0p8_exit0p05_pvalue0p4.yaml \
  "artifacts/wfa/runs/20260114_202104_optimize_q4_sanity_wfa_zscore_0p8_exit0p05_pvalue0p4"
```
- Метрики: total_pnl `179.11`, sharpe_ratio_abs `0.2855`, max_drawdown_abs `-82.22`, total_trades `3768`, total_pairs_traded `157`.
- Артефакты: `artifacts/wfa/runs/20260114_202104_optimize_q4_sanity_wfa_zscore_0p8_exit0p05_pvalue0p4/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `candidate` (новый лидер по Sharpe/PnL на Q4 sanity).

#### Сводка фильтрации пар (WFA sanity, exit0p05 + pvalue0p4)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_202104_optimize_q4_sanity_wfa_zscore_0p8_exit0p05_pvalue0p4/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 69
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 913
  hurst: 864
  kpss: 1614
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2547
  after_hurst: 1683
  after_kpss: 69
  after_market_microstructure: 69
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 916
  hurst: 455
  kpss: 1704
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2180
  after_hurst: 1725
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1103
  hurst: 913
  kpss: 1468
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2456
  after_hurst: 1543
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore_exit=0.04, pvalue=0.4)
Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/main_2024_optimize_dynamic_zscore_0p8_exit0p04_pvalue0p4.yaml \
  "artifacts/wfa/runs/20260114_204846_optimize_q4_sanity_wfa_zscore_0p8_exit0p04_pvalue0p4"
```
- Метрики: total_pnl `177.89`, sharpe_ratio_abs `0.2835`, max_drawdown_abs `-82.16`, total_trades `3768`, total_pairs_traded `157`.
- Артефакты: `artifacts/wfa/runs/20260114_204846_optimize_q4_sanity_wfa_zscore_0p8_exit0p04_pvalue0p4/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `rejected` (хуже exit0p06 + pvalue0p4 по Sharpe и PnL).

#### Сводка фильтрации пар (WFA sanity, exit0p04 + pvalue0p4)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_204846_optimize_q4_sanity_wfa_zscore_0p8_exit0p04_pvalue0p4/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 69
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 913
  hurst: 864
  kpss: 1614
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2547
  after_hurst: 1683
  after_kpss: 69
  after_market_microstructure: 69
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 916
  hurst: 455
  kpss: 1704
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2180
  after_hurst: 1725
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1103
  hurst: 913
  kpss: 1468
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2456
  after_hurst: 1543
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore_exit=0.06, pvalue=0.4)
Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/main_2024_optimize_dynamic_zscore_0p8_exit0p06_pvalue0p4.yaml \
  "artifacts/wfa/runs/20260114_205609_optimize_q4_sanity_wfa_zscore_0p8_exit0p06_pvalue0p4"
```
- Метрики: total_pnl `182.31`, sharpe_ratio_abs `0.2905`, max_drawdown_abs `-81.50`, total_trades `3768`, total_pairs_traded `157`.
- Артефакты: `artifacts/wfa/runs/20260114_205609_optimize_q4_sanity_wfa_zscore_0p8_exit0p06_pvalue0p4/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `candidate` (новый лидер по Sharpe/PnL на Q4 sanity).

#### Сводка фильтрации пар (WFA sanity, exit0p06 + pvalue0p4)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_205609_optimize_q4_sanity_wfa_zscore_0p8_exit0p06_pvalue0p4/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 69
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 913
  hurst: 864
  kpss: 1614
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2547
  after_hurst: 1683
  after_kpss: 69
  after_market_microstructure: 69
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 916
  hurst: 455
  kpss: 1704
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2180
  after_hurst: 1725
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1103
  hurst: 913
  kpss: 1468
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2456
  after_hurst: 1543
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore_exit=0.05, pvalue=0.45)
Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/main_2024_optimize_dynamic_zscore_0p8_exit0p05_pvalue0p45.yaml \
  "artifacts/wfa/runs/20260114_210901_optimize_q4_sanity_wfa_zscore_0p8_exit0p05_pvalue0p45"
```
- Метрики: total_pnl `179.46`, sharpe_ratio_abs `0.2846`, max_drawdown_abs `-82.22`, total_trades `3831`, total_pairs_traded `160`.
- Артефакты: `artifacts/wfa/runs/20260114_210901_optimize_q4_sanity_wfa_zscore_0p8_exit0p05_pvalue0p45/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `rejected` (ниже лидера exit0p06 + pvalue0p4 по Sharpe).

#### Сводка фильтрации пар (WFA sanity, exit0p05 + pvalue0p45)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_210901_optimize_q4_sanity_wfa_zscore_0p8_exit0p05_pvalue0p45/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 72
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 789
  hurst: 902
  kpss: 1697
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2671
  after_hurst: 1769
  after_kpss: 72
  after_market_microstructure: 72
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 815
  hurst: 464
  kpss: 1796
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2281
  after_hurst: 1817
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 969
  hurst: 956
  kpss: 1559
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2590
  after_hurst: 1634
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore_exit=0.05, pvalue=0.5)
Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/main_2024_optimize_dynamic_zscore_0p8_exit0p05_pvalue0p5.yaml \
  "artifacts/wfa/runs/20260114_211724_optimize_q4_sanity_wfa_zscore_0p8_exit0p05_pvalue0p5"
```
- Метрики: total_pnl `182.40`, sharpe_ratio_abs `0.2785`, max_drawdown_abs `-87.69`, total_trades `4005`, total_pairs_traded `166`.
- Артефакты: `artifacts/wfa/runs/20260114_211724_optimize_q4_sanity_wfa_zscore_0p8_exit0p05_pvalue0p5/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `rejected` (Sharpe ниже exit0p06 + pvalue0p4).

#### Сводка фильтрации пар (WFA sanity, exit0p05 + pvalue0p5)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_211724_optimize_q4_sanity_wfa_zscore_0p8_exit0p05_pvalue0p5/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 73
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 685
  hurst: 930
  kpss: 1772
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2775
  after_hurst: 1845
  after_kpss: 73
  after_market_microstructure: 73
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 23
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 730
  hurst: 472
  kpss: 1871
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2366
  after_hurst: 1894
  after_kpss: 23
  after_market_microstructure: 23
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 78
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 855
  hurst: 994
  kpss: 1632
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2704
  after_hurst: 1710
  after_kpss: 78
  after_market_microstructure: 78
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.75, zscore_exit=0.05, pvalue=0.4)
Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/main_2024_optimize_dynamic_zscore_0p75_exit0p05_pvalue0p4.yaml \
  "artifacts/wfa/runs/20260114_212610_optimize_q4_sanity_wfa_zscore_0p75_exit0p05_pvalue0p4"
```
- Метрики: total_pnl `119.22`, sharpe_ratio_abs `0.1715`, max_drawdown_abs `-138.69`, total_trades `4956`, total_pairs_traded `157`.
- Артефакты: `artifacts/wfa/runs/20260114_212610_optimize_q4_sanity_wfa_zscore_0p75_exit0p05_pvalue0p4/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `rejected` (значимо хуже лидера по Sharpe/PnL).

#### Сводка фильтрации пар (WFA sanity, zscore0p75 + exit0p05 + pvalue0p4)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_212610_optimize_q4_sanity_wfa_zscore_0p75_exit0p05_pvalue0p4/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 69
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 913
  hurst: 864
  kpss: 1614
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2547
  after_hurst: 1683
  after_kpss: 69
  after_market_microstructure: 69
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 916
  hurst: 455
  kpss: 1704
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2180
  after_hurst: 1725
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1103
  hurst: 913
  kpss: 1468
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2456
  after_hurst: 1543
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore=0.85, zscore_exit=0.05, pvalue=0.4)
Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/main_2024_optimize_dynamic_zscore_0p85_exit0p05_pvalue0p4.yaml \
  "artifacts/wfa/runs/20260114_213413_optimize_q4_sanity_wfa_zscore_0p85_exit0p05_pvalue0p4"
```
- Метрики: total_pnl `24.97`, sharpe_ratio_abs `0.0417`, max_drawdown_abs `-100.42`, total_trades `2920`, total_pairs_traded `157`.
- Артефакты: `artifacts/wfa/runs/20260114_213413_optimize_q4_sanity_wfa_zscore_0p85_exit0p05_pvalue0p4/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `rejected` (слабая доходность и Sharpe).

#### Сводка фильтрации пар (WFA sanity, zscore0p85 + exit0p05 + pvalue0p4)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_213413_optimize_q4_sanity_wfa_zscore_0p85_exit0p05_pvalue0p4/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 69
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 913
  hurst: 864
  kpss: 1614
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2547
  after_hurst: 1683
  after_kpss: 69
  after_market_microstructure: 69
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 21
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 916
  hurst: 455
  kpss: 1704
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2180
  after_hurst: 1725
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 75
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1103
  hurst: 913
  kpss: 1468
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2456
  after_hurst: 1543
  after_kpss: 75
  after_market_microstructure: 75
```

### WFA sanity (Q4 2023, dynamic selection, ssd_top_n=5000, zscore_exit=0.05, hurst<=0.7)
Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/main_2024_optimize_dynamic_zscore_0p8_exit0p05_hurst0p7.yaml \
  "artifacts/wfa/runs/20260114_203041_optimize_q4_sanity_wfa_zscore_0p8_exit0p05_hurst0p7"
```
- Метрики: total_pnl `193.25`, sharpe_ratio_abs `0.2831`, max_drawdown_abs `-98.87`, total_trades `4112`, total_pairs_traded `166`.
- Артефакты: `artifacts/wfa/runs/20260114_203041_optimize_q4_sanity_wfa_zscore_0p8_exit0p05_hurst0p7/` (strategy_metrics.csv, equity_curve.csv, daily_pnl.csv, trade_statistics.csv, CointegrationStrategy_performance_report.png, filter_reasons_*.csv).
- Статус: `candidate` (PnL выше, но Sharpe и просадка хуже лидера).

#### Сводка фильтрации пар (WFA sanity, exit0p05 + hurst0p7)
Источник: `filter_reasons_*.csv` + `trade_statistics.csv` из `artifacts/wfa/runs/20260114_203041_optimize_q4_sanity_wfa_zscore_0p8_exit0p05_hurst0p7/`.

```yaml
step: 1
period: 08/02-10/31
candidates_total: 5000
passed_pairs: 72
dropped_by_reason:
  low_correlation: 47
  beta: 1455
  mean_crossings: 0
  half_life: 38
  pvalue: 1039
  hurst: 665
  kpss: 1684
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2421
  after_hurst: 1756
  after_kpss: 72
  after_market_microstructure: 72
---
step: 2
period: 09/01-11/30
candidates_total: 5000
passed_pairs: 22
dropped_by_reason:
  low_correlation: 680
  beta: 1211
  mean_crossings: 0
  half_life: 13
  pvalue: 1017
  hurst: 348
  kpss: 1709
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2079
  after_hurst: 1731
  after_kpss: 22
  after_market_microstructure: 22
---
step: 3
period: 10/01-12/30
candidates_total: 5000
passed_pairs: 84
dropped_by_reason:
  low_correlation: 111
  beta: 1330
  mean_crossings: 0
  half_life: 0
  pvalue: 1262
  hurst: 765
  kpss: 1448
  market_microstructure: 0
dropped_other: 0
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2297
  after_hurst: 1532
  after_kpss: 84
  after_market_microstructure: 84
```

### Failed/archived runs (WFA sanity / smoke)
- `20260114_190126_optimize_q4_sanity_wfa_zscore_0p8_zscore0p85`: артефакты пустые, метрики не сформированы (падение до фикса /dev/shm и process pool). Статус: `legacy/archived`.
- `20260114_192052_smoke_fullcpu`: прерван из-за проблем с process pool (до фикса /dev/shm), метрики не сформированы. Статус: `legacy/archived`.
