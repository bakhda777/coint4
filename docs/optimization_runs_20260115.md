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

### Скрининг прогоны (12 конфигов, Q4 2023, 3 шага)

#### selgrid_20260115_exit0p06_pv0p30_kpss0p05_h0p60_c0p40_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `143.69302162824897`, sharpe_ratio_abs `0.27314050816343166`, max_drawdown_abs `-79.68192919081775`, total_trades `3081`, total_pairs_traded `127.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p30_kpss0p05_h0p60_c0p40_hl0p001-100/`.
- Статус: `candidate` (скрининг, требуется ранжирование).

Сводка фильтрации пар (из run.log, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 53
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2279
  after_hurst: 1311
  after_kpss: 53
  after_market_microstructure: 53
---
step: 2
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 18
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 1955
  after_hurst: 1452
  after_kpss: 18
  after_market_microstructure: 18
---
step: 3
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 62
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2166
  after_hurst: 1151
  after_kpss: 62
  after_market_microstructure: 62
```

#### selgrid_20260115_exit0p06_pv0p30_kpss0p05_h0p65_c0p40_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `171.7658534933871`, sharpe_ratio_abs `0.2779683162414848`, max_drawdown_abs `-81.50394513848005`, total_trades `3629`, total_pairs_traded `151.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p30_kpss0p05_h0p65_c0p40_hl0p001-100/`.
- Статус: `candidate` (скрининг, требуется ранжирование).

Сводка фильтрации пар (из run.log, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 63
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
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 21
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
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 75
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

#### selgrid_20260115_exit0p06_pv0p30_kpss0p05_h0p70_c0p40_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `194.05798188436893`, sharpe_ratio_abs `0.28467978957289264`, max_drawdown_abs `-98.15751286386512`, total_trades `4053`, total_pairs_traded `164.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p30_kpss0p05_h0p70_c0p40_hl0p001-100/`.
- Статус: `candidate` (скрининг, требуется ранжирование).

Сводка фильтрации пар (из run.log, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 70
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2279
  after_hurst: 1653
  after_kpss: 70
  after_market_microstructure: 70
---
step: 2
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 22
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 1955
  after_hurst: 1622
  after_kpss: 22
  after_market_microstructure: 22
---
step: 3
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 84
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2166
  after_hurst: 1432
  after_kpss: 84
  after_market_microstructure: 84
```

#### selgrid_20260115_exit0p06_pv0p40_kpss0p05_h0p60_c0p40_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `155.1245172986528`, sharpe_ratio_abs `0.28890415758866667`, max_drawdown_abs `-79.68192919081775`, total_trades `3220`, total_pairs_traded `133.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p40_kpss0p05_h0p60_c0p40_hl0p001-100/`.
- Статус: `candidate` (скрининг, требуется ранжирование).

Сводка фильтрации пар (из run.log, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 59
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2547
  after_hurst: 1476
  after_kpss: 59
  after_market_microstructure: 59
---
step: 2
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 18
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2180
  after_hurst: 1632
  after_kpss: 18
  after_market_microstructure: 18
---
step: 3
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 62
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2456
  after_hurst: 1351
  after_kpss: 62
  after_market_microstructure: 62
```

#### selgrid_20260115_exit0p06_pv0p40_kpss0p05_h0p65_c0p40_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `182.31011454897816`, sharpe_ratio_abs `0.2904549901652444`, max_drawdown_abs `-81.50394513848005`, total_trades `3768`, total_pairs_traded `157.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p40_kpss0p05_h0p65_c0p40_hl0p001-100/`.
- Статус: `candidate` (скрининг, требуется ранжирование).

Сводка фильтрации пар (из run.log, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 69
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
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 21
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
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 75
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

#### selgrid_20260115_exit0p06_pv0p40_kpss0p05_h0p70_c0p40_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `205.19897558176308`, sharpe_ratio_abs `0.2964668369958856`, max_drawdown_abs `-98.15751286386512`, total_trades `4192`, total_pairs_traded `170.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p40_kpss0p05_h0p70_c0p40_hl0p001-100/`.
- Статус: `candidate` (скрининг, требуется ранжирование).

Сводка фильтрации пар (из run.log, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 76
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2547
  after_hurst: 1861
  after_kpss: 76
  after_market_microstructure: 76
---
step: 2
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 22
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2180
  after_hurst: 1815
  after_kpss: 22
  after_market_microstructure: 22
---
step: 3
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 84
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2456
  after_hurst: 1659
  after_kpss: 84
  after_market_microstructure: 84
```

#### selgrid_20260115_exit0p06_pv0p50_kpss0p05_h0p60_c0p40_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `157.42060935604968`, sharpe_ratio_abs `0.27697302662393686`, max_drawdown_abs `-85.16032832342898`, total_trades `3457`, total_pairs_traded `142.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p50_kpss0p05_h0p60_c0p40_hl0p001-100/`.
- Статус: `candidate` (скрининг, требуется ранжирование).

Сводка фильтрации пар (из run.log, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 63
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2775
  after_hurst: 1628
  after_kpss: 63
  after_market_microstructure: 63
---
step: 2
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 20
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2366
  after_hurst: 1794
  after_kpss: 20
  after_market_microstructure: 20
---
step: 3
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 65
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2704
  after_hurst: 1509
  after_kpss: 65
  after_market_microstructure: 65
```

#### selgrid_20260115_exit0p06_pv0p50_kpss0p05_h0p65_c0p40_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `185.59489094410674`, sharpe_ratio_abs `0.28329610252749765`, max_drawdown_abs `-86.9823438799358`, total_trades `4005`, total_pairs_traded `166.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p50_kpss0p05_h0p65_c0p40_hl0p001-100/`.
- Статус: `candidate` (скрининг, требуется ранжирование).

Сводка фильтрации пар (из run.log, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 73
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
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 23
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
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 78
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

#### selgrid_20260115_exit0p06_pv0p50_kpss0p05_h0p70_c0p40_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `209.34538990914007`, sharpe_ratio_abs `0.2916378768426878`, max_drawdown_abs `-103.63591160532087`, total_trades `4429`, total_pairs_traded `179.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p50_kpss0p05_h0p70_c0p40_hl0p001-100/`.
- Статус: `candidate` (скрининг, требуется ранжирование).

Сводка фильтрации пар (из run.log, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 80
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2775
  after_hurst: 2032
  after_kpss: 80
  after_market_microstructure: 80
---
step: 2
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 24
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2366
  after_hurst: 1990
  after_kpss: 24
  after_market_microstructure: 24
---
step: 3
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 87
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2704
  after_hurst: 1834
  after_kpss: 87
  after_market_microstructure: 87
```

#### selgrid_20260115_exit0p06_pv0p40_kpss0p03_h0p65_c0p40_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `244.291346857819`, sharpe_ratio_abs `0.2514737986324764`, max_drawdown_abs `-133.9009118170652`, total_trades `5828`, total_pairs_traded `242.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p40_kpss0p03_h0p65_c0p40_hl0p001-100/`.
- Статус: `candidate` (скрининг, требуется ранжирование).

Сводка фильтрации пар (из run.log, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 106
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2547
  after_hurst: 1683
  after_kpss: 106
  after_market_microstructure: 106
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
  after_pvalue: 2180
  after_hurst: 1725
  after_kpss: 30
  after_market_microstructure: 30
---
step: 3
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 120
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2456
  after_hurst: 1543
  after_kpss: 120
  after_market_microstructure: 120
```

#### selgrid_20260115_exit0p06_pv0p40_kpss0p07_h0p65_c0p40_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `127.59895065487945`, sharpe_ratio_abs `0.23031910699519506`, max_drawdown_abs `-72.61445464350982`, total_trades `3015`, total_pairs_traded `123.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p40_kpss0p07_h0p65_c0p40_hl0p001-100/`.
- Статус: `candidate` (скрининг, требуется ранжирование).

Сводка фильтрации пар (из run.log, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 53
remaining_after_stage:
  after_low_correlation: 4953
  after_beta: 3498
  after_mean_crossings: 3498
  after_half_life: 3460
  after_pvalue: 2547
  after_hurst: 1683
  after_kpss: 53
  after_market_microstructure: 53
---
step: 2
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 14
remaining_after_stage:
  after_low_correlation: 4320
  after_beta: 3109
  after_mean_crossings: 3109
  after_half_life: 3096
  after_pvalue: 2180
  after_hurst: 1725
  after_kpss: 14
  after_market_microstructure: 14
---
step: 3
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 62
remaining_after_stage:
  after_low_correlation: 4889
  after_beta: 3559
  after_mean_crossings: 3559
  after_half_life: 3559
  after_pvalue: 2456
  after_hurst: 1543
  after_kpss: 62
  after_market_microstructure: 62
```

#### selgrid_20260115_exit0p06_pv0p40_kpss0p05_h0p65_c0p35_hl0p001-100
- Метрики (strategy_metrics.csv): total_pnl `182.31011454897816`, sharpe_ratio_abs `0.2904549901652444`, max_drawdown_abs `-81.50394513848005`, total_trades `3768`, total_pairs_traded `157.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p40_kpss0p05_h0p65_c0p35_hl0p001-100/`.
- Статус: `candidate` (скрининг, требуется ранжирование).

Сводка фильтрации пар (из run.log, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 69
remaining_after_stage:
  after_low_correlation: 4970
  after_beta: 3512
  after_mean_crossings: 3512
  after_half_life: 3470
  after_pvalue: 2548
  after_hurst: 1684
  after_kpss: 69
  after_market_microstructure: 69
---
step: 2
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 21
remaining_after_stage:
  after_low_correlation: 4447
  after_beta: 3206
  after_mean_crossings: 3206
  after_half_life: 3190
  after_pvalue: 2218
  after_hurst: 1763
  after_kpss: 21
  after_market_microstructure: 21
---
step: 3
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 75
remaining_after_stage:
  after_low_correlation: 4895
  after_beta: 3564
  after_mean_crossings: 3564
  after_half_life: 3564
  after_pvalue: 2456
  after_hurst: 1543
  after_kpss: 75
  after_market_microstructure: 75
```

### Прогоны, прерванные вручную
- `selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p60_c0p35_hl0p01-60`: `aborted` (ручная остановка в шаге 2).
- `selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p70_c0p35_hl0p001-100`: `aborted` (ручная остановка в шаге 2).
- `selgrid_20260115_exit0p06_pv0p30_kpss0p05_h0p60_c0p40_hl0p001-100`: `aborted` (ручная остановка в шаге 1).

### Очереди для продолжения
- `selected_runs.csv` (27 конфигов, широкая выборка по pvalue/kpss/hurst + корреляция/half-life).
- `screening_runs.csv` (12 конфигов, скрининг завершен 12/12; очередь закрыта).

## Итоги на текущий момент
- Лучший по Sharpe среди завершенных: `pv=0.40, kpss=0.05, hurst=0.70, corr=0.40` (Sharpe `0.2965`, PnL `205.20`, DD `-98.16`).
- Лучший по PnL среди завершенных: `pv=0.40, kpss=0.03, hurst=0.65, corr=0.40` (Sharpe `0.2515`, PnL `244.29`, DD `-133.90`).
- Скрининг `screening_runs.csv` завершен; следующий шаг — `selected_runs.csv` или узкая сетка вокруг топов.
