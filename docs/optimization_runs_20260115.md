# Журнал прогонов оптимизации (2026-01-15)

Назначение: фиксировать параметры запусков, артефакты и метрики для повторов и сравнения кандидатов.

## Статусы
- `active` — идет выполнение.
- `candidate` — выбран для валидации.
- `rejected` — отклонен по результатам валидации.
- `aborted` — прерван вручную/по ошибке.
- `legacy/archived` — устаревший или остановленный прогон.

## Обновление (2026-01-16)
- Ночные прогоны прерваны из-за отключения сервера; все активные/планируемые запуски требуют возобновления.
- Возобновление через очередь (из `coint4/`):
  ```bash
  PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py \
    --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv
  ```
- Rollup индекс прогонов: `coint4/artifacts/wfa/aggregate/rollup/` (генерация `scripts/optimization/build_run_index.py`).
- Новые возобновленные прогоны фиксируются в `docs/optimization_runs_20260116.md`.

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

## Strict PV grid (dynamic selection, Q4 2023, 3 шага)

Базовый конфиг для генерации: `coint4/configs/main_2024_optimize_dynamic_zscore_0p8_exit0p06_pvalue0p4.yaml`.

Диапазоны (32 конфига, строгий отбор):
- `coint_pvalue_threshold`: 0.05, 0.10
- `kpss_pvalue_threshold`: 0.05, 0.10
- `max_hurst_exponent`: 0.55, 0.60
- `min_correlation`: 0.45, 0.50
- `half_life`: min 0.01/0.05, max 60

Артефакты и агрегация:
- Сгенерированные конфиги: `coint4/configs/selection_grid_20260115_strictpv/`
- Manifest: `coint4/configs/selection_grid_20260115_strictpv/manifest.csv`
- Агрегатор запусков: `coint4/artifacts/wfa/aggregate/20260115_selgrid_strictpv/`
  - `strictpv_configs.txt`, `strictpv_runs.csv`, `strictpv_run_log.txt`

Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/selection_grid_20260115_strictpv/selgrid_20260115_strictpv_exit0p06_pv0p05_kpss0p05_h0p55_c0p45_hl0p01-60.yaml \
  artifacts/wfa/runs/20260115_selgrid_strictpv/selgrid_20260115_strictpv_exit0p06_pv0p05_kpss0p05_h0p55_c0p45_hl0p01-60
```

Статус: `active` (запуск в очереди).

Запуск после перехода на `n_jobs: -1` (каждый прогон использует все ядра, запуск по одному):
```bash
cat artifacts/wfa/aggregate/20260115_selgrid_strictpv/strictpv_configs.txt | \
  xargs -P 1 -I {} bash -lc 'cfg=\"$1\"; run_id=$(basename \"$cfg\" .yaml); ./run_wfa_fullcpu.sh \"$cfg\" \"artifacts/wfa/runs/20260115_selgrid_strictpv/$run_id\"' _ {}
```

Доп. конфиг (отключение top-N, верхний лимит валидатора):
- `coint4/configs/selection_grid_20260115_strictpv/selgrid_20260115_strictpv_exit0p06_pv0p05_kpss0p05_h0p55_c0p50_hl0p01-60_ssdall.yaml` (`ssd_top_n: 500000`).

Команда (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/selection_grid_20260115_strictpv/selgrid_20260115_strictpv_exit0p06_pv0p05_kpss0p05_h0p55_c0p50_hl0p01-60_ssdall.yaml \
  artifacts/wfa/runs/20260115_selgrid_strictpv/selgrid_20260115_strictpv_exit0p06_pv0p05_kpss0p05_h0p55_c0p50_hl0p01-60_ssdall
```

Статус: `planned` (ожидает запуска).

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

## SSD top-N sweep (dynamic selection, Q4 2023, 3 шага)

Базовый конфиг: `coint4/configs/main_2024_optimize_dynamic_zscore_0p8_exit0p06_pvalue0p4.yaml`.

Диапазоны (6 конфигов):
- `ssd_top_n`: 5000, 10000, 25000, 50000, 100000, 200000

Артефакты и агрегация:
- Конфиги: `coint4/configs/ssd_topn_sweep_20260115/`
- Manifest: `coint4/configs/ssd_topn_sweep_20260115/manifest.csv`
- Агрегатор: `coint4/artifacts/wfa/aggregate/20260115_ssd_topn_sweep/` (`run_log.txt`, `run_queue.csv`, `configs.txt`)
- Прогоны: `coint4/artifacts/wfa/runs/20260115_ssd_topn_sweep/`

Команда (из `coint4/`, `n_jobs: -1` в конфигах, запуск по одному для полной загрузки CPU):
```bash
cat configs/ssd_topn_sweep_20260115/ssd_topn_configs.txt | \
  xargs -P 1 -I {} bash -lc 'cfg="$1"; run_id=$(basename "$cfg" .yaml); ./run_wfa_fullcpu.sh "$cfg" "artifacts/wfa/runs/20260115_ssd_topn_sweep/$run_id" > "artifacts/wfa/runs/20260115_ssd_topn_sweep/$run_id/run.log" 2>&1' _ {}
```

Legacy/archived (старый запуск, `n_jobs: 8`, 2 прогона параллельно):
```bash
cat configs/ssd_topn_sweep_20260115/ssd_topn_configs.txt | \
  xargs -P 2 -I {} bash -lc 'cfg="$1"; run_id=$(basename "$cfg" .yaml); ./run_wfa_fullcpu.sh "$cfg" "artifacts/wfa/runs/20260115_ssd_topn_sweep/$run_id" > "artifacts/wfa/runs/20260115_ssd_topn_sweep/$run_id/run.log" 2>&1' _ {}
```

Статус: `partial` (1/6 completed; 2 stalled; 3 planned; активных процессов нет — требуется перезапуск ssd10000/ssd25000 после перехода на `n_jobs: -1`).

### SSD top-N sweep (subset 4 values, 20260115_4vals)

Диапазоны (4 конфига):
- `ssd_top_n`: 5000, 10000, 15000, 25000

Артефакты и агрегация:
- Конфиги: `coint4/configs/ssd_topn_sweep_20260115_4vals/`
- Manifest: `coint4/configs/ssd_topn_sweep_20260115_4vals/manifest.csv`
- Агрегатор: `coint4/artifacts/wfa/aggregate/20260115_ssd_topn_sweep_4vals/` (`run_log.txt`, `run_queue.csv`, `configs.txt`)
- Прогоны: `coint4/artifacts/wfa/runs/20260115_ssd_topn_sweep_4vals/`

Команда (из `coint4/`, `n_jobs: -1` в конфигах, каждый прогон использует все ядра):
```bash
cat configs/ssd_topn_sweep_20260115_4vals/ssd_topn_configs.txt | \
  xargs -P 1 -I {} bash -lc 'cfg="$1"; run_id=$(basename "$cfg" .yaml); ./run_wfa_fullcpu.sh "$cfg" "artifacts/wfa/runs/20260115_ssd_topn_sweep_4vals/$run_id" > "artifacts/wfa/runs/20260115_ssd_topn_sweep_4vals/$run_id/run.log" 2>&1' _ {}
```

Статус: `completed` (4/4, лучший Sharpe в sweep — 0.6502 на ssd25000).

### Выполненные прогоны (Q4 2023, 3 шага)

#### ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd5000
- Метрики (strategy_metrics.csv): total_pnl `182.31`, sharpe_ratio_abs `0.2905`, max_drawdown_abs `-81.50`, total_trades `3768`, total_pairs_traded `157.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_ssd_topn_sweep_4vals/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd5000/`.
- Статус: `rejected` (Sharpe ниже 1).

Сводка фильтрации пар (Q4 2023):
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

#### ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd10000
- Метрики (strategy_metrics.csv): total_pnl `489.63`, sharpe_ratio_abs `0.4998`, max_drawdown_abs `-111.53`, total_trades `5898`, total_pairs_traded `249.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_ssd_topn_sweep_4vals/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd10000/`.
- Статус: `rejected` (Sharpe ниже 1).

Сводка фильтрации пар (Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 10000
passed_pairs: 103
remaining_after_stage:
  after_low_correlation: 9431
  after_beta: 6370
  after_mean_crossings: 6370
  after_half_life: 6320
  after_pvalue: 4730
  after_hurst: 3251
  after_kpss: 103
  after_market_microstructure: 103
---
step: 2
period: 10/31-11/30
candidates_total: 10000
passed_pairs: 36
remaining_after_stage:
  after_low_correlation: 7803
  after_beta: 5461
  after_mean_crossings: 5461
  after_half_life: 5444
  after_pvalue: 3794
  after_hurst: 3002
  after_kpss: 36
  after_market_microstructure: 36
---
step: 3
period: 11/30-12/30
candidates_total: 10000
passed_pairs: 118
remaining_after_stage:
  after_low_correlation: 9558
  after_beta: 6637
  after_mean_crossings: 6637
  after_half_life: 6633
  after_pvalue: 4331
  after_hurst: 2807
  after_kpss: 118
  after_market_microstructure: 118
```

#### ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd15000
- Метрики (strategy_metrics.csv): total_pnl `614.21`, sharpe_ratio_abs `0.5392`, max_drawdown_abs `-175.19`, total_trades `8033`, total_pairs_traded `348.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_ssd_topn_sweep_4vals/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd15000/`.
- Статус: `rejected` (Sharpe ниже 1).

Сводка фильтрации пар (Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 15000
passed_pairs: 147
remaining_after_stage:
  after_low_correlation: 13404
  after_beta: 8792
  after_mean_crossings: 8792
  after_half_life: 8742
  after_pvalue: 6833
  after_hurst: 4747
  after_kpss: 147
  after_market_microstructure: 147
---
step: 2
period: 10/31-11/30
candidates_total: 15000
passed_pairs: 62
remaining_after_stage:
  after_low_correlation: 10830
  after_beta: 7361
  after_mean_crossings: 7361
  after_half_life: 7338
  after_pvalue: 5316
  after_hurst: 4278
  after_kpss: 62
  after_market_microstructure: 62
---
step: 3
period: 11/30-12/30
candidates_total: 15000
passed_pairs: 149
remaining_after_stage:
  after_low_correlation: 14038
  after_beta: 9495
  after_mean_crossings: 9495
  after_half_life: 9486
  after_pvalue: 6094
  after_hurst: 3965
  after_kpss: 149
  after_market_microstructure: 149
```

#### ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000
- Метрики (strategy_metrics.csv): total_pnl `1205.94`, sharpe_ratio_abs `0.6502`, max_drawdown_abs `-199.31`, total_trades `11041`, total_pairs_traded `520.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_ssd_topn_sweep_4vals/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000/`.
- Статус: `candidate` (лидер sweep, Sharpe все еще ниже 1).

Сводка фильтрации пар (Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 25000
passed_pairs: 238
remaining_after_stage:
  after_low_correlation: 20216
  after_beta: 13215
  after_mean_crossings: 13215
  after_half_life: 13164
  after_pvalue: 10517
  after_hurst: 7389
  after_kpss: 238
  after_market_microstructure: 238
---
step: 2
period: 10/31-11/30
candidates_total: 25000
passed_pairs: 104
remaining_after_stage:
  after_low_correlation: 14874
  after_beta: 10035
  after_mean_crossings: 10035
  after_half_life: 10009
  after_pvalue: 7486
  after_hurst: 6030
  after_kpss: 104
  after_market_microstructure: 104
---
step: 3
period: 11/30-12/30
candidates_total: 25000
passed_pairs: 190
remaining_after_stage:
  after_low_correlation: 22285
  after_beta: 15291
  after_mean_crossings: 15291
  after_half_life: 15258
  after_pvalue: 9573
  after_hurst: 6358
  after_kpss: 190
  after_market_microstructure: 190
```


### SSD top-N sweep (subset 3 values, 20260115_3vals)

Диапазоны (3 конфига):
- `ssd_top_n`: 30000, 40000, 50000

Артефакты и агрегация:
- Конфиги: `coint4/configs/ssd_topn_sweep_20260115_3vals/`
- Manifest: `coint4/configs/ssd_topn_sweep_20260115_3vals/manifest.csv`
- Агрегатор: `coint4/artifacts/wfa/aggregate/20260115_ssd_topn_sweep_3vals/` (`run_log.txt`, `run_queue.csv`, `configs.txt`)
- Прогоны: `coint4/artifacts/wfa/runs/20260115_ssd_topn_sweep_3vals/`

Команда (из `coint4/`, `n_jobs: -1` в конфигах, каждый прогон использует все ядра):
```bash
cat configs/ssd_topn_sweep_20260115_3vals/ssd_topn_configs.txt | \
  xargs -P 1 -I {} bash -lc 'cfg="$1"; run_id=$(basename "$cfg" .yaml); ./run_wfa_fullcpu.sh "$cfg" "artifacts/wfa/runs/20260115_ssd_topn_sweep_3vals/$run_id" > "artifacts/wfa/runs/20260115_ssd_topn_sweep_3vals/$run_id/run.log" 2>&1' _ {}
```

Статус: `partial` (1/3 completed; 1 stalled; 1 planned).

Примечание:
- `ssd40000`: run.log остановился на WF-шаге 3 (2026-01-15 20:19); требуется перезапуск.

### Выполненные прогоны (Q4 2023, 3 шага)

#### ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd30000
- Метрики (strategy_metrics.csv): total_pnl `1154.73`, sharpe_ratio_abs `0.5348`, max_drawdown_abs `-201.94`, total_trades `12419`, total_pairs_traded `605.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_ssd_topn_sweep_3vals/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd30000/`.
- Статус: `rejected` (Sharpe ниже 1; хуже ssd25000).

Сводка фильтрации пар (Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 30000
passed_pairs: 296
remaining_after_stage:
  after_low_correlation: 23024
  after_beta: 15165
  after_mean_crossings: 15165
  after_half_life: 15114
  after_pvalue: 11985
  after_hurst: 8464
  after_kpss: 296
  after_market_microstructure: 296
---
step: 2
period: 10/31-11/30
candidates_total: 30000
passed_pairs: 115
remaining_after_stage:
  after_low_correlation: 16249
  after_beta: 10995
  after_mean_crossings: 10995
  after_half_life: 10967
  after_pvalue: 8224
  after_hurst: 6612
  after_kpss: 115
  after_market_microstructure: 115
---
step: 3
period: 11/30-12/30
candidates_total: 30000
passed_pairs: 207
remaining_after_stage:
  after_low_correlation: 25306
  after_beta: 17351
  after_mean_crossings: 17351
  after_half_life: 17289
  after_pvalue: 10734
  after_hurst: 7089
  after_kpss: 207
  after_market_microstructure: 207
```

## Sharpe target (strict signals + tradeability filter, Q4 2023, 3 шага)

Базовый шаблон: строгая сетка p-value (pv=0.05/kpss=0.05/hurst=0.55/corr=0.50/half-life 0.01-60) + ужесточенные сигналы.

Артефакты и агрегация:
- Конфиги: `coint4/configs/sharpe_target_20260115/`
- Агрегатор: `coint4/artifacts/wfa/aggregate/20260115_sharpe_target/` (`run_log.txt`, `metrics_partial.csv`)
- Прогоны: `coint4/artifacts/wfa/runs/20260115_sharpe_target_strict_z1p2/`, `coint4/artifacts/wfa/runs/20260115_sharpe_target_strict_z1p4/`

Команды (из `coint4/`):
```bash
./run_wfa_fullcpu.sh configs/sharpe_target_20260115/sharpe_target_strict_z1p2.yaml \
  artifacts/wfa/runs/20260115_sharpe_target_strict_z1p2

./run_wfa_fullcpu.sh configs/sharpe_target_20260115/sharpe_target_strict_z1p4.yaml \
  artifacts/wfa/runs/20260115_sharpe_target_strict_z1p4
```

### Выполненные прогоны (Q4 2023, 3 шага)

#### 20260115_sharpe_target_strict_z1p2
- Метрики (strategy_metrics.csv): total_pnl `-20.04`, sharpe_ratio_abs `-0.1289`, max_drawdown_abs `-48.85`, total_trades `207`, total_pairs_traded `44.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_sharpe_target_strict_z1p2/`.
- Статус: `rejected` (слишком строгие сигналы, отрицательный Sharpe).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 19
remaining_after_stage:
  after_low_correlation: 4921
  after_beta: 3469
  after_mean_crossings: 3469
  after_half_life: 3407
  after_pvalue: 1004
  after_hurst: 435
  after_kpss: 19
  after_market_microstructure: 19
---
step: 2
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 5
remaining_after_stage:
  after_low_correlation: 4006
  after_beta: 2868
  after_mean_crossings: 2868
  after_half_life: 2831
  after_pvalue: 746
  after_hurst: 420
  after_kpss: 5
  after_market_microstructure: 5
---
step: 3
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 20
remaining_after_stage:
  after_low_correlation: 4850
  after_beta: 3533
  after_mean_crossings: 3533
  after_half_life: 3503
  after_pvalue: 894
  after_hurst: 324
  after_kpss: 20
  after_market_microstructure: 20
```

#### 20260115_sharpe_target_strict_z1p4
- Метрики (strategy_metrics.csv): total_pnl `-4.67`, sharpe_ratio_abs `-0.0346`, max_drawdown_abs `-25.65`, total_trades `118`, total_pairs_traded `44.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_sharpe_target_strict_z1p4/`.
- Статус: `rejected` (строгие сигналы снизили торговую активность без роста Sharpe).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 5000
passed_pairs: 19
remaining_after_stage:
  after_low_correlation: 4921
  after_beta: 3469
  after_mean_crossings: 3469
  after_half_life: 3407
  after_pvalue: 1004
  after_hurst: 435
  after_kpss: 19
  after_market_microstructure: 19
---
step: 2
period: 10/31-11/30
candidates_total: 5000
passed_pairs: 5
remaining_after_stage:
  after_low_correlation: 4006
  after_beta: 2868
  after_mean_crossings: 2868
  after_half_life: 2831
  after_pvalue: 746
  after_hurst: 420
  after_kpss: 5
  after_market_microstructure: 5
---
step: 3
period: 11/30-12/30
candidates_total: 5000
passed_pairs: 20
remaining_after_stage:
  after_low_correlation: 4850
  after_beta: 3533
  after_mean_crossings: 3533
  after_half_life: 3503
  after_pvalue: 894
  after_hurst: 324
  after_kpss: 20
  after_market_microstructure: 20
```

## Quality universe (20260115_500k)
- Скрипт: `coint4/scripts/universe/build_quality_universe.py`.
- Период качества: 2023-01-01..2023-09-30, bar 15m, min_history 180d, coverage>=0.9, avg_daily_turnover>=500k, max_days_since_last=14.
- Результат: 50 символов включено, 226 исключено (short_history 43, low_coverage 58, low_turnover 216).
- Артефакты: `coint4/artifacts/universe/quality_universe_20260115/` (quality_report.csv, exclude_symbols.yaml, included_symbols.txt, excluded_symbols.txt, quality_summary.json).

### WFA: quality universe 500k + z0p8 corr0p45 (Q4 2023)
- Конфиги: `coint4/configs/quality_runs_20260115/`.
- Агрегатор: `coint4/artifacts/wfa/aggregate/20260115_quality_universe_500k/` (`metrics_partial.csv`, `run_log.txt`).
- Прогоны: `20260115_quality_universe_500k_z0p8_corr0p45`.
- Команда:
  - `./run_wfa_fullcpu.sh configs/quality_runs_20260115/quality_500k_z0p8_corr0p45.yaml artifacts/wfa/runs/20260115_quality_universe_500k_z0p8_corr0p45`

#### 20260115_quality_universe_500k_z0p8_corr0p45
- Метрики (strategy_metrics.csv): total_pnl `145.72`, sharpe_ratio_abs `0.3551`, max_drawdown_abs `-13.80`, total_trades `203`, total_pairs_traded `9.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_quality_universe_500k_z0p8_corr0p45/`.
- Худшая пара по PnL: `ETHUSDC-ETHUSDT` (`-0.91`).
- Статус: `rejected` (меньше пар и Sharpe ниже 200k/250k).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 1225
passed_pairs: 3
remaining_after_stage:
  after_low_correlation: 1014
  after_beta: 674
  after_mean_crossings: 674
  after_half_life: 660
  after_pvalue: 133
  after_hurst: 64
  after_kpss: 3
  after_market_microstructure: 3
---
step: 2
period: 10/31-11/30
candidates_total: 1225
passed_pairs: 2
remaining_after_stage:
  after_low_correlation: 714
  after_beta: 488
  after_mean_crossings: 488
  after_half_life: 483
  after_pvalue: 105
  after_hurst: 53
  after_kpss: 2
  after_market_microstructure: 2
---
step: 3
period: 11/30-12/30
candidates_total: 1275
passed_pairs: 4
remaining_after_stage:
  after_low_correlation: 1007
  after_beta: 698
  after_mean_crossings: 698
  after_half_life: 697
  after_pvalue: 88
  after_hurst: 38
  after_kpss: 4
  after_market_microstructure: 4
```

## Quality universe (20260115_250k)
- Скрипт: `coint4/scripts/universe/build_quality_universe.py`.
- Период качества: 2023-01-01..2023-09-30, bar 15m, min_history 180d, coverage>=0.9, avg_daily_turnover>=250k, max_days_since_last=14.
- Результат: 71 символов включено, 205 исключено (short_history 43, low_coverage 58, low_turnover 187).
- Артефакты: `coint4/artifacts/universe/quality_universe_20260115_250k/` (quality_report.csv, exclude_symbols.yaml, included_symbols.txt, excluded_symbols.txt, quality_summary.json).

### WFA: quality universe + strictpv base (Q4 2023)
- Конфиги: `coint4/configs/quality_runs_20260115/`.
- Агрегатор: `coint4/artifacts/wfa/aggregate/20260115_quality_universe/` (`metrics_partial.csv`, `run_log.txt`).
- Прогоны: `20260115_quality_universe_z0p8`, `20260115_quality_universe_z1p2_hl0p05_mc3_c0p5`.
- Команды:
  - `./run_wfa_fullcpu.sh configs/quality_runs_20260115/quality_strictpv_z0p8.yaml artifacts/wfa/runs/20260115_quality_universe_z0p8`
  - `./run_wfa_fullcpu.sh configs/quality_runs_20260115/quality_strictpv_z1p2_hl0p05_mc3_c0p5.yaml artifacts/wfa/runs/20260115_quality_universe_z1p2_hl0p05_mc3_c0p5`

#### 20260115_quality_universe_z0p8
- Метрики (strategy_metrics.csv): total_pnl `153.62`, sharpe_ratio_abs `0.3724`, max_drawdown_abs `-16.84`, total_trades `294`, total_pairs_traded `13.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_quality_universe_z0p8/`.
- Худшая пара по PnL: `GMTUSDT-SANDUSDT` (`-3.79`).
- Статус: `candidate` (ниже Sharpe=1; качество пары улучшено, но рост Sharpe не достигнут).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 2485
passed_pairs: 6
remaining_after_stage:
  after_low_correlation: 2028
  after_beta: 1309
  after_mean_crossings: 1309
  after_half_life: 1292
  after_pvalue: 282
  after_hurst: 138
  after_kpss: 6
  after_market_microstructure: 6
---
step: 2
period: 10/31-11/30
candidates_total: 2485
passed_pairs: 2
remaining_after_stage:
  after_low_correlation: 1393
  after_beta: 943
  after_mean_crossings: 943
  after_half_life: 937
  after_pvalue: 214
  after_hurst: 116
  after_kpss: 2
  after_market_microstructure: 2
---
step: 3
period: 11/30-12/30
candidates_total: 2556
passed_pairs: 5
remaining_after_stage:
  after_low_correlation: 1981
  after_beta: 1362
  after_mean_crossings: 1362
  after_half_life: 1361
  after_pvalue: 161
  after_hurst: 63
  after_kpss: 5
  after_market_microstructure: 5
```

#### 20260115_quality_universe_z1p2_hl0p05_mc3_c0p5
- Метрики (strategy_metrics.csv): total_pnl `17.64`, sharpe_ratio_abs `0.2266`, max_drawdown_abs `-5.21`, total_trades `52`, total_pairs_traded `11.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_quality_universe_z1p2_hl0p05_mc3_c0p5/`.
- Худшая пара по PnL: `GMTUSDT-SANDUSDT` (`-1.46`).
- Статус: `rejected` (сильная фильтрация и более строгий вход снизили Sharpe).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 2485
passed_pairs: 5
remaining_after_stage:
  after_low_correlation: 1963
  after_beta: 1259
  after_mean_crossings: 1259
  after_half_life: 1240
  after_pvalue: 268
  after_hurst: 133
  after_kpss: 5
  after_market_microstructure: 5
---
step: 2
period: 10/31-11/30
candidates_total: 2485
passed_pairs: 1
remaining_after_stage:
  after_low_correlation: 1294
  after_beta: 877
  after_mean_crossings: 877
  after_half_life: 871
  after_pvalue: 206
  after_hurst: 112
  after_kpss: 1
  after_market_microstructure: 1
---
step: 3
period: 11/30-12/30
candidates_total: 2556
passed_pairs: 5
remaining_after_stage:
  after_low_correlation: 1959
  after_beta: 1359
  after_mean_crossings: 1359
  after_half_life: 1355
  after_pvalue: 158
  after_hurst: 63
  after_kpss: 5
  after_market_microstructure: 5
```

### WFA: quality universe 250k + corr0.45 alignment (Q4 2023)
- Конфиги: `coint4/configs/quality_runs_20260115/`.
- Агрегатор: `coint4/artifacts/wfa/aggregate/20260115_quality_universe/` (`metrics_partial.csv`, `run_log.txt`).
- Прогоны: `20260115_quality_universe_250k_z1p0_exit0p15_corr0p45`, `20260115_quality_universe_250k_z0p8_corr0p45`.
- Команды:
  - `./run_wfa_fullcpu.sh configs/quality_runs_20260115/quality_250k_z1p0_exit0p15_corr0p45.yaml artifacts/wfa/runs/20260115_quality_universe_250k_z1p0_exit0p15_corr0p45`
  - `./run_wfa_fullcpu.sh configs/quality_runs_20260115/quality_250k_z0p8_corr0p45.yaml artifacts/wfa/runs/20260115_quality_universe_250k_z0p8_corr0p45`

#### 20260115_quality_universe_250k_z1p0_exit0p15_corr0p45
- Метрики (strategy_metrics.csv): total_pnl `129.25`, sharpe_ratio_abs `0.3241`, max_drawdown_abs `-10.10`, total_trades `107`, total_pairs_traded `13.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_quality_universe_250k_z1p0_exit0p15_corr0p45/`.
- Худшая пара по PnL: `GMTUSDT-SANDUSDT` (`-1.56`).
- Статус: `rejected` (Sharpe ниже базового z0p8; exit 0.15/entry 1.0 не улучшили профиль).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 2485
passed_pairs: 6
remaining_after_stage:
  after_low_correlation: 2028
  after_beta: 1309
  after_mean_crossings: 1309
  after_half_life: 1292
  after_pvalue: 282
  after_hurst: 138
  after_kpss: 6
  after_market_microstructure: 6
---
step: 2
period: 10/31-11/30
candidates_total: 2485
passed_pairs: 2
remaining_after_stage:
  after_low_correlation: 1393
  after_beta: 943
  after_mean_crossings: 943
  after_half_life: 937
  after_pvalue: 214
  after_hurst: 116
  after_kpss: 2
  after_market_microstructure: 2
---
step: 3
period: 11/30-12/30
candidates_total: 2556
passed_pairs: 5
remaining_after_stage:
  after_low_correlation: 1981
  after_beta: 1362
  after_mean_crossings: 1362
  after_half_life: 1361
  after_pvalue: 161
  after_hurst: 63
  after_kpss: 5
  after_market_microstructure: 5
```

#### 20260115_quality_universe_250k_z0p8_corr0p45
- Метрики (strategy_metrics.csv): total_pnl `153.62`, sharpe_ratio_abs `0.3724`, max_drawdown_abs `-16.84`, total_trades `294`, total_pairs_traded `13.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_quality_universe_250k_z0p8_corr0p45/`.
- Худшая пара по PnL: `GMTUSDT-SANDUSDT` (`-3.79`).
- Статус: `candidate` (результат совпал с базовым z0p8; выравнивание min_correlation_threshold не изменило Sharpe).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 2485
passed_pairs: 6
remaining_after_stage:
  after_low_correlation: 2028
  after_beta: 1309
  after_mean_crossings: 1309
  after_half_life: 1292
  after_pvalue: 282
  after_hurst: 138
  after_kpss: 6
  after_market_microstructure: 6
---
step: 2
period: 10/31-11/30
candidates_total: 2485
passed_pairs: 2
remaining_after_stage:
  after_low_correlation: 1393
  after_beta: 943
  after_mean_crossings: 943
  after_half_life: 937
  after_pvalue: 214
  after_hurst: 116
  after_kpss: 2
  after_market_microstructure: 2
---
step: 3
period: 11/30-12/30
candidates_total: 2556
passed_pairs: 5
remaining_after_stage:
  after_low_correlation: 1981
  after_beta: 1362
  after_mean_crossings: 1362
  after_half_life: 1361
  after_pvalue: 161
  after_hurst: 63
  after_kpss: 5
  after_market_microstructure: 5
```

## Quality universe (20260115_200k)
- Скрипт: `coint4/scripts/universe/build_quality_universe.py`.
- Период качества: 2023-01-01..2023-09-30, bar 15m, min_history 180d, coverage>=0.9, avg_daily_turnover>=200k, max_days_since_last=14.
- Результат: 77 символов включено, 199 исключено (short_history 43, low_coverage 58, low_turnover 177).
- Артефакты: `coint4/artifacts/universe/quality_universe_20260115_200k/` (quality_report.csv, exclude_symbols.yaml, included_symbols.txt, excluded_symbols.txt, quality_summary.json).

### WFA: quality universe 200k + z1p0/exit0p15 (Q4 2023)
- Конфиги: `coint4/configs/quality_runs_20260115/`.
- Агрегатор: `coint4/artifacts/wfa/aggregate/20260115_quality_universe_200k/` (`metrics_partial.csv`, `run_log.txt`).
- Прогоны: `20260115_quality_universe_200k_z1p0_exit0p15`, `20260115_quality_universe_200k_z1p0_exit0p15_blacklist`, `20260115_quality_universe_200k_z0p8_corr0p45`, `20260115_quality_universe_200k_z0p8_corr0p45_signal_strict`, `20260115_quality_universe_200k_z0p8_corr0p45_tradeable`, `20260115_quality_universe_200k_z0p8_corr0p5_hl0p05-45`, `20260115_quality_universe_200k_z0p9_exit0p1`, `20260115_quality_universe_200k_z1p0_exit0p1`.
- Команды:
  - `./run_wfa_fullcpu.sh configs/quality_runs_20260115/quality_200k_z1p0_exit0p15.yaml artifacts/wfa/runs/20260115_quality_universe_200k_z1p0_exit0p15`
  - `./run_wfa_fullcpu.sh configs/quality_runs_20260115/quality_200k_z1p0_exit0p15_blacklist.yaml artifacts/wfa/runs/20260115_quality_universe_200k_z1p0_exit0p15_blacklist`

#### 20260115_quality_universe_200k_z1p0_exit0p15
- Метрики (strategy_metrics.csv): total_pnl `141.80`, sharpe_ratio_abs `0.3379`, max_drawdown_abs `-19.96`, total_trades `177`, total_pairs_traded `18.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_quality_universe_200k_z1p0_exit0p15/`.
- Худшая пара по PnL: `DOGEUSDC-ETHWUSDT` (`-5.66`).
- Статус: `candidate` (Sharpe ниже 250k z0p8; шум снизили, но Sharpe>1 не достигнут).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 2926
passed_pairs: 10
remaining_after_stage:
  after_low_correlation: 2355
  after_beta: 1544
  after_mean_crossings: 1544
  after_half_life: 1527
  after_pvalue: 381
  after_hurst: 186
  after_kpss: 10
  after_market_microstructure: 10
---
step: 2
period: 10/31-11/30
candidates_total: 2926
passed_pairs: 3
remaining_after_stage:
  after_low_correlation: 1574
  after_beta: 1083
  after_mean_crossings: 1083
  after_half_life: 1077
  after_pvalue: 265
  after_hurst: 147
  after_kpss: 3
  after_market_microstructure: 3
---
step: 3
period: 11/30-12/30
candidates_total: 3003
passed_pairs: 5
remaining_after_stage:
  after_low_correlation: 2346
  after_beta: 1627
  after_mean_crossings: 1627
  after_half_life: 1625
  after_pvalue: 188
  after_hurst: 72
  after_kpss: 5
  after_market_microstructure: 5
```

#### 20260115_quality_universe_200k_z1p0_exit0p15_blacklist
- Метрики (strategy_metrics.csv): total_pnl `128.38`, sharpe_ratio_abs `0.3090`, max_drawdown_abs `-14.77`, total_trades `135`, total_pairs_traded `14.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_quality_universe_200k_z1p0_exit0p15_blacklist/`.
- Худшая пара по PnL: `DOGEUSDC-ETHWUSDT` (`-5.66`).
- Статус: `rejected` (blacklist GMTUSDT/SANDUSDT не улучшил Sharpe).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 2775
passed_pairs: 8
remaining_after_stage:
  after_low_correlation: 2217
  after_beta: 1436
  after_mean_crossings: 1436
  after_half_life: 1420
  after_pvalue: 358
  after_hurst: 176
  after_kpss: 8
  after_market_microstructure: 8
---
step: 2
period: 10/31-11/30
candidates_total: 2775
passed_pairs: 3
remaining_after_stage:
  after_low_correlation: 1462
  after_beta: 998
  after_mean_crossings: 998
  after_half_life: 992
  after_pvalue: 239
  after_hurst: 133
  after_kpss: 3
  after_market_microstructure: 3
---
step: 3
period: 11/30-12/30
candidates_total: 2850
passed_pairs: 3
remaining_after_stage:
  after_low_correlation: 2211
  after_beta: 1515
  after_mean_crossings: 1515
  after_half_life: 1513
  after_pvalue: 171
  after_hurst: 64
  after_kpss: 3
  after_market_microstructure: 3
```

#### 20260115_quality_universe_200k_z0p8_corr0p45
- Метрики (strategy_metrics.csv): total_pnl `181.00`, sharpe_ratio_abs `0.4105`, max_drawdown_abs `-15.32`, total_trades `457`, total_pairs_traded `18.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_quality_universe_200k_z0p8_corr0p45/`.
- Худшая пара по PnL: `GMTUSDT-SANDUSDT` (`-3.79`).
- Статус: `candidate` (лучший Sharpe среди quality-universe на 2023 Q4).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 2926
passed_pairs: 10
remaining_after_stage:
  after_low_correlation: 2355
  after_beta: 1544
  after_mean_crossings: 1544
  after_half_life: 1527
  after_pvalue: 381
  after_hurst: 186
  after_kpss: 10
  after_market_microstructure: 10
---
step: 2
period: 10/31-11/30
candidates_total: 2926
passed_pairs: 3
remaining_after_stage:
  after_low_correlation: 1574
  after_beta: 1083
  after_mean_crossings: 1083
  after_half_life: 1077
  after_pvalue: 265
  after_hurst: 147
  after_kpss: 3
  after_market_microstructure: 3
---
step: 3
period: 11/30-12/30
candidates_total: 3003
passed_pairs: 5
remaining_after_stage:
  after_low_correlation: 2346
  after_beta: 1627
  after_mean_crossings: 1627
  after_half_life: 1625
  after_pvalue: 188
  after_hurst: 72
  after_kpss: 5
  after_market_microstructure: 5
```

#### 20260115_quality_universe_200k_z0p8_corr0p45_signal_strict
- Метрики (strategy_metrics.csv): total_pnl `181.00`, sharpe_ratio_abs `0.4105`, max_drawdown_abs `-15.32`, total_trades `457`, total_pairs_traded `18.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_quality_universe_200k_z0p8_corr0p45_signal_strict/`.
- Худшая пара по PnL: `GMTUSDT-SANDUSDT` (`-3.79`).
- Статус: `rejected` (строгие сигналы не улучшили Sharpe, метрики совпали с базой).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 2926
passed_pairs: 10
remaining_after_stage:
  after_low_correlation: 2355
  after_beta: 1544
  after_mean_crossings: 1544
  after_half_life: 1527
  after_pvalue: 381
  after_hurst: 186
  after_kpss: 10
  after_market_microstructure: 10
---
step: 2
period: 10/31-11/30
candidates_total: 2926
passed_pairs: 3
remaining_after_stage:
  after_low_correlation: 1574
  after_beta: 1083
  after_mean_crossings: 1083
  after_half_life: 1077
  after_pvalue: 265
  after_hurst: 147
  after_kpss: 3
  after_market_microstructure: 3
---
step: 3
period: 11/30-12/30
candidates_total: 3003
passed_pairs: 5
remaining_after_stage:
  after_low_correlation: 2346
  after_beta: 1627
  after_mean_crossings: 1627
  after_half_life: 1625
  after_pvalue: 188
  after_hurst: 72
  after_kpss: 5
  after_market_microstructure: 5
```

#### 20260115_quality_universe_200k_z0p8_corr0p45_tradeable
- Метрики (strategy_metrics.csv): total_pnl `153.13`, sharpe_ratio_abs `0.3681`, max_drawdown_abs `-15.06`, total_trades `278`, total_pairs_traded `12.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_quality_universe_200k_z0p8_corr0p45_tradeable/`.
- Худшая пара по PnL: `ETHUSDC-ETHUSDT` (`-0.91`).
- Статус: `rejected` (tradeability фильтры и denylist сократили пары и ухудшили Sharpe).
- Примечание: curated denylist `coint4/configs/quality_runs_20260115/denylist_symbols_20260115.yaml`, tradeability filter включен (min_volume=1m, days_live=30, bid_ask<=0.2, funding<=0.02).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 2628
passed_pairs: 6
remaining_after_stage:
  after_low_correlation: 2083
  after_beta: 1334
  after_mean_crossings: 1334
  after_half_life: 1318
  after_pvalue: 323
  after_hurst: 160
  after_kpss: 6
  after_market_microstructure: 6
---
step: 2
period: 10/31-11/30
candidates_total: 2628
passed_pairs: 3
remaining_after_stage:
  after_low_correlation: 1362
  after_beta: 924
  after_mean_crossings: 924
  after_half_life: 918
  after_pvalue: 215
  after_hurst: 118
  after_kpss: 3
  after_market_microstructure: 3
---
step: 3
period: 11/30-12/30
candidates_total: 2701
passed_pairs: 3
remaining_after_stage:
  after_low_correlation: 2081
  after_beta: 1415
  after_mean_crossings: 1415
  after_half_life: 1414
  after_pvalue: 166
  after_hurst: 63
  after_kpss: 3
  after_market_microstructure: 3
```

#### 20260115_quality_universe_200k_z0p8_corr0p5_hl0p05-45
- Метрики (strategy_metrics.csv): total_pnl `81.72`, sharpe_ratio_abs `0.4000`, max_drawdown_abs `-13.09`, total_trades `436`, total_pairs_traded `16.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_quality_universe_200k_z0p8_corr0p5_hl0p05-45/`.
- Худшая пара по PnL: `GMTUSDT-SANDUSDT` (`-3.79`).
- Статус: `rejected` (Sharpe ниже базового 200k z0p8/corr0.45, PnL заметно ниже).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 2926
passed_pairs: 9
remaining_after_stage:
  after_low_correlation: 2280
  after_beta: 1487
  after_mean_crossings: 1487
  after_half_life: 1466
  after_pvalue: 365
  after_hurst: 181
  after_kpss: 9
  after_market_microstructure: 9
---
step: 2
period: 10/31-11/30
candidates_total: 2926
passed_pairs: 2
remaining_after_stage:
  after_low_correlation: 1462
  after_beta: 1006
  after_mean_crossings: 1006
  after_half_life: 997
  after_pvalue: 253
  after_hurst: 141
  after_kpss: 2
  after_market_microstructure: 2
---
step: 3
period: 11/30-12/30
candidates_total: 3003
passed_pairs: 5
remaining_after_stage:
  after_low_correlation: 2301
  after_beta: 1612
  after_mean_crossings: 1612
  after_half_life: 1607
  after_pvalue: 185
  after_hurst: 72
  after_kpss: 5
  after_market_microstructure: 5
```

#### 20260115_quality_universe_200k_z0p9_exit0p1
- Метрики (strategy_metrics.csv): total_pnl `162.56`, sharpe_ratio_abs `0.3832`, max_drawdown_abs `-12.28`, total_trades `278`, total_pairs_traded `18.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_quality_universe_200k_z0p9_exit0p1/`.
- Худшая пара по PnL: `FILUSDT-SANDUSDT` (`-2.66`).
- Статус: `rejected` (ниже базового z0p8/corr0.45 по Sharpe).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 2926
passed_pairs: 10
remaining_after_stage:
  after_low_correlation: 2355
  after_beta: 1544
  after_mean_crossings: 1544
  after_half_life: 1527
  after_pvalue: 381
  after_hurst: 186
  after_kpss: 10
  after_market_microstructure: 10
---
step: 2
period: 10/31-11/30
candidates_total: 2926
passed_pairs: 3
remaining_after_stage:
  after_low_correlation: 1574
  after_beta: 1083
  after_mean_crossings: 1083
  after_half_life: 1077
  after_pvalue: 265
  after_hurst: 147
  after_kpss: 3
  after_market_microstructure: 3
---
step: 3
period: 11/30-12/30
candidates_total: 3003
passed_pairs: 5
remaining_after_stage:
  after_low_correlation: 2346
  after_beta: 1627
  after_mean_crossings: 1627
  after_half_life: 1625
  after_pvalue: 188
  after_hurst: 72
  after_kpss: 5
  after_market_microstructure: 5
```

#### 20260115_quality_universe_200k_z1p0_exit0p1
- Метрики (strategy_metrics.csv): total_pnl `141.80`, sharpe_ratio_abs `0.3379`, max_drawdown_abs `-19.96`, total_trades `177`, total_pairs_traded `18.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_quality_universe_200k_z1p0_exit0p1/`.
- Худшая пара по PnL: `DOGEUSDC-ETHWUSDT` (`-5.66`).
- Статус: `rejected` (не лучше z1p0/exit0p15, Sharpe ниже базового z0p8).

Сводка фильтрации пар (из filter_reasons_*.csv, Q4 2023):
```yaml
step: 1
period: 10/01-10/31
candidates_total: 2926
passed_pairs: 10
remaining_after_stage:
  after_low_correlation: 2355
  after_beta: 1544
  after_mean_crossings: 1544
  after_half_life: 1527
  after_pvalue: 381
  after_hurst: 186
  after_kpss: 10
  after_market_microstructure: 10
---
step: 2
period: 10/31-11/30
candidates_total: 2926
passed_pairs: 3
remaining_after_stage:
  after_low_correlation: 1574
  after_beta: 1083
  after_mean_crossings: 1083
  after_half_life: 1077
  after_pvalue: 265
  after_hurst: 147
  after_kpss: 3
  after_market_microstructure: 3
---
step: 3
period: 11/30-12/30
candidates_total: 3003
passed_pairs: 5
remaining_after_stage:
  after_low_correlation: 2346
  after_beta: 1627
  after_mean_crossings: 1627
  after_half_life: 1625
  after_pvalue: 188
  after_hurst: 72
  after_kpss: 5
  after_market_microstructure: 5
```

## Итоги на текущий момент
- Лучший по Sharpe (строгий pv): `pv=0.05, kpss=0.05, hurst=0.55, corr=0.50, hl=0.01-60` (Sharpe `0.3776`, PnL `108.97`, DD `-29.67`).
- Лучший по Sharpe среди широкой сетки: `pv=0.40, kpss=0.05, hurst=0.70, corr=0.40` (Sharpe `0.2965`, PnL `205.20`, DD `-98.16`).
- Лучший по PnL среди широкой сетки: `pv=0.40, kpss=0.03, hurst=0.65, corr=0.40` (Sharpe `0.2515`, PnL `244.29`, DD `-133.90`).
- Sharpe target (z1p2/z1p4): отрицательный Sharpe → отклонены, требуется смягчить сигналы или изменить окно.
- Quality universe: z0p8 Sharpe `0.3724` (PnL `153.62`), z1p2 Sharpe `0.2266` → Sharpe>1 не достигнут, требуется донастройка фильтров/сигналов.
- Quality universe 200k: z0p8/corr0.45 Sharpe `0.4105` (PnL `181.00`), signal_strict Sharpe `0.4105` (без изменений), z0p9/exit0p1 Sharpe `0.3832`, corr0.5/hl0.05-45 Sharpe `0.4000`, tradeable+denylist Sharpe `0.3681`, z1p0/exit0p1 Sharpe `0.3379`, blacklist Sharpe `0.3090` → лучший кандидат в рамках quality-universe, требуется иной срез tradeability/сигналов.
- Quality universe 500k: z0p8/corr0.45 Sharpe `0.3551` (PnL `145.72`) → ниже 200k/250k, пар меньше.
