# Журнал прогонов оптимизации (2026-01-17)

Назначение: smoke WFA для проверки логирования команд.

NOTE: Значения `sharpe_ratio_abs` в записях до фикса annualization (2026-01-18) являются raw (15m с `annualizing_factor=365`) и занижены примерно в √96 раз. Для актуальных значений используйте `coint4/artifacts/wfa/aggregate/rollup/run_index.*`.

## Статусы
- `active` — идет выполнение.
- `candidate` — выбран для валидации.
- `rejected` — отклонен по результатам валидации.
- `aborted` — прерван вручную/по ошибке.
- `legacy/archived` — устаревший или остановленный прогон.

## Обновления (2026-01-17)

### WFA очередь (next5_fast, signal/risk sweeps)
- Очередь: `coint4/artifacts/wfa/aggregate/20260117_next5_fast/run_queue_next5_fast.csv`.
- Запуск: `COINT_FILTER_BACKEND=threads ./run_wfa_fullcpu.sh <config> <results_dir>` по одному прогону; очередь используется как список и для ручного статуса (1 running, остальные planned).
- Параллельность: `1` (последовательно), CPU загружается через `backtest.n_jobs=-1` в конфиге.
- Конфиги: `coint4/configs/_tmp_fast_next10/*.yaml` (обновлено `backtest.n_jobs: -1`).
- Артефакты: `coint4/artifacts/wfa/runs/20260117_next5_fast/`.
- Логи: `coint4/artifacts/wfa/runs/20260117_next5_fast/<run>/run.log`.
- Нагрузка CPU (snapshot): coint2 ~130-160% CPU при 8 vCPU (системно ~15-18% user, load avg ~1.2) - есть запас для ускорения внутри одного прогона.
- Прогон 1: `configs/_tmp_fast_next10/signal_sweep_20260116_z0p75_exit0p04_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260116_z0p75_exit0p04_ssd25000` (done in `943.68s`, end `2026-01-17T14:01:27Z`).
- Метрики (strategy_metrics.csv): total_pnl `703.04`, sharpe_ratio_abs `0.4280`, max_drawdown_abs `-256.19`, total_trades `5595`, total_pairs_traded `197`, total_costs `0.0`, win_rate `0.6264`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260117_135140.csv`, `coint4/results/filter_reasons_20260117_135600.csv`, `coint4/results/filter_reasons_20260117_140117.csv`.
- Сводка причин отсева (по категориям, rows):
```yaml
step_1:
  total_rows: 24906
  pvalue: 7829
  beta_out_of_range: 6518
  low_correlation: 6063
  hurst_too_high: 2375
  kpss: 2049
  half_life: 72
step_2:
  total_rows: 24954
  low_correlation: 12320
  pvalue: 5447
  beta_out_of_range: 4185
  kpss: 1814
  hurst_too_high: 1140
  half_life: 48
step_3:
  total_rows: 24943
  pvalue: 11764
  beta_out_of_range: 6688
  low_correlation: 3384
  hurst_too_high: 1840
  kpss: 1212
  half_life: 55
```
- Прогон 2: `configs/_tmp_fast_next10/signal_sweep_20260116_z0p75_exit0p08_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260116_z0p75_exit0p08_ssd25000` (status `completed`).
- Метрики (strategy_metrics.csv): total_pnl `707.41`, sharpe_ratio_abs `0.4315`, max_drawdown_abs `-256.19`, total_trades `5598`, total_pairs_traded `197`, win_rate `0.6264`.
- Прогон 3: `configs/_tmp_fast_next10/signal_sweep_20260116_z0p8_exit0p06_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260116_z0p8_exit0p06_ssd25000` (done `2026-01-17T16:27:43Z`).
- Метрики (strategy_metrics.csv): total_pnl `771.63`, sharpe_ratio_abs `0.5860`, max_drawdown_abs `-146.72`, total_trades `4334`, total_pairs_traded `197`, win_rate `0.6813`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260117_161818.csv`.
- Прогон 4: `configs/_tmp_fast_next10/signal_sweep_20260116_z0p85_exit0p06_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260116_z0p85_exit0p06_ssd25000` (done `2026-01-17T18:05:14Z`).
- Метрики (strategy_metrics.csv): total_pnl `815.67`, sharpe_ratio_abs `0.6345`, max_drawdown_abs `-132.02`, total_trades `3406`, total_pairs_traded `197`, win_rate `0.6593`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260117_175527.csv`.
- Прогон 5: `configs/_tmp_fast_next10/signal_sweep_20260116_z0p85_exit0p08_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260116_z0p85_exit0p08_ssd25000` (done `2026-01-17T18:40:49Z`).
- Метрики (strategy_metrics.csv): total_pnl `821.12`, sharpe_ratio_abs `0.6410`, max_drawdown_abs `-128.32`, total_trades `3406`, total_pairs_traded `197`, win_rate `0.6593`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260117_183103.csv`.
- Прогон 6: `configs/_tmp_fast_next10/risk_sweep_20260116_stop2p5_time3p5.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/risk_sweep_20260116_stop2p5_time3p5` (parallel, done `2026-01-17T18:40:49Z`).
- Метрики (strategy_metrics.csv): total_pnl `771.63`, sharpe_ratio_abs `0.5860`, max_drawdown_abs `-146.72`, total_trades `4334`, total_pairs_traded `197`, win_rate `0.6813`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260117_183518.csv`.
- В очереди (planned): `signal_sweep_20260117_z0p9_exit0p08_ssd25000`, `signal_sweep_20260117_z0p9_exit0p1_ssd25000`, `signal_sweep_20260117_z0p95_exit0p08_ssd25000` (конфиги в `coint4/configs/_tmp_fast_next10/`).
- Примечание: попытка с `COINT_FILTER_BACKEND=processes` упала на `PermissionError: [Errno 13] Permission denied` (semlock); перезапуск на threads.
- Legacy/archived: запуск очереди через `COINT_FILTER_BACKEND=threads bash scripts/optimization/watch_wfa_queue.sh --queue artifacts/wfa/aggregate/20260117_next5_fast/run_queue_next5_fast.csv --parallel 1` остановлен, оставлен только для истории.
- Статус: `active` (последовательный запуск для анализа между прогонами).
- Прогон 7: `configs/_tmp_fast_next10/signal_sweep_20260117_z0p9_exit0p08_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260117_z0p9_exit0p08_ssd25000` (parallel, done `2026-01-17T19:21:24Z`).
- Метрики (strategy_metrics.csv): total_pnl `741.27`, sharpe_ratio_abs `0.5908`, max_drawdown_abs `-146.05`, total_trades `2675`, total_pairs_traded `197`, win_rate `0.6374`.
- Прогон 8: `configs/_tmp_fast_next10/signal_sweep_20260117_z0p9_exit0p1_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260117_z0p9_exit0p1_ssd25000` (parallel, done `2026-01-17T19:21:51Z`).
- Метрики (strategy_metrics.csv): total_pnl `742.94`, sharpe_ratio_abs `0.5933`, max_drawdown_abs `-146.08`, total_trades `2675`, total_pairs_traded `197`, win_rate `0.6374`.
- Прогон 9: `configs/_tmp_fast_next10/signal_sweep_20260117_z0p95_exit0p08_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260117_z0p95_exit0p08_ssd25000` (parallel, done `2026-01-17T19:21:25Z`).
- Метрики (strategy_metrics.csv): total_pnl `685.02`, sharpe_ratio_abs `0.5665`, max_drawdown_abs `-114.16`, total_trades `2157`, total_pairs_traded `197`, win_rate `0.6154`.
- Фильтрация пар (batch): `coint4/results/filter_reasons_20260117_192115.csv`, `coint4/results/filter_reasons_20260117_192116.csv`, `coint4/results/filter_reasons_20260117_192142.csv` (три файла из параллельного запуска).
- Прогон 10: `configs/_tmp_fast_next10/risk_sweep_20260117_stop2p5_time2p0_z0p85_exit0p08_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/risk_sweep_20260117_stop2p5_time2p0_z0p85_exit0p08_ssd25000` (parallel, done `2026-01-17T20:23:37Z`).
- Метрики (strategy_metrics.csv): total_pnl `821.12`, sharpe_ratio_abs `0.6410`, max_drawdown_abs `-128.32`, total_trades `3406`, total_pairs_traded `197`, win_rate `0.6593`.
- Прогон 11: `configs/_tmp_fast_next10/risk_sweep_20260117_stop3p0_time2p5_z0p85_exit0p08_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/risk_sweep_20260117_stop3p0_time2p5_z0p85_exit0p08_ssd25000` (parallel, done `2026-01-17T20:23:38Z`).
- Метрики (strategy_metrics.csv): total_pnl `821.12`, sharpe_ratio_abs `0.6410`, max_drawdown_abs `-128.32`, total_trades `3406`, total_pairs_traded `197`, win_rate `0.6593`.
- Прогон 12: `configs/_tmp_fast_next10/risk_sweep_20260117_stop3p5_time3p0_z0p85_exit0p08_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/risk_sweep_20260117_stop3p5_time3p0_z0p85_exit0p08_ssd25000` (parallel, done `2026-01-17T20:23:35Z`).
- Метрики (strategy_metrics.csv): total_pnl `821.12`, sharpe_ratio_abs `0.6410`, max_drawdown_abs `-128.32`, total_trades `3406`, total_pairs_traded `197`, win_rate `0.6593`.
- Фильтрация пар (batch): `coint4/results/filter_reasons_20260117_201752.csv`, `coint4/results/filter_reasons_20260117_201753.csv`, `coint4/results/filter_reasons_20260117_201754.csv` (три файла из параллельного запуска).
- Примечание: метрики идентичны для всех трех risk вариантов — вероятно, риск-параметры не активируются в этих данных/режимах.
- Прогон 13: `configs/_tmp_fast_next10/signal_sweep_20260117_z0p85_exit0p06_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260117_z0p85_exit0p06_ssd25000` (parallel, done `2026-01-17T20:43:08Z`).
- Метрики (strategy_metrics.csv): total_pnl `815.67`, sharpe_ratio_abs `0.6345`, max_drawdown_abs `-132.02`, total_trades `3406`, total_pairs_traded `197`, win_rate `0.6593`.
- Прогон 14: `configs/_tmp_fast_next10/signal_sweep_20260117_z0p85_exit0p1_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260117_z0p85_exit0p1_ssd25000` (parallel, done `2026-01-17T20:43:10Z`).
- Метрики (strategy_metrics.csv): total_pnl `821.86`, sharpe_ratio_abs `0.6434`, max_drawdown_abs `-124.82`, total_trades `3406`, total_pairs_traded `197`, win_rate `0.6484`.
- Прогон 15: `configs/_tmp_fast_next10/signal_sweep_20260117_z0p8_exit0p06_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260117_z0p8_exit0p06_ssd25000` (parallel, done `2026-01-17T20:43:05Z`).
- Метрики (strategy_metrics.csv): total_pnl `771.63`, sharpe_ratio_abs `0.5860`, max_drawdown_abs `-146.72`, total_trades `4334`, total_pairs_traded `197`, win_rate `0.6813`.
- Фильтрация пар (batch): `coint4/results/filter_reasons_20260117_204254.csv`, `coint4/results/filter_reasons_20260117_204258.csv`, `coint4/results/filter_reasons_20260117_204300.csv` (три файла из параллельного запуска).
- Прогон 16: `configs/_tmp_fast_next10/signal_sweep_20260117_z0p85_exit0p12_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260117_z0p85_exit0p12_ssd25000` (parallel, done `2026-01-17T21:04:44Z`).
- Метрики (strategy_metrics.csv): total_pnl `855.78`, sharpe_ratio_abs `0.6789`, max_drawdown_abs `-95.32`, total_trades `3406`, total_pairs_traded `197`, win_rate `0.6484`.
- Прогон 17: `configs/_tmp_fast_next10/signal_sweep_20260117_z0p85_exit0p09_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260117_z0p85_exit0p09_ssd25000` (parallel, done `2026-01-17T21:04:43Z`).
- Метрики (strategy_metrics.csv): total_pnl `822.87`, sharpe_ratio_abs `0.6441`, max_drawdown_abs `-124.82`, total_trades `3406`, total_pairs_traded `197`, win_rate `0.6593`.
- Прогон 18: `configs/_tmp_fast_next10/signal_sweep_20260117_z0p9_exit0p1_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260117_z0p9_exit0p1_ssd25000` (parallel, done `2026-01-17T21:04:50Z`).
- Метрики (strategy_metrics.csv): total_pnl `742.94`, sharpe_ratio_abs `0.5933`, max_drawdown_abs `-146.08`, total_trades `2675`, total_pairs_traded `197`, win_rate `0.6374`.
- Прогон 19: `configs/_tmp_fast_next10/signal_sweep_20260117_z0p8_exit0p1_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260117_z0p8_exit0p1_ssd25000` (parallel, done `2026-01-17T21:04:50Z`).
- Метрики (strategy_metrics.csv): total_pnl `780.90`, sharpe_ratio_abs `0.5965`, max_drawdown_abs `-139.51`, total_trades `4334`, total_pairs_traded `197`, win_rate `0.6703`.
- Фильтрация пар (batch): `coint4/results/filter_reasons_20260117_210407.csv`, `coint4/results/filter_reasons_20260117_210408.csv`, `coint4/results/filter_reasons_20260117_210414.csv`, `coint4/results/filter_reasons_20260117_210440.csv` (четыре файла из параллельного запуска).
- Прогон 20: `configs/_tmp_fast_next10/pair_sweep_20260117_corr0p6_z0p85_exit0p12_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_corr0p6_z0p85_exit0p12_ssd25000` (parallel, done `2026-01-17T21:29:08Z`).
- Метрики (strategy_metrics.csv): total_pnl `666.32`, sharpe_ratio_abs `0.7452`, max_drawdown_abs `-95.32`, total_trades `3264`, total_pairs_traded `188`, win_rate `0.6484`.
- Прогон 21: `configs/_tmp_fast_next10/pair_sweep_20260117_pv0p03_top800_z0p85_exit0p12_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_pv0p03_top800_z0p85_exit0p12_ssd25000` (parallel, done `2026-01-17T21:29:09Z`).
- Метрики (strategy_metrics.csv): total_pnl `789.56`, sharpe_ratio_abs `0.6637`, max_drawdown_abs `-98.47`, total_trades `3118`, total_pairs_traded `179`, win_rate `0.6703`.
- Прогон 22: `configs/_tmp_fast_next10/pair_sweep_20260117_hurst0p52_z0p85_exit0p12_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_hurst0p52_z0p85_exit0p12_ssd25000` (parallel, done `2026-01-17T21:29:13Z`).
- Метрики (strategy_metrics.csv): total_pnl `744.46`, sharpe_ratio_abs `0.6037`, max_drawdown_abs `-90.68`, total_trades `3040`, total_pairs_traded `177`, win_rate `0.6703`.
- Прогон 23: `configs/_tmp_fast_next10/pair_sweep_20260117_ssd15000_z0p85_exit0p12_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_ssd15000_z0p85_exit0p12_ssd25000` (parallel, done `2026-01-17T21:29:08Z`).
- Метрики (strategy_metrics.csv): total_pnl `404.72`, sharpe_ratio_abs `0.6275`, max_drawdown_abs `-44.95`, total_trades `2099`, total_pairs_traded `120`, win_rate `0.6703`.
- Фильтрация пар (batch): `coint4/results/filter_reasons_20260117_212748.csv`, `coint4/results/filter_reasons_20260117_212804.csv`, `coint4/results/filter_reasons_20260117_212904.csv`, `coint4/results/filter_reasons_20260117_212339.csv` (четыре файла из параллельного запуска).
- Прогон 24: `configs/_tmp_fast_next10/pair_sweep_20260117_corr0p6_pv0p03_top800_z0p85_exit0p12_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_corr0p6_pv0p03_top800_z0p85_exit0p12_ssd25000` (parallel, done `2026-01-17T21:50:29Z`).
- Метрики (strategy_metrics.csv): total_pnl `621.19`, sharpe_ratio_abs `0.7190`, max_drawdown_abs `-91.63`, total_trades `3031`, total_pairs_traded `173`, win_rate `0.6703`.
- Прогон 25: `configs/_tmp_fast_next10/pair_sweep_20260117_corr0p6_hurst0p52_z0p85_exit0p12_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_corr0p6_hurst0p52_z0p85_exit0p12_ssd25000` (parallel, done `2026-01-17T21:50:34Z`).
- Метрики (strategy_metrics.csv): total_pnl `555.34`, sharpe_ratio_abs `0.6492`, max_drawdown_abs `-90.68`, total_trades `2905`, total_pairs_traded `169`, win_rate `0.6703`.
- Прогон 26: `configs/_tmp_fast_next10/pair_sweep_20260117_pv0p03_top800_kpss0p03_z0p85_exit0p12_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_pv0p03_top800_kpss0p03_z0p85_exit0p12_ssd25000` (parallel, done `2026-01-17T21:50:27Z`).
- Метрики (strategy_metrics.csv): total_pnl `946.25`, sharpe_ratio_abs `0.5384`, max_drawdown_abs `-193.91`, total_trades `4743`, total_pairs_traded `279`, win_rate `0.6484`.
- Прогон 27: `configs/_tmp_fast_next10/pair_sweep_20260117_corr0p55_z0p85_exit0p12_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_corr0p55_z0p85_exit0p12_ssd25000` (parallel, done `2026-01-17T21:50:35Z`).
- Метрики (strategy_metrics.csv): total_pnl `666.32`, sharpe_ratio_abs `0.7452`, max_drawdown_abs `-95.32`, total_trades `3264`, total_pairs_traded `188`, win_rate `0.6484`.
- Фильтрация пар (batch): `coint4/results/filter_reasons_20260117_214946.csv`, `coint4/results/filter_reasons_20260117_214948.csv`, `coint4/results/filter_reasons_20260117_215025.csv`, `coint4/results/filter_reasons_20260117_214838.csv` (четыре файла из параллельного запуска).
- Прогон 28: `configs/_tmp_fast_next10/pair_sweep_20260117_corr0p7_pv0p02_top500_z0p85_exit0p12_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_corr0p7_pv0p02_top500_z0p85_exit0p12_ssd25000` (parallel, done `2026-01-17T22:20:59Z`).
- Метрики (strategy_metrics.csv): total_pnl `456.81`, sharpe_ratio_abs `0.6618`, max_drawdown_abs `-78.36`, total_trades `2583`, total_pairs_traded `143`, win_rate `0.6593`.
- Прогон 29: `configs/_tmp_fast_next10/pair_sweep_20260117_corr0p65_hurst0p5_kpss0p03_z0p85_exit0p12_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_corr0p65_hurst0p5_kpss0p03_z0p85_exit0p12_ssd25000` (parallel, done `2026-01-17T22:20:57Z`).
- Метрики (strategy_metrics.csv): total_pnl `620.39`, sharpe_ratio_abs `0.5028`, max_drawdown_abs `-126.15`, total_trades `3412`, total_pairs_traded `209`, win_rate `0.6264`.
- Прогон 30: `configs/_tmp_fast_next10/pair_sweep_20260117_ssd8000_pv0p03_top600_z0p85_exit0p12_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_ssd8000_pv0p03_top600_z0p85_exit0p12_ssd25000` (parallel, done `2026-01-17T22:20:56Z`).
- Метрики (strategy_metrics.csv): total_pnl `247.31`, sharpe_ratio_abs `0.4533`, max_drawdown_abs `-37.73`, total_trades `1175`, total_pairs_traded `63`, win_rate `0.5604`.
- Прогон 31: `configs/_tmp_fast_next10/pair_sweep_20260117_hl0p1_30_corr0p6_z0p85_exit0p12_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_hl0p1_30_corr0p6_z0p85_exit0p12_ssd25000` (parallel, done `2026-01-17T22:21:05Z`).
- Метрики (strategy_metrics.csv): total_pnl `547.88`, sharpe_ratio_abs `0.6830`, max_drawdown_abs `-95.32`, total_trades `3104`, total_pairs_traded `180`, win_rate `0.6484`.
- Фильтрация пар (batch): `coint4/results/filter_reasons_20260117_221723.csv`, `coint4/results/filter_reasons_20260117_222005.csv`, `coint4/results/filter_reasons_20260117_222056.csv`, `coint4/results/filter_reasons_20260117_221549.csv` (четыре файла из параллельного запуска).
- Прогон 32: `configs/_tmp_fast_next10/pair_sweep_20260117_corr0p75_pv0p015_top300_hurst0p48_kpss0p03_z0p85_exit0p12.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_corr0p75_pv0p015_top300_hurst0p48_kpss0p03_z0p85_exit0p12` (parallel, done `2026-01-17T22:45:49Z`).
- Метрики (strategy_metrics.csv): total_pnl `95.37`, sharpe_ratio_abs `0.1718`, max_drawdown_abs `-105.85`, total_trades `955`, total_pairs_traded `58`, win_rate `0.6154`.
- Прогон 33: `configs/_tmp_fast_next10/pair_sweep_20260117_corr0p7_pv0p01_top200_kpss0p02_hl0p2_20_z0p85_exit0p12.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_corr0p7_pv0p01_top200_kpss0p02_hl0p2_20_z0p85_exit0p12` (parallel, done `2026-01-17T22:45:51Z`).
- Метрики (strategy_metrics.csv): total_pnl `623.38`, sharpe_ratio_abs `0.4579`, max_drawdown_abs `-230.50`, total_trades `4024`, total_pairs_traded `237`, win_rate `0.6264`.
- Прогон 34: `configs/_tmp_fast_next10/pair_sweep_20260117_ssd4000_corr0p65_pv0p02_top400_z0p85_exit0p12.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_ssd4000_corr0p65_pv0p02_top400_z0p85_exit0p12` (parallel, done `2026-01-17T22:45:50Z`).
- Метрики (strategy_metrics.csv): total_pnl `61.89`, sharpe_ratio_abs `0.2783`, max_drawdown_abs `-25.44`, total_trades `621`, total_pairs_traded `31`, win_rate `0.5604`.
- Прогон 35: `configs/_tmp_fast_next10/pair_sweep_20260117_corr0p7_hurst0p5_kpss0p02_cross1_z0p85_exit0p12.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/pair_sweep_20260117_corr0p7_hurst0p5_kpss0p02_cross1_z0p85_exit0p12` (parallel, done `2026-01-17T22:45:57Z`).
- Метрики (strategy_metrics.csv): total_pnl `747.15`, sharpe_ratio_abs `0.5444`, max_drawdown_abs `-130.58`, total_trades `4615`, total_pairs_traded `276`, win_rate `0.7253`.
- Фильтрация пар (batch): `coint4/results/filter_reasons_20260117_224102.csv`, `coint4/results/filter_reasons_20260117_224233.csv`, `coint4/results/filter_reasons_20260117_224245.csv`, `coint4/results/filter_reasons_20260117_224547.csv` (четыре файла из параллельного запуска).

### Smoke WFA (command logging verification)
- Конфиг: `coint4/configs/main_2024_smoke.yaml` (max_steps=1, n_jobs=-1).
- Команда: `COINT_FILTER_BACKEND=processes ./run_wfa_fullcpu.sh configs/main_2024_smoke.yaml artifacts/wfa/runs/logging_smoke_20260117_072821`.
- Артефакты: `coint4/artifacts/wfa/runs/logging_smoke_20260117_072821/`.
- Командный лог: `coint4/artifacts/wfa/runs/logging_smoke_20260117_072821/run.commands.log`.
- Фильтрация пар: `coint4/results/filter_reasons_20260117_072907.csv`.
- Метрики (strategy_metrics.csv): total_pnl `-52.40`, sharpe_ratio_abs `-0.6205`, max_drawdown_abs `-66.69`, total_trades `394`, total_pairs_traded `108`, total_costs `0.0`.
- Статус: `completed`.

Сводка фильтрации пар (из stdout):
```yaml
step: 1
candidates_total: 2000
passed_pairs: 108
remaining_after_stage:
  after_low_correlation: 1302
  after_beta: 920
  after_mean_crossings: 920
  after_half_life: 427
  after_pvalue: 162
  after_hurst: 108
  after_kpss: 108
  after_market_microstructure: 108
```

### WFA очередь (next4_fast, предложенные прогоны)
- Очередь: `coint4/artifacts/wfa/aggregate/20260118_next4_fast/run_queue_next4_fast.csv`.
- Цель: проверить устойчивость лидера (z0p85/exit0p12) на соседних z-порогах и ужесточить фильтры для снижения DD.
- Параллельность: `1` (каждый прогон использует `backtest.n_jobs: -1` для полной загрузки CPU).
- Конфиги:
  - `coint4/configs/_tmp_fast_next4_20260118/signal_sweep_20260118_z0p9_exit0p12_ssd25000.yaml`
  - `coint4/configs/_tmp_fast_next4_20260118/signal_sweep_20260118_z0p8_exit0p12_ssd25000.yaml`
  - `coint4/configs/_tmp_fast_next4_20260118/pair_sweep_20260118_corr0p5_z0p85_exit0p12_ssd25000.yaml`
  - `coint4/configs/_tmp_fast_next4_20260118/pair_sweep_20260118_pv0p03_top800_kpss0p03_corr0p6_z0p85_exit0p12_ssd25000.yaml`
- Статус: `completed`.
- Прогон 1: `signal_sweep_20260118_z0p9_exit0p12_ssd25000` → `coint4/artifacts/wfa/runs/20260118_next4_fast/signal_sweep_20260118_z0p9_exit0p12_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `776.81`, sharpe_ratio_abs `0.6296`, max_drawdown_abs `-113.19`, total_trades `2675`, total_pairs_traded `197`, win_rate `0.6374`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260118_104444.csv`, `coint4/results/filter_reasons_20260118_104903.csv`, `coint4/results/filter_reasons_20260118_105424.csv`.
- Сводка причин отсева (по категориям, rows):
```yaml
step_1:
  total_rows: 24906
  pvalue: 7829
  beta_out_of_range: 6518
  low_correlation: 6063
  hurst_too_high: 2375
  kpss: 2049
  half_life: 72
step_2:
  total_rows: 24954
  low_correlation: 12320
  pvalue: 5447
  beta_out_of_range: 4185
  kpss: 1814
  hurst_too_high: 1140
  half_life: 48
step_3:
  total_rows: 24943
  pvalue: 11764
  beta_out_of_range: 6688
  low_correlation: 3384
  hurst_too_high: 1840
  kpss: 1212
  half_life: 55
```
- Прогон 2: `signal_sweep_20260118_z0p8_exit0p12_ssd25000` → `coint4/artifacts/wfa/runs/20260118_next4_fast/signal_sweep_20260118_z0p8_exit0p12_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `814.56`, sharpe_ratio_abs `0.6300`, max_drawdown_abs `-110.02`, total_trades `4334`, total_pairs_traded `197`, win_rate `0.6813`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260118_110039.csv`, `coint4/results/filter_reasons_20260118_110506.csv`, `coint4/results/filter_reasons_20260118_111028.csv`.
- Сводка причин отсева: идентична прогону 1.
- Прогон 3: `pair_sweep_20260118_corr0p5_z0p85_exit0p12_ssd25000` → `coint4/artifacts/wfa/runs/20260118_next4_fast/pair_sweep_20260118_corr0p5_z0p85_exit0p12_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `855.78`, sharpe_ratio_abs `0.6789`, max_drawdown_abs `-95.32`, total_trades `3406`, total_pairs_traded `197`, win_rate `0.6484`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260118_111646.csv`, `coint4/results/filter_reasons_20260118_112105.csv`, `coint4/results/filter_reasons_20260118_112623.csv`.
- Сводка причин отсева: идентична прогону 1.
- Прогон 4: `pair_sweep_20260118_pv0p03_top800_kpss0p03_corr0p6_z0p85_exit0p12_ssd25000` → `coint4/artifacts/wfa/runs/20260118_next4_fast/pair_sweep_20260118_pv0p03_top800_kpss0p03_corr0p6_z0p85_exit0p12_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `776.62`, sharpe_ratio_abs `0.5265`, max_drawdown_abs `-193.91`, total_trades `4634`, total_pairs_traded `269`, win_rate `0.6484`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260118_113141.csv`, `coint4/results/filter_reasons_20260118_113505.csv`, `coint4/results/filter_reasons_20260118_113945.csv`.
- Сводка причин отсева (по категориям, rows):
```yaml
step_1:
  total_rows: 24877
  pvalue: 7818
  low_correlation: 7796
  beta_out_of_range: 5851
  hurst_too_high: 1838
  kpss: 1524
  half_life: 50
step_2:
  total_rows: 24939
  low_correlation: 15048
  pvalue: 4432
  beta_out_of_range: 3405
  kpss: 1206
  hurst_too_high: 811
  half_life: 37
step_3:
  total_rows: 24914
  pvalue: 11802
  beta_out_of_range: 6271
  low_correlation: 4493
  hurst_too_high: 1434
  kpss: 869
  half_life: 45
```
- Итог: лучшая комбинация в блоке — `pair_sweep_20260118_corr0p5_z0p85_exit0p12_ssd25000` (sharpe `0.6789`, DD `-95.32`), z0p8/z0p9 дают чуть меньший Sharpe и больший DD, pv0p03/top800/kpss0p03 ухудшил DD до `-193.91` при росте числа пар.
- Примечание: `pair_sweep_20260118_corr0p5_z0p85_exit0p12_ssd25000` повторяет метрики лидера z0p85/exit0p12; вероятно, ослабление `min_correlation` не сработало из-за `backtest.min_correlation_threshold: 0.6`.

### WFA очередь (leader_validation_20260118, проверка лидера)
- Очередь: `coint4/artifacts/wfa/aggregate/20260118_leader_validation/run_queue.csv`.
- Цель: валидация лидера (z0p85/exit0p12/ssd25000) и резерва (z0p8/exit0p12/ssd25000) на 5 шагах WFA с окном `2023-10-01` → `2024-04-30`.
- Параллельность: `1`.
- Конфиги:
  - `coint4/configs/leader_validation_20260118/leader_validate_20260118_z0p85_exit0p12_ssd25000.yaml`
  - `coint4/configs/leader_validation_20260118/leader_validate_20260118_z0p8_exit0p12_ssd25000.yaml`
- Статус: `completed`.
- Прогон 1: `leader_validate_20260118_z0p85_exit0p12_ssd25000` → `coint4/artifacts/wfa/runs/20260118_leader_validation/leader_validate_20260118_z0p85_exit0p12_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `1007.18`, sharpe_ratio_abs `0.5745`, max_drawdown_abs `-95.32`, total_trades `5137`, total_pairs_traded `293`, win_rate `0.6291`.
- Фильтрация пар (step 1-5): `coint4/results/filter_reasons_20260118_133749.csv`, `coint4/results/filter_reasons_20260118_134206.csv`, `coint4/results/filter_reasons_20260118_134718.csv`, `coint4/results/filter_reasons_20260118_135129.csv`, `coint4/results/filter_reasons_20260118_135500.csv`.
- Сводка причин отсева (по категориям, rows):
```yaml
step_1:
  total_rows: 24906
  pvalue: 7829
  beta_out_of_range: 6518
  low_correlation: 6063
  hurst_too_high: 2375
  kpss: 2049
  half_life: 72
step_2:
  total_rows: 24954
  pvalue: 5447
  beta_out_of_range: 4185
  low_correlation: 12320
  hurst_too_high: 1140
  kpss: 1814
  half_life: 48
step_3:
  total_rows: 24943
  pvalue: 11764
  beta_out_of_range: 6688
  low_correlation: 3384
  hurst_too_high: 1840
  kpss: 1212
  half_life: 55
step_4:
  total_rows: 24958
  pvalue: 9234
  beta_out_of_range: 5602
  low_correlation: 7860
  hurst_too_high: 1129
  kpss: 1080
  half_life: 53
step_5:
  total_rows: 24940
  pvalue: 4057
  beta_out_of_range: 2489
  low_correlation: 16093
  hurst_too_high: 1048
  kpss: 1237
  half_life: 16
```
- Прогон 2: `leader_validate_20260118_z0p8_exit0p12_ssd25000` → `coint4/artifacts/wfa/runs/20260118_leader_validation/leader_validate_20260118_z0p8_exit0p12_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `941.02`, sharpe_ratio_abs `0.5238`, max_drawdown_abs `-110.02`, total_trades `6573`, total_pairs_traded `293`, win_rate `0.6225`.
- Фильтрация пар (step 1-5): `coint4/results/filter_reasons_20260118_140114.csv`, `coint4/results/filter_reasons_20260118_140532.csv`, `coint4/results/filter_reasons_20260118_141049.csv`, `coint4/results/filter_reasons_20260118_141502.csv`, `coint4/results/filter_reasons_20260118_141838.csv`.
- Сводка причин отсева: идентична прогону 1 (step 1-5).
- Итог: лидер z0p85 сохраняет преимущество по Sharpe/DD, z0p8 даёт меньший Sharpe при большем числе сделок.

### WFA очередь (corr_ab_20260118, согласование корреляции)
- Очередь: `coint4/artifacts/wfa/aggregate/20260118_corr_ab/run_queue.csv`.
- Цель: сравнить согласованность `pair_selection.min_correlation` и `backtest.min_correlation_threshold` на уровнях 0.50 и 0.65 (3 шага WFA, короткий период).
- Параллельность: `1`.
- Конфиги:
  - `coint4/configs/corr_ab_20260118/corr_ab_20260118_corr0p50_thr0p50_z0p85_exit0p12_ssd25000.yaml`
  - `coint4/configs/corr_ab_20260118/corr_ab_20260118_corr0p65_thr0p65_z0p85_exit0p12_ssd25000.yaml`
- Статус: `completed`.
- Прогон 1: `corr_ab_20260118_corr0p50_thr0p50_z0p85_exit0p12_ssd25000` → `coint4/artifacts/wfa/runs/20260118_corr_ab/corr_ab_20260118_corr0p50_thr0p50_z0p85_exit0p12_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `855.78`, sharpe_ratio_abs `0.6789`, max_drawdown_abs `-95.32`, total_trades `3406`, total_pairs_traded `197`, win_rate `0.6484`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260118_144138.csv`, `coint4/results/filter_reasons_20260118_144556.csv`, `coint4/results/filter_reasons_20260118_145110.csv`.
- Сводка причин отсева (по категориям, rows):
```yaml
step_1:
  total_rows: 24906
  pvalue: 7829
  beta_out_of_range: 6518
  low_correlation: 6063
  hurst_too_high: 2375
  kpss: 2049
  half_life: 72
step_2:
  total_rows: 24954
  pvalue: 5447
  beta_out_of_range: 4185
  low_correlation: 12320
  hurst_too_high: 1140
  kpss: 1814
  half_life: 48
step_3:
  total_rows: 24943
  pvalue: 11764
  beta_out_of_range: 6688
  low_correlation: 3384
  hurst_too_high: 1840
  kpss: 1212
  half_life: 55
```
- Прогон 2: `corr_ab_20260118_corr0p65_thr0p65_z0p85_exit0p12_ssd25000` → `coint4/artifacts/wfa/runs/20260118_corr_ab/corr_ab_20260118_corr0p65_thr0p65_z0p85_exit0p12_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `651.02`, sharpe_ratio_abs `0.7323`, max_drawdown_abs `-88.17`, total_trades `3188`, total_pairs_traded `183`, win_rate `0.6484`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260118_145645.csv`, `coint4/results/filter_reasons_20260118_150000.csv`, `coint4/results/filter_reasons_20260118_150503.csv`.
- Сводка причин отсева (по категориям, rows):
```yaml
step_1:
  total_rows: 24910
  pvalue: 6576
  beta_out_of_range: 5465
  low_correlation: 8880
  hurst_too_high: 2093
  kpss: 1865
  half_life: 31
step_2:
  total_rows: 24964
  pvalue: 3329
  beta_out_of_range: 2921
  low_correlation: 16572
  hurst_too_high: 851
  kpss: 1256
  half_life: 35
step_3:
  total_rows: 24943
  pvalue: 10573
  beta_out_of_range: 5951
  low_correlation: 5422
  hurst_too_high: 1799
  kpss: 1162
  half_life: 36
```
- Итог: порог 0.65 снижает DD и повышает Sharpe, но уменьшает PnL и число пар/сделок; 0.50 даёт больший охват.

### WFA очередь (baseline_20260118, фиксация базы с учетом costs)
- Очередь: `coint4/artifacts/wfa/aggregate/20260118_baseline/run_queue.csv`.
- Цель: baseline 5-step WFA для z0p85/exit0p12/corr0p65/ssd25000 с фиксацией учёта `total_costs`.
- Параллельность: `1`.
- Конфиги:
  - `coint4/configs/baseline_20260118/baseline_20260118_z0p85_exit0p12_corr0p65_ssd25000.yaml`
- Статус: `completed`.
- Прогон: `baseline_20260118_z0p85_exit0p12_corr0p65_ssd25000` → `coint4/artifacts/wfa/runs/20260118_baseline/baseline_20260118_z0p85_exit0p12_corr0p65_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `763.24`, sharpe_ratio_abs `0.5875` (до фикса annualization), max_drawdown_abs `-88.17`, total_trades `4815`, total_pairs_traded `271`, win_rate `0.6093`, total_costs `189.73`.
- Метрики (rollup recomputed): sharpe_ratio_abs `5.7560` (см. `coint4/artifacts/wfa/aggregate/rollup/run_index.*`).

### WFA очередь (turnover_sweep_20260118, снижение churn)
- Очередь: `coint4/artifacts/wfa/aggregate/20260118_turnover_sweep/run_queue.csv`.
- Цель: снизить turnover через entry/exit + min_hold/cooldown (3 шага WFA).
- Параллельность: `1`.
- Конфиги:
  - `coint4/configs/turnover_sweep_20260118/turnover_sweep_20260118_entry0p95_exit0p08_hold120_cd120_corr0p65_ssd25000.yaml`
  - `coint4/configs/turnover_sweep_20260118/turnover_sweep_20260118_entry0p95_exit0p1_hold120_cd120_corr0p65_ssd25000.yaml`
  - `coint4/configs/turnover_sweep_20260118/turnover_sweep_20260118_entry1p05_exit0p08_hold120_cd120_corr0p65_ssd25000.yaml`
  - `coint4/configs/turnover_sweep_20260118/turnover_sweep_20260118_entry1p05_exit0p1_hold120_cd120_corr0p65_ssd25000.yaml`
  - `coint4/configs/turnover_sweep_20260118/turnover_sweep_20260118_entry1p15_exit0p08_hold120_cd120_corr0p65_ssd25000.yaml`
  - `coint4/configs/turnover_sweep_20260118/turnover_sweep_20260118_entry1p15_exit0p1_hold120_cd120_corr0p65_ssd25000.yaml`
- Статус: `completed`.
- Лучший прогон: `turnover_sweep_20260118_entry0p95_exit0p1_hold120_cd120_corr0p65_ssd25000` → `coint4/artifacts/wfa/runs/20260118_turnover_sweep/turnover_sweep_20260118_entry0p95_exit0p1_hold120_cd120_corr0p65_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `481.02`, sharpe_ratio_abs `0.5954` (до фикса annualization), max_drawdown_abs `-120.81`, total_trades `2020`, total_pairs_traded `183`, win_rate `0.6154`, total_costs `81.42`.
- Метрики (rollup recomputed): sharpe_ratio_abs `5.8340`.

### WFA очередь (quality_sweep_20260118, качество пар)
- Очередь: `coint4/artifacts/wfa/aggregate/20260118_quality_sweep/run_queue.csv`.
- Цель: усилить фильтры качества пар через corr и строгий пресет (3 шага WFA).
- Параллельность: `1`.
- Конфиги:
  - `coint4/configs/quality_sweep_20260118/quality_sweep_20260118_corr0p65_z0p85_exit0p12_ssd25000.yaml`
  - `coint4/configs/quality_sweep_20260118/quality_sweep_20260118_corr0p7_z0p85_exit0p12_ssd25000.yaml`
  - `coint4/configs/quality_sweep_20260118/quality_sweep_20260118_corr0p75_z0p85_exit0p12_ssd25000.yaml`
  - `coint4/configs/quality_sweep_20260118/quality_sweep_20260118_corr0p70_strict_z0p85_exit0p12_ssd25000.yaml`
- Статус: `completed`.
- Лучший прогон: `quality_sweep_20260118_corr0p65_z0p85_exit0p12_ssd25000` → `coint4/artifacts/wfa/runs/20260118_quality_sweep/quality_sweep_20260118_corr0p65_z0p85_exit0p12_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `651.02`, sharpe_ratio_abs `0.7323` (до фикса annualization), max_drawdown_abs `-88.17`, total_trades `3188`, total_pairs_traded `183`, win_rate `0.6484`, total_costs `134.39`.
- Метрики (rollup recomputed): sharpe_ratio_abs `7.1746`.

### WFA очередь (risk_sweep_20260118, сглаживание риска)
- Очередь: `coint4/artifacts/wfa/aggregate/20260118_risk_sweep/run_queue.csv`.
- Цель: проверить более консервативные risk/kelly/позиции (3 шага WFA).
- Параллельность: `1`.
- Конфиги:
  - `coint4/configs/risk_sweep_20260118/risk_sweep_20260118_risk0p012_pos12_margin0p45_kelly0p2_z0p85_exit0p12_corr0p65_ssd25000.yaml`
  - `coint4/configs/risk_sweep_20260118/risk_sweep_20260118_risk0p01_pos10_margin0p4_kelly0p15_z0p85_exit0p12_corr0p65_ssd25000.yaml`
  - `coint4/configs/risk_sweep_20260118/risk_sweep_20260118_risk0p008_pos8_margin0p35_kelly0p15_z0p85_exit0p12_corr0p65_ssd25000.yaml`
- Статус: `completed`.
- Прогон 1: `risk_sweep_20260118_risk0p008_pos8_margin0p35_kelly0p15_z0p85_exit0p12_corr0p65_ssd25000` → `coint4/artifacts/wfa/runs/20260118_risk_sweep/risk_sweep_20260118_risk0p008_pos8_margin0p35_kelly0p15_z0p85_exit0p12_corr0p65_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `650.27`, sharpe_ratio_abs `0.7341` (до фикса annualization), max_drawdown_abs `-88.17`, total_trades `3188`, total_pairs_traded `183`, win_rate `0.6484`, total_costs `134.39`.
- Метрики (rollup recomputed): sharpe_ratio_abs `7.1928`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260118_191017.csv`, `coint4/results/filter_reasons_20260118_191332.csv`, `coint4/results/filter_reasons_20260118_191828.csv`.
- Сводка причин отсева (по категориям, rows):
```yaml
step_1:
  total_rows: 24910
  pvalue: 6576
  beta_out_of_range: 5465
  low_correlation: 8880
  hurst_too_high: 2093
  kpss: 1865
  half_life: 31
step_2:
  total_rows: 24964
  pvalue: 3329
  beta_out_of_range: 2921
  low_correlation: 16572
  hurst_too_high: 851
  kpss: 1256
  half_life: 35
step_3:
  total_rows: 24943
  pvalue: 10573
  beta_out_of_range: 5951
  low_correlation: 5422
  hurst_too_high: 1799
  kpss: 1162
  half_life: 36
```
- Прогон 2: `risk_sweep_20260118_risk0p012_pos12_margin0p45_kelly0p2_z0p85_exit0p12_corr0p65_ssd25000` → `coint4/artifacts/wfa/runs/20260118_risk_sweep/risk_sweep_20260118_risk0p012_pos12_margin0p45_kelly0p2_z0p85_exit0p12_corr0p65_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `650.51`, sharpe_ratio_abs `0.7327` (до фикса annualization), max_drawdown_abs `-88.17`, total_trades `3188`, total_pairs_traded `183`, win_rate `0.6484`, total_costs `134.39`.
- Метрики (rollup recomputed): sharpe_ratio_abs `7.1791`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260118_192402.csv`, `coint4/results/filter_reasons_20260118_192718.csv`, `coint4/results/filter_reasons_20260118_193217.csv`.
- Прогон 3: `risk_sweep_20260118_risk0p01_pos10_margin0p4_kelly0p15_z0p85_exit0p12_corr0p65_ssd25000` → `coint4/artifacts/wfa/runs/20260118_risk_sweep/risk_sweep_20260118_risk0p01_pos10_margin0p4_kelly0p15_z0p85_exit0p12_corr0p65_ssd25000`.
- Метрики (strategy_metrics.csv): total_pnl `653.63`, sharpe_ratio_abs `0.7368` (до фикса annualization), max_drawdown_abs `-88.17`, total_trades `3188`, total_pairs_traded `183`, win_rate `0.6593`, total_costs `134.39`.
- Метрики (rollup recomputed): sharpe_ratio_abs `7.2187`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260118_193751.csv`, `coint4/results/filter_reasons_20260118_194107.csv`, `coint4/results/filter_reasons_20260118_194605.csv`.
- Примечание: сводка причин отсева для прогонов 2-3 идентична прогону 1 (одинаковые фильтры).
- Итог: метрики почти не изменились при изменении risk-параметров (возможно, портфельные ограничения слабо влияют в Numba-бэктесте).

### WFA очередь (shortlist_20260118, 5-step shortlist)
- Очередь: `coint4/artifacts/wfa/aggregate/20260118_shortlist/run_queue.csv`.
- Цель: 5-step WFA для shortlist (baseline corr0.65, corr0.70 strict, turnover best) с обновлённым annualization Sharpe.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиги:
  - `coint4/configs/shortlist_20260118/shortlist_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000.yaml`
  - `coint4/configs/shortlist_20260118/shortlist_20260118_corr0p7_z0p85_exit0p12_ssd25000.yaml`
  - `coint4/configs/shortlist_20260118/shortlist_20260118_entry0p95_exit0p1_hold120_cd120_corr0p65_ssd25000.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `shortlist_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000` → `coint4/artifacts/wfa/runs/20260118_shortlist/shortlist_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000`.
- Метрики (rollup recomputed): sharpe_ratio_abs `5.7560`, total_pnl `763.24`, max_drawdown_abs `-88.17`, total_trades `4815`, total_pairs_traded `271`, total_costs `189.73`, win_rate `0.6093`.
- Прогон 2: `shortlist_20260118_corr0p7_z0p85_exit0p12_ssd25000` → `coint4/artifacts/wfa/runs/20260118_shortlist/shortlist_20260118_corr0p7_z0p85_exit0p12_ssd25000`.
- Метрики (rollup recomputed): sharpe_ratio_abs `5.7302`, total_pnl `681.93`, max_drawdown_abs `-88.17`, total_trades `4649`, total_pairs_traded `258`, total_costs `179.86`, win_rate `0.6159`.
- Прогон 3: `shortlist_20260118_entry0p95_exit0p1_hold120_cd120_corr0p65_ssd25000` → `coint4/artifacts/wfa/runs/20260118_shortlist/shortlist_20260118_entry0p95_exit0p1_hold120_cd120_corr0p65_ssd25000`.
- Метрики (rollup recomputed): sharpe_ratio_abs `4.9631`, total_pnl `600.39`, max_drawdown_abs `-120.81`, total_trades `3003`, total_pairs_traded `271`, total_costs `115.87`, win_rate `0.5828`.
- Итог: baseline и corr0.7 дают почти одинаковый Sharpe > 5 с DD ~ -88; turnover-версия снижает Sharpe и увеличивает DD, но уменьшает сделки/издержки.

### WFA очередь (holdout_20260118, 5-step holdout)
- Очередь: `coint4/artifacts/wfa/aggregate/20260118_holdout/run_queue.csv`.
- Цель: holdout 2024-05-01 → 2024-12-31 для top-1/2 (baseline + corr0.7) без подбора.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиги:
  - `coint4/configs/holdout_20260118/holdout_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000.yaml`
  - `coint4/configs/holdout_20260118/holdout_20260118_corr0p7_z0p85_exit0p12_ssd25000.yaml`
- Статус: `completed` (запуск на 85.198.90.128).
- Прогон 1: `holdout_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000` → `coint4/artifacts/wfa/runs/20260118_holdout/holdout_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000`.
- Метрики (rollup recomputed): sharpe_ratio_abs `-3.4066`, total_pnl `-324.10`, max_drawdown_abs `-358.78`, total_trades `8641`, total_pairs_traded `501`, total_costs `256.47`, win_rate `0.4570`.
- Прогон 2: `holdout_20260118_corr0p7_z0p85_exit0p12_ssd25000` → `coint4/artifacts/wfa/runs/20260118_holdout/holdout_20260118_corr0p7_z0p85_exit0p12_ssd25000`.
- Метрики (rollup recomputed): sharpe_ratio_abs `-3.2651`, total_pnl `-306.85`, max_drawdown_abs `-346.38`, total_trades `8497`, total_pairs_traded `490`, total_costs `252.81`, win_rate `0.4570`.
- Итог: обе holdout-версии дают отрицательный Sharpe и PnL → нужна диагностика режима/фильтров.

### WFA очередь (stress_20260118, 5-step stress costs)
- Очередь: `coint4/artifacts/wfa/aggregate/20260118_stress/run_queue.csv`.
- Цель: стресс-издержки на shortlist (slippage x2, commission +50%). Funding +50% аппроксимирован через рост комиссий/слиппеджа, т.к. Numba-бэктест не моделирует funding отдельно.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиги:
  - `coint4/configs/stress_20260118/stress_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000.yaml`
  - `coint4/configs/stress_20260118/stress_20260118_corr0p7_z0p85_exit0p12_ssd25000.yaml`
- Статус: `completed` (запуск на 85.198.90.128).
- Прогон 1: `stress_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000` → `coint4/artifacts/wfa/runs/20260118_stress/stress_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000`.
- Метрики (rollup recomputed): sharpe_ratio_abs `4.6636`, total_pnl `616.53`, max_drawdown_abs `-91.36`, total_trades `4815`, total_pairs_traded `271`, total_costs `337.29`, win_rate `0.5629`.
- Прогон 2: `stress_20260118_corr0p7_z0p85_exit0p12_ssd25000` → `coint4/artifacts/wfa/runs/20260118_stress/stress_20260118_corr0p7_z0p85_exit0p12_ssd25000`.
- Метрики (rollup recomputed): sharpe_ratio_abs `4.5746`, total_pnl `542.81`, max_drawdown_abs `-91.36`, total_trades `4649`, total_pairs_traded `258`, total_costs `319.76`, win_rate `0.5894`.
- Итог: стресс-издержки снижают Sharpe относительно базового прогона, но остаются > 4 на WFA‑периоде.
