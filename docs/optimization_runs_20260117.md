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

### Holdout diagnostics (20260118)
- Отчёт: `docs/holdout_diagnostics_20260118.md`.
- Окно теста фактически 2024-05-02 → 2024-09-28 (ограничение `max_steps=5`).
- Steps 1-4 отрицательные; самые слабые шаги 2-3.
- Перекрытие пар с shortlist очень низкое (13/501, Jaccard ~1.7%); пересекающиеся пары дают +48.7 PnL, остальные -370.2.
- Смещение причин отсева: holdout доминирует pvalue, shortlist доминирует low_correlation.

### WFA очередь (stability_20260119, shortlist with stability filter)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_stability_shortlist/run_queue.csv`.
- Цель: проверить устойчивость пар через окно шагов (pair_stability_window=3, min_steps=2) и ужесточить pvalue/half-life/hurst + tradeability фильтры.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиги:
  - `coint4/configs/stability_20260119/stability_20260119_pv0p03_hurst0p5_hl0p05_40_corr0p65_ssd15000.yaml`
  - `coint4/configs/stability_20260119/stability_20260119_pv0p02_hurst0p48_hl0p05_30_corr0p7_ssd12000.yaml`
  - `coint4/configs/stability_20260119/stability_20260119_liquid_pv0p03_hurst0p5_hl0p05_40_corr0p65_ssd10000.yaml`
- Статус: `completed` (запуск на 85.198.90.128).
- Прогон 1: `stability_20260119_pv0p03_hurst0p5_hl0p05_40_corr0p65_ssd15000` → `coint4/artifacts/wfa/runs/20260119_stability_shortlist/stability_20260119_pv0p03_hurst0p5_hl0p05_40_corr0p65_ssd15000`.
- Метрики (rollup recomputed): sharpe_ratio_abs `5.9245`, total_pnl `76.90`, max_drawdown_abs `-26.90`, total_trades `597`, total_pairs_traded `34`, total_costs `18.64`, win_rate `0.6452`.
- Прогон 2: `stability_20260119_pv0p02_hurst0p48_hl0p05_30_corr0p7_ssd12000` → `coint4/artifacts/wfa/runs/20260119_stability_shortlist/stability_20260119_pv0p02_hurst0p48_hl0p05_30_corr0p7_ssd12000`.
- Метрики (rollup recomputed): sharpe_ratio_abs `4.2141`, total_pnl `16.68`, max_drawdown_abs `-7.54`, total_trades `79`, total_pairs_traded `4`, total_costs `3.72`, win_rate `0.3871`.
- Прогон 3: `stability_20260119_liquid_pv0p03_hurst0p5_hl0p05_40_corr0p65_ssd10000` → `coint4/artifacts/wfa/runs/20260119_stability_shortlist/stability_20260119_liquid_pv0p03_hurst0p5_hl0p05_40_corr0p65_ssd10000`.
- Метрики (rollup recomputed): sharpe_ratio_abs `2.4103`, total_pnl `27.83`, max_drawdown_abs `-29.48`, total_trades `378`, total_pairs_traded `21`, total_costs `13.50`, win_rate `0.6129`.
- Итог: фильтр стабильности + ужесточённые пороги сильно сокращают число пар/сделок; критерии `total_pairs_traded >= 100` не выполняются. Нужна корректировка порогов (ослабить стабильность/фильтры).

### WFA очередь (stability_relaxed_20260119, shortlist relaxed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_stability_relaxed/run_queue.csv`.
- Цель: увеличить число пар до >=100 при сохранении стабильности (window=2/min=1), ослабив пороги.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиги:
  - `coint4/configs/stability_20260119_relaxed/stability_relaxed_20260119_pv0p04_hurst0p55_hl0p02_60_corr0p65_ssd20000.yaml`
  - `coint4/configs/stability_20260119_relaxed/stability_relaxed_20260119_pv0p05_hurst0p55_hl0p02_60_corr0p6_ssd25000.yaml`
- Статус: `completed` (запуск на 85.198.90.128).
- Прогон 1: `stability_relaxed_20260119_pv0p04_hurst0p55_hl0p02_60_corr0p65_ssd20000` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed/stability_relaxed_20260119_pv0p04_hurst0p55_hl0p02_60_corr0p65_ssd20000`.
- Метрики (rollup recomputed): sharpe_ratio_abs `3.9899`, total_pnl `148.26`, max_drawdown_abs `-77.76`, total_trades `1592`, total_pairs_traded `78`, total_costs `44.80`, win_rate `0.4355`.
- Прогон 2: `stability_relaxed_20260119_pv0p05_hurst0p55_hl0p02_60_corr0p6_ssd25000` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed/stability_relaxed_20260119_pv0p05_hurst0p55_hl0p02_60_corr0p6_ssd25000`.
- Метрики (rollup recomputed): sharpe_ratio_abs `2.5860`, total_pnl `100.02`, max_drawdown_abs `-64.87`, total_trades `815`, total_pairs_traded `41`, total_costs `24.39`, win_rate `0.3478`.
- Итог: пары выросли (78 в лучшем варианте), но всё ещё ниже порога 100; требуется дополнительное ослабление фильтров/стабильности или увеличение SSD/universe.

### WFA очередь (stability_relaxed2_20260119, ещё более мягкие фильтры)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_stability_relaxed2/run_queue.csv`.
- Цель: довести `total_pairs_traded >= 100` при сохранении pair_stability (window=2/min=1); ослабить corr/pvalue/hurst и увеличить `ssd_top_n`.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиги:
  - `coint4/configs/stability_20260119_relaxed2/stability_relaxed2_20260119_pv0p06_hurst0p6_hl0p02_60_corr0p55_ssd30000.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `stability_relaxed2_20260119_pv0p06_hurst0p6_hl0p02_60_corr0p55_ssd30000` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed2/stability_relaxed2_20260119_pv0p06_hurst0p6_hl0p02_60_corr0p55_ssd30000`.
- Метрики (rollup recomputed): sharpe_ratio_abs `2.1286`, total_pnl `84.13`, max_drawdown_abs `-74.42`, total_trades `1010`, total_pairs_traded `52`, total_costs `31.36`, win_rate `0.4022`.
- Итог: число пар всё ещё ниже порога 100; требуется дальнейшее ослабление фильтров (например, window=1/min=1 или рост `ssd_top_n`/universe).

### WFA очередь (stability_relaxed3_20260119, window=1/min=1)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_stability_relaxed3/run_queue.csv`.
- Цель: поднять `total_pairs_traded >= 100` за счёт window=1/min=1 и расширенных порогов corr/pvalue/ssd; проверить градиент ослабления.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиги:
  - `coint4/configs/stability_20260119_relaxed3/stability_relaxed3_20260119_pv0p06_hurst0p6_hl0p02_60_corr0p55_ssd30000_st1.yaml`
  - `coint4/configs/stability_20260119_relaxed3/stability_relaxed3_20260119_pv0p07_hurst0p62_hl0p02_60_corr0p5_ssd40000_st1.yaml`
  - `coint4/configs/stability_20260119_relaxed3/stability_relaxed3_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_st1.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `stability_relaxed3_20260119_pv0p06_hurst0p6_hl0p02_60_corr0p55_ssd30000_st1` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed3/stability_relaxed3_20260119_pv0p06_hurst0p6_hl0p02_60_corr0p55_ssd30000_st1`.
- Метрики (rollup recomputed): sharpe_ratio_abs `2.0432`, total_pnl `80.68`, max_drawdown_abs `-74.42`, total_trades `989`, total_pairs_traded `51`, total_costs `29.07`, win_rate `0.3804`.
- Прогон 2: `stability_relaxed3_20260119_pv0p07_hurst0p62_hl0p02_60_corr0p5_ssd40000_st1` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed3/stability_relaxed3_20260119_pv0p07_hurst0p62_hl0p02_60_corr0p5_ssd40000_st1`.
- Метрики: 0 сделок/0 пар (KPSS фильтр занулил пары после Hurst; см. run.log).
- Прогон 3: `stability_relaxed3_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_st1` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed3/stability_relaxed3_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_st1`.
- Метрики: 0 сделок/0 пар (KPSS фильтр занулил пары после Hurst; см. run.log).
- Итог: window=1/min=1 не дал роста пар; при более мягких конфигх KPSS-фильтр стал бутылочным горлышком.

### WFA очередь (stability_relaxed4_20260119, ослабление KPSS)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_stability_relaxed4/run_queue.csv`.
- Цель: снять бутылочное горлышко KPSS (0.03–0.05) и вернуть пары на corr0.5/0.45 при ssd 40–50k.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиги:
  - `coint4/configs/stability_20260119_relaxed4/stability_relaxed4_20260119_pv0p07_hurst0p62_hl0p02_60_corr0p5_ssd40000_kpss0p05.yaml`
  - `coint4/configs/stability_20260119_relaxed4/stability_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_kpss0p03.yaml`
  - `coint4/configs/stability_20260119_relaxed4/stability_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `stability_relaxed4_20260119_pv0p07_hurst0p62_hl0p02_60_corr0p5_ssd40000_kpss0p05` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed4/stability_relaxed4_20260119_pv0p07_hurst0p62_hl0p02_60_corr0p5_ssd40000_kpss0p05`.
- Метрики (rollup recomputed): sharpe_ratio_abs `3.0707`, total_pnl `325.01`, max_drawdown_abs `-158.63`, total_trades `3065`, total_pairs_traded `159`, total_costs `96.16`, win_rate `0.3770`.
- Прогон 2: `stability_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_kpss0p03` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed4/stability_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_kpss0p03`.
- Метрики (rollup recomputed): sharpe_ratio_abs `3.0732`, total_pnl `538.74`, max_drawdown_abs `-181.16`, total_trades `5067`, total_pairs_traded `270`, total_costs `167.87`, win_rate `0.5033`.
- Прогон 3: `stability_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed4/stability_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03`.
- Метрики (rollup recomputed): sharpe_ratio_abs `3.0920`, total_pnl `554.44`, max_drawdown_abs `-168.23`, total_trades `5147`, total_pairs_traded `272`, total_costs `175.02`, win_rate `0.5232`.
- Итог: ослабление KPSS сняло бутылочное горлышко; пары 159–272, Sharpe ~3.07–3.09. Лучший по Sharpe/PnL — corr0.45 + ssd50000 + kpss0.03; стоит подтверждать holdout/stress.

### WFA очередь (relaxed4_holdout_20260119, holdout топ‑2)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed4_holdout/run_queue.csv`.
- Цель: holdout 2024-05-01 → 2024-12-31 для top‑1/2 relaxed4.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиги:
  - `coint4/configs/holdout_20260119_relaxed4/holdout_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03.yaml`
  - `coint4/configs/holdout_20260119_relaxed4/holdout_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_kpss0p03.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `holdout_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03` → `coint4/artifacts/wfa/runs/20260119_relaxed4_holdout/holdout_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03`.
- Метрики (rollup recomputed): sharpe_ratio_abs `-1.3228`, total_pnl `-98.19`, max_drawdown_abs `-218.02`, total_trades `8514`, total_pairs_traded `567`, total_costs `91.45`, win_rate `0.4768`.
- Прогон 2: `holdout_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_kpss0p03` → `coint4/artifacts/wfa/runs/20260119_relaxed4_holdout/holdout_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_kpss0p03`.
- Метрики (rollup recomputed): sharpe_ratio_abs `-1.0760`, total_pnl `-78.69`, max_drawdown_abs `-202.09`, total_trades `8432`, total_pairs_traded `560`, total_costs `90.23`, win_rate `0.4768`.
- Итог: holdout снова отрицательный, несмотря на рост числа пар; нужна дополнительная диагностика режима/фильтров.

### WFA очередь (relaxed4_stress_20260119, стресс-издержки топ‑2)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed4_stress/run_queue.csv`.
- Цель: стресс-издержки (commission 0.0006, slippage 0.0010, stress_multiplier 2.0) для top‑1/2 relaxed4 на WFA окне.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиги:
  - `coint4/configs/stress_20260119_relaxed4/stress_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03.yaml`
  - `coint4/configs/stress_20260119_relaxed4/stress_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_kpss0p03.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `stress_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03` → `coint4/artifacts/wfa/runs/20260119_relaxed4_stress/stress_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03`.
- Метрики (rollup recomputed): sharpe_ratio_abs `2.3860`, total_pnl `426.44`, max_drawdown_abs `-176.75`, total_trades `5147`, total_pairs_traded `272`, total_costs `311.15`, win_rate `0.4834`.
- Прогон 2: `stress_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_kpss0p03` → `coint4/artifacts/wfa/runs/20260119_relaxed4_stress/stress_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_kpss0p03`.
- Метрики (rollup recomputed): sharpe_ratio_abs `2.3816`, total_pnl `416.13`, max_drawdown_abs `-189.92`, total_trades `5067`, total_pairs_traded `270`, total_costs `298.43`, win_rate `0.4702`.
- Итог: стресс-издержки уменьшают Sharpe, но остаётся >2 на WFA окне.

### Holdout diagnostics (20260119 relaxed4)
- Отчёт: `docs/holdout_diagnostics_20260119_relaxed4.md`.
- Перекрытие пар с WFA минимальное (4 пересечения, Jaccard ~0.0048); убыток формируется в основном вне пересечения.
- Holdout чаще режется по pvalue, что указывает на развал коинтеграции в 2024H2.

### WFA очередь (stability_relaxed5_20260119, усиление стабильности)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_stability_relaxed5/run_queue.csv`.
- Цель: увеличить устойчивость пар между шагами (window=2/3) при сохранении corr0.45/ssd50000/kpss0.03.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиги:
  - `coint4/configs/stability_20260119_relaxed5/stability_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w2m1.yaml`
  - `coint4/configs/stability_20260119_relaxed5/stability_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `stability_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w2m1` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed5/stability_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w2m1`.
- Метрики (rollup recomputed): sharpe_ratio_abs `3.0706`, total_pnl `550.62`, max_drawdown_abs `-168.23`, total_trades `5193`, total_pairs_traded `275`, total_costs `178.27`, win_rate `0.5232`.
- Прогон 2: `stability_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed5/stability_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2`.
- Метрики (rollup recomputed): sharpe_ratio_abs `5.8116`, total_pnl `958.72`, max_drawdown_abs `-188.98`, total_trades `6519`, total_pairs_traded `383`, total_costs `236.95`, win_rate `0.5738`.
- Итог: window=3/min=2 даёт лучший Sharpe и рост пар; перейти к holdout/stress для w3m2.

### WFA очередь (relaxed5_holdout_20260119, holdout w3m2)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed5_holdout/run_queue.csv`.
- Цель: holdout 2024-05-01 → 2024-12-31 для w3m2.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/holdout_20260119_relaxed5/holdout_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `holdout_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2` → `coint4/artifacts/wfa/runs/20260119_relaxed5_holdout/holdout_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2`.
- Метрики (rollup recomputed): sharpe_ratio_abs `-1.7710`, total_pnl `-220.63`, max_drawdown_abs `-419.86`, total_trades `17110`, total_pairs_traded `1018`, total_costs `464.87`, win_rate `0.3689`.
- Итог: holdout резко отрицательный, несмотря на высокий Sharpe в WFA.

### WFA очередь (relaxed5_stress_20260119, стресс w3m2)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed5_stress/run_queue.csv`.
- Цель: стресс-издержки (commission 0.0006, slippage 0.0010, stress_multiplier 2.0) для w3m2.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/stress_20260119_relaxed5/stress_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `stress_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2` → `coint4/artifacts/wfa/runs/20260119_relaxed5_stress/stress_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2`.
- Метрики (rollup recomputed): sharpe_ratio_abs `4.7573`, total_pnl `783.53`, max_drawdown_abs `-214.79`, total_trades `6519`, total_pairs_traded `383`, total_costs `421.25`, win_rate `0.5410`.
- Итог: стресс-издержки снижают Sharpe, но остаётся высокое значение на WFA.

### Holdout diagnostics (20260119 relaxed5 w3m2)
- Отчёт: `docs/holdout_diagnostics_20260119_relaxed5.md`.
- Пересечение пар с WFA низкое (16 пересечений, Jaccard ~0.0116); убыток формируется вне пересечения.
- Holdout доминируется pvalue → подтверждает нестабильность коинтеграции в 2024H2.

### WFA очередь (relaxed5_holdout_fixed_20260119, фиксированный universe w3m2)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed5_holdout_fixed/run_queue.csv`.
- Цель: повторить holdout w3m2 с фиксированным universe (383 пары из WFA) для проверки устойчивости пар.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/holdout_20260119_relaxed5_fixed/holdout_relaxed5_fixed_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2.yaml`
- Источник пар: `coint4/artifacts/universe/20260119_relaxed5_w3m2_fixed/pairs_universe.yaml`.
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `holdout_relaxed5_fixed_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2` → `coint4/artifacts/wfa/runs/20260119_relaxed5_holdout_fixed/holdout_relaxed5_fixed_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2`.
- Метрики (rollup recomputed): sharpe_ratio_abs `1.2471`, total_pnl `13.94`, max_drawdown_abs `-8.34`, total_trades `352`, total_pairs_traded `18`, total_costs `7.16`, win_rate `0.2295`.
- Фильтрация (run.log): pvalue отсев ~32–47%, kpss ~19–31%, hurst ~4–14%; pair_stability (window=3/min=2) часто режет 6–7 пар до 0–1.
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_120517.csv`, `coint4/results/filter_reasons_20260119_120549.csv`, `coint4/results/filter_reasons_20260119_120621.csv`, `coint4/results/filter_reasons_20260119_120650.csv`, `coint4/results/filter_reasons_20260119_120722.csv`.
- Итог: Sharpe > 1, но число пар/сделок слишком низкое для критериев стабильности; большая часть фиксированного universe не проходит фильтры в holdout.

### WFA очередь (relaxed5_holdout_fixed_w1m1_20260119, фиксированный universe без жёсткой стабильности)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed5_holdout_fixed_w1m1/run_queue.csv`.
- Цель: снять строгий pair_stability (window=1/min=1) и проверить рост числа пар в holdout при фиксированном universe.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/holdout_20260119_relaxed5_fixed/holdout_relaxed5_fixed_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w1m1.yaml`
- Источник пар: `coint4/artifacts/universe/20260119_relaxed5_w3m2_fixed/pairs_universe.yaml`.
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `holdout_relaxed5_fixed_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w1m1` → `coint4/artifacts/wfa/runs/20260119_relaxed5_holdout_fixed_w1m1/holdout_relaxed5_fixed_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w1m1`.
- Метрики (rollup recomputed): sharpe_ratio_abs `-0.0188`, total_pnl `-0.20`, max_drawdown_abs `-11.73`, total_trades `253`, total_pairs_traded `11`, total_costs `3.47`, win_rate `0.1885`.
- Фильтрация (run.log): pvalue отсев ~32–47%, kpss ~19–31%, hurst ~4–14%; pair_stability (window=1/min=1) всё ещё режет до 0–2 пар.
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_121535.csv`, `coint4/results/filter_reasons_20260119_121607.csv`, `coint4/results/filter_reasons_20260119_121637.csv`, `coint4/results/filter_reasons_20260119_121709.csv`, `coint4/results/filter_reasons_20260119_121742.csv`.
- Итог: отключение жёсткой стабильности не улучшило статистику; holdout остаётся отрицательным и с малым числом пар.

### WFA очередь (stability_relaxed6_20260119, ослабление pvalue/kpss/hurst)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_stability_relaxed6/run_queue.csv`.
- Цель: слегка ослабить cointegration фильтры (pvalue 0.12, kpss 0.05, hurst 0.70) и проверить рост числа пар при сохранении Sharpe.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/stability_20260119_relaxed6/stability_relaxed6_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `stability_relaxed6_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed6/stability_relaxed6_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2`.
- Метрики (rollup recomputed): sharpe_ratio_abs `5.3588`, total_pnl `894.35`, max_drawdown_abs `-123.46`, total_trades `5193`, total_pairs_traded `303`, total_costs `183.24`, win_rate `0.5109`.
- Фильтрация (run.log): pvalue отсев ~11–34%, kpss ~9–17%, hurst ~2.7–5.4%.
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_122615.csv`, `coint4/results/filter_reasons_20260119_122727.csv`, `coint4/results/filter_reasons_20260119_122849.csv`.
- Итог: Sharpe остаётся высоким, число пар > 300; можно перейти к holdout/стресс для этой конфигурации.

### WFA очередь (relaxed6_holdout_20260119, holdout w3m2)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed6_holdout/run_queue.csv`.
- Цель: holdout 2024-05-01 → 2024-12-31 для relaxed6 (pvalue 0.12, kpss 0.05, hurst 0.70).
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/holdout_20260119_relaxed6/holdout_relaxed6_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `holdout_relaxed6_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2` → `coint4/artifacts/wfa/runs/20260119_relaxed6_holdout/holdout_relaxed6_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2`.
- Метрики (rollup recomputed): sharpe_ratio_abs `-2.1972`, total_pnl `-267.02`, max_drawdown_abs `-413.50`, total_trades `13268`, total_pairs_traded `779`, total_costs `359.55`, win_rate `0.3245`.
- Фильтрация (run.log): pvalue отсев ~22–40%, kpss ~13–21%, hurst ~5–8%.
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_123006.csv`, `coint4/results/filter_reasons_20260119_123119.csv`, `coint4/results/filter_reasons_20260119_123905.csv`.
- Итог: holdout сильно отрицательный несмотря на хороший WFA; требуется переоценка устойчивости в 2024H2.

### Holdout diagnostics (20260119 relaxed6)
- Отчёт: `docs/holdout_diagnostics_20260119_relaxed6.md`.
- Перекрытие пар с WFA минимальное (intersection 6, Jaccard ~0.0056); убыток формируется вне пересечения.

### WFA очередь (relaxed6_holdout_fixed_20260119, фиксированный universe w3m2)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed6_holdout_fixed/run_queue.csv`.
- Цель: повторить holdout relaxed6 с фиксированным universe (303 пары из WFA) для проверки влияния overlap.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/holdout_20260119_relaxed6_fixed/holdout_relaxed6_fixed_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2.yaml`
- Источник пар: `coint4/artifacts/universe/20260119_relaxed6_w3m2_fixed/pairs_universe.yaml`.
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `holdout_relaxed6_fixed_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2` → `coint4/artifacts/wfa/runs/20260119_relaxed6_holdout_fixed/holdout_relaxed6_fixed_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2`.
- Метрики (rollup recomputed): sharpe_ratio_abs `3.7672`, total_pnl `24.98`, max_drawdown_abs `-4.68`, total_trades `141`, total_pairs_traded `7`, total_costs `3.19`, win_rate `0.2213`.
- Фильтрация (run.log): pvalue отсев ~25–43%, kpss ~21–37%, hurst ~5–12%; pair_stability (window=3/min=2) режет до 0–1 пар.
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_125706.csv`, `coint4/results/filter_reasons_20260119_125734.csv`, `coint4/results/filter_reasons_20260119_125803.csv`, `coint4/results/filter_reasons_20260119_125830.csv`, `coint4/results/filter_reasons_20260119_125859.csv`.
- Итог: Sharpe высокий, но статистика слишком мала (7 пар, 141 сделка); фиксированный universe не решает проблему охвата.

### WFA очередь (stability_relaxed7_20260119, train=90d)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_stability_relaxed7/run_queue.csv`.
- Цель: увеличить окно обучения до 90 дней при relaxed6 фильтрах (pvalue 0.12, kpss 0.05, hurst 0.70) и проверить устойчивость.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/stability_20260119_relaxed7/stability_relaxed7_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2_t90.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `stability_relaxed7_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2_t90` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed7/stability_relaxed7_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2_t90`.
- Метрики (rollup recomputed): sharpe_ratio_abs `5.2468`, total_pnl `914.61`, max_drawdown_abs `-201.41`, total_trades `2622`, total_pairs_traded `177`, total_costs `126.71`, win_rate `0.6230`.
- Фильтрация (run.log): pvalue отсев ~18–39%, kpss ~9–17%, hurst ~3.8–6.9%.
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_130951.csv`, `coint4/results/filter_reasons_20260119_131139.csv`, `coint4/results/filter_reasons_20260119_131324.csv`, `coint4/results/filter_reasons_20260119_131503.csv`, `coint4/results/filter_reasons_20260119_131640.csv`.
- Итог: Sharpe остаётся высоким при уменьшении числа пар; следующий шаг — holdout с train=90d.

### WFA очередь (relaxed7_holdout_20260119, holdout w3m2 train=90d)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed7_holdout/run_queue.csv`.
- Цель: holdout 2024-05-01 → 2024-12-31 для relaxed7 (train=90d).
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/holdout_20260119_relaxed7/holdout_relaxed7_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2_t90.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `holdout_relaxed7_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2_t90` → `coint4/artifacts/wfa/runs/20260119_relaxed7_holdout/holdout_relaxed7_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2_t90`.
- Метрики (rollup recomputed): sharpe_ratio_abs `-0.2010`, total_pnl `-20.95`, max_drawdown_abs `-172.98`, total_trades `6509`, total_pairs_traded `339`, total_costs `240.36`, win_rate `0.4130`.
- Фильтрация (run.log): pvalue отсев ~24–37%, kpss ~11–21%, hurst ~6.7–9.9%.
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_132542.csv`, `coint4/results/filter_reasons_20260119_132844.csv`, `coint4/results/filter_reasons_20260119_133203.csv`, `coint4/results/filter_reasons_20260119_133442.csv`, `coint4/results/filter_reasons_20260119_133736.csv`.
- Итог: holdout остаётся отрицательным при хорошем WFA; требуется пересмотр подхода к universe/фильтрам для 2024H2.

### Universe build (relaxed8_strict_preholdout_20260119)
- Период: `2023-07-01` → `2024-04-30` (пред‑holdout окно).
- Критерии: `configs/criteria_strict.yaml` (pvalue 0.05, hl 5–200, hurst 0.2–0.6, min_cross 10, beta_drift 0.15).
- Символы: `artifacts/universe/SYMBOLS_20250824.txt`, limit_symbols `200`, top_n `200`, diversify_by_base `true`, max_per_base `3`.
- Результат: selected pairs `110` из tested `5356`.
- Артефакты:
  - `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout/pairs_universe.yaml`
  - `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout/UNIVERSE_REPORT.md`
  - `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout/universe_metrics.csv`
  - `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout/REJECTION_BREAKDOWN.yaml`

### Universe build (relaxed8_strict_preholdout_v2_20260119)
- Период: `2023-07-01` → `2024-04-30` (пред‑holdout окно).
- Критерии: default (pvalue 0.05, hl 5–200, min_cross 10, beta_drift 0.15; без hurst в prefilter).
- Символы: `artifacts/universe/SYMBOLS_20250824.txt`, limit_symbols `300`, top_n `250`, diversify_by_base `true`, max_per_base `4`.
- Результат: selected pairs `250` из tested `13203`.
- Артефакты:
  - `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/pairs_universe.yaml`
  - `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/UNIVERSE_REPORT.md`
  - `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/universe_metrics.csv`
  - `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/REJECTION_BREAKDOWN.yaml`

### WFA очередь (stability_relaxed8_20260119, fixed universe + train=90d)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_stability_relaxed8/run_queue.csv`.
- Цель: проверить fixed universe (110 пар) в pre‑holdout WFA при train=90d.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/stability_20260119_relaxed8/stability_relaxed8_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2_t90_fixed.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `stability_relaxed8_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2_t90_fixed` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed8/stability_relaxed8_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2_t90_fixed`.
- Метрики (strategy_metrics): sharpe_ratio_abs `0.0000`, total_pnl `0.00`, max_drawdown_abs `0.00` (торговых пар 0).
- Фильтрация (run.log): после KPSS остаётся 2 пары, затем стабильность/фильтры режут до 0 (все пары вырезаны).
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_142542.csv`, `coint4/results/filter_reasons_20260119_142611.csv`, `coint4/results/filter_reasons_20260119_142640.csv`, `coint4/results/filter_reasons_20260119_142709.csv`, `coint4/results/filter_reasons_20260119_142738.csv`.
- Итог: fixed universe + текущие фильтры дают 0 пар → требуется смягчение фильтров/стабильности перед продолжением.

### WFA очередь (stability_relaxed8_loose_20260119, fixed universe + фильтры мягче)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_stability_relaxed8_loose/run_queue.csv`.
- Цель: смягчить фильтры (pvalue 0.2, kpss 0.1, hurst 0.8, corr 0.4, w2m1) и проверить, дают ли fixed‑пары торговую статистику.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/stability_20260119_relaxed8_loose/stability_relaxed8_loose_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss0p1_w2m1_t90_fixed.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `stability_relaxed8_loose_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss0p1_w2m1_t90_fixed` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed8_loose/stability_relaxed8_loose_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss0p1_w2m1_t90_fixed`.
- Метрики (strategy_metrics): sharpe_ratio_abs `0.0000`, total_pnl `0.00`, max_drawdown_abs `0.00` (торговых пар 0).
- Фильтрация (run.log): на каждом шаге KPSS режет до 0 пар даже при kpss=0.1.
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_143531.csv`, `coint4/results/filter_reasons_20260119_143559.csv`, `coint4/results/filter_reasons_20260119_143628.csv`, `coint4/results/filter_reasons_20260119_143657.csv`, `coint4/results/filter_reasons_20260119_143727.csv`.
- Итог: KPSS остаётся основным стоп‑фильтром; для fixed‑universe нужен обход/ослабление KPSS.

### WFA очередь (stability_relaxed8_nokpss_20260119, fixed universe без KPSS)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_stability_relaxed8_nokpss/run_queue.csv`.
- Цель: отключить KPSS (kpss=1.0) и проверить, появятся ли торгуемые пары в fixed‑universe.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/stability_20260119_relaxed8_nokpss/stability_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `stability_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed` → `coint4/artifacts/wfa/runs/20260119_stability_relaxed8_nokpss/stability_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed`.
- Метрики (rollup recomputed): sharpe_ratio_abs `1.8550`, total_pnl `104.11`, max_drawdown_abs `-60.19`, total_trades `1242`, total_pairs_traded `35`, total_costs `60.23`, win_rate `0.4967`.
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_144355.csv`, `coint4/results/filter_reasons_20260119_144426.csv`, `coint4/results/filter_reasons_20260119_144457.csv`, `coint4/results/filter_reasons_20260119_144529.csv`, `coint4/results/filter_reasons_20260119_144600.csv`.
- Итог: без KPSS появляются сделки и Sharpe > 1, но число пар (`35`) ниже порога стабильности → требуется валидация на holdout.

### WFA очередь (relaxed8_nokpss_holdout_20260119, holdout w2m1 train=90d)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed8_nokpss_holdout/run_queue.csv`.
- Цель: holdout 2024-05-01 → 2024-12-31 для relaxed8_nokpss (fixed universe, kpss=1.0).
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/holdout_20260119_relaxed8_nokpss/holdout_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `holdout_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed` → `coint4/artifacts/wfa/runs/20260119_relaxed8_nokpss_holdout/holdout_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed`.
- Метрики (strategy_metrics): sharpe_ratio_abs `3.2090`, total_pnl `145.84`, max_drawdown_abs `-40.29`, total_trades `2252`, total_pairs_traded `64`, total_costs `61.12`, win_rate `0.5563`.
- Фильтрация (run.log): после KPSS остаётся 25–51 пар; pair stability в шагах 2–5 режет до 21–34 (window=2, min_steps=1).
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_150306.csv`, `coint4/results/filter_reasons_20260119_150339.csv`, `coint4/results/filter_reasons_20260119_150413.csv`, `coint4/results/filter_reasons_20260119_150446.csv`, `coint4/results/filter_reasons_20260119_150521.csv`.
- Итог: holdout подтверждён, Sharpe > 3 и положительный PnL; число пар выросло до 64, но всё ещё ниже желаемого порога стабильности.

### WFA очередь (relaxed8_nokpss_stress_holdout_20260119, stress costs)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed8_nokpss_stress_holdout/run_queue.csv`.
- Цель: стресс-издержки для relaxed8_nokpss holdout (commission 0.0006, slippage 0.0010, slippage_stress_multiplier 2.0).
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/stress_20260119_relaxed8_nokpss/stress_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `stress_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed` → `coint4/artifacts/wfa/runs/20260119_relaxed8_nokpss_stress_holdout/stress_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed`.
- Метрики (strategy_metrics): sharpe_ratio_abs `2.1692`, total_pnl `98.35`, max_drawdown_abs `-44.29`, total_trades `2252`, total_pairs_traded `64`, total_costs `108.65`, win_rate `0.5232`.
- Фильтрация (run.log): после KPSS остаётся 25–51 пар; pair stability в шагах 2–5 режет до 21–34 (window=2, min_steps=1).
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_152305.csv`, `coint4/results/filter_reasons_20260119_152338.csv`, `coint4/results/filter_reasons_20260119_152411.csv`, `coint4/results/filter_reasons_20260119_152444.csv`, `coint4/results/filter_reasons_20260119_152518.csv`.
- Итог: стресс снижает Sharpe/PnL, но остаётся > 1; концентрация по PnL высокая (top‑5 пар ~71% суммарного PnL) → нужен рост числа устойчивых пар.

### WFA очередь (relaxed8_nokpss_u250_holdout_20260119, expanded universe)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed8_nokpss_u250_holdout/run_queue.csv`.
- Цель: повторить holdout relaxed8_nokpss на expanded universe (250 пар, limit_symbols=300).
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/holdout_20260119_relaxed8_nokpss_u250/holdout_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250.yaml`
- Источник пар: `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/pairs_universe.yaml`.
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `holdout_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250` → `coint4/artifacts/wfa/runs/20260119_relaxed8_nokpss_u250_holdout/holdout_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250`.
- Метрики (strategy_metrics): sharpe_ratio_abs `4.2025`, total_pnl `421.60`, max_drawdown_abs `-72.64`, total_trades `6572`, total_pairs_traded `168`, total_costs `174.35`, win_rate `0.6225`.
- Фильтрация (run.log): после KPSS 76–139 пар; pair stability режет до 52–93 (window=2, min_steps=1).
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_160121.csv`, `coint4/results/filter_reasons_20260119_160156.csv`, `coint4/results/filter_reasons_20260119_160230.csv`, `coint4/results/filter_reasons_20260119_160304.csv`, `coint4/results/filter_reasons_20260119_160338.csv`.
- Итог: цель достигнута (pairs ≥ 100) и Sharpe > 1; концентрация снизилась (top‑5 пар ~43% PnL).

### WFA очередь (relaxed8_nokpss_u250_stress_holdout_20260119, stress costs)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed8_nokpss_u250_stress_holdout/run_queue.csv`.
- Цель: стресс-издержки для relaxed8_nokpss_u250 (commission 0.0006, slippage 0.0010, slippage_stress_multiplier 2.0).
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиг:
  - `coint4/configs/stress_20260119_relaxed8_nokpss_u250/stress_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `stress_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250` → `coint4/artifacts/wfa/runs/20260119_relaxed8_nokpss_u250_stress_holdout/stress_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250`.
- Метрики (strategy_metrics): sharpe_ratio_abs `2.8931`, total_pnl `289.30`, max_drawdown_abs `-75.32`, total_trades `6572`, total_pairs_traded `168`, total_costs `309.95`, win_rate `0.5828`.
- Фильтрация (run.log): после KPSS 76–139 пар; pair stability режет до 52–93 (window=2, min_steps=1).
- Файлы причин фильтрации: `coint4/results/filter_reasons_20260119_162055.csv`, `coint4/results/filter_reasons_20260119_162129.csv`, `coint4/results/filter_reasons_20260119_162203.csv`, `coint4/results/filter_reasons_20260119_162237.csv`, `coint4/results/filter_reasons_20260119_162311.csv`.
- Итог: стресс снижает Sharpe/PnL, но остаётся > 1; концентрация по PnL умеренная (top‑5 пар ~61% PnL) → пригодно для финальной валидации.

### WFA очередь (relaxed8_nokpss_u250_turnover_stress_20260119, снижение churn)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed8_nokpss_u250_turnover_stress/run_queue.csv`.
- Цель: снизить turnover/издержки при сохранении Sharpe>1 (повышение entry, понижение exit, увеличение hold/cooldown).
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиги:
  - `coint4/configs/stress_20260119_relaxed8_nokpss_u250_turnover/stress_relaxed8_nokpss_u250_z0p95_exit0p08_hold120_cd120.yaml`
  - `coint4/configs/stress_20260119_relaxed8_nokpss_u250_turnover/stress_relaxed8_nokpss_u250_z1p05_exit0p08_hold120_cd120.yaml`
  - `coint4/configs/stress_20260119_relaxed8_nokpss_u250_turnover/stress_relaxed8_nokpss_u250_z1p0_exit0p06_hold180_cd180.yaml`
- Статус: `planned`.

### WFA очередь (relaxed8_nokpss_u250_topk_20260119, ограничение пар)
- Очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed8_nokpss_u250_topk/run_queue.csv`.
- Цель: снизить turnover/издержки через ограничение числа пар (max_pairs=10/20) на holdout + stress.
- Параллельность: `8` (nproc на 85.198.90.128).
- Конфиги:
  - `coint4/configs/holdout_20260119_relaxed8_nokpss_u250_topk/holdout_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top10.yaml`
  - `coint4/configs/holdout_20260119_relaxed8_nokpss_u250_topk/holdout_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top20.yaml`
  - `coint4/configs/stress_20260119_relaxed8_nokpss_u250_topk/stress_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top10.yaml`
  - `coint4/configs/stress_20260119_relaxed8_nokpss_u250_topk/stress_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top20.yaml`
- Статус: `completed` (запуск на 85.198.90.128, очередь обновлена).
- Прогон 1: `holdout_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top10` → `coint4/artifacts/wfa/runs/20260119_relaxed8_nokpss_u250_topk/holdout_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top10`.
  - Метрики: sharpe_ratio_abs `2.7421`, total_pnl `75.78`, max_drawdown_abs `-27.63`, total_trades `798`, total_pairs_traded `26`, total_costs `23.61`.
- Прогон 2: `holdout_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top20` → `coint4/artifacts/wfa/runs/20260119_relaxed8_nokpss_u250_topk/holdout_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top20`.
  - Метрики: sharpe_ratio_abs `4.5614`, total_pnl `178.29`, max_drawdown_abs `-32.09`, total_trades `1693`, total_pairs_traded `53`, total_costs `46.37`.
- Прогон 3: `stress_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top10` → `coint4/artifacts/wfa/runs/20260119_relaxed8_nokpss_u250_topk/stress_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top10`.
  - Метрики: sharpe_ratio_abs `2.0825`, total_pnl `57.42`, max_drawdown_abs `-31.75`, total_trades `798`, total_pairs_traded `26`, total_costs `41.97`.
- Прогон 4: `stress_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top20` → `coint4/artifacts/wfa/runs/20260119_relaxed8_nokpss_u250_topk/stress_relaxed8_nokpss_20260119_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top20`.
  - Метрики: sharpe_ratio_abs `3.6497`, total_pnl `142.23`, max_drawdown_abs `-37.60`, total_trades `1693`, total_pairs_traded `53`, total_costs `82.43`.
- Фильтрация (run.log): файлы причин отсева `coint4/results/filter_reasons_20260119_221818.csv`, `coint4/results/filter_reasons_20260119_221856.csv`, `coint4/results/filter_reasons_20260119_221857.csv`, `coint4/results/filter_reasons_20260119_221934.csv`, `coint4/results/filter_reasons_20260119_221935.csv`, `coint4/results/filter_reasons_20260119_222011.csv`, `coint4/results/filter_reasons_20260119_222012.csv`, `coint4/results/filter_reasons_20260119_222013.csv`, `coint4/results/filter_reasons_20260119_222049.csv`, `coint4/results/filter_reasons_20260119_222051.csv`.
- Итог: сделки снижены в 3.9–8.2 раза (1693/798 vs 6572), Sharpe остаётся > 1 даже в стресс‑режиме; топ‑20 выглядит балансом между Sharpe и turnover.
