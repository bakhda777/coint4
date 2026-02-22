## 1) TL;DR проекта
- Это система парного трейдинга крипты с walk-forward/backtest пайплайном: фильтрация пар -> бэктест по шагам WFA -> агрегированные метрики (`coint4/src/coint2/pipeline/walk_forward_orchestrator.py`, `coint4/src/coint2/pipeline/filters.py`).
- Текущая рабочая цель: подобрать устойчивые параметры (robust Sharpe при контроле DD) на наборе OOS-окон и обновлять shortlist для следующего batch (`coint4/scripts/optimization/rank_multiwindow_robust_runs.py`, `coint4/scripts/optimization/autopilot_budget1000.py`).
- Базовый бюджет стратегии: `portfolio.initial_capital=1000.0` (`coint4/configs/prod_final_budget1000.yaml`).
- WFA считает raw-метрики в `strategy_metrics.csv` из агрегированного `aggregated_pnl` и `equity_curve` (`coint4/src/coint2/pipeline/walk_forward_orchestrator.py`).
- Источник истины для межранового сравнения Sharpe/DD/PnL: `canonical_metrics.json` (если есть) и rollup `run_index.csv` с пересчётом Sharpe из `equity_curve.csv` (`coint4/src/coint2/core/canonical_metrics.py`, `coint4/src/coint2/ops/run_index.py`, `coint4/scripts/optimization/build_run_index.py`).
- Тяжёлые WFA запускаются через remote-скрипты/очереди, артефакты складываются в `coint4/artifacts/wfa/runs*/...`, индексы в `coint4/artifacts/wfa/aggregate/rollup/`.

## 2) Архитектура репо (минимальная карта)
- Корень Git: `/home/claudeuser/coint4`.
- App-root: `coint4/` (Poetry, CLI, scripts, artifacts).
- Ключевые директории:

```text
coint4/src/coint2/
  core/         # ядро метрик, PnL, kernels
  engine/       # backtest engines (Numba/Base)
  pipeline/     # WFA orchestration + pair filters
  ops/          # run_queue/run_index
  utils/        # config/logging/visualization

coint4/scripts/
  optimization/ # queue/watch/ranking/autopilot/loop orchestrator
  remote/       # запуск jobs на VPS + power/sync
  ralph/        # strict launcher для ralph-tui

coint4/configs/ # YAML конфиги стратегии/оптимизаций
tasks/          # PRD/ralph task JSON
coint4/artifacts/
  wfa/runs/|runs_clean/      # run-артефакты
  wfa/aggregate/<group>/     # run_queue.csv
  wfa/aggregate/rollup/      # run_index.csv|json|md
```

- Где лежат конфиги стратегии:
  - Основной прод-конфиг: `coint4/configs/prod_final_budget1000.yaml`.
  - Автопилот/свипы: `coint4/configs/autopilot/*.yaml`, `coint4/configs/budget*/...`.
- Точки входа прогона:
  - CLI: `./.venv/bin/coint2 walk-forward --config ... --results-dir ...` -> `coint4/src/coint2/cli/__init__.py` -> `run_walk_forward`.
  - Один ран: `coint4/run_wfa_fullcpu.sh`.
  - Очередь: `coint4/scripts/optimization/run_wfa_queue.py` + `coint4/scripts/optimization/watch_wfa_queue.sh`.
  - VPS orchestration: `coint4/scripts/remote/run_server_job.sh`.
  - Batch loop: `coint4/scripts/optimization/loop_orchestrator/orchestrate.py`.

## 3) Как считаются метрики (формулы + пути до кода)
- Sharpe (канонический):
  - Формула: `Sharpe = sqrt(periods_per_year) * mean(excess_returns) / std(excess_returns, ddof=1)`.
  - `excess_returns = returns - risk_free_rate`.
  - Реализация: `annualized_sharpe_ratio()` в `coint4/src/coint2/core/sharpe.py`.
- Sharpe из equity curve:
  - Доходности: `r_t = (equity_t - equity_{t-1}) / equity_{t-1}`.
  - Annualization (rollup/canonical): `periods_per_year = days_per_year * (86400 / median_period_seconds)`.
  - Реализация: `compute_equity_sharpe_from_equity_curve_csv()` и `compute_sharpe_ratio_abs_from_equity_curve_csv()` в `coint4/src/coint2/core/sharpe.py`.
- Drawdown:
  - По equity (доля): `drawdown_t = (equity_t - cummax(equity)_t) / cummax(equity)_t`, берётся `min`.
  - Реализация: `max_drawdown_on_equity()` в `coint4/src/coint2/core/performance.py`.
  - По cumulative PnL (абсолют): `drawdown_t = cumulative_pnl_t - cummax(cumulative_pnl)_t`, берётся `min`.
  - Реализация: `max_drawdown()` в `coint4/src/coint2/core/performance.py`.
- PnL / equity curve:
  - Вход/выход и cost в Numba kernel: `cost = trade_units * (|y| + |beta*x|) * (commission + slippage)`.
  - Закрытие: `price_pnl = position * units * (spread_now - entry_spread)`, `pnl = price_pnl - cost`; на входе `pnl = -cost`.
  - Реализация: `calculate_positions_and_pnl_full()` в `coint4/src/coint2/core/numba_kernels.py`.
  - Компоненты cost (`commission_costs`, `slippage_costs`) раскладываются в `NumbaPairBacktester.run()` (`coint4/src/coint2/engine/numba_engine.py`).
  - Equity формируется через `Portfolio.record_daily_pnl()` (`coint4/src/coint2/core/portfolio.py`).
- Какие метрики пишутся в отчёты WFA и где:
  - `strategy_metrics.csv`, `daily_pnl.csv`, `equity_curve.csv`, `trade_statistics.csv`, `trades_log.csv`.
  - Сохранение: `run_walk_forward()` в `coint4/src/coint2/pipeline/walk_forward_orchestrator.py`.
  - PNG-отчёт: `create_performance_report()` в `coint4/src/coint2/utils/visualization.py`.

## 4) Walk-Forward Analysis (WFA)
- Окна train/test:
  - В конфиге: `walk_forward.training_period_days`, `walk_forward.testing_period_days`, `walk_forward.start_date`, `walk_forward.end_date`, `walk_forward.max_steps`.
  - Пример baseline: `training=90d`, `testing=15d`, `max_steps=5` в `coint4/configs/prod_final_budget1000.yaml`.
  - Планирование в коде: `run_walk_forward()` (`coint4/src/coint2/pipeline/walk_forward_orchestrator.py`):
    - `training_start = current_test_start - training_period_days`
    - `training_end = current_test_start - 1 bar`
    - `testing_start = current_test_start`
    - `testing_end = testing_start + testing_period_days`
    - затем `current_test_start = testing_end`.
- Что такое fold:
  - По факту fold = один WF-шаг (tuple из `training_start, training_end, testing_start, testing_end`).
  - В коде используются термины `walk_forward_steps` / `WF-шаг`.
- Агрегация по фолдам:
  - `step_pnl` каждого шага конкатенируется в `aggregated_pnl`.
  - Одновременно обновляется `portfolio.equity_curve` через `record_daily_pnl()`.
  - Финальные метрики считаются на агрегированных сериях после всех шагов.
- Что такое best run и по какому критерию:
  - В самом `walk_forward_orchestrator` best-run не выбирается (он только считает один run).
  - Best-run выбирается после rollup:
    - robust ranking: `robust_sharpe_window = min(holdout_sharpe, stress_sharpe)`, score по худшему окну (`coint4/scripts/optimization/rank_multiwindow_robust_runs.py`).
    - в autopilot: `score = worst_robust_sharpe - DD_penalty` с risk gates (`coint4/scripts/optimization/autopilot_budget1000.py`).

## 5) Параметры, которые мы оптимизируем
- Ключевые параметры (текущий batch-loop, bridge11):

| Параметр | Что означает | Где используется | Текущий диапазон/шаг |
|---|---|---|---|
| `portfolio.risk_per_position_pct` | риск на позицию относительно equity | `Portfolio.calculate_position_risk_capital()` (`coint4/src/coint2/core/portfolio.py`), передаётся из WFA orchestrator | step `0.001`, min `0.003`, max `0.014` (`coint4/configs/autopilot/budget1000_batch_loop_bridge11_20260217.yaml`) |
| `backtest.pair_stop_loss_usd` | USD stop-loss на пару (unrealized) | `calculate_positions_and_pnl_full()` (`coint4/src/coint2/core/numba_kernels.py`) | step `0.35`, min `1.2`, max `4.8` |
| `backtest.max_var_multiplier` | верхний clamp адаптивного порога входа (volatility scaling) | `BasePairBacktester`/Numba kernel (`coint4/src/coint2/engine/base_engine.py`, `coint4/src/coint2/core/numba_kernels.py`) | step `0.003`, min `1.001`, max `1.055` |
| `pair_selection.max_pairs` | максимум пар на шаг WFA | slice `active_pairs[:max_pairs]` в `coint4/src/coint2/pipeline/walk_forward_orchestrator.py` | step `3`, min `12`, max `48` |
| `pair_selection.min_correlation` | нижний фильтр корреляции | pair filter (`coint4/src/coint2/pipeline/filters.py`) | step `0.025`, min `0.16`, max `0.58` |
| `pair_selection.coint_pvalue_threshold` | порог p-value коинтеграции | pair filter (`coint4/src/coint2/pipeline/filters.py`) | step `0.025`, min `0.12`, max `0.5` |
| `filter_params.min_beta` | нижняя граница `abs(beta)` для пар | фильтр beta в `coint4/src/coint2/pipeline/filters.py` | step `0.002`, min `0.0005`, max `0.028` |
| `filter_params.max_hurst_exponent` | верхний порог Hurst (mean-reversion gate) | фильтр Hurst в `coint4/src/coint2/pipeline/filters.py` | step `0.02`, min `0.55`, max `0.9` |

- Часто фиксированные рабочие параметры (baseline prod):
  - `portfolio.initial_capital=1000.0`.
  - `walk_forward.training_period_days=90`, `walk_forward.testing_period_days=15`, `walk_forward.max_steps=5`.
  - `walk_forward.pairs_file=configs/universe/pruned_v2_pairs_universe.yaml`.
  - `backtest.zscore_entry_threshold=1.15`, `backtest.zscore_exit=0.08`, `backtest.rolling_window=96`, `backtest.time_stop_multiplier=1.5`, `backtest.pair_stop_loss_zscore=3.0`.
  - Источник: `coint4/configs/prod_final_budget1000.yaml`.
- Параметры, которые по guardrail нельзя ослаблять в queue-run:
  - `walk_forward.max_steps` должен быть явно задан и `<=5` в `watch_wfa_queue.sh`.
  - `backtest.max_var_multiplier` валидируется как `>1.0` (`coint4/src/coint2/engine/base_engine.py`), в bridge11 отдельный комментарий про strict `>1.0`.

## 6) Как запускаются прогоны сейчас
- Базовый запуск одного WFA:
  - Из `coint4/`: `./.venv/bin/coint2 walk-forward --config <cfg> --results-dir <dir>`.
  - Обёртка: `bash run_wfa_fullcpu.sh <config_path> <results_dir>` (пишет `run.commands.log`, `worker.pid`).
- Запуск очереди:
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py --queue <run_queue.csv> --statuses planned,stalled --parallel <n> --postprocess true`.
  - Выключение постпроцесса (редко, для диагностики): `--postprocess false`.
  - Наблюдение/heartbeat: `bash scripts/optimization/watch_wfa_queue.sh --queue <run_queue.csv>`.
- Запуск на VPS (текущий рабочий путь):
  - `bash scripts/remote/run_server_job.sh bash scripts/optimization/watch_wfa_queue.sh --queue artifacts/wfa/aggregate/<group>/run_queue.csv`.
  - По умолчанию `STOP_AFTER=1` (авто-shutdown VPS), `SYNC_BACK=1`.
- Ralph-TUI (если используется):
  - Старт: `make ralph` или `bash coint4/scripts/ralph/run_tui_strict.sh`.
  - Tracker: `.ralph-tui/config.toml` -> `tracker=json`, `path=.ralph-tui/prd.json`.
  - Состояние сессии: `.ralph-tui/session-meta.json`, `.ralph-tui/session.json`, `.ralph-tui/iterations/*`, `.ralph-tui/progress.md`.
  - Продолжение: обычный `ralph-tui resume`; strict launcher при stale-state чистит локальные `session*.json` и запускает свежий `ralph-tui run` с тем же PRD.
- Где сохраняются артефакты прогонов:
  - Runs: `coint4/artifacts/wfa/runs/<run_group>/<run_id>/` и `coint4/artifacts/wfa/runs_clean/<run_group>/<run_id>/`.
  - Queue: `coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv`.
  - Rollup: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` (`.json`, `.md`).
- Типовая структура одного run-dir:
  - `strategy_metrics.csv`, `equity_curve.csv`, `daily_pnl.csv`, `trade_statistics.csv`, `trades_log.csv`.
  - `run.log`, `run.commands.log`, `worker.pid`.
  - Для queue-run с postprocess: `config_snapshot.yaml`, `git_commit.txt`, `canonical_metrics.json`.
- Как сравниваются прогоны:
  - Построение индекса: `coint4/scripts/optimization/build_run_index.py`.
  - Ranking: `coint4/scripts/optimization/rank_robust_runs.py` и `coint4/scripts/optimization/rank_multiwindow_robust_runs.py`.

## 7) Инфраструктура и ограничения
- Где выполняются прогоны:
  - Тяжёлые WFA/оптимизации: VPS `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh`.
  - Локальный сервер: только подготовка/постпроцесс/доки.
- Что важно для стабильности:
  - После запуска VPS должен выключаться (`STOP_AFTER=1` default в `run_server_job.sh`).
  - Для несовпадающего локального/remote кода использовать `SYNC_UP=1` (tracked files через `git ls-files | rsync`).
  - Для ручных запусков отдельный `sync_queue_status.py` обычно не нужен: `build_run_index.py` теперь авто-синхронизирует `planned/running/stalled -> completed` по наличию `strategy_metrics.csv`.
- Автозапуск/батчи/ретраи:
  - `run_wfa_queue.py`: меняет статусы `planned/stalled -> running -> completed|stalled`; при `completed` автоматически пишет `config_snapshot.yaml`, `git_commit.txt`, пересчитывает `canonical_metrics.json`.
  - `watch_wfa_queue.sh`: heartbeat, stale reset `running -> stalled`, guardrail `max_steps<=5`.
  - После завершения очереди `run_wfa_queue.py` автоматически вызывает `build_run_index.py` и обновляет `artifacts/wfa/aggregate/rollup/run_index.(csv|json|md)`.
  - `run_server_job.sh`: fallback API->SSH, best-effort shutdown при ошибках, retry sync_back для `rsync rc=11`.
  - `loop_orchestrator/orchestrate.py`: infinite loop, lease/heartbeat, exponential infra backoff (`_exp_backoff_sec`), события в SQLite.
- Known issues (факты по коду/докам):
  - Повторяющиеся предупреждения в последних run.log: нулевой буфер train/test и `pyarrow memory_map` fallback (`docs/optimization_state.md`).
  - Для legacy run-dir (до включения автопостпроцесса) `config_snapshot.yaml`/`git_commit.txt`/`canonical_metrics.json` могут отсутствовать.
  - Market microstructure в filters считает OHLCV-proxy метрики; orderbook/funding-метрики зависят от наличия отдельного кэша.

## 8) Определение “успеха” (текущая функция цели)
- Что оптимизируем сейчас (в активном batch-loop bridge11):
  - По окну: `robust_sharpe_window = min(holdout_sharpe, stress_sharpe)`.
  - По варианту: `worst_robust = min_windows(robust_sharpe_window)`.
  - Score: `score = worst_robust - max(0, worst_dd_pct - dd_target_pct) * dd_penalty`.
  - Источник: `coint4/scripts/optimization/autopilot_budget1000.py` + `coint4/configs/autopilot/budget1000_batch_loop_bridge11_20260217.yaml`.
- Ограничения (текущий цикл bridge11):
  - `min_windows=3`, `min_trades=200`, `min_pairs=20`, `max_dd_pct=0.14`, `dd_target_pct=0.09`, `dd_penalty=18.0`.
- Ограничения по затратам/риску в исполнении:
  - `commission_pct=0.0004`, `slippage_pct=0.0005`, `pair_stop_loss_usd`, `pair_stop_loss_zscore`, `portfolio_daily_stop_pct` и лимиты notional/position size в `coint4/configs/prod_final_budget1000.yaml`.
- Критерии остановки search в bridge11:
  - `max_rounds=5`, `no_improvement_rounds=2`, `min_improvement=0.02`, `require_all_knobs_before_stop=true`.
- TODO (неоднозначность критерия):
  - Нет одного глобального “канонического” objective для всех пайплайнов: в `docs/optimization_state.md` фигурирует DD-gate `<=0.15`, а в текущем bridge11 — `<=0.14` и penalty-модель.
  - Где искал: `docs/optimization_state.md`, `coint4/configs/autopilot/*.yaml`, `coint4/scripts/optimization/*rank*.py`, `coint4/scripts/optimization/autopilot_budget1000.py`.

## 9) Где смотреть правду (наблюдаемость)
- Важные логи:
  - На run уровне: `<results_dir>/run.log`, `<results_dir>/run.commands.log`, `<results_dir>/worker.pid`.
  - На queue уровне: `<queue_dir>/run_queue.log`, `<queue_dir>/run_queue.watch.log`.
  - На loop уровне: `coint4/artifacts/loop_orchestrator/orchestrator.log`, `coint4/artifacts/loop_orchestrator/orchestrator.sqlite`.
- Файлы/метаданные, которые однозначно описывают run:
  - `run_id` и `run_group` (из пути `artifacts/wfa/runs*/<run_group>/<run_id>/` и в `run_index.csv`).
  - `config_path` (из `run_queue.csv` и `run_index.csv`).
  - Метрики: `strategy_metrics.csv`; канонизация: `canonical_metrics.json` (обязательно для новых queue-run с postprocess).
  - Rollup truth-table: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`.
- Commit hash / config snapshot:
  - Для VPS sync есть `SYNCED_FROM_COMMIT.txt` (пишется `run_server_job.sh` при `SYNC_UP=1`, может быть fetched через `coint4/scripts/vps/vps_fetch_results.py`).
  - Для новых queue-run: в run-dir пишутся `config_snapshot.yaml` (точная копия конфига) и `git_commit.txt` (`git rev-parse HEAD` или `unknown` вне git).
- Как понять, что прогон завершился успешно:
  - `run_wfa_queue.py` ставит `completed`, если `returncode==0` и существует `strategy_metrics.csv`.
  - В `run_queue.csv` статус `completed` и в rollup `metrics_present=true`.
  - Для watcher-сценария финальный `DONE rc=0` в `run_queue.watch.log`.
- Примечание по legacy:
  - Для старых/ручных прогонов без queue-runner возможны неполные metadata; при следующей пересборке rollup статусы подтягиваются автоматически через `build_run_index.py`.

## 10) Список вопросов к владельцу (чтобы закрыть неизвестные)
1. Какой DD-gate считать каноническим для автоподбора сейчас: `0.14` (bridge11) или `0.15` (optimization_state)?
2. Нужна ли единая функция цели для всех скриптов (ranker/autopilot/manual), или допускаются разные по циклам?
3. Должен ли `canonical_metrics.json` быть обязательным артефактом для каждого run перед сравнением?
4. Фиксируем ли правило annualization навсегда как `365` (calendar), или нужен переход на trading-day convention для части отчётов?
5. Нужен ли обязательный `config_snapshot.yaml` в каждом run-dir (сейчас присутствует не везде)?
6. Нужно ли жёстко писать `git commit hash` в каждый run-dir (не только `SYNCED_FROM_COMMIT.txt` на уровне VPS repo)?
7. Какой минимальный набор sanity-gates обязателен для “успешного” кандидата: trades/pairs/windows сейчас стабильный или временный?
8. Для queue-run policy `max_steps<=5`: это постоянный guardrail или только для текущего бюджета/инфры?
9. Можно ли запускать full-span сценарии через тот же queue pipeline, или только отдельным контуром?
10. Нужен ли реальный источник orderbook/funding для market microstructure (сейчас в filters используются OHLCV-proxy и optional cache)?
11. Приоритет между критериями robust Sharpe и drawdown: что важнее при конфликте (особенно для final cutover)?
12. Какие ограничения по turnover/cost_ratio являются hard-gate на этапе отбора (в коде нет единого обязательного порога)?
13. Какой run-tree считать основным для анализа: `artifacts/wfa/runs/` или `artifacts/wfa/runs_clean/` при смешанных циклах?
14. Какой официальный workflow ralph: strict fresh-run launcher vs `ralph-tui resume` как основной способ продолжения?
15. Нужен ли автоматический postprocess после каждого queue-run (sync_queue_status + recompute canonical + rebuild rollup) как обязательный шаг CI/CD?
