# Paper-to-Code Parity Blueprint (v1)

## Цель
Формализовать контроль соответствия между исследовательской спецификацией (paper) и исполняемым кодом, чтобы выпуск в production проходил только при **100% parity** по обязательным контрольным пунктам.

## Формальное определение parity
- `Parity Control Item (PCI)` — атомарное требование с 4 полями: `paper claim`, `code contract`, `evidence`, `pass criterion`.
- `Parity Score = passed_mandatory_pci / mandatory_pci * 100`.
- Критерий готовности к release: `Parity Score = 100%`, без открытых `critical`/`high` расхождений.
- Любой PCI без проверяемого evidence считается `fail-closed` (не выполнен).

## Source of Truth
1. Paper/spec требования: `papers/` + утвержденный PRD/спека для конкретной фичи.
2. Конфигурационный контракт: `coint4/src/coint2/utils/config.py` + YAML в `coint4/configs/`.
3. Исполнение стратегии: `coint4/src/coint2/pipeline/`, `coint4/src/coint2/engine/`, `coint4/src/coint2/core/`.
4. Метрики и rollup: `coint4/src/coint2/core/sharpe.py`, `coint4/src/coint2/core/performance.py`, `coint4/src/coint2/ops/run_index.py`.
5. Артефакты запуска: `coint4/artifacts/wfa/runs/<run_group>/<run_id>/` + `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`.

## Матрица контроля parity (обязательные PCI)
| PCI | Paper claim (что должно быть) | Code contract (где реализовано) | Evidence (что проверяем) | Pass criterion |
|---|---|---|---|---|
| `PCI-01` | Без lookahead в сигналах и статистиках | `coint4/src/coint2/engine/base_engine.py`, `coint4/src/coint2/core/numba_kernels.py` | `coint4/tests/test_lookahead_bias_fix.py` | Тесты green; сигналы формируются только по историческим данным |
| `PCI-02` | Формальные фильтры отбора пар (corr, coint p-value, beta, half-life, hurst) | `coint4/src/coint2/pipeline/filters.py`, `coint4/src/coint2/utils/config.py` | `run.log` + `filter_reasons.csv`/логи фильтра | Все пары в run проходят заявленные фильтры; пороги читаются из config |
| `PCI-03` | Сигнальный контракт вход/выход по z-score и stop/time guards | `coint4/src/coint2/core/numba_kernels.py`, `coint4/src/coint2/engine/numba_engine.py` | `coint4/tests/test_numba_debug_mode.py`, `trades_log.csv` | Переходы позиций соответствуют правилам; нет невозможных состояний |
| `PCI-04` | Cost-aware PnL (commission+slippage) | `coint4/src/coint2/core/numba_kernels.py`, `coint4/src/coint2/engine/numba_engine.py` | `trade_statistics.csv`, колонки `costs/commission_costs/slippage_costs` | При наличии сделок costs > 0 и согласованы с параметрами config |
| `PCI-05` | Риск-контур портфеля (лимиты позиций/капитала) | `coint4/src/coint2/core/portfolio.py`, `coint4/src/coint2/pipeline/walk_forward_orchestrator.py` | `strategy_metrics.csv`, `trade_statistics.csv` | Лимиты не нарушены; метрики риска присутствуют |
| `PCI-06` | Walk-forward с корректными train/test шагами и ограничением queue-run | `coint4/src/coint2/pipeline/walk_forward_orchestrator.py`, `coint4/scripts/optimization/watch_wfa_queue.sh` | `run_queue.csv`, `run_queue.watch.log` | Для queue: `walk_forward.max_steps` задан явно и `<=5`; шаги валидны |
| `PCI-07` | Единая каноническая методика Sharpe/coverage/DD | `coint4/src/coint2/core/sharpe.py`, `coint4/src/coint2/core/performance.py`, `coint4/src/coint2/ops/run_index.py` | `coint4/tests/unit/test_sharpe.py`, `coint4/tests/unit/core/test_coverage_metrics.py`, `scripts/optimization/check_sharpe_consistency.py` | Raw vs recomputed Sharpe в допуске; coverage в диапазоне [0,1] |
| `PCI-08` | Постпроцесс run-артефактов и rollup обязателен | `coint4/scripts/optimization/run_wfa_queue.py`, `coint4/scripts/optimization/build_run_index.py` | `config_snapshot.yaml`, `git_commit.txt`, `canonical_metrics.json`, `run_index.csv` | Для completed runs присутствуют snapshot/commit/canonical; rollup пересобран |
| `PCI-09` | Повторяемость вычислений | `coint4/src/coint2/utils/determinism.py` | `coint4/tests/determinism/test_repeatability.py` | Повторный запуск с тем же seed дает эквивалентный результат |
| `PCI-10` | Паритет reference vs numba на контрольном наборе | `coint4/tests/test_numba_parity.py` | pytest-результат parity-теста | Тест parity green (либо зафиксирован approved waiver с датой) |

## Чеклист исполнения parity

### 1. Перед реализацией paper-гипотезы
- [ ] У каждого нового требования есть `PCI-ID` и измеримый `pass criterion`.
- [ ] Для каждого `PCI-ID` определен конкретный code owner и файл-контракт.
- [ ] Для каждого `PCI-ID` определен машинно-проверяемый evidence (test/script/artifact).

### 2. Перед запуском WFA/очереди
- [ ] Конфиг валиден через `AppConfig` (`coint4/src/coint2/utils/config.py`).
- [ ] `walk_forward.max_steps` задан явно и `<=5` для queue-run.
- [ ] Запуск идет по каноническому пути (heavy runs только на VPS `85.198.90.128`).
- [ ] Локально пройден минимум `make lint` и `make test`.

### 3. После каждого блока прогонов
- [ ] Статусы очереди синхронизированы (`sync_queue_status.py` при необходимости).
- [ ] Rollup пересобран (`build_run_index.py`).
- [ ] Для completed run присутствуют `strategy_metrics.csv`, `equity_curve.csv`, `canonical_metrics.json`, `config_snapshot.yaml`, `git_commit.txt`.
- [ ] Проверена консистентность Sharpe (`check_sharpe_consistency.py`).
- [ ] Обновлены `docs/optimization_state.md` и дневник `docs/optimization_runs_YYYYMMDD.md`.

### 4. Release gate (100% parity)
- [ ] Все обязательные `PCI-01..PCI-10` имеют статус `PASS`.
- [ ] `Parity Score = 100%`.
- [ ] Нет активных waiver старше 7 дней.
- [ ] Нет `critical/high` расхождений между paper claim и кодом/артефактами.

## Канонический набор команд проверки
Из корня репозитория:

```bash
make lint
make test
make ci
```

Из `coint4/`:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/optimization/check_sharpe_consistency.py \
  --queue artifacts/wfa/aggregate/<group>/run_queue.csv

PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py \
  --output-dir artifacts/wfa/aggregate/rollup
```

## Правило обработки расхождений
- Если любой обязательный PCI не проходит: статус релиза `blocked`.
- Разблокировка только через исправление + повторную проверку evidence.
- Временный waiver допускается только с owner, датой истечения и явным риском в `docs/optimization_state.md`.
