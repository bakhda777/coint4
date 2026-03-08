# Agent Playbook (coint4)

## 1) Source of truth
- Перед любым решением читать `docs/project_context.md` и текущие machine artifacts.
- Если документация и runtime расходятся, приоритет у machine truth:
  - metrics truth: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`
  - promotion truth: `coint4/artifacts/wfa/aggregate/.autonomous/fullspan_decision_state.json`
  - remote execution truth: `coint4/artifacts/wfa/aggregate/.autonomous/remote_runtime_state.json`
- `docs/optimization_state.md` и `docs/best_params_latest.yaml` являются производными snapshot-файлами.
- `coint4/artifacts/wfa/aggregate/.autonomous/candidate.csv` это scratchpad селектора, а не source of truth.

## 2) Канонические entrypoints
- Online runtime: `make loop` -> `coint4/scripts/optimization/autonomous_wfa_driver.sh`
- Planner-only path: `make loop-plan` -> `coint4/scripts/optimization/autonomous_optimize.py`
- Local -> VPS launch adapter: `coint4/scripts/optimization/run_wfa_queue_powered.py`
- Remote worker: `coint4/scripts/optimization/run_wfa_queue.py`
- Canonical postprocess / winner evaluation: `coint4/scripts/optimization/run_fullspan_decision_cycle.py`
- Manual/debug only: `coint4/scripts/optimization/watch_wfa_queue.sh`

## 3) Итерационный цикл
`plan -> queue dispatch -> sync status -> rebuild rollup -> fullspan decision cycle -> decision memo`

### Step A: planning
- Ответственный: `orchestrator` / `research`.
- Planner генерирует следующий batch и handoff, но не считается каноническим online runtime.
- Для planner-only режима использовать `make loop-plan`.

### Step B: queue execution
- Ответственный: `ops`.
- Канонический запуск очереди идёт через `autonomous_wfa_driver.sh` или `run_wfa_queue_powered.py`.
- Тяжёлые WFA/optimization runs выполнять только на `85.198.90.128`.
- `watch_wfa_queue.sh` оставлен для manual/debug сценариев и исторической совместимости.

Базовая команда powered-run:
- `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue_powered.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv --postprocess true`

### Step C: canonical postprocess
- После выполнения очереди canonical chain всегда одна:
  - `sync_queue_status.py`
  - `build_run_index.py`
  - `run_fullspan_decision_cycle.py`
- Финальный winner/evidence определяется только из `run_index.csv` через `fullspan_v1`.
- `rank_multiwindow_robust_runs.py` остаётся diagnostic/research инструментом и не является final promotion gate.

Базовая команда:
- `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_fullspan_decision_cycle.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv --contains <group_or_tag>`

### Step D: decision memo
- Ответственный: `orchestrator`.
- После каждого блока прогонов обновлять:
  - `docs/optimization_state.md`
  - `docs/optimization_runs_YYYYMMDD.md`

## 4) Winner contract
- Final promotion / live cutover определяется только `fullspan_v1`.
- Канонические strict defaults:
  - `min_windows=1`
  - `min_trades=200`
  - `min_pairs=20`
  - `max_dd_pct=0.20`
  - `min_pnl=0`
  - `tail_worst_gate_pct=0.20`
- Primary ranking key в fullspan-режиме: `score_fullspan_v1`.
- `avg_robust_sharpe` допустим только как diagnostic key.
- `PROMOTE_ELIGIBLE` и `ALLOW_PROMOTE` выставляет только `promotion_gatekeeper_agent.py`.

## 5) Invariants
- Нельзя менять определение метрик и окна WFA внутри активного цикла без отдельного решения.
- Нельзя логировать секреты, ключи, токены.
- Heavy artifacts в `coint4/artifacts/wfa/runs/**` не коммитятся.
- `docs/best_params_latest.yaml` не должен откатываться к placeholder после публикации winner.

## 6) Planner role
- `autonomous_optimize.py` это planner-only модуль для offline batch planning и supervised handoff.
- По умолчанию planner идёт single-pass; repeated in-process режим нужен только при явном `--until-done` или `make loop-plan LOOP_PLAN_REPEAT=1`.
- Если нужен online autonomous runtime, использовать только `make loop`.
