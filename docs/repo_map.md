# Repo Map (canonical)

Обновлено: 2026-03-07.

## 1) Канонические корни
- Git root: `/home/claudeuser/coint4`
- App root (Poetry/CLI/скрипты): `/home/claudeuser/coint4/coint4`
- Docs root: `/home/claudeuser/coint4/docs`
- Ссылка для удобства: `/home/claudeuser/coint4/coint4/docs -> ../docs`

Правило: runtime-код, очереди, rollup и operational state живут только в app-root; `docs/` хранят human-readable summaries и published snapshots.

## 2) Канонические entrypoints
- Online/autonomous runtime: `coint4/scripts/optimization/autonomous_wfa_driver.sh`
- Local -> VPS dispatch adapter: `coint4/scripts/optimization/run_wfa_queue_powered.py`
- Remote queue worker: `coint4/scripts/optimization/run_wfa_queue.py`
- Canonical postprocess/winner cycle: `coint4/scripts/optimization/run_fullspan_decision_cycle.py`
- Offline planner only: `coint4/scripts/optimization/autonomous_optimize.py`
- Manual/debug only: `coint4/scripts/optimization/watch_wfa_queue.sh`

## 3) Machine source of truth

### Metrics truth
- `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`
- Optional derived siblings:
  - `coint4/artifacts/wfa/aggregate/rollup/run_index.json`
  - `coint4/artifacts/wfa/aggregate/rollup/run_index.md`

### Promotion truth
- `coint4/artifacts/wfa/aggregate/.autonomous/fullspan_decision_state.json`

### Remote execution truth
- `coint4/artifacts/wfa/aggregate/.autonomous/remote_runtime_state.json`

### Scratch / non-canonical runtime helpers
- `coint4/artifacts/wfa/aggregate/.autonomous/candidate.csv`
- `coint4/artifacts/wfa/aggregate/.autonomous/candidate_pool.csv`
- `coint4/artifacts/wfa/aggregate/.autonomous/decision_notes.jsonl`
- `coint4/artifacts/wfa/aggregate/.autonomous/driver.log`
- `coint4/artifacts/wfa/aggregate/.autonomous/driver_state.txt`

Правило: `candidate.csv` и similar selector outputs не являются source of truth для winner/promote.

## 4) Tracked vs untracked

### Tracked (держим в Git)
- `coint4/configs/**`
- `docs/**`
- `coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv`
- `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`
- `coint4/artifacts/wfa/aggregate/rollup/run_index.json`
- `coint4/artifacts/wfa/aggregate/rollup/run_index.md`

### Operational/local state (не источник публикации)
- `coint4/artifacts/wfa/aggregate/.autonomous/**`
- `coint4/artifacts/live/**`

### Heavy/untracked (в Git не коммитим)
- `coint4/artifacts/wfa/runs/<run_group>/<run_id>/**`
- `coint4/artifacts/wfa/aggregate/**/logs/**`
- `coint4/outputs/**`

## 5) Human-readable derived files
- `docs/optimization_state.md`: текущий human summary, derived from machine state.
- `docs/optimization_runs_YYYYMMDD.md`: дневник прогонов и решений.
- `docs/best_params_latest.yaml`: published winner snapshot; не должен откатываться к `No winner yet`, если уже есть published winner.

## 6) VPS sync boundary
- Цель sync с `origin/main` и с VPS: доставить файлы, нужные для корректного runtime.
- Полное совпадение дерева `1 в 1` не требуется.
- `MANIFEST_MISMATCH` относится только к code/runtime sync drift, а не к отсутствию локальных `.autonomous` файлов на VPS.
