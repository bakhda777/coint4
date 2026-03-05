# Repo Map (canonical)

Обновлено: 2026-03-05.

## 1) Канонические корни
- Git root: `/home/claudeuser/coint4`
- App root (Poetry/CLI/скрипты): `/home/claudeuser/coint4/coint4`
- Docs root: `/home/claudeuser/coint4/docs`
- Ссылка для удобства: `/home/claudeuser/coint4/coint4/docs -> ../docs`

Правило: новые рабочие данные и запусковые артефакты писать только в app-root (`/home/claudeuser/coint4/coint4/...`).

## 2) Карта по зонам ответственности

| Зона | Канонический путь | Что хранится |
|---|---|---|
| Код стратегии/движка | `coint4/src/` | runtime-код, метрики, ядро пайплайна |
| Тесты | `coint4/tests/` | unit/integration/regression |
| Конфиги | `coint4/configs/` | YAML конфиги WFA/holdout/stress/live |
| Orchestration (автономный цикл) | `coint4/scripts/optimization/` | `autonomous_wfa_driver.sh`, queue-runner, ranker, gatekeeper/auditor/agents |
| Remote/VPS orchestration | `coint4/scripts/remote/` | `run_server_job.sh`, power on/off, remote verify/run/fetch |
| Операционный отчёт 10m | `coint4/scripts/dev/autonomous_10m_report.sh` | срез состояния драйвера/очередей |
| Документация | `docs/` | state, run-logs, контракты |

## 3) Артефакты: tracked vs untracked

### Tracked (держим в Git)
- `coint4/configs/**`
- `docs/**`
- `coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv`
- `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`
- `coint4/artifacts/wfa/aggregate/rollup/run_index.json`
- `coint4/artifacts/wfa/aggregate/rollup/run_index.md`

### Untracked/тяжёлые (в Git не коммитим)
- `coint4/artifacts/wfa/runs/<run_group>/<run_id>/**`
- `coint4/artifacts/wfa/aggregate/**/logs/**`
- `coint4/outputs/**`
- runtime live-логи/снапшоты в `coint4/artifacts/live/**`

## 4) Автономный runtime state

Служебное состояние автономного контура:
- `coint4/artifacts/wfa/aggregate/.autonomous/driver.log`
- `coint4/artifacts/wfa/aggregate/.autonomous/driver_state.txt`
- `coint4/artifacts/wfa/aggregate/.autonomous/candidate.csv`
- `coint4/artifacts/wfa/aggregate/.autonomous/orphan_queues.csv`
- `coint4/artifacts/wfa/aggregate/.autonomous/decision_notes.jsonl`
- `coint4/artifacts/wfa/aggregate/.autonomous/fullspan_decision_state.json`

Это operational state; для long-term фиксации решений использовать `docs/optimization_state.md` и `docs/optimization_runs_YYYYMMDD.md`.
