# Процессный контракт автономного fullspan-цикла

Дата: 2026-03-05

## 1) Цель процесса
Дойти до `PROMOTE_ELIGIBLE` только через strict fullspan-контракт и подтверждения.

Правило: до выполнения всех контрактных условий система работает в режиме `FAIL_CLOSED` и продолжает `search`.

## 2) Воронка (каноника)
Единый объект управления скоростью:

1. `generated` — все сгенерированные строки очередей.
2. `executable` — строки с валидным конфигом (или уже `running`).
3. `completed` — завершённые прогоны.
4. `strict_pass` — очереди с `strict_pass_count > 0`.
5. `confirm_ready` — strict-pass + достаточные независимые подтверждения.
6. `promote_eligible` — окончательно разрешённый promote через gatekeeper.

Источник метрик воронки: `artifacts/wfa/aggregate/.autonomous/process_slo_state.json`.

## 3) KPI процесса
Минимальный набор KPI:

- `throughput_completed_per_hour`
- `strict_pass_rate`
- `confirm_conversion_rate`
- `promote_conversion_rate`
- `lead_time_to_promote_min` (если доступен)

Они считаются автоматом агентом `process_slo_guard_agent.py`.

## 4) WIP-лимиты
WIP-лимиты задаются env и контролируются автоматом:

- `PROCESS_WIP_SEARCH_MAX` (default: `600`)
- `PROCESS_WIP_CONFIRM_MAX` (default: `6`)
- `PROCESS_STALLED_RATIO_WARN` (default: `0.60`)

При нарушениях генерируются process-alert события.

## 5) SLA
Контрольные SLA:

- `PROCESS_SLA_STRICT_PASS_SEC` (default: `21600`) — время до первого strict-pass.
- `PROCESS_SLA_CONFIRM_PENDING_SEC` (default: `SLA_CONFIRM_PENDING_SEC` или `7200`) — ожидание confirm.
- `PROCESS_SLA_NO_RUNNER_PENDING_SEC` (default: `900`) — pending при отсутствии локального runner.

## 6) Decision rights (кто может открыть promote)
- `autonomous_wfa_driver.sh` не имеет права открывать cutover/promote.
- Только `promotion_gatekeeper_agent.py` может выставить `PROMOTE_ELIGIBLE` и `ALLOW_PROMOTE`.
- `contract_auditor_agent.py` имеет право только понижать решение в `FAIL_CLOSED` при нарушении контракта.

## 7) Формат гипотезы для каждого batch
Каждый новый batch должен иметь краткую гипотезу:

- `hypothesis_id`
- `expected_gate_impact` (какой gate должен улучшиться)
- `kill_criterion` (когда прекращаем направление)
- `owner_agent`

Если в цикле не видно явной гипотезы, это считается процессным debt и подлежит авто-эскалации в отчёте.

## 8) Автоматизация процесса
Включён агент `process_slo_guard_agent.py` + systemd timer:

- service: `autonomous-process-slo-guard-agent.service`
- timer: `autonomous-process-slo-guard-agent.timer`

Артефакты:

- state: `artifacts/wfa/aggregate/.autonomous/process_slo_state.json`
- events: `artifacts/wfa/aggregate/.autonomous/process_slo_events.jsonl`
- log: `artifacts/wfa/aggregate/.autonomous/process_slo_guard.log`

## 9) Встраивание в 10m отчёт
`scripts/dev/autonomous_10m_report.sh` должен показывать:

- воронку,
- KPI,
- активные process-alerts.

Это обязательный слой наблюдаемости процесса, а не только технического состояния runner.

## 10) Surrogate runtime contract
Surrogate-слой считается корректно встроенным только если одновременно выполняется следующее:

- `gate_surrogate_state.json` существует, свежий и имеет `summary.mode=active` при наличии required inputs.
- `search_director_directive.json` существует и содержит актуальный overlay (`gate_surrogate_mode`, `gate_surrogate_ts`).
- `autonomous_wfa_driver.sh` пишет runtime evidence для `SURROGATE_REJECT`, `SURROGATE_REFINE`, `SURROGATE_ALLOW` в `driver.log` и/или `decision_notes.jsonl`.
- `fullspan_decision_state.json.runtime_metrics` содержит counters:
  - `surrogate_reject_count`
  - `surrogate_refine_count`
  - `surrogate_allow_count`

Источник истины для проверки:

- `artifacts/wfa/aggregate/.autonomous/gate_surrogate_state.json`
- `artifacts/wfa/aggregate/.autonomous/search_director_directive.json`
- `artifacts/wfa/aggregate/.autonomous/driver.log`
- `artifacts/wfa/aggregate/.autonomous/decision_notes.jsonl`
- `artifacts/wfa/aggregate/.autonomous/fullspan_decision_state.json`

Диагностика ветки surrogate через `probe_autonomous_markers.py` должна различать:

- `evidence_present` — ветка реально исполняется.
- `no_eligible_case` — сейчас нет очередей с `decision in {reject, refine}`.
- `broken_branch` — eligible очереди есть, state/directive свежие, но runtime evidence отсутствует.
- `stale_or_inconclusive` — нет права делать вывод из-за устаревшего state/directive или driver ещё не дошёл до eligible очереди в свежем окне наблюдения.

## 11) Calibration contract
Пороги `reject/refine` калибруются отдельным агентом, но не меняются произвольно:

- calibrator пишет только `surrogate_calibration_state.json`;
- `gate_surrogate_agent.py` читает его fail-safe и откатывается к safe defaults при отсутствии/невалидности state;
- minimum sample guard: `sample_size >= 100`;
- hysteresis: изменение порогов за одно применение не больше `0.05`;
- apply interval guard: не чаще одного применения за `86400` секунд.

Это означает, что ошибка calibrator не имеет права открыть promote-path или ослабить fail-closed поведение.
