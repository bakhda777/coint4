# PRD 07: Optimization Params Closed Loop v1 (runs_clean -> Codex -> WFA batches)

## Overview
Цель: найти максимально устойчивые параметры для live, используя последние clean результаты.

Objective:
- maximize worst-window robust Sharpe (как в `scripts/optimization/rank_multiwindow_robust_runs.py`)

Risk gates:
- `worst_dd_pct <= 0.14`
- `min_windows=3`
- `min_trades=200`
- `min_pairs=20`

Knobs (LLM может менять):
- `portfolio.risk_per_position_pct`
- `backtest.pair_stop_loss_usd`
- `backtest.max_var_multiplier`
- `pair_selection.min_correlation`
- `pair_selection.coint_pvalue_threshold`
- `pair_selection.max_pairs`
- `filter_params.min_beta`
- `filter_params.max_hurst_exponent`

WFA policy:
- 3 OOS окна
- батч = 20 параметр-сетов (60 queue entries)
- stop criterion: только LLM decision `next_action=stop`
- тяжёлые прогоны: только VPS `85.198.90.128` с auto poweroff

## Quality Gates
Для каждого цикла/задачи:
- `make lint`
- `make test`
- пересборка rollup
- rank для проверки лучшего кандидата

## User Stories

### US-LOOP-001: Preflight: доступы, безопасность, quality gates
**Description:** Как оператор, я хочу убедиться, что loop может безопасно запускать тяжёлые WFA удалённо и локально выполняются только лёгкие шаги.

**Acceptance Criteria:**
- Подтверждено: тяжёлые WFA/оптимизации выполняются только на VPS `85.198.90.128`.
- Доступ к VPS без пароля работает: `ssh root@85.198.90.128 'echo ok'`.
- Ключ Serverspace доступен безопасно: env `SERVSPACE_API_KEY` или `.secrets/serverspace_api_key`.
- Из корня репо проходят quality gates: `make lint` и `make test`.

### US-LOOP-002: Baseline: rebuild rollup + снимок топ clean кандидатов
**Description:** Как аналитик, я хочу пересобрать rollup (включая runs_clean) и зафиксировать baseline перед запуском нового loop.

**Acceptance Criteria:**
- Выполнен rebuild rollup:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
- Снят baseline top:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_multiwindow_robust_runs.py --top 10 --max-dd-pct 0.14 --min-windows 3 --min-trades 200 --min-pairs 20 --contains budget1000 --include-noncompleted`
- Baseline записан в `docs/optimization_runs_20260219.md`.

### US-LOOP-003: Запуск closed-loop оптимизации до stop от LLM
**Description:** Как оператор, я хочу запустить цикл: анализ clean -> новый батч -> удалённый прогон -> анализ -> ... пока LLM не скажет stop.

**Acceptance Criteria:**
- Loop запущен:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/autonomous_optimize.py --until-done --use-codex-exec`
- В logs есть маркеры удалённого исполнения (`run_wfa_queue_powered.py`, `compute-host=85.198.90.128`, `--poweroff true`, `--wait-completion true`).
- Loop завершён только по решению LLM (`next_action=stop` + `stop_reason`).
- После stop `docs/best_params_latest.yaml` больше не содержит placeholder `# No winner yet`.
- `docs/final_report_latest.md` содержит `Status: done`, winner и score.
- VPS выключен после завершения.

### US-LOOP-004: Фиксация результата: docs + экспорт winner для live
**Description:** Как владелец системы, я хочу получить финальные оптимальные параметры и понятную документацию, что именно выигрывает и почему.

**Acceptance Criteria:**
- Обновлён `docs/optimization_state.md` (дата, stop_reason, winner, ссылки на best params/report).
- Обновлён `docs/optimization_runs_20260219.md` (baseline, run_groups, финальный winner).
- Создан кандидат прод-конфига: `coint4/configs/prod_final_budget1000_bestparams_20260219.yaml`.
- Подтверждено, что тяжёлые артефакты из `coint4/artifacts/wfa/runs/**` не добавлены в git.
