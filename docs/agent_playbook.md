# Agent Playbook (coint4)

## 1) Source of truth
- Перед любым решением все агенты читают `docs/project_context.md`.
- Если `project_context` и текущие артефакты расходятся, приоритет у фактических артефактов запуска:
  - `coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv`
  - `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`
  - `coint4/artifacts/wfa/runs*/<run_group>/<run_id>/strategy_metrics.csv`

## 2) Итерационный цикл
`baseline -> batch -> ranking -> decision memo -> stop criteria`

### Step A: baseline
- Ответственный: `orchestrator` (+ `research` по запросу).
- Что делаем:
  - фиксируем baseline-конфиг и baseline objective/gates;
  - проверяем, что метрики и окна WFA не меняются внутри итерации.
- Артефакты:
  - baseline config path в `docs/optimization_state.md`;
  - ссылка на активный конфиг цикла (сейчас: `coint4/configs/autopilot/budget1000_batch_loop_bridge11_20260217.yaml`).

### Step B: batch
- Ответственный: `backtest`/`ops`.
- Что делаем:
  - генерируем и запускаем очередь прогонов;
  - выполняем queue/watch через `scripts/optimization/run_wfa_queue_powered.py`;
  - после выполнения обязательно прогоняем единый postprocess: `sync_queue_status.py -> build_run_index.py -> rank_multiwindow_robust_runs.py --fullspan-policy-v1`.
  - heavy запуск делаем только вручную через `coint4/scripts/batch/run_heavy_queue.sh` (policy: `docs/operating_mode_no_exec.md`).
- Обязательные команды:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue_powered.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv --postprocess true`
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv --dry-run --statuses planned,stalled --parallel 1`
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv`
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_multiwindow_robust_runs.py --run-index artifacts/wfa/aggregate/rollup/run_index.csv --contains <group_or_tag> --fullspan-policy-v1 --min-windows 1 --min-trades 200 --min-pairs 20 --max-dd-pct 0.50 --min-pnl 0 --initial-capital 1000 --tail-quantile 0.20 --tail-q-soft-loss-pct 0.03 --tail-worst-soft-loss-pct 0.10 --tail-q-penalty 2.0 --tail-worst-penalty 1.0 --tail-worst-gate-pct 0.20`

### Powered execution
- Для реальных запусков всегда используем:
  `PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue_powered.py`.
- Для запуска используйте только `run_wfa_queue_powered.py`; все логи смотрите по пути из `powered: log_file=...` в stderr/stdout запуска.
- В powered-run не используйте `tee`/внешние пересылки логов: внутренний лог-файл скрипта всегда создаётся автоматически и содержит диагностику.
- Пример команды запуска:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue_powered.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv --parallel 1 --postprocess true`
- Проверки после запуска:
  - `grep -R \"canonical_metrics.json\" artifacts/wfa/runs/<run_group>` (для проверенных completed run)
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
- Результирующие логи запуска:
  - `artifacts/wfa/aggregate/<group>/logs/powered_run_YYYYMMDD_HHMMSS.log`

### Run queue (canonical postprocess)
- Оба режима (`run_wfa_queue.py` и `watch_wfa_queue.sh`) считаются каноничными для queue-run.
- Запускаем queue через `scripts/optimization/run_wfa_queue.py`.
- В `run_wfa_queue.py` флаг `--postprocess` включён по-умолчанию (`true`).
- После каждого `completed run` внутри очереди (код возврата 0 и наличие `strategy_metrics.csv`) выполняется постпроцесс:
  - `config_snapshot.yaml`
  - `git_commit.txt`
  - `canonical_metrics.json`
- После завершения очереди (когда нет `planned/running`) очередь пересобирает `artifacts/wfa/aggregate/rollup/run_index.csv` через `scripts/optimization/build_run_index.py`.
- Независимо от режима запуска (watch/powered/manual) блок считается завершённым только после явной цепочки:
  - `sync_queue_status.py` для соответствующего `run_queue.csv`;
  - `build_run_index.py` для rollup;
  - `rank_multiwindow_robust_runs.py --fullspan-policy-v1` для финального verdict.
- В рамках `decision memo` отмечаем:
  - что именно запускали, какой run_group и с какими параметрами;
  - выигравшие конфиги и метрики;
  - что переносим в следующую итерацию и почему.

### Step C: ranking
- Ответственный: `research` + `risk`.
- Что делаем:
  - ранжируем по fullspan-контракту `score_fullspan_v1`;
  - проверяем risk-gates.
- Базовая команда:
  - `PYTHONPATH=coint4/src coint4/.venv/bin/python coint4/scripts/optimization/rank_multiwindow_robust_runs.py --run-index coint4/artifacts/wfa/aggregate/rollup/run_index.csv --fullspan-policy-v1 --min-windows 1 --min-trades 200 --min-pairs 20 --max-dd-pct 0.50 --min-pnl 0 --initial-capital 1000 --tail-quantile 0.20 --tail-q-soft-loss-pct 0.03 --tail-worst-soft-loss-pct 0.10 --tail-q-penalty 2.0 --tail-worst-penalty 1.0 --tail-worst-gate-pct 0.20`

### Step D: decision memo
- Ответственный: `orchestrator`.
- Что делаем:
  - фиксируем, что запускали, кто победил, почему, и что дальше.
- Куда пишем:
  - `docs/optimization_runs_YYYYMMDD.md` (дневник итераций),
  - `docs/optimization_state.md` (текущее состояние/следующий шаг).

### Step E: stop criteria
- Ответственный: `orchestrator` (+ `risk` на валидацию).
- Источник критериев остановки: `coint4/configs/autopilot/budget1000_batch_loop_bridge11_20260217.yaml`, секция `search`.
- На текущем bridge11:
  - `search.max_rounds = 5`
  - `search.no_improvement_rounds = 2`
  - `search.min_improvement = 0.02`
  - `search.require_all_knobs_before_stop = true`
  - `search.min_queue_entries = 60`
- Дополнительные gates (не для остановки цикла, а для pass кандидата):
  - `selection.min_windows = 3`
  - `selection.min_trades = 200`
  - `selection.min_pairs = 20`
  - `selection.max_dd_pct = 0.14`
  - `selection.dd_target_pct = 0.09`
  - `selection.dd_penalty = 18.0`

## 3) Decision memo template
Использовать этот шаблон в `docs/optimization_runs_YYYYMMDD.md`:

```md
## Decision Memo: <iteration_id>

Date (UTC): <YYYY-MM-DDTHH:MM:SSZ>
Owner: orchestrator
Source context: docs/project_context.md

### Baseline
- Config: <path to baseline config>
- Frozen WFA/metrics contract: <unchanged / approved change ref>

### Batch executed
- Run group(s): <group1, group2, ...>
- Queue file(s): <path(s)>
- Host: <hostname/ip>
- ALLOW_HEAVY_RUN: <1|0|n/a>
- Result: <success|fail>
- Commands:
  - `<command 1>`
  - `<command 2>`
- Execution notes: <short>

### Postprocess
- sync_queue_status: <done yes/no + command>
- build_run_index: <done yes/no + command>
- ranker_fullspan_v1: <done yes/no + command>
- Source of truth: `artifacts/wfa/aggregate/rollup/run_index.csv`

### Ranking summary
- Candidate 1: <variant/run_id> | worst_robust_sh=<...> | worst_dd=<...> | windows=<...> | trades_min=<...> | pairs_min=<...>
- Candidate 2: ...
- Candidate 3: ...

### Risk verdict
- PASS/FAIL per candidate with one-line reason.

### Decision
- Winner: <variant/run_id>
- Why: <2-4 bullets>
- Promote to baseline next round: <yes/no + path>

### Next actions
1. <concrete next step>
2. <concrete next step>
3. <optional>

### Stop criteria check
- max_rounds reached: <yes/no>
- no_improvement_rounds reached: <yes/no>
- min_improvement satisfied: <yes/no>
- require_all_knobs_before_stop satisfied: <yes/no>
```

## 4) Invariants
- Нельзя менять определение метрик (Sharpe/DD/PnL) и окна WFA внутри активной итерации без отдельного решения владельца.
- Нельзя логировать/печатать секреты, ключи, токены.
- Heavy artifacts в `coint4/artifacts/wfa/runs*/` не коммитятся.

## 5) Autonomous Autopilot (systemd)
- Один автономный цикл запускается скриптом:
  - `coint4/scripts/optimization/autonomous_optimize.py`
- Цикл: `batch -> powered run -> rollup -> rank -> next batch`, до stop-criteria.
- Итоговые артефакты:
  - `docs/best_params_latest.yaml`
  - `docs/final_report_latest.md`
- Состояние и основной лог:
  - `coint4/artifacts/optimization_state/autonomous_state.json`
  - `coint4/artifacts/optimization_state/autonomous_service.log`
- Проверка статуса systemd:
  - `systemctl status coint4-autopilot.timer`
  - `systemctl status coint4-autopilot.service`
  - `journalctl -u coint4-autopilot.service -n 200 --no-pager`
