# Optimization state

Last updated: 2026-02-24

## Rollup snapshot (source of truth)

Rollup индекс: `coint4/artifacts/wfa/aggregate/rollup/run_index.{csv,json,md}`.
Текущий снимок rollup: `2026-02-24 06:33:55Z` (из `coint4/artifacts/wfa/aggregate/rollup/run_index.md`).

Текущий baseline (alt-holdout top50 sens, rollup):
- `run_id=holdout_relaxed8_nokpss_20260123_alt20220901_20230430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1` (`run_group=20260123_relaxed8_nokpss_u250_churnfix_alt_top50_sens`)
- Sharpe `8.6494`, |DD| `0.0129`, trades `3666` (из `sharpe_ratio_abs`, `max_drawdown_on_equity`, `total_trades`)
- Run artifacts: `coint4/artifacts/wfa/runs/20260123_relaxed8_nokpss_u250_churnfix_alt_top50_sens/holdout_relaxed8_nokpss_20260123_alt20220901_20230430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1`

Current stage: **US-LOOP-004 (docs + live export) закрыт + выполнены VPS confirm-replay top-10 BL11 и fullspan replay top-3 BL11**: `decision_id=us-loop-003-stop-20260220T0149Z-infra-block`, `stop_reason=INFRA_BLOCKED_SANDBOX_NETWORK: codex backend and serverspace api unreachable; powered runner repeats RC4`. Live winner для cutover пока остаётся baseline `run_group=20260213_budget1000_dd_sprint08_stoplossusd_micro` (`run_id=holdout_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91`, `score=3.530254`, `worst_dd_pct=0.132205`) до отдельного решения по промоуту BL11.

**Prod config лидер**: `pruned_v2` (168 пар, universe: `coint4/configs/universe/pruned_v2_pairs_universe.yaml`), full-span holdout Sharpe **2.24**, stress **1.83**. Max DD -53.0% (было -83.1%). Все 3 OOS-окна прибыльны.

**Denylisted symbols**: AFCUSDT, CITYUSDT, ERTHAUSDT, FLOWUSDT, GALFTUSDT, HFTUSDC, HFTUSDT, INTERUSDT, IZIUSDT, JUVUSDT, KDAUSDT

**Operating mode (обязательно)**: `OPERATING MODE: CODE-ONLY / NO-EXEC` (`docs/operating_mode_no_exec.md`).
- Heavy execution разрешён только вручную и только при прохождении guardrails (`ALLOW_HEAVY_RUN=1`, hostname allowlist, RAM/CPU пороги).
- Канонический ручной runner: `coint4/scripts/batch/run_heavy_queue.sh`.

## Gates / Stop condition

Короткая памятка “что считать годным кандидатом и когда останавливаемся”. Метрики берём из rollup `coint4/artifacts/wfa/aggregate/rollup/run_index.*` (и, для tail-risk, из run-артефактов), тяжёлое исполнение — только на VPS `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh`.

Единый prefilter-gate (использует менеджер при выборе top-кандидатов из rollup): `tools/sprint_manager.py` (`CANDIDATE_GATE_V1`):
- trades: `total_trades >= 200`;
- DD: `abs(max_drawdown_on_equity) <= 0.15`;
- tail proxy: `tail_loss_worst_pair_pnl >= -200` и `tail_loss_worst_period_pnl >= -200` (где `200 = 0.20 * initial_capital`, `initial_capital=1000`);
- fail-closed: если tail-метрик нет/они невалидны — кандидат отклоняется.

**Gates (качество кандидата, fail-closed):**
- Минимум сделок/ширина:
  - shortlist (short OOS): `total_trades >= 10`, `total_pairs_traded >= 1`;
  - promote (fullspan v1): `total_trades >= 200`, `total_pairs_traded >= 20`.
- Ограничение просадки:
  - shortlist (budget=$1000): `max_drawdown_on_equity <= 0.15`;
  - promote (fullspan v1): `worst_dd_pct <= 0.50`.
- Tail-loss (robust step PnL, budget=$1000):
  - hard gate: `worst_step_pnl >= -200` (то есть `>= -0.20 * initial_capital`);
  - rollup proxy (для prefilter в `tools/sprint_manager.py:CANDIDATE_GATE_V1`): `tail_loss_worst_pair_pnl >= -200` и `tail_loss_worst_period_pnl >= -200`.
  - soft thresholds (penalty в `score_fullspan_v1`): `q20_step_pnl >= -30` и `worst_step_pnl >= -100`.
- Экономика/валидность (для promote):
  - `metrics_present=true` для holdout+stress;
  - `worst_robust_pnl >= 0`.

**Подтверждения (перед promote/cutover):**
- Не принимать финальное решение по одному `run_group` или только по short OOS.
- Кандидат считается подтверждённым, если strict-pass по `Fullspan selection policy v1` получен минимум в **2 независимых** `run_group` и есть хотя бы один `confirm-replay/fullspan replay` прогон на VPS.

**Stop condition (когда прекращаем текущий цикл/ветку поиска):**
- Немедленно отклонять кандидата при провале любого hard gate (trades/ширина/|DD|/tail-loss/metrics_present/economic).
- Завершать текущий search-space (переход к новому `rXX`), если **2 итерации подряд** не дают улучшения top-метрик (`score_fullspan_v1` или `worst_robust_sharpe`/`worst_robust_sh`) относительно baseline, либо “улучшения” достигаются только через узкий режим, который нарушает gates ширины.
- Завершать оптимизационный цикл (готовить decision memo/cutover) только при strict-pass по fullspan v1 + подтверждениям выше.

## FS-009 update (2026-02-22, tailguard_r02 final)

- Heavy queue `coint4/artifacts/wfa/aggregate/20260222_tailguard_r02/run_queue.csv` добежала полностью на VPS `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh`:
  - итог очереди: `completed=32/32`;
  - результаты синхронизированы обратно (`SYNC_BACK=1`);
  - VPS выключен (проверка: SSH timeout после завершения).
- Канонический postprocess выполнен:
  1. `sync_queue_status.py` -> `metrics_present=32, missing=0`;
  2. `build_run_index.py --output-dir artifacts/wfa/aggregate/rollup --no-auto-sync-status` -> `Run index entries: 8119`;
  3. strict/diagnostic rank (`rank_multiwindow_robust_runs.py`, `--fullspan-policy-v1`, `--min-windows 1`) по `--contains 20260222_tailguard_r02`.
- Strict (`tail_worst_gate_pct=0.20`) и diagnostic (`0.21`) дали одинаковый top-1:
  - `variant_id=tailguard_risk0p0055_risk0p0055_slusd1p81_max_var_multiplier1p0065_mp21_corr0p335_pv0p39`
  - `score=1.043`, `worst_robust_sh=1.142`, `worst_dd_pct=0.291`, `worst_pnl=1147.93`, `worst_step_pnl=-198.90`.
- Сравнение с `20260222_tailguard_r01`: улучшения нет (метрики top-1 совпали).
- Вердикт цикла: `r02` зафиксирован как завершённый без прогресса к цели `Sharpe>3`; следующий шаг — `r03` с новым search-space (не локальный fine-tune tailguard вокруг текущего контроля).

### FS-009 / r03 contract (start)

- Режим: `optimization-only` (live/cutover не выполняем до явного достижения целевых метрик).
- Baseline для сравнения в `r03`:
  - `variant_id=tailguard_risk0p0055_risk0p0055_slusd1p81_max_var_multiplier1p0065_mp21_corr0p335_pv0p39`
  - `score=1.043`, `worst_robust_sh=1.142`, `worst_dd_pct=0.291`, `worst_pnl=1147.93`, `worst_step_pnl=-198.90`
- Целевые условия pass:
  - `worst_robust_sh >= 3.0`
  - `worst_pnl > 0`
  - `worst_dd_pct <= 0.35` (research ceiling: `0.50`)
  - `worst_step_pnl >= -200`
  - `metrics_present=true` для holdout+stress.
- Артефакты `r03`:
  - `coint4/artifacts/wfa/aggregate/20260222_tailguard_r03/search_space.csv`
  - `coint4/artifacts/wfa/aggregate/20260222_tailguard_r03/search_space.md`

## FS-009 update (2026-02-23, tailguard_r03 final)

- Heavy queue `coint4/artifacts/wfa/aggregate/20260222_tailguard_r03/run_queue.csv` завершена на VPS `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh`:
  - итог очереди: `completed=48/48`;
  - результаты синхронизированы обратно (`SYNC_BACK=1`);
  - VPS выключен (проверка: `ssh root@85.198.90.128` -> timeout).
- Канонический postprocess выполнен:
  1. `sync_queue_status.py --queue artifacts/wfa/aggregate/20260222_tailguard_r03/run_queue.csv` -> `metrics_present=48, missing=0`.
  2. `build_run_index.py --output-dir artifacts/wfa/aggregate/rollup --no-auto-sync-status` -> `Run index entries: 8167`.
  3. strict/diagnostic rank (`rank_multiwindow_robust_runs.py`, `--fullspan-policy-v1`, `--min-windows 1`) по `--contains 20260222_tailguard_r03`.
- Strict (`tail_worst_gate_pct=0.20`) и diagnostic (`0.21`) совпали:
  - `pass_count=2` для обоих профилей;
  - top-1: `tailguard_risk0p0055_risk0p0055_slusd1p81_max_var_multiplier1p0065_mp21_corr0p335_pv0p39_risk0p0055_slusd1p81_max_var_multiplier1p0065_mp21_corr0p335_pv0p39_z1p15_exit0p08_dstop0p02_maxpos18`
  - `score=1.043`, `worst_robust_sh=1.142`, `worst_dd_pct=0.291`, `worst_pnl=1147.93`, `worst_step_pnl=-198.90`.
- Сравнение `r03` vs `r02`/`r01`: улучшения нет (ключевые top-метрики совпадают), `gap` до цели `Sharpe=3.0` остаётся `+1.858`.
- Вердикт цикла: `r03` зафиксирован как `completed-without-improvement`; следующий шаг — цикл `r04` с новым search-space (выход из плато `1.142`).

## FS-009 update (2026-02-23, tailguard_r04 final)

- Heavy queue `coint4/artifacts/wfa/aggregate/20260223_tailguard_r04/run_queue.csv` завершена на VPS `85.198.90.128` (через `coint4/scripts/batch/run_heavy_queue.sh`, внутри — `coint4/scripts/remote/run_server_job.sh`):
  - итог очереди: `completed=26/26`;
  - результаты синхронизированы обратно (`SYNC_BACK=1`);
  - VPS выключен (проверка: `ssh root@85.198.90.128` -> timeout).
- Канонический postprocess выполнен:
  1. `sync_queue_status.py --queue artifacts/wfa/aggregate/20260223_tailguard_r04/run_queue.csv` -> `metrics_present=26, missing=0`.
  2. `build_run_index.py --output-dir artifacts/wfa/aggregate/rollup` -> `Run index entries: 8193`.
  3. ranking (fullspan single-window): `rank_multiwindow_robust_runs.py --contains 20260223_tailguard_r04 --fullspan-policy-v1 --min-windows 1 --min-trades 200 --min-pairs 20`.
- Итог ranking (`min_pairs>=20`, `min_trades>=200`, `min_pnl>=0`):
  - top-1 (единственный pass в r04): `tailguard_r04_v07_h2_quality_mild`
  - `worst_robust_sh=0.608`, `worst_dd_pct=0.190`, `worst_pnl=261.94`, `worst_step_pnl=-30.39`, `q20_step_pnl=-2.16`
- Наблюдения по гипотезам r04:
  - H1 tradeability (v01–v06) почти не сдвигает stress: мягкие пороги прижаты guardrail, а “выбитые” пары часто не торговали в stress.
  - H2 quality mild (v07) дал первый “честный” прирост robust относительно контроля r04 (stress стал положительным) при сохранении ширины (≈21 пары и тысячи сделок).
  - v08/v10 (med/kpss-only): Sharpe высокий, но это “узкий” режим (2 пары, 64 сделки) и он должен отклоняться гейтами ширины.
  - H3 stability (v11–v13) ухудшает robust и/или режет ширину (в r04 не подходит как направление).
- Исправление метрик против “декоративного Sharpe”:
  - В WFA окна теста, где не отобралось ни одной пары, раньше не попадали в `daily_pnl.csv`/`equity_curve.csv`, что завышало Sharpe/volatility.
  - Исправлено: теперь такие окна записываются как нулевой PnL (см. `coint4/src/coint2/pipeline/walk_forward_orchestrator.py`).

Следующий шаг (fast-loop по просадке):
- Для baseline `tailguard_r04_v07_h2_quality_mild` max-DD по robust daily PnL пришёлся на период **2023-09-27 → 2024-05-28** (peak→trough).
- Рекомендуемый dd-focus диапазон для ускоренной итерации WFA: **2023-06-29 → 2024-06-27** (padding: -90d до peak и +30d после trough).
- Дальше: собрать `r05_ddfocus` (комбинации вокруг v07: tradeability+quality, без трогания risk/stop/z/dstop/maxpos) и только потом подтверждать winners на fullspan.

## Рабочий цикл fullspan v1 (обязательный для каждого блока)

Цель:
- Закрывать каждый блок прогонов одним и тем же циклом: queue execution -> postprocess -> fullspan ranking -> decision memo.
- Исключить неоднозначность между short OOS shortlist и финальным fullspan promote.

Текущий статус:
- Fullspan replay top-3 BL11 завершён и перепроверен на свежем каноническом rollup (`run_index`, 8077 записей).
- Source of truth snapshot (local, 2026-02-22T17:42:16Z): `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` содержит `8077` записей.
- Расхождение с VPS-снимком (`7927`) трактуется как срез другой рабочей копии/момента синхронизации; для локального решения и документации каноническим считается локальный `run_index` в репозитории.
- Strict `fullspan_v1` (`--min-pnl 0`, `--tail-worst-gate-pct 0.20`) дал `No variants matched`:
  - `..._slusd1p81` отклонён по hard gate `worst_step_pnl=-209.15 < -200`.
  - `..._pv0p365` и `..._max_pairs24p0` отклонены по economic gate (`worst_robust_pnl < 0`), у `..._max_pairs24p0` дополнительно провален worst-step gate.
- Policy decision (2026-02-22): строгий gate `tail_worst_gate_pct=0.20` остаётся обязательным для final promote; `0.21` допустим только в diagnostic/research режиме.
- Baseline lock (2026-02-22): до первого strict-pass по `fullspan_v1` baseline winner остаётся `20260213_budget1000_dd_sprint08_stoplossusd_micro/holdout_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91`; promote в live запрещён (fail-closed).
- Для bridge11 controller-профиля временно отключён DSR-gate (`selection.min_dsr: null` в `coint4/configs/autopilot/budget1000_batch_loop_bridge11_20260217.yaml`) до согласованной нормализации DSR.
- План возврата DSR-gate:
  - нормализовать интерпретацию DSR (или привести к шкале вероятности, или зафиксировать иной канонический порог и единицы);
  - добавить регрессионный тест ранжирования с `min_dsr` на synthetic run_index;
  - вернуть `selection.min_dsr` в bridge-профиль только после подтверждения на confirm/fullspan-блоке без ложных reject.
- Контракт postprocess восстановлен в коде:
  - `scripts/optimization/run_wfa_queue.py` снова поддерживает `--postprocess` и `--rollup-*`;
  - `scripts/optimization/build_run_index.py` синхронизирует queue-статусы по метрикам (с опцией `--no-auto-sync-status`).
- Добавлен единый оркестратор цикла: `scripts/optimization/run_fullspan_decision_cycle.py`.
- Финальный promote в live не выполнялся автоматически (режим fail-closed сохранён, `selection_mode=shortlist_only`).

Обязательный postprocess (после каждого блока прогонов):
1. `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/<run_group>/run_queue.csv`
2. `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup --no-auto-sync-status`
3. `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_multiwindow_robust_runs.py --run-index artifacts/wfa/aggregate/rollup/run_index.csv --contains <run_group_or_tag> --fullspan-policy-v1 --min-windows 1 --min-trades 200 --min-pairs 20 --max-dd-pct 0.50 --min-pnl 0 --min-coverage-ratio 0.95 --initial-capital 1000 --tail-quantile 0.20 --tail-q-soft-loss-pct 0.03 --tail-worst-soft-loss-pct 0.10 --tail-q-penalty 2.0 --tail-worst-penalty 1.0 --tail-worst-gate-pct 0.20`

Важно (после фикса WFA “нулевых окон”, 2026-02-23):
- Sharpe/volatility/tail-метрики могут заметно измениться, потому что окна без отобранных пар теперь записываются как 0-PnL на весь testing_slice. Сравнение run'ов “до/после” невалидно без перезапуска baseline под новым кодом.
- Для fullspan single-window прогонов использовать `--min-windows 1` (иначе ранкер будет ожидать OOS-теги `_oosYYYYMMDD_YYYYMMDD` и отбрасывать).
- Добавлены coverage-метрики в `strategy_metrics.csv` и rollup `run_index.csv`: `expected_test_days`, `observed_test_days`, `coverage_ratio`, `zero_pnl_days`, `zero_pnl_days_pct`, `missing_test_days`.
- В ранкере `rank_multiwindow_robust_runs.py` добавлен coverage-gate `--min-coverage-ratio` (default: `0.95`, fail-closed на missing/non-finite), чтобы “разреженные” прогоны не проскальзывали с декоративным Sharpe.

Канонические gate-пороги отбора (fullspan v1):
- `min_windows >= 1`
- `min_trades >= 200`
- `min_pairs >= 20`
- `max_dd_pct <= 0.50`
- `worst_robust_pnl >= 0`
- `worst_step_pnl >= -0.20 * initial_capital` (для `$1000`: `>= -200`)

Следующие шаги:
1. Для каждого нового run_group фиксировать в дневнике `selection_policy=fullspan_v1`, `selection_mode`, `promotion_verdict`, `rejection_reason` (если есть).
2. Не продвигать winner из short OOS без отдельного fullspan replay по команде выше.
3. Поддерживать единый формат отчётности в `docs/optimization_runs_YYYYMMDD.md` и `docs/optimization_state.md`.
4. Референс перехода на новый шаблон отчётности: `docs/optimization_runs_20260222.md`.

### Update: NO-EXEC rails + tailguard execution attempt (2026-02-22 UTC)

- Введён единый guardrail-контракт (`coint4/src/coint2/ops/heavy_guardrails.py`) и подключён к heavy entrypoints (`run_wfa_queue.py`, `watch_wfa_queue.sh`, `run_wfa_fullcpu.sh`, optuna CLI/UI entrypoints).
- Добавлен канонический ручной runner: `coint4/scripts/batch/run_heavy_queue.sh`.
- Попытка запуска `20260222_tailguard_r01`:
  - `runner=watch` корректно отклонён по контракту `max_steps<=5` (конфиги fullspan имеют `max_steps=null`);
  - `runner=queue` упёрся в infra-блокер `SSH not ready after 15 minutes` (`run_server_job.sh`), VPS отправлен в shutdown.
- Текущее состояние очереди `20260222_tailguard_r01`: `planned=10`, `completed=0`; postprocess/promotion для блока не выполнялись.

## Критерий shortlist для short OOS (не финальный promote, budget=$1000)

Этот блок применяется только для pre-filter/shortlist в коротких OOS-циклах. Финальный выбор winner для live/cutover выполняется только по контракту из раздела `Fullspan selection policy v1`.

Source of truth для ранжирования: `canonical_metrics.json` (пересчитан из `equity_curve.csv`) и/или rollup `run_index.csv` (поле `sharpe_ratio_abs` уже канонизировано через пересчёт из `equity_curve.csv`; детали: `docs/sharpe_audit.md`).

Objective (robust Sharpe):
- Для одного окна: `robust_sharpe = min(sharpe_ratio_abs_holdout, sharpe_ratio_abs_stress)`.
- Для multi-window: `robust_sharpe = min(robust_sharpe_window_i)` (worst-window robust Sharpe).

DD-gate для $1000 (short OOS shortlist):
- Проходит только если `max_drawdown_on_equity <= 0.15` (то есть worst DD не хуже -15%), эквивалентно `max_drawdown_abs >= -150` при `initial_capital=1000`.
- Этот порог не переопределяет fullspan hard-gate `worst_dd_pct <= 0.50`; для промоута действует только fullspan-контракт ниже.

Sanity-gates (анти no-op, минимальные):
- `total_trades >= 10` и (если метрика присутствует) `total_pairs_traded >= 1`.
- `equity_curve.csv` должен содержать минимум 2 точки (иначе Sharpe/DD по curve не определены, а `sharpe_ratio_abs` в канонизации может стать 0).

## Fullspan selection policy v1

Канонический источник: `docs/fullspan_selection_policy.md` (этот раздел в state — оперативная выжимка для ежедневной работы).

Область действия:
- Этот контракт обязателен для выбора победителя в fullspan-контуре (`walk_forward.max_steps=null` или эквивалентный полный диапазон дат по данным).
- Short OOS (например, `max_steps<=5`) используется только как shortlist/pre-filter и не может быть финальным критерием промоута.
- Любая неоднозначность между short OOS и fullspan трактуется fail-closed в пользу fullspan-контракта.

Source of truth:
- `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` (канонизированные Sharpe/DD/PnL).
- Перешаговая диагностика tail-risk из run-артефактов (`daily_pnl.csv` holdout/stress); для шага используется `step_robust_pnl = min(step_pnl_holdout, step_pnl_stress)`.

Objective (формально, v1):
- Базовый сигнал: `worst_robust_sharpe = min_window(min(sharpe_holdout, sharpe_stress))`.
- Tail-компоненты:
  - `worst_step_pnl = min(step_robust_pnl)`;
  - `q20_step_pnl = quantile(step_robust_pnl, 0.20)`.
- Итоговый score:
  - `score_fullspan_v1 = worst_robust_sharpe - 2.0 * max(0, (-q20_step_pnl / initial_capital) - 0.03) - 1.0 * max(0, (-worst_step_pnl / initial_capital) - 0.10)`.
- Tie-break: `worst_robust_pnl` (выше лучше) -> `worst_dd_pct` (ниже лучше) -> `avg_robust_sharpe` (выше лучше).

Hard-gates (обязательные):
- DD gate: `worst_dd_pct = max_window(max(abs(dd_holdout), abs(dd_stress))) <= 0.50`.
- Worst-step gate: `worst_step_pnl >= -0.20 * initial_capital` (для `$1000`: не хуже `-200` на худшем шаге).
- Economic gate: `worst_robust_pnl >= 0`.
- Sanity gate: `metrics_present=true` для holdout+stress; `total_trades>=200`; `total_pairs_traded>=20`.

Запрет (явный):
- Нельзя выбирать winner только по `avg_sharpe`/`avg_robust_sharpe` на коротком OOS.
- Любой кандидат с "хорошим средним Sharpe" и провалом `worst_step_pnl`/`worst_robust_pnl`/`worst_dd_pct` должен быть отклонён.

Статистические метрики устойчивости:
- Нормативно: использовать `PSR` и `DSR` как обязательные diagnostics в ранжировании fullspan.
- Временный fallback (safe default v1 до добавления PSR/DSR в rollup): `stats_mode=fallback_no_psr_dsr`, решение валидно только при документированной причине `PSR_DSR_UNAVAILABLE_V1` и прохождении всех hard-gates выше.
- Если одновременно недоступны `PSR/DSR` и tail-step данные, применяется fail-closed: только `selection_mode=shortlist_only`, `promotion_verdict=reject`, `rejection_reason=TAIL_DATA_MISSING_OR_PSR_DSR_UNAVAILABLE`.

Recent updates (2026-02-20):

### US-LOOP-004 finalization (docs + winner export for live)
- Финальный stop зафиксирован как fail-closed по валидному LLM-решению: `decision_id=us-loop-003-stop-20260220T0149Z-infra-block`.
- Stop reason: `INFRA_BLOCKED_SANDBOX_NETWORK: codex backend and serverspace api unreachable; powered runner repeats RC4`.
- Winner для cutover/live зафиксирован без изменений относительно baseline US-LOOP-002:
  - `run_group=20260213_budget1000_dd_sprint08_stoplossusd_micro`
  - `run_id=holdout_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91` (paired stress: `stress_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91`)
  - `score=3.530254` (`worst_robust_sharpe`), `worst_dd_pct=0.132205`.
- Почему winner: лучший `worst_robust_sharpe` при выполнении gate (`max_dd_pct<=0.14`, `min_windows=3`, `min_trades>=200`, `min_pairs>=20`), tie-break сохранён за baseline sprint08 для continuity.
- Источники для оператора:
  - `docs/best_params_latest.yaml`
  - `docs/final_report_latest.md`
  - `docs/optimization_runs_20260219.md`
- Экспортирован кандидат prod-конфига для live: `coint4/configs/prod_final_budget1000_bestparams_20260219.yaml`.

### Baseline refresh before next loop
- Пересобран canonical rollup:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
  - `run_index` после пересборки: `1800` записей.
- Проверено покрытие `runs_clean` в `run_index.csv`: `20` строк, `20/20 status=completed`, `20/20 metrics_present=true`.
- Baseline snapshot (global multi-window robust ranking, DD-gate `<=0.14`) остаётся прежним:
  - `worst_robust_sharpe=3.530254`, `worst_dd_pct=0.132205`, `windows=3`.
  - baseline выбран как `run_group=20260213_budget1000_dd_sprint08_stoplossusd_micro`, `variant_id=prod_final_budget1000_risk0p006_slusd1p91`.
  - worst-window pair: `holdout_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91` + `stress_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91`.
- Примечание по tie-break: у `20260214_budget1000_dd_sprint09_hurst_slusd1p91` тот же `worst_robust_sharpe`; для continuity оставлен baseline из sprint08 (`20260213...`).
- Snapshot по `runs_clean` (`20260215_confirm_shortlist`): `10/10` пар имеют `robust_sharpe=0` и `PnL=0` (no-op), поэтому clean-shortlist не используется как baseline для нового loop до разбора причины no-op.
- Дневник блока: `docs/optimization_runs_20260220.md`.

### VPS confirm replay (20260220_confirm_top10_bl11)
- Собрана confirm-очередь top-10: `coint4/artifacts/wfa/aggregate/20260220_confirm_top10_bl11/run_queue.csv` (`60` run'ов = `10` вариантов × `3` окна × holdout+stress).
- Для воспроизводимости подтянуты отсутствовавшие локально YAML из VPS (`20260217_budget1000_bl11_r02_max_pairs`, `r03_pv`, `r05_slusd`) через `coint4/scripts/remote/run_server_job.sh` + `SYNC_BACK`.
- Heavy run выполнен на VPS `85.198.90.128`:
  - `SYNC_UP=1 UPDATE_CODE=0 STOP_AFTER=1 SYNC_BACK=1 bash coint4/scripts/remote/run_server_job.sh bash -lc 'bash scripts/optimization/watch_wfa_queue.sh --queue artifacts/wfa/aggregate/20260220_confirm_top10_bl11/run_queue.csv --parallel 12'`
  - Итог очереди: `60/60 completed`; `sync_queue_status.py` -> `no changes`.
  - VPS выключен автоматически (`STOP_AFTER=1`, SSH после завершения недоступен).
- Пересобран rollup после confirm:
  - `run_index` после пересборки: `8071` записей.
- Ранжирование confirm-группы воспроизвело исходный shortlist 1:1:
  - лидер: `..._slusd1p81` (`worst_robust_sharpe=4.504620`, `worst_dd_pct=0.085423`, `worst_pnl=252.894`).
  - далее: `..._max_pairs24p0` (`4.365635`) и `..._pv0p365` (`4.311314`).
- Статус промоута: результаты зафиксированы и подтверждены на VPS; live winner в state не переключался автоматически.

### VPS fullspan replay top-3 BL11 (20260220_top3_fullspan_wfa)
- Fullspan-окно собрано по фактическому диапазону данных parquet: `2022-03-01` -> `2025-06-30` (`walk_forward.max_steps=null`).
- Очередь: `coint4/artifacts/wfa/aggregate/20260220_top3_fullspan_wfa/run_queue.csv` (`6` run'ов = top-3 × holdout+stress).
- Heavy run выполнен на VPS `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh` + `run_wfa_queue.py --parallel 6`.
- Результат очереди: `6/6 completed`; проверка: `Sharpe consistency OK (6 run(s))`; VPS выключен автоматически (`STOP_AFTER=1`).
- Пересобран rollup после fullspan:
  - `run_index` после пересборки: `8077` записей.
- Fullspan robust ranking (по `robust_sharpe=min(holdout,stress)`) показал:
  - #1 `..._slusd1p81`: holdout/stress Sharpe `1.463/1.088`, robust PnL `1500.25`.
  - #2 `..._pv0p365`: holdout/stress Sharpe `1.367/-0.342`, robust PnL `-210.92`.
  - #3 `..._max_pairs24p0`: holdout/stress Sharpe `1.478/-0.346`, robust PnL `-215.28`.
- Строгая проверка canonical fullspan-policy-v1 (postprocess-команда из раздела выше):
  - результат: `No variants matched fullspan policy v1 (missing_tail=0, worst_step_gate_failed=1)`.
  - диагностически: `..._slusd1p81` имеет `worst_step_pnl=-209.15` (ниже порога `-200`), `..._max_pairs24p0` имеет `worst_step_pnl=-255.48`, `..._pv0p365` проваливает economic gate (`worst_robust_pnl=-210.92`).
- Вывод по промоуту: для `20260220_top3_fullspan_wfa` зафиксирован fail-closed verdict (`selection_mode=shortlist_only`, `promotion_verdict=reject`), live winner в state не переключался автоматически.
- Дополнительная VPS-перепроверка (2026-02-22 UTC, `SYNC_UP=1`, isolated fetch):
  - `build_run_index.py` на VPS: `Run index entries: 7927` (снимок сервера).
  - strict ranker на VPS повторил тот же verdict: `No variants matched fullspan policy v1`.
  - diagnostic `tail_worst_gate_pct=0.21` на VPS пропускает только `..._slusd1p81`.
  - targeted verify на VPS для postprocess-сценариев: `3 passed`, `4 failed` (`test_build_run_index_auto_sync`, `test_run_wfa_queue_postprocess*`), что указывает на несоответствие текущего `run_wfa_queue.py` ожиданиям тестов postprocess CLI.
  - артефакты/rollup подтянуты в `.remote_fetch_20260222/` без перезаписи рабочей копии.
- После фикса контрактов (2026-02-22 UTC):
  - локально: `tests/scripts/test_build_run_index_auto_sync.py + tests/scripts/test_run_wfa_queue_postprocess.py + tests/utils/test_run_index.py` -> `7 passed`;
  - VPS (`SYNC_UP=1`): те же 3 файла -> `7 passed`.
  - канонический цикл воспроизводится новой командой-оркестратором и подтверждает тот же fail-closed verdict (strict no-pass, research pass only `..._slusd1p81`).

Recent updates (2026-02-17):

### Batch loop BL9 (executor pass, bridge09)
- Запуск выполнен командой:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/autopilot_budget1000.py --config configs/autopilot/budget1000_batch_loop_bridge09_20260216.yaml --reset`
- Heavy execution: только VPS `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh` с `STOP_AFTER=1` (shutdown после каждого раунда подтверждён).
- Исполненные раунды BL9: `r01_vm`, `r02_max_pairs`, `r03_corr`, `r04_pv` (каждый `72/72 completed`).
- Явный post-sync после прогона:
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/20260216_budget1000_bl9_r*/run_queue.csv`
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
- Prefix totals (`20260216_budget1000_bl9_r*`): `completed=288`, `stalled=0`, `planned=228`, `total=516`.
- Stop reason (controller state): `search_space_exhausted: round=5, reason=Queue generation produced 0 entries for run_group=20260216_budget1000_bl9_r05_pv`.
- Отчёт: `docs/budget1000_autopilot_final_20260217.md`.
- Блокер/уточнение: `BL-ANL.completionNotes` содержал placeholder `Completed by agent`; исполнитель использовал safe default из `metadata.latestAnalystReview.recommendedNextConfig` и предыдущего progress log.

Recent updates (2026-02-16):

### Post-cycle consistency (closed-loop finalize)
- Проверен state `coint4/artifacts/wfa/aggregate/20260216_budget1000_cl_autopilot/state.json`: `done=true`, `stop_reason=max_rounds_reached: max_rounds=3`, history run_group = `r01_risk -> r02_slusd -> r03_vm`.
- Выполнен адресный sync очередей цикла:
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/20260216_budget1000_cl_r01_risk/run_queue.csv --queue artifacts/wfa/aggregate/20260216_budget1000_cl_r02_risk/run_queue.csv --queue artifacts/wfa/aggregate/20260216_budget1000_cl_r02_slusd/run_queue.csv --queue artifacts/wfa/aggregate/20260216_budget1000_cl_r03_risk/run_queue.csv --queue artifacts/wfa/aggregate/20260216_budget1000_cl_r03_vm/run_queue.csv`
- Изменения в статусах: `20260216_budget1000_cl_r02_risk/run_queue.csv` обновлён `6/6 -> completed`; в остальных очередях изменений нет.
- Пересобран canonical rollup:
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
  - `run_index` после пересборки: `2106` записей.
- Для финального выбора использовать актуальный `run_index` по всем completed run_group префикса `20260216_budget1000_cl_*` (не только `current_best` из state): итоговый winner `run_group=20260216_budget1000_cl_r02_risk`, `score=2.8563555379`, `worst_robust_sharpe=3.2694375191`, `worst_dd_pct=0.2016352476`.

### Budget1000 autopilot (VPS WFA -> postprocess -> max_rounds)
- Конфиг автопилота: `coint4/configs/autopilot/budget1000.yaml`.
- Controller state: `coint4/artifacts/wfa/aggregate/20260215_budget1000_ap_autopilot/state.json`.
- Итоговый отчёт: `docs/budget1000_autopilot_final_20260216.md` (завершение по `max_rounds=3`, не по stop-condition).
- Выполнено 9 очередей (3 раунда × 3 knobs): `20260215_budget1000_ap_r{01,02,03}_{risk,slusd,vm}` (по 30 run'ов на очередь: 3 окна × 5 значений × holdout+stress).
- Best candidate (DD gate отключён, использовался soft penalty `dd_target_pct=0.15`, `dd_penalty=5.0`):
  - variant_id: `prod_final_budget1000_risk0p019_slusd6p5_slusd4p5_vm1p0035_risk0p015`
  - config_path (sample): `coint4/configs/budget1000_autopilot/20260215_budget1000_ap_r03_risk/holdout_prod_final_budget1000_oos20220601_20230430_risk0p019_oos20220601_20230430_slusd6p5_oos20220601_20230430_slusd4p5_oos20220601_20230430_vm1p0035_oos20220601_20230430_risk0p015.yaml`
  - knobs: `risk_per_position_pct=0.015`, `pair_stop_loss_usd=4.5`, `max_var_multiplier=1.0035`
  - metrics: `score=2.357188`, `worst_robust_sharpe=3.310223`, `worst_dd_pct=0.340607` (34.1%; DD-gate `<=15%` НЕ проходит).

### Budget1000 autopilot follow-up (APF-04: итоги + сравнение)
- Новый итоговый отчёт follow-up: `docs/budget1000_autopilot_followup_final_20260216.md`.
- Follow-up queue: `coint4/artifacts/wfa/aggregate/20260216_budget1000_ap2_r01_risk/run_queue.csv`.
- Выполнен локальный постпроцесс и пересборка canonical rollup (`run_index.csv`, entries=1890).
- Состояние `20260216_budget1000_ap2_autopilot`: `done=false`, `history=[]`; по очереди `planned=30`, `metrics_present=False` (completed-run'ов нет).
- Ранжирование с DD-first ограничениями (`min_windows=3`, `max_dd_pct=0.25`) по `20260216_budget1000_ap2*` вернуло `No variants matched`.
- Зафиксирован fallback-кандидат для продолжения follow-up (из последней завершённой autopilot-семьи):
  - run_group: `20260215_budget1000_ap_r03_slusd`
  - variant_id: `prod_final_budget1000_risk0p019_slusd6p5_slusd4p5_vm1p0035_risk0p015_slusd2p5`
  - metrics: `score=1.646785`, `worst_robust_sharpe=2.288574`, `worst_dd_pct=0.230224` (DD gate `<=0.25` проходит).
  - sample_config_path: `coint4/configs/budget1000_autopilot/20260215_budget1000_ap_r03_slusd/holdout_prod_final_budget1000_oos20220601_20230430_risk0p019_oos20220601_20230430_slusd6p5_oos20220601_20230430_slusd4p5_oos20220601_20230430_vm1p0035_oos20220601_20230430_risk0p015_oos20220601_20230430_slusd2p5.yaml`
- Сравнение с циклом `20260215_budget1000_ap_autopilot` (завершён по `max_rounds=3`):
  - max_rounds best: `score=2.357188`, `worst_robust_sharpe=3.310223`, `worst_dd_pct=0.340607` (DD gate `<=0.25` не проходит).
  - follow-up fallback: `score=1.646785`, `worst_robust_sharpe=2.288574`, `worst_dd_pct=0.230224` (DD gate `<=0.25` проходит).
  - Δ (follow-up - max_rounds): `score=-0.710403`, `worst_robust_sharpe=-1.021648`, `worst_dd_pct=-0.110383` (улучшение DD на `11.04` п.п.).

### Budget1000 closed-loop autopilot (APF-05: resume до done=true)
- Конфиг: `coint4/configs/autopilot/budget1000_closed_loop_20260216.yaml`.
- Controller state: `coint4/artifacts/wfa/aggregate/20260216_budget1000_cl_autopilot/state.json`.
- Статус: `done=true`.
- Stop reason (факт из `state.json`): `max_rounds_reached: max_rounds=3`.
- Heavy выполнен только через `coint4/scripts/remote/run_server_job.sh` на `85.198.90.128`; VPS выключался автоматически после каждого раунда (`STOP_AFTER=1`).
- История контроллера (`state.history`): `20260216_budget1000_cl_r01_risk -> 20260216_budget1000_cl_r02_slusd -> 20260216_budget1000_cl_r03_vm`.
- Для финального сравнения (после sync статусов) учтены все completed run_group префикса: `r01_risk`, `r02_slusd`, `r02_risk`, `r03_vm`, `r03_risk`.
- Лучший кандидат closed-loop:
  - run_group: `20260216_budget1000_cl_r02_risk`
  - variant_id: `prod_final_budget1000_risk0p019_slusd6p5_slusd4p5_vm1p0035_risk0p015_slusd2p5_risk0p011_risk0p009`
  - metrics: `score=2.856356`, `worst_robust_sharpe=3.269438`, `worst_dd_pct=0.201635`
  - sample_config_path: `coint4/configs/budget1000_autopilot/20260216_budget1000_cl_r02_risk/holdout_prod_final_budget1000_risk0p019_slusd6p5_slusd4p5_vm1p0035_risk0p015_slusd2p5_risk0p011_oos20220601_20230430_risk0p009.yaml`
- Итоговые отчёты:
  - closed-loop финал: `docs/budget1000_autopilot_closed_loop_final_20260216.md`
  - controller summary: `docs/budget1000_autopilot_final_20260216.md`
- Сравнение с предыдущим циклом (`20260216_budget1000_ap2_autopilot`, baseline из follow-up):
  - предыдущий цикл (APF-04 fallback): `score=1.646785`, `worst_robust_sharpe=2.288574`, `worst_dd_pct=0.230224`.
  - closed-loop winner: `score=2.856356`, `worst_robust_sharpe=3.269438`, `worst_dd_pct=0.201635`.
  - Δ (closed-loop - previous): `score=+1.209570`, `worst_robust_sharpe=+0.980863`, `worst_dd_pct=-0.028588` (улучшение DD на `2.86` п.п.).

### Следующий шаг (после closed-loop)
- Зафиксировать winner-конфиг в рабочем cutover-пакете и выполнить confirmatory holdout+stress replay на VPS `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh` (`STOP_AFTER=1`) перед обновлением `configs/prod_final_budget1000.yaml`.
- По clean-cycle: верифицировать причину baseline no-op (`0` сделок) и только после этого запускать sweeps.

Recent updates (2026-02-15):

### Clean Cycle TOP-10 kickoff (prep)
- Цель: отделить decision-making от "старой партии" (`coint4/artifacts/wfa/runs/**`) и ранжировать только по `canonical_metrics.json`, пересчитанным из `equity_curve.csv`.
- Док процесса и guardrails: `docs/clean_cycle_top10.md`.
- Source of truth по константам цикла: `coint4/scripts/optimization/clean_cycle_top10/definitions.py` (`CYCLE_NAME=20260215_clean_top10`, `FIXED_WINDOWS.*`).
- Seed TOP-10: `coint4/artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_manifest.json`.
  - Примечание: `select_top10.py` выбирает TOP-N *runs* из `run_index.csv` (без дедупликации по `config_sha256`) → сейчас 10 строк, 5 уникальных конфигов.
- Baseline post-processing (локально, после sync_back) для цикла `20260216_clean_top10`:
  - `baseline_run_queue.csv`: `10/10 completed` (синхронизация статусов не требовалась).
  - `canonical_metrics.json` записан для baseline results_dir (10/10) через `recompute_canonical_metrics.py --bar-minutes 15` (в `equity_curve.csv` только 1 точка, нет timestamp deltas).
  - Baseline freeze sentinel `BASELINE_FROZEN.txt` создан и проверен (OK; WARN только про дополнительные `walk_forward.*` ключи в baseline-конфигах).
  - Построен baseline-only rollup (rows=10): `coint4/artifacts/wfa/aggregate/clean_cycle_top10/20260216_clean_top10/rollup_clean_cycle_top10.(csv|md)` (явный opt-in `--allow-baseline-only`).
  - Raw vs canonical diff: `missing_raw=0`, `missing_canonical=0`, `over_threshold=0` (selected=10, `compare_metrics.py`).
  - Важно: все baseline метрики вышли `0` (0 сделок; `equity_curve.csv` содержит только стартовую точку) → перед sweeps нужно проверить, что baseline batch действительно прогонялся корректно.
- Baseline post-processing (локально, после sync_back):
  - `baseline_run_queue.csv`: `10/10 completed`.
  - `canonical_metrics.json` записан для baseline results_dir (10/10).
  - Baseline freeze sentinel `BASELINE_FROZEN.txt` создан и проверен.
  - `verify_baseline_frozen.py`: OK; WARN только про дополнительные `walk_forward.*` ключи в исходных holdout-конфигах (`enabled/min_training_samples/pairs_file`).
  - Финальный clean rollup baseline+sweeps построен: `coint4/artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/rollup_clean_cycle_top10.(csv|md)` (rows=13, skipped=0).
  - Raw vs canonical diff: `over_threshold=0` (selected=13, `compare_metrics.py`).
  - Sweeps post-processing (локально): `sweeps_run_queue.csv` -> `3/3 completed`, `canonical_metrics.json` записан для sweeps results_dir (3/3).
    - Примечание: в sweeps `equity_curve.csv` содержит только 1 точку (нет timestamp deltas), поэтому canonical пересчитан с явным `--bar-minutes 15`; метрики sweeps = 0 (0 сделок).
- Дневник/следующие шаги: `docs/optimization_runs_20260215.md`.
  - Для цикла `20260216_clean_top10`: `docs/optimization_runs_20260216.md`.

### VPS baseline WFA (queue10)
- Очередь: `coint4/artifacts/wfa/aggregate/20260215_baseline_queue10/run_queue.csv` → `10/10 completed`.
- Исполнение: только на VPS `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh` + `scripts/optimization/watch_wfa_queue.sh` (на этом сервере тяжёлое не запускаем).
- Rollup индекс пересобран: `coint4/artifacts/wfa/aggregate/rollup/run_index.(csv|json|md)`.

### VPS confirmatory queue (clean_cycle_top10 shortlist: holdout+stress)
- Очередь: `coint4/artifacts/wfa/aggregate/clean_cycle_top10/20260215_confirm_shortlist/run_queue.csv` → `20/20 completed` (holdout + stress).
- Исполнение: VPS `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh` + `scripts/optimization/watch_wfa_queue.sh` (с `SYNC_UP=1`, `STOP_AFTER=1`).
- Результаты: `coint4/artifacts/wfa/runs_clean/20260215_confirm_shortlist/confirm/{holdout,stress}/...` (для shortlist валидации robust-метрики).

### VPS sweeps WFA (dd_sprint10 min_beta)
- Очередь: `coint4/artifacts/wfa/aggregate/20260214_budget1000_dd_sprint10_minbeta_slusd1p91/run_queue.csv` → `30/30 completed`.
- Исполнение: только на VPS `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh` + `scripts/optimization/watch_wfa_queue.sh`.
- Примечание: на VPS `git pull` падал из-за dirty worktree, поэтому запуск делался с `SYNC_UP=1` (rsync tracked файлов) и `UPDATE_CODE=0`.

Recent updates (2026-02-13):

### DD Analysis & Fix (pair_stop_loss_usd + pruned_v2)
- **Root cause**: HFTUSDC-KDAUSDT lost -$6,489 in Jul-Aug 2023 (68x leverage from beta amplification on $0.35/$0.56 tokens)
- **CRITICAL BUG FOUND**: `pair_stop_loss_usd` was configured ($7.5) but NOT implemented in numba kernel! Only worked in live/paper engine.
- **Code fix**: Added `pair_stop_loss_usd` parameter to `calculate_positions_and_pnl_full()` in `numba_kernels.py`, propagated through `numba_backtest_engine_full.py` and `numba_engine.py`
- **Universe pruning v2**: removed HFT (HFTUSDC, HFTUSDT) and KDAUSDT from denylist — 191→168 pairs (23 pairs removed)
- **Full-span v2 results** (Jun 2022 → Jun 2025, 75 steps):
  - Holdout: Sharpe **2.24** (was 1.90, +18%), PnL $25,402, Max DD **-53.0%** (was -83.1%)
  - Stress: Sharpe **1.83** (was 1.59, +15%), PnL $18,906, Max DD -66.0%
  - Worst pair: **-$2,237** (was -$6,489)
- **OOS validation v2** (3 windows, pruned_v2):
  - Window A holdout: Sharpe 4.61, PnL $593, DD -9.8%
  - Window B holdout: Sharpe 1.64, PnL $291, DD -36.6%
  - Window C holdout: Sharpe 4.57, PnL $1,293, DD -21.4%
  - Avg holdout: 3.61, avg stress: 3.36. All 3 windows profitable.
- **DD sprints (multi-window, pruned_v2, risk=0.006)**:
  - Ключевой рычаг — `pair_stop_loss_usd`: снижение stop-loss с 5.0 до ~1.5–2.25 резко улучшает worst-DD и robust Sharpe.
  - Новый лидер по DD<=15%: `pair_stop_loss_usd=1.91` → worst-window robust Sharpe `3.530`, worst DD `-13.2%` (worst DD окно: `20231001-20240930`).
  - Sprint07 (`20260213_budget1000_dd_sprint07_maxbeta_slusd2`): cap по `filter_params.max_beta` при `pair_stop_loss_usd=2.0` не улучшил Sharpe/DD (лучшее при DD<=15%: `max_beta=20` → worst robust `2.211`, worst DD `14.3%`).
  - Sprint08 (`20260213_budget1000_dd_sprint08_stoplossusd_micro`): micro-sweep `pair_stop_loss_usd=1.85..2.0` нашёл улучшение лидера (новый лидер `slusd=1.91`).
  - Детали/очереди: `docs/optimization_runs_20260213.md`.

### Previous: OOS validation + pruned universe v1 (191 pairs)
- **OOS validation (3 окна, original 250 пар)**: Window A Sharpe -0.61, Window B -$874 DD>100%, Window C Sharpe 3.95. Gate FAIL: 1/3 прибыльных.
- **Pair attribution**: FLOWUSDT-JUVUSDT уничтожил Window B (-$738.90, 84.5% потерь). Фан-токены — источники катастрофических потерь.
- **Universe pruning v1**: удалены 59 пар (фан-токены). Pruned universe: 191 пара.
- **Re-test pruned v1 (3 окна)**: все 3 окна прибыльны — holdout Sharpe A=5.40, B=2.25, C=5.51 (avg 4.39).
- **Notional sweep**: no-op при $1K (avg entry notional $15-34). Set 100 as guardrail.
- **Full-span v1**: holdout Sharpe 1.90, PnL $29,836, Max DD -83.1%.
- **Prod config v1**: `pruned191` universe, max_notional=100.

Signal sprints (2026-02-13):
- Signal sprint19 (hold/cooldown sweep under `ts1p5`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint19: максимум по robust-метрике остаётся на baseline `hold300/cd300` (Sharpe `4.424/4.119`); `hold60` уходит в отрицательный Sharpe, `hold600/900` резко режут PnL и ухудшают cost_ratio.
- Signal sprint20 (min_spread_move_sigma sweep under `ts1p5`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint20: новый лидер `ms0p1` (min_spread_move_sigma=0.1) — Sharpe `4.572/4.277` (лучше baseline `ms0p2` = `4.424/4.119`).
- Signal sprint21 (corr/pvalue sweep under `ms0p1`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint21: loosen/tighten `corr/pvalue` ухудшает Sharpe; лидер остаётся `ms0p1` (baseline `corr0p34_pv0p35`).
- Signal sprint22 (time_stop_multiplier sweep under `ms0p1`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint22: `ts1p5` остаётся лучшим (чуть лучше `ts1p0`); новый лидер не найден.
- Signal sprint23 (pair_stop_loss_zscore sweep under `ms0p1+ts1p5`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint23: лидер остаётся `slz3p0` (=3.0); `2.0–2.5` ломает Sharpe через churn/издержки, `3.5–4.0` ухудшает Sharpe и раздувает DD.
- Signal sprint24 (protections toggles sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint24: все защиты должны оставаться включенными; `market_regime_detection=false` особенно разрушает Sharpe и DD; лидер остаётся `ms0p1`.
- Signal sprint25 (rolling_window sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint25: `rolling_window=96` остаётся явным максимумом Sharpe; `48/144/192/288` резко ухудшают метрики, вплоть до отрицательного PnL в stress.
- Signal sprint26 (z-entry sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint26: `z=1.15` остаётся локальным максимумом; `z=0.9–1.0` ухудшает Sharpe через churn, `z=1.30–1.45` режет PnL и снижает Sharpe.
- Signal sprint27 (structural-break intensity sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint27: baseline `sb_base` (= прежние Numba константы) остаётся лучшим; изменение min_correlation или мультипликаторов ухудшает robust Sharpe.
- Signal sprint28 (market-regime clamp sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint28: базовый clamp `rg0p5to1p5` остаётся лучшим; расширение верхней границы или сужение диапазона резко ухудшает Sharpe.
- Signal sprint29 (max_pairs sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint29: `max_pairs=24` остаётся лучшим; уменьшение/увеличение числа торгуемых пар ухудшает Sharpe через потерю диверсификации или рост издержек.
- Signal sprint30 (training_period_days sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint30: лидер остаётся baseline `tr90` (training=90d); `tr60` ломает стратегию (отрицательные Sharpe/PnL), а `120–240d` ухудшают robust Sharpe и повышают stress cost_ratio.
- Signal sprint31 (max_active_positions sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint31: лидер остаётся baseline `ap18` (max_active_positions=18); уменьшение лимита (`12/16`) ухудшает Sharpe, увеличение (`20/24`) не улучшает (почти идентичные метрики и чуть хуже robust Sharpe).
- Signal sprint32 (lookback_days sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint32: все варианты дали идентичные метрики → `pair_selection.lookback_days` сейчас не влияет на WFA (окно данных задаётся `training_start..testing_end`).
- Signal sprint33 (testing_period_days sweep under `ms0p1+ts1p5+slz3p0`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint33: новый лидер `tp15` — Sharpe `5.142/4.899` (robust `4.899`), но горизонт теста при `max_steps=5` становится короче (≈75 дней) → нужно подтверждение на сопоставимом горизонте (например, `tp15 + max_steps=10`).
- Signal sprint34 (training_period_days sweep under `tp15`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint34: лидер по robust остаётся на `tr90` (Sharpe `5.142/4.899`); `tr180` второй (Sharpe `4.402/4.200`), `tr60` ломает стратегию через огромный DD и высокий stress cost_ratio.
- Signal sprint35 (testing_period_days sweep under `tp15_tr90`, max_steps=null) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint35: на полном горизонте extended OOS `tp15` остаётся лучшим по robust (Sharpe `3.326/3.117`, robust `3.117`), но рост Sharpe в sprint33/34 был существенно завязан на укороченный тест при `max_steps=5`.
- Signal sprint36 (pair stability sweep under full-horizon `tp15_tr90`, max_steps=null) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint36: новый full-horizon лидер `psw1m1` (pair_stability window=1, min_steps=1) — Sharpe `3.544/3.342` (robust `3.342`).
- Signal sprint37 (max_hurst_exponent sweep under full-horizon `tp15_tr90 + psw1m1`) завершён: `10/10 completed`, `Sharpe consistency OK (10 run(s))`.
- Итог sprint37: новый full-horizon лидер `hx0p70` (max_hurst_exponent=0.70) — Sharpe `3.576/3.375` (robust `3.375`).

Recent updates (2026-02-12):
- Проверена целостность последних `$1000` прогонов: для очередей `20260131_budget1000_*` обязательные артефакты присутствуют; Sharpe consistency check пройден.
- Подготовлена новая очередь из 10 конфигов: `coint4/artifacts/wfa/aggregate/20260212_budget1000_tlow_extended_sharpe_recover10/run_queue.csv`.
- Новые конфиги: `coint4/configs/budget_20260212_1000_tlow_extended_sharpe_recover10/*.yaml` (варианты r1-r5, holdout/stress).
- Гипотеза: улучшить Sharpe через более широкий пул пар (`corr/pvalue`, `max_pairs`) и снижение нелинейности sizing (`min/max_notional`).
- Запуск на `85.198.90.128` завершён: `10/10 completed`.
- Лучший вариант: `r4` (min Sharpe holdout/stress = `1.482`), пары выросли до `53`, но DD остался за гейтом (`~ -356`) и stress cost_ratio `0.62 > 0.5`.
- Дополнительный sweep без ограничений по рисковым гейтам: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_unbounded_minnot/run_queue.csv` (`12/12 completed`).
- Лучший вариант в Max-Sharpe режиме: `u6` (holdout/stress Sharpe `2.775/2.425`, pairs `58`, stress PnL `668.69`).
- Наблюдение по min_notional: в вариантах `u1-u3` (`min_notional` 0.5/1/2) метрики идентичны — в этой зоне параметр не лимитирует; рост Sharpe получен за счёт комбинированной смены режима (`z/ms/corr/pvalue/max_pairs`).
- Дополнительный signal sprint around `u6`: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint1/run_queue.csv` (`10/10 completed`).
- Новый лидер по robust-метрике `min(Sharpe_holdout, Sharpe_stress)`: `v1` (`3.338/3.007`), выше `u6` (`2.775/2.425`).
- Целостность результатов `signal_sprint1`: `Sharpe consistency OK (10 run(s))`, обязательные артефакты есть в `10/10`, в `run.log` нет `Traceback/ERROR`.
- Signal sprint2 (local search around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint2/run_queue.csv` (`10/10 completed`).
- Итог sprint2: ни один `s1-s5` не улучшил `v1`; лучший robust `min_sharpe` у `s5` = `2.067` (хуже `v1` = `3.007`). Понижение `z/ms` относительно `v1` ухудшает Sharpe.
- Infra: устранена коррупция memory-mapped кэша при параллельных WFA (lock + atomic replace + range-keyed cache filename для consolidated parquet).
- Signal sprint3 (z fine sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint3/run_queue.csv` (`10/10 completed`).
- Итог sprint3: лучший `zf4` (z=1.15) совпал с `v1` (Sharpe `3.338/3.007`), остальные `z=1.12-1.16` хуже → по `z` достигнут локальный максимум.
- Signal sprint4 (exit sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint4/run_queue.csv` (`10/10 completed`).
- Итог sprint4: лучший `ex3` (exit=0.08) совпал с `v1` (Sharpe `3.338/3.007`), остальные `exit=0.06-0.10` хуже → по `exit` достигнут локальный максимум.
- Signal sprint5 (ms sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint5/run_queue.csv` (`10/10 completed`).
- Итог sprint5: лучший `ms3` (ms=0.20) совпал с `v1` (Sharpe `3.338/3.007`), остальные `ms=0.16-0.24` хуже → по `ms` достигнут локальный максимум.
- Signal sprint6 (max_pairs sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint6/run_queue.csv` (`10/10 completed`).
- Итог sprint6: лучший `mp4` (max_pairs=24) совпал с `v1` (Sharpe `3.338/3.007`), остальные max_pairs хуже → по `max_pairs` локальный максимум на `24` (для max-Sharpe режима).
- Signal sprint7 (stop_loss_zscore sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint7/run_queue.csv` (`10/10 completed`).
- Итог sprint7: лучший `slz3p0` (stop_loss_z=3.0) совпал с `v1`; `2.0-2.5` убивает edge через churn/издержки, `3.5-4.0` раздувает DD → локальный максимум по stop_loss на `3.0`.
- Signal sprint8 (max_active_positions sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint8/run_queue.csv` (`10/10 completed`).
- Итог sprint8: лучший `ap18` (max_active_positions=18) совпал с `v1`; `ap24` почти идентичен, но чуть хуже; `ap6-ap14` дают просадку Sharpe.
- Signal sprint9 (max_var_multiplier sweep around `v1`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint9/run_queue.csv` (`10/10 completed`).
- Итог sprint9: Sharpe резко растёт при уменьшении `max_var_multiplier`; лучший `vm1` (1.10) уже лучше `v1` (Sharpe `3.650/3.306` vs `3.338/3.007`).
- Signal sprint10 (max_var_multiplier fine sweep) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint10/run_queue.csv` (`10/10 completed`).
- Итог sprint10: новый лидер `vmf101` (max_var_multiplier=1.01) — Sharpe `4.348/4.043`, PnL `2153.76/1908.57`.
- Signal sprint11 (adaptive+regime+struct toggles) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint11/run_queue.csv` (`10/10 completed`).
- Итог sprint11: `market_regime_detection` и `structural_break_protection` должны оставаться включенными; `adaptive_thresholds=false` почти не хуже, но лидер всё равно `at1vm101` (= `vmf101`).
- Signal sprint12 (z sweep under max_var_multiplier=1.01) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint12/run_queue.csv` (`10/10 completed`).
- Итог sprint12: по `z` локальный максимум остался на `z=1.15` (`z1p15` совпадает с лидером `vmf101`).
- Signal sprint13 (exit sweep under `vmf101`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint13/run_queue.csv` (`10/10 completed`).
- Итог sprint13: по `exit` локальный максимум остался на `exit=0.08` (`ex08` совпадает с лидером `vmf101`).
- Signal sprint14 (stop_loss_zscore sweep under `vmf101`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint14/run_queue.csv` (`10/10 completed`).
- Итог sprint14: по `pair_stop_loss_zscore` локальный максимум остался на `3.0` (`slz3p0` совпадает с лидером `vmf101`).
- Signal sprint15 (max_var_multiplier ultra-fine sweep) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint15/run_queue.csv` (`10/10 completed`).
- Итог sprint15: новый лидер `vm1005` (max_var_multiplier=1.005) — Sharpe `4.378/4.074`, PnL `2188.25/1940.54`.
- Signal sprint16 (max_var_multiplier refine around 1.005) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint16/run_queue.csv` (`10/10 completed`).
- Итог sprint16: новый лидер `vm10055` (max_var_multiplier=1.0055) — Sharpe `4.380/4.076`, PnL `2190.02/1942.05`.
- Signal sprint17 (z micro-sweep under `vm10055`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint17/run_queue.csv` (`10/10 completed`).
- Итог sprint17: `z=1.15` остаётся локальным максимумом, новый лидер не найден.
- Signal sprint18 (time_stop_multiplier sweep under `vm10055`) завершён: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint18/run_queue.csv` (`10/10 completed`).
- Итог sprint18: новый лидер `ts1p5` (`time_stop_multiplier=1.5`) — Sharpe `4.424/4.119`, PnL `2230.75/1978.76`.
- Детали: `docs/optimization_runs_20260212.md`.

Recent updates (2026-01-31):
- Extended OOS (2023-05-01 → 2024-04-30) для top20/top30 завершён: stress cost_ratio > 1.0, слабый Sharpe (см. `docs/optimization_runs_20260130.md`).
- Turnover-grid extended OOS (top10/top15, ms0.25/0.30, hold/cd 240) завершён: провал по парам и стресс-издержкам (см. `docs/optimization_runs_20260131.md`).
- Stop-condition: extended OOS stress cost_ratio > 0.5 и пары < 50 → оптимизацию в этом направлении останавливаем.
- Сформирован обзор по $1000: `docs/budget1000_overview_20260131.md`.
- Очередь tlow extended OOS для $1000 выполнена: risk0p0175 Sharpe 1.70/1.36, cost_ratio 0.26/0.60, DD ~-32%, pairs 36; risk0p015 отрицательный (см. `docs/optimization_runs_20260131.md`).
- Refine‑очередь (z=1.25/1.30, ms=0.30/0.35, hold/cd=300) выполнена: DD снизился до ~-17…-18%, но stress cost_ratio 0.53–0.56 и пары 36 (см. `docs/optimization_runs_20260131.md`).
- Refine2‑очередь (z=1.35/1.40, ms=0.35/0.40, hold/cd=360) выполнена: PnL/Sharpe ухудшились, stress cost_ratio 1.85–1.86, пары 36 — направление закрываем (см. `docs/optimization_runs_20260131.md`).
- Tradeability+basecap3 очередь (corr 0.35/0.40, pv 0.25/0.20, пары basecap3=102) выполнена: Sharpe < 0, PnL отрицательный, pairs 26–27; tradeM/tradeS совпали → ветку закрываем (см. `docs/optimization_runs_20260131.md`).
- Candidate configs (live): `coint4/configs/prod_candidate_relaxed8_nokpss_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2.yaml` (primary) и `coint4/configs/prod_candidate_relaxed8_nokpss_u250_top20_z1p00_exit0p06_hold180_cd180_ms0p2.yaml` (fallback).
- Legacy: план paper/forward (не используем; paper trading не делаем): `docs/paper_forward_plan_20260131.md`.

Next steps:
- Clean Cycle TOP-10 (cycle `20260216_clean_top10`): baseline post-processing сделан, но метрики baseline = 0 → сначала проверить корректность baseline batch, затем только делать sweeps на VPS `85.198.90.128` и пересобрать финальный `rollup_clean_cycle_top10.*` (см. `docs/optimization_runs_20260216.md`, `docs/clean_cycle_top10.md`).
- Зафиксировать DD-оптимум: `pair_stop_loss_usd=1.85` (multi-window worst-DD `-13.2%`, worst robust Sharpe `3.448`) и прогнать full-span holdout+stress для подтверждения.
- Live cutover кандидата: см. `docs/production_checklist.md` и `AGENTS.md`.
- Если extended OOS обязателен для $1000: текущие попытки (tradeability+basecap3) дали отрицательные метрики → целесообразнее фиксировать stop‑condition и переходить к live cutover (paper не делаем).
- Исправление `total_costs` для Numba-бэктеста выполнено; метрики обновлены (done).
- Baseline WFA (5 шагов) выполнен и зафиксирован (done).
- Turnover sweep завершён, лучшая комбинация entry 0.95 / exit 0.10 (done).
- Quality sweep завершён, лучший Sharpe при corr 0.65 (done).
- Risk sweep завершён, существенных отличий не выявлено (done).
- Shortlist WFA (5 шагов) завершён на 85.198.90.128; топ Sharpe: baseline `5.7560`, corr0.7 `5.7302`, turnover `4.9631` (см. rollup).
- Holdout WFA (2024-05-01 → 2024-12-31, фактический тест до 2024-09-28) завершён: Sharpe `-3.41/-3.27`, PnL `-324/-307` (baseline/corr0.7).
- Stress costs WFA завершён: Sharpe `4.66/4.57`, PnL `616/543` (baseline/corr0.7).
- Диагностика holdout завершена: см. `docs/holdout_diagnostics_20260118.md` + CSV в `coint4/results/holdout_20260118_*`.
- Добавлен фильтр стабильности пар (pair_stability_window_steps/min_steps) в WFA.
- Stability shortlist (20260119) завершён: Sharpe 2.41–5.92, но total_pairs_traded 4–34 (ниже порога 100).
- Stability_relaxed WFA завершён: Sharpe `3.99/2.59`, total_pairs_traded `78/41` (ещё ниже порога 100).
- Stability_relaxed2 WFA завершён: Sharpe `2.13`, total_pairs_traded `52`, total_trades `1010`, PnL `84.13`.
- Stability_relaxed3 WFA завершён: лучший Sharpe `2.04`, total_pairs_traded `51`; 2 конфига дали 0 пар из-за KPSS фильтра.
- Stability_relaxed4 WFA завершён: Sharpe `3.07–3.09`, total_pairs_traded `159–272`; лучший вариант corr0.45 + ssd50000 + kpss0.03.
- Holdout relaxed4 завершён: Sharpe `-1.32/-1.08`, PnL `-98.19/-78.69` (corr0.45/corr0.5).
- Stress relaxed4 завершён: Sharpe `2.386/2.382`, PnL `426/416`.
- Диагностика holdout relaxed4 завершена: пересечение пар 4 (Jaccard ~0.0048), доминирует pvalue; см. `docs/holdout_diagnostics_20260119_relaxed4.md`.
- Stability_relaxed5 WFA завершён: w3m2 Sharpe `5.81`, pairs `383`, PnL `958.72` (corr0.45, ssd50000, kpss0.03, window=3/min=2).
- Holdout relaxed5 (w3m2) завершён: Sharpe `-1.77`, PnL `-220.63`, pairs `1018`.
- Stress relaxed5 (w3m2) завершён: Sharpe `4.76`, PnL `783.53`.
- Диагностика holdout w3m2 завершена: пересечение пар `16` (Jaccard ~0.0116), доминирует pvalue; см. `docs/holdout_diagnostics_20260119_relaxed5.md`.
- Подготовлен фиксированный universe из WFA w3m2 (383 пары) для повторного holdout; очередь: `coint4/artifacts/wfa/aggregate/20260119_relaxed5_holdout_fixed/run_queue.csv`.
- Holdout relaxed5 fixed universe (w3m2) завершён: Sharpe `1.25`, PnL `13.94`, pairs `18`, total_trades `352` (низкая статистика).
- Holdout fixed universe (window=1/min=1) завершён: Sharpe `-0.02`, PnL `-0.20`, pairs `11`, total_trades `253`.
- Stability_relaxed6 WFA завершён: Sharpe `5.36`, pairs `303`, PnL `894.35` (pvalue 0.12, kpss 0.05, hurst 0.70).
- Holdout relaxed6 (w3m2) завершён: Sharpe `-2.20`, PnL `-267.02`, pairs `779`, total_trades `13268`.
- Диагностика holdout relaxed6 завершена: пересечение пар `6` (Jaccard ~0.0056); см. `docs/holdout_diagnostics_20260119_relaxed6.md`.
- Holdout relaxed6 fixed universe завершён: Sharpe `3.77`, PnL `24.98`, pairs `7`, total_trades `141`.
- Stability_relaxed7 WFA завершён: Sharpe `5.25`, pairs `177`, PnL `914.61` (train=90d).
- Holdout relaxed7 (train=90d) завершён: Sharpe `-0.20`, PnL `-20.95`, pairs `339`, total_trades `6509`.
- Собран новый universe (relaxed8 строгий, 110 пар) для пред‑holdout периода: `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout/pairs_universe.yaml`.
- Собран expanded universe (relaxed8 strict pre‑holdout v2, 250 пар) с limit_symbols=300: `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/pairs_universe.yaml`.
- Подготовлен более мягкий criteria-файл для возможного следующего расширения universe: `coint4/configs/criteria_relaxed8_nokpss_universe.yaml`.
- WFA relaxed8 fixed universe завершён: 0 торгуемых пар, Sharpe `0.00` (фильтры режут до нуля).
- WFA relaxed8_loose завершён: 0 торгуемых пар (KPSS режет до нуля даже при kpss=0.1).
- WFA relaxed8_nokpss завершён: Sharpe `1.86`, pairs `35`, PnL `104.11` (kpss=1.0).
- Holdout relaxed8_nokpss завершён: Sharpe `3.21`, PnL `145.84`, pairs `64`, total_trades `2252`.
- Stress relaxed8_nokpss holdout завершён: Sharpe `2.17`, PnL `98.35`, pairs `64`, total_trades `2252`, costs `108.65`.
- Holdout relaxed8_nokpss_u250 завершён: Sharpe `4.20`, PnL `421.60`, pairs `168`, total_trades `6572`.
- Stress holdout u250 завершён: Sharpe `2.89`, PnL `289.30`, pairs `168`, total_trades `6572`, costs `309.95`.
- Итоговый кандидат: `docs/candidate_relaxed8_u250_20260119.md`.
- Запланирован turnover stress grid (u250) для снижения числа сделок: `coint4/artifacts/wfa/aggregate/20260119_relaxed8_nokpss_u250_turnover_stress/run_queue.csv`.
- Top-k лимит пар (max_pairs=10/20) выполнен для u250 holdout + stress: top10 Sharpe `2.74/2.08`, trades `798`; top20 Sharpe `4.56/3.65`, trades `1693`, costs `46.37/82.43` (holdout/stress).
- Turnover grid поверх top20 завершён: лучший вариант z1.05/exit0.08/hold120/cd120 → holdout Sharpe `3.85`, stress Sharpe `3.44`, trades `630`, costs `18.04/32.07`.
- Baseline u250 turnover grid завершён: лучший вариант z0.95/exit0.08/hold120/cd120 → holdout Sharpe `4.52`, stress Sharpe `3.70`, PnL `447.87/366.56`, trades `3936`.
- Candidate sweep по риск-параметрам завершён: метрики почти не меняются, max_active_positions даёт минимальные отличия; оставляем baseline z0.95/0.08/120/120.
- Sharpe annualization: WFA использует `annualizing_factor * (24*60/bar_minutes)`; base_engine приведён к динамическому periods_per_year по шагу данных.
- Micro-grid u250 (entry/exit/hold/cd + max_pairs 50/100/150) завершён: лучший min‑Sharpe у z0.95/exit0.06/hold120/cd120 (4.54/3.73); exit0.10 практически идентичен. Очередь: `coint4/artifacts/wfa/aggregate/20260121_relaxed8_nokpss_u250_search/run_queue.csv`.
- Кандидат обновлён: `docs/candidate_relaxed8_u250_20260121.md`.
- Numba: включены cooldown/min_hold/min_spread_move/stop-loss + выход по |z|<=z_exit; портфельная симуляция использует позиции вместо PnL-сигналов.
- Очередь churnfix micro-grid (u250) подготовлена: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix/run_queue.csv` (результаты 0 сделок из-за адаптивных порогов).
- Churnfix v2 (после фикса адаптивных порогов) завершён: 0 сделок в holdout/stress, требуется диагностика порогов/volatility (`coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_v2/run_queue.csv`).
- Sanity no-adapt завершён: 0 сделок даже при entry 0.75 (см. `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_sanity/run_queue.csv`).
- Sanity v2 (current-bar signals) завершён: 0 сделок, Sharpe 0.00 (см. `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_sanity_v2/run_queue.csv`).
- Numba выравнен с базовой логикой по std (guard 1e-6, без min_volatility clamp в z-score) и принимает beta/mu/sigma напрямую.
- Sanity v3 завершён: 0 сделок (см. `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_sanity_v3/run_queue.csv`).
- Найдено: min_spread_move_sigma блокировал входы из-за NaN в last_flat_spread при fastmath; исправлено через last_flat_valid.
- Sanity v4 завершён: сделки восстановились, но turnover очень высокий (26k+), stress Sharpe отрицательный (см. `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_sanity_v4/run_queue.csv`).
- Churnfix v3 (holdout+stress) завершён: Sharpe 5.0–8.0, PnL 583–1072, trades 20k–28k.
- Churnfix top‑k завершён: top20 Sharpe 7.74/6.84, PnL 898/793, trades 4841; top50 Sharpe 7.63/6.43, PnL 1083/914, trades 11823 (holdout/stress).
- Churnfix msgrid завершён: ms0p2/ms0p3 на hold180 дают метрики близкие к ms0p1; hold240 снижает Sharpe/PNL.
- Alt holdout (2022-09-01 → 2023-04-30) завершён: top50/full идентичны (≈47 пар), Sharpe 7.96/6.72, PnL 941/794; top20 чуть хуже.
- Sensitivity top50 завершён: лучший Sharpe у z1.00/exit0.06 (9.01/7.64) при PnL 1115/946; z0.95/exit0.08 даёт максимум PnL (1180/997).
- Basecap3 завершён: Sharpe 4.87/3.80, PnL 674/526, pairs 71 — слишком жёстко.
- Новый лучший компромисс (robust): top50/z1.00/exit0.06/hold180/cd180/ms0.1 → Sharpe 9.01/7.64, PnL 1115/946, trades 11414.
- Кандидат обновлён: `docs/candidate_relaxed8_u250_20260122.md`.
- Канонический конфиг: `coint4/configs/candidate_20260123_relaxed8_nokpss_u250_top50/candidate_relaxed8_nokpss_20260123_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1.yaml` (PnL‑альтернатива рядом).
- Alt-holdout top50 sens завершён: z1.00/exit0.06 Sharpe 8.65/7.48, PnL 1049/907; z0.95/exit0.08 чуть ниже (см. `coint4/artifacts/wfa/aggregate/20260123_relaxed8_nokpss_u250_churnfix_alt_top50_sens/run_queue.csv`).
- OOS 2023-05 → 2023-12: лучше z0.95/exit0.08 (Sharpe 4.24/2.94, PnL 810/561); z1.00 ниже (2.63/1.55, PnL 556/326).
- OOS 2025-01 → 2025-06: лучше z1.00/exit0.06 (Sharpe 3.83/2.61, PnL 400/271); z0.95 ниже.
- OOS top30/top40 (z1.00/ms0p1): top30 лучше top40, но уступает top50 по Sharpe/PNL; turnover снижается ~34% (2025H1), поэтому оставляем top50 как primary.
- По шагам WFA (daily_pnl срезы) есть отрицательные минимумы на обоих OOS периодах; детали в `docs/optimization_runs_20260122.md`.
- Концентрация на новых OOS умеренная: top10 ≈ 44–50%, top20 ≈ 63–68%; отрицательных пар 50–69 из 141–145.
- Churngrid min_spread_move_sigma завершён: ms0p2 лучше ms0p15 в базовом holdout, но OOS 2025H1 для ms0p2 хуже (Sharpe 2.66/1.44, PnL 278/149), поэтому оставляем ms0p1 как primary.
- Decision matrix (summary):
  - Primary (z1.00/ms0p1): лучший на OOS 2025H1, сильный базовый holdout, стабильнее ms0p2.
  - PnL alt (z0.95/exit0.08): лучший на OOS 2023H2, но слабее на 2025H1.
- Sharpe sanity: скрипт `scripts/optimization/check_sharpe_consistency.py` прошёл для ключевых очередей (32 прогона).
- Base cap: текущий pairs_universe уже max_per_base=4; basecap3 ухудшал метрики, поэтому отдельный cap 5/8 не даёт нового эффекта.
- Следующий шаг: финальная проверка концентрации/устойчивости и решение о live cutover (paper не делаем).

Legacy context:
Current stage: Leader holdout WFA (2024-05-01 → 2024-12-31, max_steps=5) via artifacts/wfa/aggregate/20260116_leader_holdout/run_queue.csv (parallel=1, n_jobs=-1). Additional: next5_fast WFA (manual sequential runs; queue file artifacts/wfa/aggregate/20260117_next5_fast/run_queue_next5_fast.csv used for status, backtest.n_jobs=-1, COINT_FILTER_BACKEND=threads). Current next5_fast run: none (latest best by Sharpe: pair_sweep_20260117_corr0p55_z0p85_exit0p12_ssd25000); queued: none.

Progress:
- ssd5000 completed
- ssd10000 completed
- ssd25000 completed
- ssd50000 active (WF step 3/3, step 2: 122 pairs, P&L +355.58)
- leader_validation completed (Sharpe 0.5255, PnL 1388.71, DD -199.31)
- leader_holdout active: coint4/configs/best_config__leader_holdout_ssd25000__20260116_211943.yaml → artifacts/wfa/runs/20260116_leader_holdout/best_config__leader_holdout_ssd25000__20260116_211943 (COINT_FILTER_BACKEND=processes)

Parallel stage:
- Piogoga grid (leader filters, zscore sweep) via artifacts/wfa/aggregate/20260116_piogoga_grid/run_queue.csv (parallel=16, n_jobs=1).
- Signal grid (16 configs, z=0.75/0.8/0.85/0.9 × exit=0.04/0.06/0.08/0.1) via artifacts/wfa/aggregate/20260116_signal_grid/run_queue.csv (parallel=16, n_jobs=1).
- SSD sweep (6 values) via artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv (queue has running statuses; check worker activity before resume).
- Leader validation (post-analysis, single run) completed: artifacts/wfa/runs/20260116_leader_validation/.
- Патч: фильтрация пар теперь параллельная (n_jobs из backtest, backend threads; `COINT_FILTER_BACKEND=processes` с spawn для OpenMP‑безопасности) — цель полной загрузки CPU.

After ssd50000 DONE:
1) Update docs/optimization_runs_20260116.md (metrics + filtering summary for ssd50000).
2) Update rollup in artifacts/wfa/aggregate/rollup/.
3) Start SSD sweep (3 values) queue: artifacts/wfa/aggregate/20260115_ssd_topn_sweep_3vals/run_queue.csv.
4) Update this file with the new stage and next steps.

After signal grid DONE:
1) Update docs/optimization_runs_20260116.md (metrics + filtering summary for signal grid).
2) Update rollup in artifacts/wfa/aggregate/rollup/.

After piogoga grid DONE:
1) Update docs/optimization_runs_20260116.md (metrics + filtering summary for piogoga grid).
2) Update rollup in artifacts/wfa/aggregate/rollup/.
3) Start leader validation queue: artifacts/wfa/aggregate/20260116_leader_validation/run_queue.csv.

After leader holdout DONE:
1) Update docs/optimization_runs_20260116.md (metrics + filtering summary for leader holdout).
2) Update rollup in artifacts/wfa/aggregate/rollup/.
3) Decide next stage (risk sweep vs signal grid refinement) based on holdout Sharpe/PnL/DD.

Notes:
- NOTE: Sharpe в записях/артефактах до фикса annualization (2026-01-18) занижен примерно в √96 раз для 15m; для актуальных значений используйте `coint4/artifacts/wfa/aggregate/rollup/run_index.*`.
- 2026-01-19: normalized_backtester Sharpe приведён к annualization 365*96 и учитывает нулевые доходности (см. `coint4/src/coint2/core/numba_kernels_v2.py`).
- 2026-01-18: shortlist WFA completed на 85.198.90.128, артефакты синхронизированы, сервер выключен.
- 2026-01-18: holdout + stress WFA завершены на 85.198.90.128, артефакты синхронизированы, сервер выключен.
- 2026-01-17: smoke WFA для проверки логирования команд (config main_2024_smoke.yaml, results artifacts/wfa/runs/logging_smoke_20260117_072821).
- 2026-01-17: next5_fast completed for signal_sweep_20260116_z0p85_exit0p06_ssd25000 (PnL 815.67, Sharpe 0.6345, DD -132.02).
- 2026-01-17: next5_fast completed for signal_sweep_20260116_z0p85_exit0p08_ssd25000 (PnL 821.12, Sharpe 0.6410, DD -128.32).
- 2026-01-17: next5_fast completed for risk_sweep_20260116_stop2p5_time3p5 (PnL 771.63, Sharpe 0.5860, DD -146.72).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p9_exit0p08_ssd25000 (PnL 741.27, Sharpe 0.5908, DD -146.05).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p9_exit0p1_ssd25000 (PnL 742.94, Sharpe 0.5933, DD -146.08).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p95_exit0p08_ssd25000 (PnL 685.02, Sharpe 0.5665, DD -114.16).
- 2026-01-17: next5_fast completed for risk_sweep_20260117_stop2p5_time2p0_z0p85_exit0p08_ssd25000 (PnL 821.12, Sharpe 0.6410, DD -128.32).
- 2026-01-17: next5_fast completed for risk_sweep_20260117_stop3p0_time2p5_z0p85_exit0p08_ssd25000 (PnL 821.12, Sharpe 0.6410, DD -128.32).
- 2026-01-17: next5_fast completed for risk_sweep_20260117_stop3p5_time3p0_z0p85_exit0p08_ssd25000 (PnL 821.12, Sharpe 0.6410, DD -128.32).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p85_exit0p06_ssd25000 (PnL 815.67, Sharpe 0.6345, DD -132.02).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p85_exit0p1_ssd25000 (PnL 821.86, Sharpe 0.6434, DD -124.82).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p8_exit0p06_ssd25000 (PnL 771.63, Sharpe 0.5860, DD -146.72).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p85_exit0p12_ssd25000 (PnL 855.78, Sharpe 0.6789, DD -95.32).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p85_exit0p09_ssd25000 (PnL 822.87, Sharpe 0.6441, DD -124.82).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p9_exit0p1_ssd25000 (PnL 742.94, Sharpe 0.5933, DD -146.08).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p8_exit0p1_ssd25000 (PnL 780.90, Sharpe 0.5965, DD -139.51).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p6_z0p85_exit0p12_ssd25000 (PnL 666.32, Sharpe 0.7452, DD -95.32).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_pv0p03_top800_z0p85_exit0p12_ssd25000 (PnL 789.56, Sharpe 0.6637, DD -98.47).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_hurst0p52_z0p85_exit0p12_ssd25000 (PnL 744.46, Sharpe 0.6037, DD -90.68).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_ssd15000_z0p85_exit0p12_ssd25000 (PnL 404.72, Sharpe 0.6275, DD -44.95).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p6_pv0p03_top800_z0p85_exit0p12_ssd25000 (PnL 621.19, Sharpe 0.7190, DD -91.63).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p6_hurst0p52_z0p85_exit0p12_ssd25000 (PnL 555.34, Sharpe 0.6492, DD -90.68).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_pv0p03_top800_kpss0p03_z0p85_exit0p12_ssd25000 (PnL 946.25, Sharpe 0.5384, DD -193.91).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p55_z0p85_exit0p12_ssd25000 (PnL 666.32, Sharpe 0.7452, DD -95.32).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p7_pv0p02_top500_z0p85_exit0p12_ssd25000 (PnL 456.81, Sharpe 0.6618, DD -78.36).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p65_hurst0p5_kpss0p03_z0p85_exit0p12_ssd25000 (PnL 620.39, Sharpe 0.5028, DD -126.15).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_ssd8000_pv0p03_top600_z0p85_exit0p12_ssd25000 (PnL 247.31, Sharpe 0.4533, DD -37.73).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_hl0p1_30_corr0p6_z0p85_exit0p12_ssd25000 (PnL 547.88, Sharpe 0.6830, DD -95.32).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p75_pv0p015_top300_hurst0p48_kpss0p03_z0p85_exit0p12 (PnL 95.37, Sharpe 0.1718, DD -105.85).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p7_pv0p01_top200_kpss0p02_hl0p2_20_z0p85_exit0p12 (PnL 623.38, Sharpe 0.4579, DD -230.50).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_ssd4000_corr0p65_pv0p02_top400_z0p85_exit0p12 (PnL 61.89, Sharpe 0.2783, DD -25.44).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p7_hurst0p5_kpss0p02_cross1_z0p85_exit0p12 (PnL 747.15, Sharpe 0.5444, DD -130.58).

## Update 2026-02-16 16:45Z (BL-EXEC bridge06 complete)

Текущее состояние:
- Исполнен batch-loop `20260216_budget1000_bl6` по `configs/autopilot/budget1000_batch_loop_bridge06_20260216.yaml`.
- Remote heavy runs проведены на `85.198.90.128` через `scripts/remote/run_server_job.sh`, `STOP_AFTER=1`.
- Контроллер завершился по `max_rounds_reached: max_rounds=6`.
- Prefix queue status: `completed=426`, `planned=12`, `stalled=0`, `total=438`.
- Best текущего цикла: `20260216_budget1000_bl6_r01_vm` (`score=4.3447411319`, `worst_robust_sharpe=4.3874465813`, `worst_dd_pct=0.1026690906`).
- Обнаружен planned-only fallback queue `20260216_budget1000_bl6_r02_vm` (`planned=12`) без исполнения; это ожидаемо при fallback-ветке контроллера.
- Выполнен post-sync: `sync_queue_status.py` для `bl6_r*` и `build_run_index.py`.
- Rollup обновлён: `coint4/artifacts/wfa/aggregate/rollup/run_index.*` (`entries=3705`).

Что дальше:
- Передать цикл аналитику для сравнения `bl6` против completed-пула и подготовки `bl7` (или стоп-вердикта по PRD guardrail).
