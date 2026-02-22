# Fullspan Selection Policy v1

Last updated: 2026-02-22

Цель: единый контракт отбора для fullspan, чтобы winner не выбирался по среднему Sharpe на коротком OOS без учёта хвостового риска.

Статус документа:
- Этот файл является единственным каноническим контрактом отбора для `fullspan` (v1).
- При конфликте с дневниками прогонов, заметками раундов и историческими отчётами приоритет у этого файла.

## Scope

- Политика обязательна для final-promo в fullspan-контуре (`walk_forward.max_steps=null` или эквивалентный полный диапазон дат).
- Short OOS (`max_steps<=5`) разрешён только как pre-filter/shortlist.
- Если fullspan + tail-risk diagnostics недоступны, решение fail-closed: только shortlist, без промоута в live.

## Enforcement (обязательное применение)

- Финальный выбор winner для live/cutover выполнять только через fullspan-ранжирование по этому контракту (`score_fullspan_v1` + hard-gates).
- Инвариант final-rank ключа: в fullspan-режиме использовать только `score_fullspan_v1`; `avg_robust_sharpe` не может быть primary ranking key.
- Если ранкер поддерживает `--fullspan-policy-v1`, использовать его как обязательный режим.
- Если ранкер не поддерживает `--fullspan-policy-v1`, применять контракт вручную (decision memo + проверка формулы/gates) и держать verdict `NO_PROMOTE` до подтверждённого fullspan-pass.
- В автоматизациях, где `avg_robust_sharpe` присутствует, он считается диагностикой и не может быть ключом финального ранжирования.
- Для `short OOS` победитель не фиксируется как final winner без отдельного fullspan replay по этому контракту.
- Прямой запрет: нельзя выбирать winner только по среднему Sharpe на коротком OOS (`max_steps<=5`), даже при положительном PnL.
- Safe default при любой неоднозначности (нет tail-данных, смешанные окна, неполные метрики): `NO_PROMOTE` (fail-closed, только shortlist).

## Source of truth

- Rollup: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`.
- Tail-risk: `daily_pnl.csv` из paired run'ов `holdout_*` + `stress_*`.

## Hard-gates (обязательные)

- `metrics_present=true` для holdout+stress.
- `min(total_trades) >= 200` и `min(total_pairs_traded) >= 20` (или более строгие пороги в конкретном цикле).
- `worst_dd_pct = max_window(max(abs(dd_holdout), abs(dd_stress))) <= max_dd_pct`.
- `worst_robust_pnl = min_window(min(pnl_holdout, pnl_stress)) >= 0`.
- `worst_step_pnl >= -0.20 * initial_capital` (для `$1000` это `>= -200`).
- Любой провал хотя бы одного hard-gate => кандидат отклоняется и не может быть promoted в live.

## Canonical defaults (safe baseline v1)

- `min_windows=1` для fullspan replay (одно fullspan-окно допустимо).
- `min_trades=200`.
- `min_pairs=20`.
- `max_dd_pct=0.50`.
- `min_pnl=0`.
- `initial_capital=1000`.
- `tail_quantile=0.20`.
- `tail_q_soft_loss_pct=0.03`.
- `tail_worst_soft_loss_pct=0.10`.
- `tail_q_penalty=2.0`.
- `tail_worst_penalty=1.0`.
- `tail_worst_gate_pct=0.20`.
- Разрешено только ужесточение порогов для конкретного цикла; ослабление порогов требует новой ревизии политики.

## Operational profiles (research vs promote)

- `promote_profile` (обязательный для live/cutover): использовать только канонические default-пороги из этого файла, включая `tail_worst_gate_pct=0.20`, без послаблений.
- `research_profile` (диагностика/анализ): допускаются временные диагностические прогоны с ослабленными порогами (например, `tail_worst_gate_pct=0.21`), но такие результаты не могут менять `promotion_verdict` и не являются основанием для promote.
- Для аудируемости в дневнике явно помечать профиль запуска: `profile=promote_profile|research_profile`.

## Score (канонический objective)

- `worst_robust_sharpe = min_window(min(sharpe_holdout, sharpe_stress))`.
- `robust_daily_pnl_t = min(holdout_daily_pnl_t, stress_daily_pnl_t)` по пересечению дат.
- `worst_step_pnl = min(robust_daily_pnl_t)`.
- `q20_step_pnl = quantile(robust_daily_pnl_t, 0.20)`.
- `score_fullspan_v1 = worst_robust_sharpe - 2.0 * max(0, (-q20_step_pnl / initial_capital) - 0.03) - 1.0 * max(0, (-worst_step_pnl / initial_capital) - 0.10)`.

Tie-break при равном `score_fullspan_v1`:
`worst_robust_pnl` (выше лучше) -> `worst_dd_pct` (ниже лучше) -> `avg_robust_sharpe` (выше лучше).

## Decision memo contract (обязательное поле фиксации)

Для каждого fullspan-решения в `docs/optimization_runs_YYYYMMDD.md` и/или `docs/optimization_state.md` фиксировать:
- `selection_policy=fullspan_v1`
- `selection_mode=fullspan` (или `shortlist_only`, если fullspan не выполнен)
- `promotion_verdict=promote|reject`
- `rejection_reason=<code>` при `reject` (например: `TAIL_DATA_MISSING`, `WORST_STEP_GATE_FAIL`, `ECONOMIC_GATE_FAIL`)

## Statistical diagnostics (обязательные)

- При наличии данных использовать `PSR` и `DSR` как обязательные diagnostics в отчёте ранжирования fullspan.
- До нормализации шкалы `DSR` в текущем пайплайне запрещено включать `min_dsr` как hard gate в controller-конфигах (держать `min_dsr: null`); `DSR` используется как диагностическая метрика.
- Если `PSR/DSR` недоступны в текущем пайплайне, разрешён только временный fallback:
  - `stats_mode=fallback_no_psr_dsr`
  - `stats_reason=PSR_DSR_UNAVAILABLE_V1`
  - hard-gates и `score_fullspan_v1` остаются обязательными без послаблений.

## CLI (при наличии режима в ранкере)

Если в текущей ревизии ранкера доступен `--fullspan-policy-v1`, запускать его как канонический enforcement-режим:

- считает score по формуле выше;
- применяет hard-gate по `worst_step_pnl`;
- fail-closed отбрасывает вариант, если нельзя получить tail-samples из `daily_pnl.csv`.
- Для аудируемости запускать с явно заданными параметрами из секции `Canonical defaults`.

Пример:

```bash
cd coint4
PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_multiwindow_robust_runs.py \
  --run-index artifacts/wfa/aggregate/rollup/run_index.csv \
  --contains fullspan \
  --fullspan-policy-v1 \
  --min-windows 1 \
  --min-trades 200 \
  --min-pairs 20 \
  --max-dd-pct 0.50 \
  --min-pnl 0 \
  --initial-capital 1000 \
  --tail-quantile 0.20 \
  --tail-q-soft-loss-pct 0.03 \
  --tail-worst-soft-loss-pct 0.10 \
  --tail-q-penalty 2.0 \
  --tail-worst-penalty 1.0 \
  --tail-worst-gate-pct 0.20
```

Единый оркестратор цикла (рекомендуется):

```bash
cd coint4
PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_fullspan_decision_cycle.py \
  --queue artifacts/wfa/aggregate/<group>/run_queue.csv \
  --contains <group_or_tag>
```

Скрипт выполняет canonical sequence: status sync -> rollup rebuild (`--no-auto-sync-status`) -> strict promote profile -> research diagnostic profile.
