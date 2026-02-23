# План следующего цикла оптимизации (после r05b dd-focus) — цель Sharpe>3 на fullspan

Дата: 2026-02-23

## Глобальная цель (инвариант)

Достичь **`worst_robust_sh >= 3.0`** на всём доступном периоде (fullspan WFA), при ограничениях:
- `worst_pnl > 0`
- `worst_dd_pct <= 0.35` (жёстко, можно уточнить позже)
- `worst_step_pnl >= -200` (на `initial_capital=1000`)
- честные метрики: **coverage>=0.95** и нулевые окна/дни учитываются.

## Текущее состояние (кратко)

- Метрики “разреженности” починены: WFA записывает 0-PnL для окон без пар; coverage клипается в пределах `[start,end]`; end_date не overshoot’ится.
- `r05b dd-focus` (2023-06-29 → 2024-06-27) честно считается, но **строгий ранкер пустой**: `min_pairs` упирается в максимум 19.
- Решение: **не ослаблять гейты на dd-focus**, а сделать dd-focus “широким” через universe/selection.

## План (10 шагов) + критерии приёмки (DoD)

1) **Собрать wide pairs-universe для dd-focus (без изменения risk)**
- DoD:
  - Есть файл `coint4/configs/universe/ddfocus_wide_pairs_universe_v01.yaml`.
  - `pairs >= 300`.
  - В парах нет символов из denylist (`metadata.denylisted_symbols`).

2) **Сгенерировать r06 dd-focus run-group на wide-universe (14 вариантов вокруг v07)**
- DoD:
  - Папка `coint4/configs/budget1000_autopilot/20260223_tailguard_r06_ddfocus_wideuniverse/` содержит `28` YAML (holdout+stress).
  - Во всех YAML: `walk_forward.start_date=2023-06-29`, `walk_forward.end_date=2024-06-27`.
  - Во всех YAML: `walk_forward.pairs_file=configs/universe/ddfocus_wide_pairs_universe_v01.yaml`.
  - Во всех YAML: **отключён** pair-stability (`pair_selection.pair_stability_*` отсутствуют).
  - `pair_selection.max_pairs` установлен (единый) и > 21 (не risk-ось).

3) **Собрать aggregate-артефакты очереди r06 и сделать dry-run**
- DoD:
  - Есть `coint4/artifacts/wfa/aggregate/20260223_tailguard_r06_ddfocus_wideuniverse/run_queue.csv` (28 planned).
  - Есть `search_space.csv` и `search_space.md` рядом.
  - `scripts/optimization/run_wfa_queue.py --dry-run` по очереди проходит (exit=0).

4) **Зафиксировать генераторы/конфиги в git (для SYNC_UP на VPS)**
- DoD:
  - В репо есть генераторы:
    - `coint4/scripts/universe/build_ddfocus_wide_pairs_universe.py`
    - `coint4/scripts/optimization/generate_tailguard_r06_ddfocus_wideuniverse.py`
  - Новые файлы добавлены в git (tracked).

5) **Прогнать heavy WFA очередь r06 на VPS (параллельность 10)**
- DoD:
  - На VPS `85.198.90.128` очередь выполнена: `completed=28/28`.
  - `SYNC_BACK=1` вернул результаты в `coint4/artifacts/wfa/runs/20260223_tailguard_r06_ddfocus_wideuniverse/...`.
  - VPS выключен после job (SSH/ping не отвечает).

6) **Локальный mini-rollup только для r06**
- DoD:
  - `sync_queue_status.py` обновил статусы очереди r06 локально.
  - Собран `coint4/artifacts/wfa/aggregate/rollup_r06/run_index.csv` (28 entries), без legacy coverage.

7) **Ранкинг r06 dd-focus со строгими гейтами**
- DoD:
  - Запущен строгий ранкер на `rollup_r06/run_index.csv` с:
    - `--fullspan-policy-v1 --min-windows 1`
    - `--min-pairs 20 --min-trades 200 --min-pnl 0 --min-coverage-ratio 0.95`
  - Зафиксировано, сколько вариантов прошло (если 0, это допустимый результат fast-loop).
  - Для анализа обязательно выполнен диагностический ранк с ослабленным `--min-pnl` (например, `-200`) при тех же гейтах ширины/coverage, и получен top-list.

8) **Сводка результатов r06 (top-5)**
- DoD:
  - В этот `plan.md` внесены top-5 (variant_id + ключевые метрики: worst_robust_sh, worst_pnl, worst_dd_pct, worst_step_pnl, coverage, trades, pairs).
  - Указано, сколько кандидатов отсеялось по coverage и по ширине.

9) **Подготовить fullspan-confirm queue для top-3 r06 (без heavy прогона)**
- DoD:
  - Создан новый run-group `20260223_tailguard_r07_fullspan_confirm_top3`:
    - 6 YAML (top-3 * holdout+stress) с fullspan `walk_forward.start_date=2022-03-01`, `end_date=2025-06-30`.
    - `run_queue.csv` + dry-run проходит.

10) **Проверки**
- DoD:
  - `make lint` проходит.
  - `make test` проходит.

## Прогресс / заметки выполнения

- [x] Шаг 1: wide pairs-universe (`pairs=519`)
- [x] Шаг 2: r06 configs с dd-focus датами, wide pairs_file, stability off, `max_pairs=60`
- [x] Шаг 3: r06 queue/search_space + dry-run ок
- [x] Шаг 4: tracked/commit (commit `2afb3a1`)
- [x] Шаг 5: heavy r06 на VPS (28/28 completed, VPS выключен)
- [x] Шаг 6: mini-rollup r06 (`coint4/artifacts/wfa/aggregate/rollup_r06/run_index.csv`, 28 entries)
- [x] Шаг 7: strict ranking r06 (min_pnl=0 → 0 кандидатов; причина: все варианты убыточны на dd-focus)
- [x] Шаг 8: top-5 summary в этом файле
- [x] Шаг 9: fullspan-confirm queue top-3 (r07 создан + dry-run ok)
- [x] Шаг 10: lint+test (ok)

## r06 dd-focus (2023-06-29 → 2024-06-27) — результаты ранкинга

Контекст гейтов для dd-focus:
- `min_windows=1`, `min_pairs=20`, `min_trades=200`, `min_coverage_ratio=0.95`
- `min_pnl=0` (строго) → **0/14** прошли (все варианты убыточны на dd-focus).
- Диагностический ранк (ослаблено): `min_pnl=-200` (остальные гейты прежние), `fullspan-policy-v1` (tail-penalty включён).

### Top-5 (diagnostic, min_pnl=-200)

| rank | variant_id | worst_robust_sh | worst_pnl | worst_dd_pct | worst_step_pnl | coverage(min) | trades(min) | pairs(min) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `tailguard_r06_ddfocus_v02_h1_trade_mild_A` | -0.959 | -101.23 | 0.156 | -42.78 | 1.00 | 1709 | 29 |
| 2 | `tailguard_r06_ddfocus_v13_h12_tradeA_cross3` | -0.959 | -101.23 | 0.156 | -42.78 | 1.00 | 1709 | 29 |
| 3 | `tailguard_r06_ddfocus_v03_h1_trade_balanced_B` | -0.964 | -124.74 | 0.172 | -51.18 | 1.00 | 2183 | 28 |
| 4 | `tailguard_r06_ddfocus_v04_h1_trade_hard_C` | -0.964 | -124.74 | 0.172 | -51.18 | 1.00 | 2183 | 28 |
| 5 | `tailguard_r06_ddfocus_v05_h1_diag_bidask_only` | -0.968 | -98.62 | 0.163 | -42.57 | 1.00 | 1774 | 30 |

### Отсев по гейтам (r06, 14 вариантов)

- Coverage: 0 отсеяно (везде `coverage_ratio=1.0` и нулевые дни присутствуют).
- Ширина (min_pairs/min_trades): 1 отсеяно (v12 KPSS: `pairs=11`, `trades=325`).
- PnL (min_pnl=0): 14/14 отсеяно (все варианты убыточны в dd-focus, даже если проходят ширину).

## r07 fullspan confirm queue (top-3 из r06)

- Run group: `20260223_tailguard_r07_fullspan_confirm_top3`
- Configs: `coint4/configs/budget1000_autopilot/20260223_tailguard_r07_fullspan_confirm_top3/` (6 YAML)
- Queue: `coint4/artifacts/wfa/aggregate/20260223_tailguard_r07_fullspan_confirm_top3/run_queue.csv` (6 planned)
- Dry-run: ok (`scripts/optimization/run_wfa_queue.py --dry-run ...`)
