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
  - Ранкер запускается на `rollup_r06/run_index.csv` с:
    - `--fullspan-policy-v1 --min-windows 1`
    - `--min-pairs 20 --min-trades 200 --min-pnl 0 --min-coverage-ratio 0.95`
  - Есть минимум 1 прошедший вариант (ranker exit=0).

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
- [ ] Шаг 4: tracked/commit
- [ ] Шаг 5: heavy r06 на VPS
- [ ] Шаг 6: mini-rollup r06
- [ ] Шаг 7: strict ranking r06
- [ ] Шаг 8: top-5 summary в этом файле
- [ ] Шаг 9: fullspan-confirm queue top-3
- [ ] Шаг 10: lint+test

