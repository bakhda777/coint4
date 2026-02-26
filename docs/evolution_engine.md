# Evolution Engine (оптимизация): дизайн-док и контракт v1

Цель: зафиксировать **детерминированную** и **трассируемую** архитектуру “evolution engine”, который итеративно предлагает новые конфиги для WFA/holdout+stress прогонов и обновляет состояние поиска на основе rollup-метрик.

## Scope / Non-goals

В scope:
- Генерация кандидатов (конфигов) через набор операторов.
- Similarity/dedup + diversity (чтобы не тратить прогоны на дубликаты).
- Reward decomposition (явная формула score + hard-gates) и протокол выбора элиты.
- Артефакты и трассировка (state + decisions), чтобы любой шаг можно было replay.

Не в scope (v1):
- Реализация бэктестера/движка WFA (используем существующие `run_wfa_queue.py` / rollup).
- Запуск тяжёлых прогонов локально (heavy только на VPS по AGENTS.md).
- “Интеллектуальное” предсказание результатов без прогонов (ML-surrogate).

## Термины

- **Genome**: словарь `{dotted_key -> value}` по заранее объявленному knob-space.
- **Phenotype**: материализованный YAML-конфиг стратегии (base_config + genome overrides).
- **Candidate**: genome + lineage (родители/оператор/seed/generation) + materialization (пути файлов).
- **Evaluation**: агрегированные метрики варианта из rollup (`run_index.csv`), включая holdout+stress и multi-window.
- **Reward**: score как сумма/композиция терминов + hard-gates (fail-closed).
- **Similarity**: расстояние между genome’ами (MVP: параметрическое).

## Архитектура (высокоуровнево)

Engine — это **state machine** вокруг существующего WFA пайплайна:

1) `ingest`:
   - читает `run_index.csv` (rollup) + (опционально) tail-данные из `daily_pnl.csv`,
   - матчится с ранее предложенными кандидатами по `candidate_id` / `run_id` / `config_path`,
   - рассчитывает reward decomposition (компоненты + итоговый score),
   - обновляет state: history + elite + stop-условия.

2) `propose`:
   - выбирает parents (обычно elite),
   - применяет операторы (mutation/crossover/restart/coord-sweep),
   - проверяет ограничения knob-space + hard-gates “до прогона” (структурные/guardrails),
   - делает similarity-dedup (MVP: по genome),
   - формирует batch (список кандидатов для запуска).

3) `emit`:
   - пишет YAML-конфиги в `coint4/configs/...`,
   - пишет очередь `run_queue.csv` в `coint4/artifacts/wfa/aggregate/<run_group>/`,
   - фиксирует decision-артефакт (JSON) и обновляет engine state (JSON).

Принцип: **evaluation отделён** (выполняется существующими скриптами и/или на VPS), а engine — **детерминированный генератор решений**.

## Канонические артефакты и пути

Для совместимости с текущими конвенциями (см. AGENTS.md):

- Очередь прогонов (маленький артефакт, коммитабельно):
  - `coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv`
- Сгенерённые конфиги (маленькие, коммитабельно):
  - `coint4/configs/evolution/<run_group>/*.yaml`
- Тяжёлые результаты (не коммитить):
  - `coint4/artifacts/wfa/runs_clean/<run_group>/<run_id>/`
- Состояние и решения engine (маленькие, коммитабельно по умолчанию):
  - `coint4/artifacts/wfa/aggregate/<controller_group>/evolution_state.json`
  - `coint4/artifacts/wfa/aggregate/<controller_group>/decisions/<decision_id>.json`

## Детерминизм (обязательные инварианты)

1) **Единый RNG**: `numpy.random.Generator(PCG64)`; никаких `random.*`, никаких hidden global seeds.
2) **Снимок RNG state**: в state и в каждом decision сохраняется `bit_generator.state` (JSON-совместимый dict).
3) **Стабильная сортировка входов**:
   - строки rollup сортируются по `(run_group, variant_id, run_id)` перед агрегированием,
   - parents сортируются по `candidate_id` (и затем по hash) перед применением операторов.
4) **Candidate ID**:
   - `candidate_id = "evo_" + sha256(normalize(engine_id,generation,operator_id,parents,genome))[:12]`
   - normalize = JSON canonical form: `json.dumps(obj, sort_keys=True, separators=(",",":"))`
5) **Имена файлов** (config/results) не должны зависеть от wall-clock; допустимо включать `generation` и `candidate_id`.
6) **Dedup** всегда выбирает победителя детерминированно (tie-break по `candidate_id`).

## Reward decomposition (v1)

Reward считается только на основании **явно объявленных** компонентов. Любая неоднозначность/пропуск данных => **fail-closed** (кандидат не попадает в elite/promote).

### Базовые метрики (определения)

Для пары `holdout_*` и `stress_*` внутри окна:
- `robust_sharpe_window = min(sharpe_holdout, sharpe_stress)`
- `robust_pnl_window = min(pnl_holdout, pnl_stress)`
- `dd_window_pct = max(abs(dd_holdout_pct), abs(dd_stress_pct))`

Агрегация по OOS окнам (multi-window):
- `worst_robust_sharpe = min_window(robust_sharpe_window)`
- `worst_robust_pnl = min_window(robust_pnl_window)`
- `worst_dd_pct = max_window(dd_window_pct)`

### Профили reward

**1) `short_oos_v1` (по умолчанию для max_steps<=5 / shortlist)**

Hard-gates (минимальный набор):
- `metrics_present=true` для holdout+stress.
- `min_trades >= 200`, `min_pairs >= 20` (параметризуется).
- `worst_dd_pct <= max_dd_pct`.
- `worst_robust_pnl >= min_pnl` (safe default: `0`).

Score (скалярный, для ранжирования внутри прошедших hard-gates):
- `score = worst_robust_sharpe - dd_penalty(worst_dd_pct)`
- `dd_penalty(dd) = dd_penalty_k * max(0, dd - dd_target_pct)`

**2) `fullspan_v1` (обязателен для promote/cutover)**

Канонический контракт вынесен отдельно:
- `docs/fullspan_selection_policy.md`

Engine обязан:
- применять hard-gates и `score_fullspan_v1` из этого документа (без послаблений в promote_profile),
- при отсутствии tail-данных / неполных метрик ставить `promotion_verdict=NO_PROMOTE` (fail-closed).

### Что сохранять в state/decision для трассируемости

Для каждого evaluated кандидата:
- все компоненты reward (worst_robust_sharpe, worst_dd_pct, worst_robust_pnl, trades/pairs, tail metrics при наличии),
- итоговый `score` и `gate_failures[]` (если есть),
- `profile` (short_oos_v1/fullspan_v1) и все параметры порогов/penalty.

## Similarity / diversity (v1)

MVP similarity — **только по genome** (параметрам), без curve-matching.

### Distance: `genome_weighted_l1_v1`

Для каждого ключа `k`:
- если значения категориальные/булевы: `d_k = 0 если равны иначе 1`
- если числовые: `d_k = |a-b| / norm_k`, где `norm_k` берём из knob-space (`range=max-min` или `step`, safe default: `range`, иначе `1`).

Общая дистанция:
- `D(a,b) = sum_k(w_k * d_k) / sum_k(w_k)`

### Dedup policy

- Дубликат, если существует `b` в (batch ∪ history) с `D(a,b) < dedupe_threshold`.
- При коллизии сохраняем кандидата с **меньшим** `candidate_id` (детерминированный tie-break).

### Diversity (опционально, v1)

Если нужно поощрять novelty:
- `novelty(a) = min_{e in elite} D(a,e)`
- добавляем в score отдельным термином `+ novelty_weight * novelty(a)` **только на этапе отбора batch**, не для promote.

## MVP операторы (v1)

Все операторы:
- принимают `parents[]`, `rng`, `knob_space`,
- возвращают `child_genome` + `operator_id` + `parents[]` + (опционально) `notes`,
- не читают файловую систему и не используют время (кроме `decision_id` как метадаты).

### 1) `mutate_step_v1`

Выбирает 1 ключ `k` (по weights/распределению), затем:
- числовой knob: `v' = clip(v + s * step_k)`, где `s ∈ {-1,+1}` из RNG,
- булев: flip,
- categorical: выбрать соседнее/случайное из `choices`.

Квантование (детерминированно):
- float округлять до `round_to` знаков (если задано), иначе до точности `step_k`.

### 2) `crossover_uniform_v1`

Берёт 2 родителей и для каждого ключа выбирает значение из parent A/B с `p=0.5`.

### 3) `random_restart_v1`

Сэмплирует genome из prior’ов knob-space (uniform по диапазону/choices) — для “выхода из локального оптимума”.

### 4) `coordinate_sweep_v1` (микросвип вокруг elite)

Детерминированно генерирует небольшой набор точек вокруг parent:
- для набора ключей `K` строит значения `{v-step, v, v+step}` (с клипом),
- ограничивает размер декартова произведения (budget) и выдаёт первые N в **стабильном** порядке.

## Контракт decision-артефакта (JSON)

Каждый шаг `propose` обязан сохранять JSON, который валидируется схемой:
- `coint4/scripts/optimization/schemas/evolution_engine_decision.schema.json`

Decision включает:
- параметры reward decomposition + similarity policy,
- knob-space snapshot,
- rng seed + rng state,
- список proposals (кандидаты, lineage, genome, materialization).

## Fail-closed правила (обязательные)

- Нет paired `holdout_*/stress_*` => кандидат не проходит `metrics_present`.
- Нет `daily_pnl.csv` там, где требуется `fullspan_v1` => `NO_PROMOTE`.
- Любая неоднозначность в метриках/окнах/статусах => кандидат не в elite/promote.

