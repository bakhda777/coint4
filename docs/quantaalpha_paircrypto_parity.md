# QuantaAlpha Parity (Pair-Crypto) — 100% Checklist

Дата: 2026-02-26

Этот документ фиксирует, что именно значит “100% соответствие” подходу QuantaAlpha (2602.07085) **с поправкой на парную крипту**.

В paper QuantaAlpha строит факторы (alpha factors) как symbolic IR/AST поверх operator library, делает semantic consistency проверку между гипотезой ↔ IR ↔ кодом, а также вводит complexity/redundancy gates и self-evolution на уровне траекторий (mutation/crossover по сегментам траектории).

В нашем домене “фактор” = **исполняемая конфигурация парной стратегии** (pair-crypto). Поэтому symbolic IR/AST адаптируется как **ConfigPatch AST**: структурированное описание “что изменить в конфиге и зачем”, компилируемое в YAML/config.

## Определения (адаптация)

- **Hypothesis (h)**: текстовая гипотеза про механику pair-trading (например, риск/фильтры/guards/selection).
- **Semantic description (d)**: структурированная расшифровка гипотезы (ключи, операции, ожидаемые эффекты, проверки).
- **Symbolic IR (f)**: **ConfigPatch AST** (операторы над ключами конфига).
- **Executable (c)**: материализованный YAML patch и итоговый YAML config, который уходит в WFA/backtest.
- **Trajectory (τ)**: end-to-end шаг эволюции с узлами: (h, d, f, c, metrics, reflection).

## Parity Control Items (QPCI)

### QPCI-01 — ConfigPatch AST IR существует и валидируется
- Есть каноническая схема IR (JSON schema) и пример.
- IR не позволяет “свободный” произвольный YAML/код без структурного контроля.

### QPCI-02 — Компиляция IR → YAML patch → YAML config
- Есть компилятор (materializer), который:
  - fail-closed на небезопасных корнях/типах,
  - воспроизводимо применяет операции к базовому конфигу,
  - выдаёт patch/config без скрытых side-effects.

### QPCI-03 — Semantic consistency gate (h ↔ d ↔ f ↔ c)
- Есть **LLM-verifier** (codex exec) и фиксированные критерии:
  - ключи/операции в IR соответствуют описанной гипотезе и ожидаемым эффектам,
  - материализованный patch/config соответствует IR,
  - при провале — candidate отклоняется (fail-closed).

### QPCI-04 — Complexity control (структурная регуляризация)
- Для IR считается complexity score (аналог paper):
  - `SL(f)` — symbolic length (узлы AST),
  - `PC(f)` — число свободных параметров (числовые значения/окна/пороги),
  - `|F_f|` — число затронутых config keys (или roots).
- Кандидаты выше порога отклоняются/перегенерируются.

### QPCI-05 — Redundancy control (anti-crowding)
- Для IR считается redundancy score (AST similarity) относительно “зоопарка” уже принятых решений:
  - метрика должна быть структурной (не зависеть только от numeric value),
  - кандидаты выше порога отклоняются/перегенерируются.

### QPCI-06 — Diversified planning initialization
- Генерация N кандидатов (hypothesis+IR) должна обеспечивать комплементарность:
  - разные roots/механизмы,
  - низкая redundancy внутри батча (через QPCI-05).

### QPCI-07 — Trajectory-level mutation (локализация плохого сегмента)
- Mutation выбирает “сегмент” (часть IR/узел траектории), который вероятнее всего ответственен за провал (DD/trades/pairs/tails),
  и переписывает только этот сегмент, сохраняя остальное.

### QPCI-08 — Trajectory-level crossover (сегментная рекомбинация)
- Crossover комбинирует совместимые сегменты из нескольких “родительских” IR (например, risk-guards + pair-selection),
  с явной lineage/parents фиксацией.

### QPCI-09 — Артефакты траектории и история оценки
- Для каждого поколения пишется decision/state с:
  - IR/hypothesis + gate results (complexity/redundancy/semantic),
  - lineage (родители) и оператор,
  - связка к queue/config paths.

### QPCI-10 — Final factor pool (консолидация)
- Есть сборщик “factor pool” (лучшие варианты) по `run_index.csv` + decisions:
  - гипотеза + IR + метрики + lineage,
  - экспорт в JSON/MD.

## Текущий статус реализации (verified)

| QPCI | Статус | Evidence |
|---|---|---|
| QPCI-01 | PASS | `coint4/src/coint2/ops/config_patch_ast.py`, `coint4/src/coint2/ops/config_patch_ir.py`, `coint4/scripts/optimization/schemas/config_patch_ast.v1.schema.json`, `coint4/tests/utils/test_config_patch_ast.py`, `coint4/tests/utils/test_config_patch_ir.py` |
| QPCI-02 | PASS | `coint4/scripts/optimization/hypothesis_factor_dsl.py`, `coint4/tests/scripts/test_hypothesis_factor_dsl.py` |
| QPCI-03 | PASS | `coint4/scripts/optimization/semantic_consistency_gate.py`, `coint4/tests/utils/test_semantic_consistency_gate.py` |
| QPCI-04 | PASS | `coint4/src/coint2/ops/config_patch_gates.py`, `coint4/tests/utils/test_config_patch_gates.py` |
| QPCI-05 | PASS | `coint4/src/coint2/ops/config_patch_ast.py` (`redundancy_similarity`), `coint4/src/coint2/ops/config_patch_gates.py` |
| QPCI-06 | PASS | `coint4/scripts/optimization/evolve_next_batch.py` (diversity по root signatures + anti-dup) |
| QPCI-07 | PASS | `coint4/scripts/optimization/evolve_next_batch.py` (`_segment_mutation_candidate`), `coint4/tests/scripts/test_evolve_next_batch_patch_ast.py` |
| QPCI-08 | PASS | `coint4/scripts/optimization/evolve_next_batch.py` (`_segment_crossover_candidate`) |
| QPCI-09 | PASS | `coint4/scripts/optimization/evolve_next_batch.py` (decision/state lineage + gates) |
| QPCI-10 | PASS | `coint4/scripts/optimization/build_factor_pool.py`, `coint4/tests/scripts/test_build_factor_pool.py` |

## Критерий “100% parity”

Считаем “100%” для pair-crypto, когда QPCI-01..QPCI-10 реализованы и покрыты минимум:
- `make lint`
- `make test`
