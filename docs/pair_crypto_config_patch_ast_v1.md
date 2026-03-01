# Pair-Crypto ConfigPatch AST (IR) v1

В QuantaAlpha (2602.07085) ключевая часть — промежуточный **symbolic IR/AST**, который “якорит” генерацию и позволяет делать:
- semantic consistency проверки (hypothesis ↔ IR ↔ executable),
- complexity control,
- redundancy control (anti-crowding) через AST similarity.

Для нашего домена **pair-crypto** IR — это не “фактор-формула”, а **структурированный патч конфига**.

## Канонический формат IR

На практике IR v1 реализован через существующий DSL:
- schema: `coint4/scripts/optimization/schemas/pair_crypto_hypothesis_factor.dsl.v1.schema.json`
- пример: `coint4/scripts/optimization/schemas/pair_crypto_hypothesis_factor.dsl.v1.example.json`

Внутри каждой `hypotheses[].factors[]`:
- `target_key` — dotted key в YAML конфиге,
- `op` — операция (`set|scale|offset|enable|disable`),
- `value` — параметр операции (если нужен),
- `bounds` — опциональные numeric bounds.

## AST view (структурный слой)

Для complexity/redundancy gates факторы конвертируются в детерминированный AST:
- модуль: `coint4/src/coint2/ops/config_patch_ast.py`
- JSON schema (derived view): `coint4/scripts/optimization/schemas/config_patch_ast.v1.schema.json`
- пример (derived view): `coint4/scripts/optimization/schemas/config_patch_ast.v1.example.json`
- узлы дерева строятся из `target_key` (иерархия ключей) + leaf `op:<op>`.

Пример (концептуально):

```
PATCH
  risk
    daily_stop_pct
      op:set
  pair_selection
    max_pairs
      op:set
```

## Валидация / материализация

Validator + materializer DSL:
- `coint4/scripts/optimization/hypothesis_factor_dsl.py`

См. `docs/pair_crypto_hypothesis_factor_dsl_v1.md` для команд.
