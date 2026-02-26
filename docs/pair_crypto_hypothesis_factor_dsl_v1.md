# Pair-Crypto Hypothesis/Factor DSL v1

Canonical schema path:
- `coint4/scripts/optimization/schemas/pair_crypto_hypothesis_factor.dsl.v1.schema.json`

Example payload:
- `coint4/scripts/optimization/schemas/pair_crypto_hypothesis_factor.dsl.v1.example.json`

Validator:
- `coint4/scripts/optimization/hypothesis_factor_dsl.py`

## Scope
This DSL defines a strict JSON structure for LLM-generated pair-crypto hypotheses.
A payload must include:
- top-level DSL version and scope,
- one or more hypotheses,
- for each hypothesis: expected effects, factor changes, and WFA checks.

## Validation command
From `coint4/` app root:

```bash
python scripts/optimization/hypothesis_factor_dsl.py \
  --input-path scripts/optimization/schemas/pair_crypto_hypothesis_factor.dsl.v1.example.json
```

Exit code:
- `0` = valid payload
- `1` = invalid payload (schema errors printed to stderr)

## Materialize to executable YAML patch
From `coint4/` app root:

```bash
python scripts/optimization/hypothesis_factor_dsl.py \
  --input-path scripts/optimization/schemas/pair_crypto_hypothesis_factor.dsl.v1.example.json \
  --hypothesis-id HYP-TAIL_GUARD_01 \
  --base-config-path configs/prod_final_budget1000.yaml \
  --materialize-output-path artifacts/tmp/hypothesis_patch.yaml
```

Safety rules in materializer:
- only safe config roots are allowed (`backtest`, `pair_selection`, `portfolio`, `filter_params`, `walk_forward`, `data_processing`, `data_filters`, `time`, `risk`, `guards`);
- duplicate `target_key` in one hypothesis is rejected;
- `scale`/`offset` require `--base-config-path` and existing numeric base value;
- numeric `bounds` are enforced on resolved values.
