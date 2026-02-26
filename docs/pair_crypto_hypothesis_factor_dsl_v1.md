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
