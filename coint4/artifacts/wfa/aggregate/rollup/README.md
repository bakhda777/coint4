# WFA rollup index

This folder stores the consolidated index of WFA runs for analysis and ranking.

Regenerate from `coint4/`:
```bash
PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py \
  --output-dir artifacts/wfa/aggregate/rollup
```
