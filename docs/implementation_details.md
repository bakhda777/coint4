# Implementation Details & Scratchpad (Protocol v2.1.1)

## Commands (Single Source of Truth)
Test Command: UNKNOWN
Lint/Format Command: UNKNOWN
Typecheck Command: UNKNOWN
Build/CI Command: UNKNOWN

## Repo Map
- Repo root: `/home/claudeuser/coint4`
- App workspace (Poetry project): `coint4/` (venv in `coint4/.venv`, code in `coint4/src/`, tests in `coint4/tests/`)
- Configs: `coint4/configs/` (strategy/WFA configs, search spaces, overlays)
- Scripts / entrypoints:
  - CLI scripts live in `coint4/scripts/`
  - Installed CLIs via Poetry: `coint2`, `coint2-optimize`, `coint2-live`, etc. (see `coint4/pyproject.toml`)
- Artifacts / outputs:
  - Heavy run artifacts: `coint4/artifacts/wfa/runs/<run_group>/<run_id>/`
  - Queues + rollups (small, tracked): `coint4/artifacts/wfa/aggregate/**` and `coint4/artifacts/wfa/aggregate/rollup/`
  - Other outputs: `coint4/outputs/`, `coint4/results/` (generated; typically not committed)
- Docs: `docs/` (with `coint4/docs -> ../docs` symlink)

## Invariants
- No lookahead bias: all train/test splits must be time-correct; WFA windows/gaps must prevent leakage
- Determinism: if any randomness is used, seed it and record it in artifacts/logs/config; runs should be reproducible given the same inputs
- Artifact write policy: never overwrite or edit existing artifacts directories unless explicitly requested; new runs write to new `run_id` paths

## Observability
- Metrics to track (gross and net where applicable): gross PnL, net PnL, fees/slippage, Sharpe, max drawdown, turnover, trade count, pair counts
- Logs to capture per run: run_id, config path/hash, data window, universe/pairs source, code version (commit), host, and any guardrail violations

## Contract Examples (Truth Table)
| Input | State | Expected Output | Error/Exception |
|-------|-------|-----------------|-----------------|
| TODO: Load/validate `--config` (YAML) | Config file missing or invalid YAML | Fail fast; error mentions the config path and why it failed | TODO: define (SystemExit / yaml.YAMLError) |
| TODO: Run backtest/WFA with `--pairs-file` | Pairs file references symbols missing in data | Policy decided and enforced: skip-with-warning OR fail-fast | TODO: define |
| TODO: Run WFA with `--results-dir` | Results dir exists and is non-empty | Refuse to overwrite unless explicitly forced | TODO: define (FileExistsError / SystemExit) |
| TODO: Build rollup index (`build_run_index.py`) | Some run dirs missing required metrics files | Index build completes; missing runs marked incomplete and logged | TODO: define (warn-only) |
| TODO: Queue watcher / runner (`run_wfa_queue.py` / `watch_wfa_queue.sh`) | `walk_forward.max_steps` missing or exceeds guardrail | Guardrail triggers before any heavy work starts | TODO: define (SystemExit) |
| TODO: Config has invalid fee/slippage/risk params | Negative or out-of-range values | Config validation rejects and points to the bad field | TODO: define (pydantic.ValidationError) |

## Log / Scratchpad
2026-02-14: Phase 0.5 docs customization only (updated `docs/implementation.md` and `docs/implementation_details.md`). No code changes and no runtime logs/stack traces yet.
