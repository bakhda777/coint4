# Implementation Plan (Protocol v2.1.1)

## Context
Coint4 is a crypto market-neutral pairs trading research codebase focused on cointegration-based strategies. We run backtests and walk-forward analysis (WFA) to improve net Sharpe under realistic risk/drawdown and fee/slippage constraints.

## Goals
- [ ] Improve net Sharpe for market-neutral pair trading under realistic fees/slippage and drawdown constraints
- [ ] Keep the WFA/backtest pipeline reproducible, comparable across runs, and easy to audit from configs + artifacts
- [ ] Maintain a clear repo structure (canonical paths/commands) so results, logs, and rollups are unambiguous

## Non-Goals
- Rewriting the entire system/strategy without WFA evidence and a scoped migration plan
- Adding a paper-trading (demo/testnet) stage as part of the workflow (cutover is intended to go straight to live when ready)
- Committing large run artifacts or retroactively editing historical artifacts directories without an explicit request

## Definition of Done (DoD)
- [ ] Logic is covered by tests (Unit/Integration)
- [ ] Linter/Formatter passes (no warnings)
- [ ] Type checks pass (mypy/pyright/tsc --noEmit)
- [ ] Manual smoke-test verified (if applicable)
- [ ] Reproducible run verified (same inputs/config => same outputs/metrics within an agreed tolerance)
- [ ] Postprocess выполнен единообразно для блока прогонов: `sync_queue_status.py -> build_run_index.py -> rank_multiwindow_robust_runs.py`
- [ ] Источник истины для отбора явно зафиксирован и использован: `artifacts/wfa/aggregate/rollup/run_index.csv` + `rank_multiwindow_robust_runs.py --fullspan-policy-v1` + gates (`min_windows/min_trades/min_pairs/max_dd_pct/min_pnl/tail_worst_gate_pct`)
- [ ] No edits to existing artifacts directories unless explicitly requested
- [ ] Minimal smoke-run command is documented and verified (placeholder allowed): UNKNOWN
- [ ] No TODOs left without a ticket/issue link

## Phases
- [ ] **Phase 1: Config/Contracts**
- [ ] **Phase 2: Core Change + Tests** (TDD: Red -> Green)
- [ ] **Phase 3: Small Regression Run**
- [ ] **Phase 4: WFA/Hardening** (Remote runs, rollups, docs, edge cases)

## Phase Brief (Copy/Paste Per Phase)
<!-- Paste this block at the beginning of each phase to keep context clean. -->
- Phase: <!-- e.g. Phase 2: Core Logic & Tests -->
- Objective: <!-- one sentence -->
- Scope: <!-- key files/modules -->
- Interfaces: <!-- inputs/outputs/APIs touched -->
- Invariants: <!-- must not break -->
- Risks: <!-- top 2 risks -->
- Failure modes: <!-- what can go wrong in prod -->
- Observability: <!-- logs/metrics/alerts to add or verify -->
- Test focus: <!-- what to add/verify -->
- Exit criteria: <!-- what means done -->

## Current Status
- **Current Phase:** Phase 1
- **Blocking Issues:** None
- **Last Green Commit:** <!-- hash/branch -->
