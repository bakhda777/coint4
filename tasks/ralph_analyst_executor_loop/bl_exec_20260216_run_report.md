# BL-EXEC Run Report (2026-02-16 UTC)

## Scope
- Task: `BL-EXEC` (execute recommended batch + post-sync statuses/rollup).
- Recommended config: `coint4/configs/autopilot/budget1000_batch_loop_bridge01_20260216.yaml`.
- Target VPS: `85.198.90.128` via `coint4/scripts/remote/run_server_job.sh` with `STOP_AFTER=1`.

## Command Execution
- Exact BL-ANL command was attempted first and failed before queue execution.
- Blocker: nested `run_server_job.sh` call from inside remote `autopilot` had no `SERVSPACE_API_KEY` on VPS side.
- Safe fallback used (same config, no knob changes): local `autopilot` launch so heavy part still executed on VPS via helper.

## Executed Run Groups
- `20260216_budget1000_bl_r01_risk`: queue_size=36, completed=36, stalled=0, planned=0.
- `20260216_budget1000_bl_r02_slusd`: queue_size=30, completed=30, stalled=0, planned=0.
- `20260216_budget1000_bl_r03_vm`: queue_size=24, completed=24, stalled=0, planned=0.
- Total: queue_size=90, completed=90, stalled=0, planned=0.

## Mandatory Post-Sync
- `sync_queue_status.py` executed for all three queues (`no changes`, `metrics_present=36/30/24`, `missing=0`).
- `build_run_index.py` executed for rollup (`run_index entries=2196`).

## Key Metrics
- stop_reason: `max_rounds_reached (max_rounds=3)`.
- best run_group: `20260216_budget1000_bl_r03_vm`.
- best score: `3.2501`.
- best worst-window robust Sharpe: `3.5605`.
- best worst-window DD pct: `0.1710`.

## Loop Handoff
- Executor completed and returned cycle to analyst (`BL-ANL=false`, `BL-EXEC=true`, `BL-CLOSE=false` in local ralph tracker state).
