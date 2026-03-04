#!/usr/bin/env python3
"""Track per-queue progress and ETA heuristics for autonomous WFA driver."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict


def load_state(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(path: Path, state: Dict[str, dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--pending", type=int, required=True)
    parser.add_argument("--completed", type=int, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--planned", type=int, required=True)
    parser.add_argument("--running", type=int, required=True)
    parser.add_argument("--stalled", type=int, required=True)
    args = parser.parse_args()

    queue = args.queue
    pending = args.pending
    completed = args.completed
    total = args.total
    now = time.time()

    state = load_state(Path(args.state))
    entry = state.get(queue, {})

    last_ts = float(entry.get("ts", now))
    last_completed = int(entry.get("completed", completed))
    last_pending = int(entry.get("pending", pending))
    last_progress_ts = float(entry.get("progress_ts", now))
    rate_per_min = float(entry.get("rate_per_min", 0.0))

    progressed = completed > last_completed
    if progressed:
        dt = max(1e-6, now - last_ts)
        delta = max(0, completed - last_completed)
        if dt > 0:
            rate_per_min = (delta / dt) * 60.0
        last_progress_ts = now

    if pending != last_pending:
        stale_sec = 0.0
    else:
        stale_sec = max(0.0, now - float(last_progress_ts))

    eta_min = None
    if pending > 0 and rate_per_min > 0:
        eta_min = max(0.0, pending / rate_per_min)

    done = (completed >= total) or (pending <= 0)

    entry = {
        "ts": now,
        "pending": pending,
        "completed": completed,
        "total": total,
        "running": args.running,
        "planned": args.planned,
        "stalled": args.stalled,
        "progress_ts": last_progress_ts,
        "rate_per_min": rate_per_min,
        "eta_min": eta_min,
        "stale_sec": stale_sec,
        "done": done,
        "updated": now,
    }
    state[queue] = entry
    save_state(Path(args.state), state)

    stale_out = f"{stale_sec:.1f}"
    eta_out = "NA" if eta_min is None else f"{eta_min:.1f}"
    rate_out = f"{rate_per_min:.3f}"

    # Output for shell parsing.
    print(f"{queue}|{pending}|{completed}|{rate_out}|{eta_out}|{stale_out}|{int(done)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
