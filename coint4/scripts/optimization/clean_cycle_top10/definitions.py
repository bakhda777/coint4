"""Local clean-cycle definitions (C03/C04 tasks).

Kept in a small dedicated module so CLI tools can use `--use-definitions`
without requiring an external JSON file.
"""

from __future__ import annotations

from typing import Any, Dict

# NOTE: This is intentionally minimal for C03 (validator). C04 may extend it with
# CLEAN_ROOT/BASELINE_DIR/OPT_DIR/CLEAN_AGG_DIR and other constants.
FIXED_WINDOWS: Dict[str, Any] = {
    "resolved_on": "2026-02-15",
    "resolution_note": "Checked on TOP-10 from rollup run_index.csv: all 10 share the same walk_forward params.",
    "walk_forward": {
        "start_date": "2024-05-01",
        "end_date": "2024-12-31",
        "training_period_days": 90,
        "testing_period_days": 30,
        "step_size_days": 30,
        "max_steps": 5,
        "gap_minutes": 15,
        "refit_frequency": "weekly",
    },
}

