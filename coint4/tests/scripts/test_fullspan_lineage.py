from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_lineage_module(tmp_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "fullspan_lineage.py"
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(tmp_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[tmp_name] = module
    spec.loader.exec_module(module)
    return module


def _write_run_index(path: Path) -> None:
    rows = [
        {
            "run_id": "holdout_variant_x",
            "run_group": "confirm_fastlane_group_a",
            "status": "completed",
            "metrics_present": "true",
        },
        {
            "run_id": "stress_variant_x",
            "run_group": "confirm_fastlane_group_a",
            "status": "completed",
            "metrics_present": "true",
        },
        {
            "run_id": "holdout_variant_x",
            "run_group": "confirm_fastlane_group_b",
            "status": "completed",
            "metrics_present": "true",
        },
        {
            "run_id": "stress_variant_x",
            "run_group": "confirm_fastlane_group_b",
            "status": "completed",
            "metrics_present": "true",
        },
    ]
    fieldnames = ["run_id", "run_group", "status", "metrics_present"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_count_confirms_by_lineage_tracks_group_and_lineage_keys(tmp_path: Path) -> None:
    module = _load_lineage_module("fullspan_lineage_unit_test")
    registry_path = tmp_path / "confirm_lineage_registry.json"
    run_index_path = tmp_path / "run_index.csv"
    _write_run_index(run_index_path)

    module.register_confirm_queue(
        registry_path=registry_path,
        queue_rel="artifacts/wfa/aggregate/confirm_fastlane_group_a/run_queue.csv",
        source_queue_rel="artifacts/wfa/aggregate/source_a/run_queue.csv",
        candidate_uid="cand_demo",
        top_run_group="source_a",
        top_variant="variant_x",
        dispatch_id="d1",
    )
    module.register_confirm_queue(
        registry_path=registry_path,
        queue_rel="artifacts/wfa/aggregate/confirm_fastlane_group_b/run_queue.csv",
        source_queue_rel="artifacts/wfa/aggregate/source_a/run_queue.csv",
        candidate_uid="cand_demo",
        top_run_group="source_a",
        top_variant="variant_x",
        dispatch_id="d2",
    )

    stats = module.count_confirms_by_lineage(
        run_index_path=run_index_path,
        registry_path=registry_path,
        candidate_uid="cand_demo",
    )

    assert stats.confirmed_count == 2
    assert set(stats.confirmed_run_groups) == {"confirm_fastlane_group_a", "confirm_fastlane_group_b"}
    assert len(stats.confirmed_lineage_keys) == 2
    assert all(key.startswith("ln_") for key in stats.confirmed_lineage_keys)
    assert set(stats.confirmed_group_lineage_keys.keys()) == {"confirm_fastlane_group_a", "confirm_fastlane_group_b"}


def test_derive_candidate_uid_prefers_explicit_lineage_uid_and_metadata(tmp_path: Path) -> None:
    module = _load_lineage_module("fullspan_lineage_uid_priority_test")

    direct_uid = module.derive_candidate_uid(
        top_run_group="rg_demo",
        top_variant="variant_without_evo",
        top_score="1.0",
        lineage_uid="lnuid_direct_123",
    )
    assert direct_uid == "lnuid_direct_123"

    metadata_uid = module.derive_candidate_uid(
        top_run_group="rg_demo",
        top_variant="variant_without_evo",
        top_score="1.0",
        top_metadata='{"lineage_uid":"lnuid_meta_456"}',
    )
    assert metadata_uid == "lnuid_meta_456"


def test_register_confirm_queue_persists_lineage_uid(tmp_path: Path) -> None:
    module = _load_lineage_module("fullspan_lineage_register_uid_test")
    registry_path = tmp_path / "confirm_lineage_registry.json"

    out = module.register_confirm_queue(
        registry_path=registry_path,
        queue_rel="artifacts/wfa/aggregate/confirm_fastlane_group_x/run_queue.csv",
        source_queue_rel="artifacts/wfa/aggregate/source_x/run_queue.csv",
        candidate_uid="cand_demo_x",
        lineage_uid="lnuid_demo_x",
        top_run_group="source_x",
        top_variant="variant_x",
        dispatch_id="dispatch_x",
    )

    assert out["ok"] is True
    state = json.loads(registry_path.read_text(encoding="utf-8"))
    stored = state["queues"]["artifacts/wfa/aggregate/confirm_fastlane_group_x/run_queue.csv"]
    assert stored["candidate_uid"] == "cand_demo_x"
    assert stored["lineage_uid"] == "lnuid_demo_x"
