"""Tests for run queue helpers."""

from pathlib import Path

from coint2.ops.run_queue import RunQueueEntry, load_run_queue, select_by_status, write_run_queue


def test_run_queue_roundtrip(tmp_path: Path) -> None:
    entries = [
        RunQueueEntry(
            config_path="configs/test_a.yaml",
            results_dir="artifacts/wfa/runs/test_a",
            status="planned",
        ),
        RunQueueEntry(
            config_path="configs/test_b.yaml",
            results_dir="artifacts/wfa/runs/test_b",
            status="stalled",
        ),
    ]
    queue_path = tmp_path / "run_queue.csv"

    write_run_queue(queue_path, entries)
    loaded = load_run_queue(queue_path)

    assert loaded == entries


def test_select_by_status_filters(tmp_path: Path) -> None:
    entries = [
        RunQueueEntry("cfg_a.yaml", "runs/a", "planned"),
        RunQueueEntry("cfg_b.yaml", "runs/b", "completed"),
        RunQueueEntry("cfg_c.yaml", "runs/c", "stalled"),
    ]

    selected = select_by_status(entries, ["planned", "stalled"])

    assert [entry.results_dir for entry in selected] == ["runs/a", "runs/c"]

