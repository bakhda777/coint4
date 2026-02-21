from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_PAPERS = REPO_ROOT / "scripts" / "papers"
if str(SCRIPTS_PAPERS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PAPERS))

from lib_pipeline import paper_id_from_text_sha  # noqa: E402
import index_texts  # noqa: E402


def _read_last_index_stats(db_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT stats_json FROM runs WHERE kind = 'papers_index' ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None
    return json.loads(row[0])


def test_paper_id_from_text_sha_is_stable() -> None:
    sha = "a" * 64
    assert paper_id_from_text_sha(sha) == "a" * 12
    assert paper_id_from_text_sha(sha) == paper_id_from_text_sha(sha)


@pytest.mark.fast
def test_index_texts_incremental_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text_dir = tmp_path / "papers" / "text"
    db_path = tmp_path / "papers" / "db.sqlite"
    digests_dir = tmp_path / "papers" / "digests"
    text_dir.mkdir(parents=True)

    sample_file = text_dir / "sample.txt"
    sample_file.write_text(
        "Abstract\nThis is a crypto pairs trading paper with walk-forward and transaction costs.\n"
        "Results\nSharpe 1.2, drawdown 5%, turnover control included.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "index_texts.py",
            "--text-dir",
            str(text_dir),
            "--db-path",
            str(db_path),
            "--digests-dir",
            str(digests_dir),
        ],
    )
    assert index_texts.main() == 0

    stats_first = _read_last_index_stats(db_path)
    assert stats_first["total_files"] == 1
    assert stats_first["new"] == 1
    assert stats_first["updated"] == 0
    assert stats_first["skipped"] == 0

    digest_files = list(digests_dir.glob("*.txt"))
    assert len(digest_files) == 1
    digest_before = digest_files[0].read_text(encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "index_texts.py",
            "--text-dir",
            str(text_dir),
            "--db-path",
            str(db_path),
            "--digests-dir",
            str(digests_dir),
        ],
    )
    assert index_texts.main() == 0

    stats_second = _read_last_index_stats(db_path)
    assert stats_second["total_files"] == 1
    assert stats_second["new"] == 0
    assert stats_second["updated"] == 0
    assert stats_second["skipped"] == 1

    digest_after = digest_files[0].read_text(encoding="utf-8")
    assert digest_after == digest_before
