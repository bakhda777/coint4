from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

PIPELINE_SCHEMA_VERSION = "papers-pipeline-v1"
DIGEST_SCHEMA_VERSION = "digest-tags-v1"
CARD_SCHEMA_VERSION = "1.1.0"

DEFAULT_PAPERS_DIR = Path("papers")
DEFAULT_TEXT_DIR = DEFAULT_PAPERS_DIR / "text"
DEFAULT_DB_PATH = DEFAULT_PAPERS_DIR / "db.sqlite"
DEFAULT_DIGESTS_DIR = DEFAULT_PAPERS_DIR / "digests"
DEFAULT_CARDS_DIR = DEFAULT_PAPERS_DIR / "cards"
DEFAULT_SYNTHESIS_DIR = DEFAULT_PAPERS_DIR / "synthesis"
DEFAULT_LOCKS_DIR = DEFAULT_PAPERS_DIR / ".locks"

SQLITE_BUSY_TIMEOUT_MS = 10000
SQLITE_WRITE_LOCK_NAME = "sqlite-writer.lock"

DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
URL_PATTERN = re.compile(
    r"(https?://\S+|arxiv:\d{4}\.\d{4,5}(?:v\d+)?|10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    re.IGNORECASE,
)

KEYWORD_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "transaction_costs": (r"\btransaction(?:al)? costs?\b", r"\btrading costs?\b", r"\bfee(?:s)?\b"),
    "slippage": (r"\bslippage\b",),
    "funding": (r"\bfunding(?: rates?)?\b",),
    "out_of_sample": (r"\bout[- ]of[- ]sample\b", r"\boos\b"),
    "walk_forward": (r"\bwalk[- ]forward\b", r"\bwfa\b"),
    "robustness": (r"\brobust(?:ness)?\b", r"\bsensitivity(?: analysis)?\b"),
    "kalman": (r"\bkalman\b",),
    "copula": (r"\bcopula\b",),
    "rl": (r"\breinforcement learning\b", r"\brl\b"),
    "regime": (r"\bregime\b", r"\bmarket states?\b"),
    "structural_break": (r"\bstructural breaks?\b", r"\bbreakpoint(?:s)?\b"),
    "cointegration": (r"\bcointegrat(?:ion|ed|ing)\b",),
    "ou": (r"\bornstein[- ]uhlenbeck\b", r"\bou process\b", r"\bou\b"),
    "half_life": (r"\bhalf[- ]life\b",),
    "clustering": (r"\bclustering\b", r"\bk[- ]means\b"),
    "pca": (r"\bpca\b", r"\bprincipal component(?:s)?\b"),
    "stop_loss": (r"\bstop[- ]loss\b",),
    "position_sizing": (r"\bposition sizing\b", r"\bposition size\b"),
    "leverage": (r"\bleverage\b", r"\blevered\b"),
    "drawdown": (r"\bdrawdown\b", r"\bmax(?:imum)? dd\b"),
    "turnover": (r"\bturnover\b",),
    "crypto": (r"\bcrypto(?:currency|currencies)?\b", r"\bbitcoin\b", r"\bethereum\b", r"\bdigital assets?\b"),
    "perpetual": (r"\bperpetual(?:s)?\b", r"\bperp(?:s)?\b"),
    "bybit": (r"\bbybit\b",),
    "binance": (r"\bbinance\b",),
}

KEYWORD_PATTERNS = {
    key: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for key, patterns in KEYWORD_DEFINITIONS.items()
}

SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "abstract": ("abstract",),
    "method": ("method", "methods", "methodology", "approach"),
    "results": ("results", "experiments", "empirical results", "findings"),
    "conclusion": ("conclusion", "conclusions", "discussion", "summary"),
}

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            text_path TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            file_name TEXT,
            title TEXT,
            year INTEGER,
            doi TEXT,
            url TEXT,
            added_at TEXT,
            updated_at TEXT,
            text_len INTEGER,
            parse_status TEXT,
            parse_error TEXT,
            digest_path TEXT,
            digest_sha256 TEXT,
            tags_json TEXT,
            keyword_hits_json TEXT,
            priority_score INTEGER,
            priority_reasons_json TEXT,
            dup_group_id TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_text_path ON papers(text_path);
        CREATE INDEX IF NOT EXISTS idx_papers_text_sha256 ON papers(text_sha256);
        CREATE INDEX IF NOT EXISTS idx_papers_priority_score ON papers(priority_score);
        CREATE INDEX IF NOT EXISTS idx_papers_dup_group ON papers(dup_group_id);
        """,
    ),
    (
        2,
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            started_at TEXT,
            finished_at TEXT,
            kind TEXT,
            params_json TEXT,
            stats_json TEXT
        );
        """,
    ),
    (
        3,
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts
        USING fts5(paper_id UNINDEXED, title, digest, excerpt);
        """,
    ),
]


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def loads_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def paper_id_from_text_sha(text_sha256: str) -> str:
    return text_sha256[:12]


def ensure_pipeline_dirs(
    text_dir: Path = DEFAULT_TEXT_DIR,
    digests_dir: Path = DEFAULT_DIGESTS_DIR,
    cards_dir: Path = DEFAULT_CARDS_DIR,
    synthesis_dir: Path = DEFAULT_SYNTHESIS_DIR,
    locks_dir: Path = DEFAULT_LOCKS_DIR,
) -> None:
    text_dir.mkdir(parents=True, exist_ok=True)
    digests_dir.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)
    synthesis_dir.mkdir(parents=True, exist_ok=True)
    locks_dir.mkdir(parents=True, exist_ok=True)


@contextmanager
def pipeline_file_lock(
    lock_name: str = SQLITE_WRITE_LOCK_NAME,
    timeout_seconds: float = 120.0,
    poll_interval: float = 0.1,
) -> Iterator[Path]:
    DEFAULT_LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = DEFAULT_LOCKS_DIR / lock_name
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        started = time.monotonic()
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    raise TimeoutError(f"Could not acquire lock {lock_path} within {timeout_seconds:.1f}s")
                time.sleep(poll_interval)

        try:
            lock_file.seek(0)
            lock_file.truncate(0)
            lock_file.write(f"pid={os.getpid()} acquired_at={utc_now_iso()}\n")
            lock_file.flush()
            yield lock_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def sqlite_write_section(
    conn: sqlite3.Connection,
    *,
    lock_name: str = SQLITE_WRITE_LOCK_NAME,
) -> Iterator[None]:
    with pipeline_file_lock(lock_name=lock_name):
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()


def connect_db(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    read_only: bool = False,
    busy_timeout_ms: int = SQLITE_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    timeout_seconds = max(1.0, busy_timeout_ms / 1000.0)
    if read_only:
        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
    else:
        conn = sqlite3.connect(str(db_path), timeout=timeout_seconds)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        apply_migrations(conn)
    return conn


def apply_migrations(conn: sqlite3.Connection) -> None:
    with sqlite_write_section(conn):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        applied = {
            row["version"]
            for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        }
        for version, sql in MIGRATIONS:
            if version in applied:
                continue
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                (version, utc_now_iso()),
            )


def start_run(conn: sqlite3.Connection, kind: str, params: dict[str, Any]) -> str:
    run_id = uuid.uuid4().hex
    with sqlite_write_section(conn):
        conn.execute(
            "INSERT INTO runs(id, started_at, kind, params_json) VALUES(?, ?, ?, ?)",
            (run_id, utc_now_iso(), kind, dumps_json(params)),
        )
    return run_id


def finish_run(conn: sqlite3.Connection, run_id: str, stats: dict[str, Any]) -> None:
    with sqlite_write_section(conn):
        conn.execute(
            "UPDATE runs SET finished_at = ?, stats_json = ? WHERE id = ?",
            (utc_now_iso(), dumps_json(stats), run_id),
        )


def read_text_file(path: Path) -> tuple[str, str, str | None]:
    data = path.read_bytes()
    decode_error: str | None = None
    encoding = "utf-8"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        encoding = "latin-1"
        decode_error = str(exc)
        text = data.decode("latin-1", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    return text, encoding, decode_error


def normalize_text(text: str) -> str:
    value = re.sub(r"[ \t\f\v]+", " ", text)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_probable_heading(line: str) -> bool:
    value = line.strip(" :\t")
    if not value:
        return False
    if not (3 <= len(value) <= 90):
        return False
    if "." in value and len(value) < 40:
        return False
    words = value.split()
    if len(words) > 14:
        return False
    alpha = sum(1 for ch in value if ch.isalpha())
    if alpha < max(3, int(len(value) * 0.45)):
        return False
    return True


def _match_section_key(line: str) -> str | None:
    lowered = line.strip().lower().rstrip(":")
    for section_name, keywords in SECTION_KEYWORDS.items():
        if any(keyword == lowered or lowered.startswith(f"{keyword} ") for keyword in keywords):
            return section_name
    return None


def extract_section_snippets(text: str) -> dict[str, str]:
    snippets: dict[str, str] = {}
    lines = text.splitlines()
    current_section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_section
        if not current_section:
            buffer = []
            return
        body = compact_whitespace("\n".join(buffer))
        if body:
            snippets[current_section] = body[:1400]
        buffer = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_section and buffer and buffer[-1] != "":
                buffer.append("")
            continue

        possible_section = _match_section_key(stripped) if _is_probable_heading(stripped) else None
        if possible_section:
            flush()
            current_section = possible_section
            continue

        if current_section:
            buffer.append(stripped)
            if len(" ".join(buffer)) > 2200:
                flush()
                current_section = None

    flush()
    return snippets


def extract_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    paragraphs = [compact_whitespace(p) for p in parts]
    return [paragraph for paragraph in paragraphs if paragraph]


def build_digest(text: str, min_chars: int = 1000, max_chars: int = 2000) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""

    sections = extract_section_snippets(text)
    ordered_parts: list[str] = []
    for section_name in ("abstract", "method", "results", "conclusion"):
        value = sections.get(section_name)
        if value:
            ordered_parts.append(value)

    if ordered_parts:
        digest = compact_whitespace("\n\n".join(ordered_parts))
    else:
        paragraphs = extract_paragraphs(text)
        if not paragraphs:
            digest = compact_whitespace(normalized)
        else:
            head = paragraphs[:3]
            tail = paragraphs[-2:] if len(paragraphs) > 4 else []
            digest = compact_whitespace("\n\n".join(head + tail))

    if len(digest) < min_chars:
        deficit = min_chars - len(digest)
        padding = normalized[: deficit + 300]
        digest = compact_whitespace(f"{digest} {padding}")

    if len(digest) > max_chars:
        digest = digest[:max_chars]
        if " " in digest:
            digest = digest.rsplit(" ", 1)[0]
    return digest.strip()


def build_excerpt(text: str, max_chars: int = 2000) -> str:
    normalized = compact_whitespace(normalize_text(text))
    if len(normalized) <= max_chars:
        return normalized
    excerpt = normalized[:max_chars]
    if " " in excerpt:
        excerpt = excerpt.rsplit(" ", 1)[0]
    return excerpt


def _is_title_candidate(line: str) -> bool:
    value = compact_whitespace(line)
    if not (10 <= len(value) <= 180):
        return False
    if re.match(r"^(page\s+\d+|copyright|all rights reserved)", value, re.IGNORECASE):
        return False
    if re.match(r"^(https?://|doi:|arxiv:)", value, re.IGNORECASE):
        return False
    if value.count("|") > 2 or value.count("{") > 0 or value.count("}") > 0:
        return False
    letters = sum(ch.isalpha() for ch in value)
    digits = sum(ch.isdigit() for ch in value)
    if letters < max(6, int(len(value) * 0.35)):
        return False
    if digits > int(len(value) * 0.35):
        return False
    return True


def extract_metadata(text: str) -> dict[str, Any]:
    lines = [compact_whitespace(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    title = None
    for line in lines[:80]:
        if _is_title_candidate(line):
            title = line
            break

    probe = text[:2000]
    year = None
    for match in YEAR_PATTERN.finditer(probe):
        candidate = int(match.group(1))
        if 1900 <= candidate <= 2099:
            year = candidate
            break

    doi_match = DOI_PATTERN.search(probe)
    doi = doi_match.group(0) if doi_match else None

    url = None
    url_match = URL_PATTERN.search(probe)
    if url_match:
        url = url_match.group(0).rstrip(").,;")
        if DOI_PATTERN.fullmatch(url):
            url = f"https://doi.org/{url}"

    return {
        "title": title,
        "year": year,
        "doi": doi,
        "url": url,
    }


def compute_keyword_hits(text: str) -> dict[str, bool]:
    hits: dict[str, bool] = {}
    for name, patterns in KEYWORD_PATTERNS.items():
        hits[name] = any(pattern.search(text) is not None for pattern in patterns)
    return hits


def _tag_append(target: list[str], value: str) -> None:
    if value not in target:
        target.append(value)


def infer_tags(hits: dict[str, bool], text: str) -> dict[str, Any]:
    lowered = text.lower()
    tags: dict[str, Any] = {
        "method_family": [],
        "practical": [],
        "market": [],
        "_schema_version": DIGEST_SCHEMA_VERSION,
    }

    if hits.get("cointegration"):
        _tag_append(tags["method_family"], "cointegration")
    if re.search(r"\bdistance (?:metric|approach|method)\b", lowered) or re.search(r"\bssd\b", lowered):
        _tag_append(tags["method_family"], "distance")
    if hits.get("kalman"):
        _tag_append(tags["method_family"], "kalman")
    if hits.get("copula"):
        _tag_append(tags["method_family"], "copula")
    if re.search(r"\b(machine learning|xgboost|random forest|svm|neural network|deep learning)\b", lowered):
        _tag_append(tags["method_family"], "ml")
    if hits.get("rl"):
        _tag_append(tags["method_family"], "rl")
    if hits.get("regime") or hits.get("structural_break"):
        _tag_append(tags["method_family"], "regime")
    if re.search(r"\b(volatility filter|garch|realized volatility)\b", lowered):
        _tag_append(tags["method_family"], "volatility_filter")
    if hits.get("clustering"):
        _tag_append(tags["method_family"], "clustering")
    if hits.get("pca"):
        _tag_append(tags["method_family"], "pca")
    if hits.get("ou") or hits.get("half_life"):
        _tag_append(tags["method_family"], "ou_model")

    if hits.get("transaction_costs") or hits.get("slippage") or hits.get("funding") or hits.get("turnover"):
        _tag_append(tags["practical"], "execution_costs")
    if hits.get("out_of_sample"):
        _tag_append(tags["practical"], "oos")
    if hits.get("walk_forward"):
        _tag_append(tags["practical"], "wfa")
    if hits.get("robustness"):
        _tag_append(tags["practical"], "robustness")
    if hits.get("stop_loss") or hits.get("position_sizing") or hits.get("leverage") or hits.get("drawdown"):
        _tag_append(tags["practical"], "risk_controls")

    if hits.get("crypto") or hits.get("perpetual") or hits.get("bybit") or hits.get("binance"):
        _tag_append(tags["market"], "crypto")
    if re.search(r"\b(equit(?:y|ies)|stock(?:s)?|s&p|nasdaq|nyse)\b", lowered):
        _tag_append(tags["market"], "equities")
    if re.search(r"\b(fx|forex|currency pairs?)\b", lowered):
        _tag_append(tags["market"], "fx")
    if re.search(r"\b(futures?|contracts?)\b", lowered):
        _tag_append(tags["market"], "futures")

    return tags


def tags_version_matches(tags_json: str | None) -> bool:
    parsed = loads_json(tags_json, {})
    if not isinstance(parsed, dict):
        return False
    return parsed.get("_schema_version") == DIGEST_SCHEMA_VERSION


def resolve_unique_paper_id(conn: sqlite3.Connection, base_id: str, text_path: str) -> str:
    row = conn.execute("SELECT text_path FROM papers WHERE id = ?", (base_id,)).fetchone()
    if not row or row["text_path"] == text_path:
        return base_id

    suffix = sha256_text(text_path)[:6]
    candidate = f"{base_id}-{suffix}"
    row = conn.execute("SELECT text_path FROM papers WHERE id = ?", (candidate,)).fetchone()
    if not row or row["text_path"] == text_path:
        return candidate

    for index in range(1, 1000):
        candidate = f"{base_id}-{suffix}-{index}"
        row = conn.execute("SELECT text_path FROM papers WHERE id = ?", (candidate,)).fetchone()
        if not row or row["text_path"] == text_path:
            return candidate

    raise RuntimeError(f"Unable to allocate paper id for {text_path}")


def upsert_fts(conn: sqlite3.Connection, paper_id: str, title: str | None, digest: str, excerpt: str) -> None:
    conn.execute("DELETE FROM papers_fts WHERE paper_id = ?", (paper_id,))
    conn.execute(
        "INSERT INTO papers_fts(paper_id, title, digest, excerpt) VALUES(?, ?, ?, ?)",
        (paper_id, title or "", digest, excerpt),
    )


def normalize_title(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized or None


def merge_reason(reasons_json: str | None, reason: str) -> str:
    reasons = loads_json(reasons_json, [])
    if not isinstance(reasons, list):
        reasons = []
    if reason not in reasons:
        reasons.append(reason)
    return dumps_json(reasons)
