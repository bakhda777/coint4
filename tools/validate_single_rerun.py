#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_CONFIG = REPO_ROOT / "coint4" / "configs" / "main_2024_smoke.yaml"
DEST_CONFIG = (
    REPO_ROOT
    / "coint4"
    / "artifacts"
    / "wfa"
    / "reruns"
    / "Повторный прогон"
    / "configs"
    / "test_run.yaml"
)

# NOTE: contract requires this exact value (and no other YAML changes).
NEW_INITIAL_CAPITAL_LINE_SUFFIX = " 1000"
NEW_RESULTS_DIR = "artifacts/wfa/reruns/Повторный прогон/runs/test_run"

# Contract paths (relative to repo root).
APP_ROOT = REPO_ROOT / "coint4"
RUN_RESULTS_DIR = APP_ROOT / NEW_RESULTS_DIR
RUN_METRICS_CSV = RUN_RESULTS_DIR / "strategy_metrics.csv"

RUN_LOG = (
    REPO_ROOT / "coint4" / "artifacts" / "wfa" / "reruns" / "Повторный прогон" / "test_run.log"
)
VENV_PYTHON = APP_ROOT / ".venv" / "bin" / "python"


def _normalize_yaml_text(source_text: str) -> str:
    lines = source_text.splitlines(True)
    out: list[str] = []

    in_portfolio = False
    changed_results_dir = False
    changed_initial_capital = False

    for line in lines:
        stripped = line.strip()

        # Track top-level mapping sections.
        if line and not line.startswith(" ") and stripped.endswith(":"):
            in_portfolio = stripped == "portfolio:"

        if line.startswith("results_dir:"):
            out.append(f"results_dir: {NEW_RESULTS_DIR}\n")
            changed_results_dir = True
            continue

        if in_portfolio and stripped.startswith("initial_capital:"):
            prefix = line.split("initial_capital:", 1)[0] + "initial_capital:"
            out.append(f"{prefix}{NEW_INITIAL_CAPITAL_LINE_SUFFIX}\n")
            changed_initial_capital = True
            continue

        out.append(line)

    if not (changed_results_dir and changed_initial_capital):
        raise SystemExit(
            f"Normalization failed: results_dir_changed={changed_results_dir} initial_capital_changed={changed_initial_capital}"
        )

    return "".join(out)


def _read_first_row(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        cols = list(reader.fieldnames or [])
        for row in reader:
            return cols, {k: (v or "") for k, v in row.items()}
    return [], {}


def _to_float(value: str) -> float:
    return float(str(value).strip())


def main() -> int:
    # Step 2: normalized copy (text edit, no YAML reformatting).
    DEST_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    source_text = SOURCE_CONFIG.read_text(encoding="utf-8")
    normalized_text = _normalize_yaml_text(source_text)
    DEST_CONFIG.write_text(normalized_text, encoding="utf-8")

    # Step 3: run walk-forward (capture stdout+stderr to RUN_LOG).
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ".venv/bin/coint2",
        "walk-forward",
        "--config",
        "artifacts/wfa/reruns/Повторный прогон/configs/test_run.yaml",
    ]
    with RUN_LOG.open("w", encoding="utf-8") as handle:
        subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    # Step 4: validate outputs.
    if not RUN_METRICS_CSV.exists():
        print(RUN_LOG.read_text(encoding="utf-8"), end="")
        return 1

    # Step 5: parse metrics.
    cols, row = _read_first_row(RUN_METRICS_CSV)
    sharpe = None
    if "sharpe_ratio_abs" in row and row["sharpe_ratio_abs"].strip():
        sharpe = _to_float(row["sharpe_ratio_abs"])
    elif "sharpe_ratio" in row and row["sharpe_ratio"].strip():
        sharpe = _to_float(row["sharpe_ratio"])

    total_pnl = _to_float(row.get("total_pnl", "0"))
    total_return = total_pnl / 1000.0

    # Step 6: build run_index (scoped to rerun root).
    build_cmd = [
        str(VENV_PYTHON),
        "scripts/optimization/build_run_index.py",
        "--runs-dir",
        "artifacts/wfa/reruns/Повторный прогон/runs",
        "--queue-dir",
        "artifacts/wfa/reruns/Повторный прогон",
        "--output-dir",
        "artifacts/wfa/reruns/Повторный прогон",
    ]
    subprocess.run(build_cmd, cwd=str(APP_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    run_index_csv = APP_ROOT / "artifacts" / "wfa" / "reruns" / "Повторный прогон" / "run_index.csv"
    run_index_built = "YES" if run_index_csv.exists() else "NO"

    # Step 7: audit_sharpe (scoped to rerun root only).
    audit_cmd = [
        str(VENV_PYTHON),
        str(REPO_ROOT / "tools" / "audit_sharpe.py"),
        "--runs-glob",
        "artifacts/wfa/reruns/Повторный прогон/runs/*/strategy_metrics.csv",
    ]
    audit = subprocess.run(audit_cmd, cwd=str(APP_ROOT), capture_output=True, text=True, check=False)
    audit_out = (audit.stdout or "") + (audit.stderr or "")
    audit_rows_csv = None
    for line in audit_out.splitlines():
        if line.strip().startswith("rows_csv:"):
            audit_rows_csv = line.split("rows_csv:", 1)[1].strip()
            break

    audit_line = ""
    if audit_rows_csv:
        _, row2 = _read_first_row(Path(audit_rows_csv))
        audit_line = f"run_dir={row2.get('run_dir','')} metrics_path={row2.get('metrics_path','')}".strip()

    # Step 8: print EXACTLY.
    print(f"CONFIG_PATH={DEST_CONFIG.as_posix()}")
    print(f"RESULTS_DIR={RUN_RESULTS_DIR.as_posix()}")
    print(f"CSV_COLUMNS=[{', '.join(cols)}]")
    print(f"SHARPE={sharpe}")
    print(f"TOTAL_RETURN={total_return}")
    print(f"RUN_INDEX_BUILT={run_index_built}")
    print(f"AUDIT_SNIPPET={audit_line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
