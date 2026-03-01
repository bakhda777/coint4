#!/usr/bin/env python3
"""Critic/reflection step for evolution loop.

Consumes decision + run_index and produces structured reflection JSON
(`action -> result -> reflection`) suitable for trajectory memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _to_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _env_float(name: str, default: float) -> float:
    parsed = _to_float(os.environ.get(name))
    if parsed is None:
        return float(default)
    return float(parsed)


def _env_int(name: str, default: int) -> int:
    parsed = _to_int(os.environ.get(name))
    if parsed is None:
        return int(default)
    return int(parsed)


def _reflect_policy() -> dict[str, float | int]:
    """Load deterministic stop policy from env with safe defaults."""
    # By default, the reflection step must not stop the evolution loop.
    # Stopping is reserved for explicit operator intent (env enable flag).
    enable_stop = _to_bool(os.environ.get("COINT4_REFLECT_ENABLE_STOP"))
    stop_sharpe_lt = _env_float("COINT4_REFLECT_STOP_SHARPE_LT", 0.0)
    stop_dd_gt = _env_float("COINT4_REFLECT_STOP_DD_GT", 0.25)
    min_bad_rows = max(1, _env_int("COINT4_REFLECT_STOP_MIN_BAD_ROWS", 2))
    min_bad_share = _env_float("COINT4_REFLECT_STOP_MIN_BAD_SHARE", 0.5)
    min_bad_share = min(1.0, max(0.0, float(min_bad_share)))
    min_metrics_rows = max(1, _env_int("COINT4_REFLECT_STOP_MIN_METRICS_ROWS", 4))
    hard_stop_sharpe_lt = _env_float("COINT4_REFLECT_HARD_STOP_SHARPE_LT", -2.0)
    hard_stop_dd_gt = _env_float("COINT4_REFLECT_HARD_STOP_DD_GT", 0.35)
    # By default, stop policy is DD-based; negative Sharpe alone must not stop the loop.
    stop_on_sharpe = _to_bool(os.environ.get("COINT4_REFLECT_STOP_ON_SHARPE"))
    hard_stop_on_sharpe = _to_bool(os.environ.get("COINT4_REFLECT_HARD_STOP_ON_SHARPE"))
    return {
        "enable_stop": int(enable_stop),
        "stop_sharpe_lt": stop_sharpe_lt,
        "stop_dd_gt": stop_dd_gt,
        "min_bad_rows": min_bad_rows,
        "min_bad_share": min_bad_share,
        "min_metrics_rows": min_metrics_rows,
        "hard_stop_sharpe_lt": hard_stop_sharpe_lt,
        "hard_stop_dd_gt": hard_stop_dd_gt,
        "stop_on_sharpe": int(stop_on_sharpe),
        "hard_stop_on_sharpe": int(hard_stop_on_sharpe),
    }


def _read_run_index(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _matches_all(text: str, needles: Sequence[str]) -> bool:
    hay = str(text or "").lower()
    for needle in needles:
        token = str(needle or "").strip().lower()
        if token and token not in hay:
            return False
    return True


def _deterministic_reflection(
    *,
    decision: dict[str, Any],
    rows: list[dict[str, str]],
    contains: Sequence[str],
) -> dict[str, Any]:
    policy = _reflect_policy()
    run_group = str(decision.get("run_group") or "").strip()
    filtered: list[dict[str, str]] = []
    for row in rows:
        meta = " | ".join(
            (
                str(row.get("run_group") or "").strip(),
                str(row.get("run_id") or "").strip(),
                str(row.get("config_path") or "").strip(),
                str(row.get("results_dir") or "").strip(),
            )
        )
        if contains and not _matches_all(meta, contains):
            continue
        if run_group and str(row.get("run_group") or "").strip() != run_group:
            continue
        filtered.append(row)

    completed = [row for row in filtered if str(row.get("status") or "").strip().lower() == "completed"]
    metrics_rows = [row for row in completed if _to_bool(row.get("metrics_present"))]
    sharpe_values = [_to_float(row.get("worst_robust_sharpe") or row.get("sharpe_ratio_abs")) for row in metrics_rows]
    sharpe_values = [value for value in sharpe_values if value is not None]
    dd_values = [_to_float(row.get("worst_dd_pct") or row.get("max_drawdown_on_equity")) for row in metrics_rows]
    dd_values = [abs(value) for value in dd_values if value is not None]
    worst_sharpe = min(sharpe_values) if sharpe_values else None
    worst_dd = max(dd_values) if dd_values else None

    bad_rows = []
    hard_bad_rows = []
    stop_on_sharpe = bool(int(policy["stop_on_sharpe"]))
    hard_stop_on_sharpe = bool(int(policy["hard_stop_on_sharpe"]))
    for row in metrics_rows:
        sharpe = _to_float(row.get("worst_robust_sharpe") or row.get("sharpe_ratio_abs"))
        dd_raw = _to_float(row.get("worst_dd_pct") or row.get("max_drawdown_on_equity"))
        if dd_raw is None:
            continue
        dd_abs = abs(dd_raw)
        if stop_on_sharpe:
            if sharpe is None:
                continue
            is_bad = sharpe < float(policy["stop_sharpe_lt"]) and dd_abs > float(policy["stop_dd_gt"])
        else:
            is_bad = dd_abs > float(policy["stop_dd_gt"])
        if is_bad:
            bad_rows.append(row)
        if hard_stop_on_sharpe:
            if sharpe is None:
                continue
            is_hard_bad = sharpe < float(policy["hard_stop_sharpe_lt"]) and dd_abs > float(policy["hard_stop_dd_gt"])
        else:
            is_hard_bad = dd_abs > float(policy["hard_stop_dd_gt"])
        if is_hard_bad:
            hard_bad_rows.append(row)

    bad_share = (len(bad_rows) / len(metrics_rows)) if metrics_rows else 0.0
    risk_flags: list[str] = []
    if bad_rows:
        risk_flags.append(f"bad_windows={len(bad_rows)}/{len(metrics_rows)} share={bad_share:.2f}")
    if stop_on_sharpe and worst_sharpe is not None and worst_sharpe < float(policy["stop_sharpe_lt"]):
        risk_flags.append(
            f"worst_sharpe={worst_sharpe:.3f} < stop_sharpe_lt={float(policy['stop_sharpe_lt']):.3f}"
        )
    if worst_dd is not None and worst_dd > float(policy["stop_dd_gt"]):
        risk_flags.append(f"worst_dd={worst_dd:.3f} > stop_dd_gt={float(policy['stop_dd_gt']):.3f}")

    if not completed:
        next_action = "wait"
        reflection = "В filtered run_index нет completed строк; продолжать наблюдение/выполнение."
    elif hard_bad_rows:
        next_action = "stop"
        if hard_stop_on_sharpe:
            reflection = (
                f"Hard-stop: экстремально плохих окон={len(hard_bad_rows)} (из metrics={len(metrics_rows)}), "
                f"worst robust Sharpe={worst_sharpe:.3f}, worst DD={worst_dd:.3f}; остановить эволюцию."
            )
        else:
            reflection = (
                f"Hard-stop по DD: экстремально плохих окон={len(hard_bad_rows)} (из metrics={len(metrics_rows)}), "
                f"worst DD={worst_dd:.3f}; остановить эволюцию."
            )
        risk_flags.append("hard_stop_triggered")
    else:
        should_fail_closed_stop = (
            len(metrics_rows) >= int(policy["min_metrics_rows"])
            and len(bad_rows) >= int(policy["min_bad_rows"])
            and bad_share >= float(policy["min_bad_share"])
        )
        if should_fail_closed_stop and worst_sharpe is not None and worst_dd is not None:
            next_action = "stop"
            reflection = (
                f"Системно плохой батч: bad_windows={len(bad_rows)}/{len(metrics_rows)} (share={bad_share:.2f}), "
                f"worst robust Sharpe={worst_sharpe:.3f}, worst DD={worst_dd:.3f}; fail-closed stop."
            )
        else:
            next_action = "run_next_batch"
            if bad_rows:
                reflection = (
                    f"Completed={len(completed)}, metrics_present={len(metrics_rows)}, "
                    f"bad_windows={len(bad_rows)} (share={bad_share:.2f}); продолжать мутации в следующем батче."
                )
            else:
                reflection = (
                    f"Completed={len(completed)}, metrics_present={len(metrics_rows)}; "
                    "продолжать целевой батч с обновлённой политикой операторов."
                )

    # Default behavior: do not stop the evolution loop based on "bad results" alone.
    # We keep risk flags for observability, but keep iterating unless explicitly enabled.
    if next_action == "stop" and not bool(int(policy.get("enable_stop") or 0)):
        risk_flags.append("stop_suppressed")
        next_action = "run_next_batch"
        reflection = reflection.replace("остановить эволюцию", "продолжать (stop disabled)")

    return {
        "reflection_id": f"reflect_{_utc_now().replace(':', '').replace('-', '')}",
        "timestamp": _utc_now(),
        "action": str(decision.get("decision_id") or ""),
        "result": {
            "run_group": run_group,
            "filtered_rows": len(filtered),
            "completed_rows": len(completed),
            "metrics_rows": len(metrics_rows),
            "worst_sharpe": worst_sharpe,
            "worst_dd_pct": worst_dd,
            "bad_rows_count": len(bad_rows),
            "bad_rows_share": bad_share,
        },
        "next_action": next_action,
        "reflection": reflection,
        "risk_flags": risk_flags,
        "policy": policy,
    }


def _llm_reflection(
    *,
    base_payload: dict[str, Any],
    model: str,
    codex_bin: str,
    timeout_sec: int,
    repo_root: Path,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["next_action", "reflection", "risk_flags"],
        "properties": {
            "next_action": {"type": "string", "enum": ["run_next_batch", "wait", "stop"]},
            "reflection": {"type": "string", "minLength": 1, "maxLength": 2000},
            "risk_flags": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        },
    }
    prompt = (
        "You are critic/reflection engine for an optimization loop.\n"
        "Return only JSON matching schema.\n"
        f"Context: {json.dumps(base_payload, ensure_ascii=False, sort_keys=True)}"
    )

    with tempfile.TemporaryDirectory(prefix="reflect_llm_") as temp_dir:
        temp = Path(temp_dir)
        schema_path = temp / "schema.json"
        output_path = temp / "out.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

        command = [
            codex_bin,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--model",
            model,
            prompt,
        ]
        proc = subprocess.run(
            command,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(10, int(timeout_sec)),
            check=False,
        )
        if proc.returncode != 0 or not output_path.exists():
            raise RuntimeError(f"codex exec failed rc={proc.returncode}: {proc.stderr.strip()}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("invalid llm reflection payload")
        return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build reflection JSON for evolution trajectory memory.")
    parser.add_argument("--decision", required=True, help="Path to decision JSON.")
    parser.add_argument(
        "--run-index",
        default="artifacts/wfa/aggregate/rollup/run_index.csv",
        help="Path to rollup run_index CSV.",
    )
    parser.add_argument("--contains", action="append", default=[], help="Filter token over run metadata (repeatable).")
    parser.add_argument("--output-json", required=True, help="Where to write reflection JSON.")
    parser.add_argument("--llm-critic", action="store_true", help="Use codex LLM critic on top of deterministic reflection.")
    parser.add_argument("--llm-model", default="gpt-5.2", help="Model name for LLM critic.")
    parser.add_argument("--llm-codex-bin", default="codex", help="codex executable path.")
    parser.add_argument("--llm-timeout-sec", type=int, default=120, help="LLM timeout in seconds.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    decision_path = Path(args.decision)
    run_index_path = Path(args.run_index)
    output_path = Path(args.output_json)
    decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
    if not isinstance(decision_payload, dict):
        raise SystemExit("decision payload must be object")

    contains = [str(token).strip() for token in list(args.contains or []) if str(token).strip()]
    rows = _read_run_index(run_index_path)
    reflection_payload = _deterministic_reflection(decision=decision_payload, rows=rows, contains=contains)

    if args.llm_critic:
        try:
            llm_payload = _llm_reflection(
                base_payload=reflection_payload,
                model=str(args.llm_model),
                codex_bin=str(args.llm_codex_bin),
                timeout_sec=int(args.llm_timeout_sec),
                repo_root=Path.cwd(),
            )
            reflection_payload["next_action"] = llm_payload.get("next_action")
            reflection_payload["reflection"] = llm_payload.get("reflection")
            reflection_payload["risk_flags"] = list(llm_payload.get("risk_flags") or [])
            reflection_payload["critic_source"] = "llm_codex_exec"
        except Exception as exc:  # noqa: BLE001
            reflection_payload["critic_source"] = "deterministic_fallback"
            reflection_payload["critic_error"] = f"{type(exc).__name__}: {exc}"
    else:
        reflection_payload["critic_source"] = "deterministic"

    # Safety clamp: never stop unless explicitly enabled.
    if not _to_bool(os.environ.get("COINT4_REFLECT_ENABLE_STOP")):
        if str(reflection_payload.get("next_action") or "").strip() == "stop":
            flags = list(reflection_payload.get("risk_flags") or [])
            if "stop_suppressed" not in flags:
                flags.append("stop_suppressed")
            reflection_payload["risk_flags"] = flags[:20]
            reflection_payload["next_action"] = "run_next_batch"
            text = str(reflection_payload.get("reflection") or "").strip()
            if text:
                reflection_payload["reflection"] = text + " (stop disabled)"
            else:
                reflection_payload["reflection"] = "Stop suppressed by policy; continue."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reflection_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote reflection: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
