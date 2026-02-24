#!/usr/bin/env python3
"""
Sprint manager-loop for ralph-tui + beads.

Responsibilities (best-effort, no heavy compute):
  A) Collect results/metrics from typical project artifacts/logs
  B) Generate Retro report into reports/sprint_<N>_retro.md and update .ai_autopilot/state.json
  C) Create next sprint (6–10 tasks) as Beads tasks under the same epic
  D) Optional stop condition: set done=true and stop generating new sprints
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_DIR = PROJECT_ROOT / ".ai_autopilot"
DEFAULT_GOAL = AI_DIR / "goal.md"
DEFAULT_STATE = AI_DIR / "state.json"
DEFAULT_RETRO_TEMPLATE = AI_DIR / "templates" / "retro.md"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class CandidateGate:
    """Unified hard-gate for candidate selection (fail-closed by default)."""

    min_trades: int
    max_dd_abs: float
    initial_capital: float
    max_tail_bucket_loss_pct: Optional[float]
    require_tail_metrics: bool = True


# Canonical (v1) gate for "don't optimize garbage" shortlist in sprint_manager.
# NOTE: Tail gate uses rollup `tail_loss_*` bucket PnL (pair/period), not step-level tail metrics.
CANDIDATE_GATE_V1 = CandidateGate(
    min_trades=200,
    max_dd_abs=0.50,
    initial_capital=1000.0,
    max_tail_bucket_loss_pct=0.20,
    require_tail_metrics=True,
)


def _log(msg: str) -> None:
    print(f"[sprint_manager] {msg}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _today_ymd() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _is_sharpe_audit_task(task: dict[str, Any]) -> bool:
    title = str(task.get("title") or "").strip().lower()
    # Be tolerant to quote types, punctuation and prefixes like "S3:".
    return "sharpe" in title and "слишком высокий" in title


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)


def _bd_json(args: list[str], *, cwd: Path, warnings: list[str]) -> list[dict[str, Any]]:
    proc = _run(["bd", *args], cwd=cwd)
    if proc.returncode != 0:
        warnings.append(f"bd {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
        return []
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as e:
        warnings.append(f"bd {' '.join(args)} invalid json: {e}")
        return []
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    warnings.append(f"bd {' '.join(args)} unexpected payload type: {type(payload).__name__}")
    return []


def _ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _latest_by_mtime(paths: Iterable[Path]) -> Optional[Path]:
    latest: Optional[Path] = None
    latest_mtime = -1.0
    for p in paths:
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        if st.st_mtime > latest_mtime:
            latest_mtime = st.st_mtime
            latest = p
    return latest


def _find_run_index_json(repo_root: Path) -> Optional[Path]:
    preferred = repo_root / "coint4" / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.json"
    if preferred.exists():
        return preferred
    candidates: list[Path] = []
    candidates.extend(repo_root.glob("coint4/artifacts/wfa/aggregate/**/run_index.json"))
    candidates.extend(repo_root.glob("coint4/artifacts/wfa/reruns/**/run_index.json"))
    return _latest_by_mtime(candidates)


@dataclass(frozen=True)
class RunIndexRow:
    sharpe: float
    dd_abs: float
    trades: float
    tail_loss_worst_pair_pnl: Optional[float]
    tail_loss_worst_period_pnl: Optional[float]
    run_group: str
    run_id: str
    config_path: str
    results_dir: str
    raw: dict[str, Any]


def _extract_run_index_rows(payload: list[dict[str, Any]]) -> list[RunIndexRow]:
    rows: list[RunIndexRow] = []
    for r in payload:
        status = str(r.get("status") or "").strip().lower()
        if status and status != "completed":
            continue

        sharpe = _to_float(r.get("sharpe_ratio_abs") or r.get("sharpe"))
        dd = _to_float(r.get("max_drawdown_on_equity") or r.get("max_drawdown") or r.get("dd"))
        trades = _to_float(r.get("total_trades") or r.get("trades") or r.get("n_trades"))
        if sharpe is None:
            continue
        dd_abs = abs(dd) if dd is not None else float("nan")
        trades_f = trades if trades is not None else float("nan")

        tail_pair_pnl = _to_float(r.get("tail_loss_worst_pair_pnl"))
        tail_period_pnl = _to_float(r.get("tail_loss_worst_period_pnl"))

        rows.append(
            RunIndexRow(
                sharpe=float(sharpe),
                dd_abs=float(dd_abs),
                trades=float(trades_f),
                tail_loss_worst_pair_pnl=tail_pair_pnl,
                tail_loss_worst_period_pnl=tail_period_pnl,
                run_group=str(r.get("run_group") or ""),
                run_id=str(r.get("run_id") or ""),
                config_path=str(r.get("config_path") or ""),
                results_dir=str(r.get("results_dir") or ""),
                raw=r,
            )
        )
    return rows


def _candidate_gate_reason(row: RunIndexRow, gate: CandidateGate) -> Optional[str]:
    if not math.isfinite(float(row.trades)):
        return "trades missing"
    if float(row.trades) < float(gate.min_trades):
        return f"trades < {int(gate.min_trades)}"

    if not math.isfinite(float(row.dd_abs)):
        return "dd missing"
    if float(row.dd_abs) > float(gate.max_dd_abs):
        return f"|DD| > {float(gate.max_dd_abs):.2f}"

    if gate.max_tail_bucket_loss_pct is None or float(gate.max_tail_bucket_loss_pct) <= 0:
        return None

    worst_pair_pnl = row.tail_loss_worst_pair_pnl
    worst_period_pnl = row.tail_loss_worst_period_pnl
    if worst_pair_pnl is None or worst_period_pnl is None:
        return "tail metrics missing" if gate.require_tail_metrics else None

    loss_gate_abs = float(gate.initial_capital) * float(gate.max_tail_bucket_loss_pct)
    if math.isfinite(float(worst_pair_pnl)) and float(worst_pair_pnl) < -loss_gate_abs:
        return f"tail_loss_worst_pair_pnl < -{loss_gate_abs:.0f}"
    if math.isfinite(float(worst_period_pnl)) and float(worst_period_pnl) < -loss_gate_abs:
        return f"tail_loss_worst_period_pnl < -{loss_gate_abs:.0f}"
    return None


def _pick_top_candidates(
    rows: list[RunIndexRow],
    *,
    limit: int = 5,
    gate: Optional[CandidateGate] = CANDIDATE_GATE_V1,
    warnings: Optional[list[str]] = None,
) -> list[RunIndexRow]:
    pool = list(rows)
    if gate is not None:
        passed = [r for r in pool if _candidate_gate_reason(r, gate) is None]
        if passed:
            if warnings is not None and len(passed) != len(pool):
                tail_abs = (
                    float(gate.initial_capital) * float(gate.max_tail_bucket_loss_pct)
                    if gate.max_tail_bucket_loss_pct is not None
                    else None
                )
                tail_txt = f", tail_bucket_pnl>=-{tail_abs:.0f}" if tail_abs is not None else ""
                warnings.append(
                    "candidate gate applied: "
                    f"passed={len(passed)}/{len(pool)} (min_trades={gate.min_trades}, max_dd_abs={gate.max_dd_abs:.2f}{tail_txt})"
                )
            pool = passed
        else:
            if warnings is not None and pool:
                warnings.append("candidate gate applied: 0 passed; showing ungated top-by-sharpe for visibility")

    # Cheap, informative default: highest Sharpe first; break ties by DD then trades.
    ranked = sorted(pool, key=lambda x: (-x.sharpe, x.dd_abs, -x.trades))
    return ranked[: max(0, limit)]


def _format_candidates_table(rows: list[RunIndexRow]) -> str:
    if not rows:
        return "- (не найдено)\n"
    out = []
    out.append("| Sharpe | |DD| | Trades | Run group | Config |")
    out.append("|---:|---:|---:|---|---|")
    for r in rows:
        cfg = r.config_path or "-"
        rg = r.run_group or "-"
        out.append(f"| {r.sharpe:.3f} | {r.dd_abs:.3f} | {r.trades:.0f} | {rg} | {cfg} |")
    return "\n".join(out) + "\n"


def _render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for k, v in values.items():
        rendered = rendered.replace(f"{{{{{k}}}}}", v)
    return rendered


def _ensure_min_tasks(tasks: list[dict[str, Any]], *, min_count: int, warnings: list[str]) -> list[dict[str, Any]]:
    if len(tasks) >= min_count:
        return tasks
    warnings.append(f"task generator produced only {len(tasks)} tasks; padding to {min_count}")
    padded = list(tasks)
    i = len(padded) + 1
    while len(padded) < min_count:
        padded.append(
            {
                "title": f"(pad) Вспомогательная задача #{i}",
                "priority": 4,
                "description": (
                    "Цель: не терять итерацию — создать место для уточнения/декомпозиции.\n\n"
                    "Ожидаемый эффект: менеджер сможет добавить/уточнить конкретные эксперименты.\n\n"
                    "Проверка:\n- [ ] Обновить .ralph-tui/progress.md краткой заметкой, что нужно уточнить\n"
                ),
            }
        )
        i += 1
    return padded


def _generate_next_sprint_tasks(
    *,
    next_sprint: int,
    state: dict[str, Any],
    last_results: Optional[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """
    Return list of dicts: {title, description, priority}.
    Keep tasks small (one ralph iteration each), cheap, and informative.
    """
    focus_bits: list[str] = []
    if last_results and isinstance(last_results.get("best"), dict):
        best = last_results["best"]
        sharpe = _to_float(best.get("sharpe"))
        dd = _to_float(best.get("dd_abs"))
        trades = _to_float(best.get("trades"))
        if sharpe is not None:
            focus_bits.append(f"baseline Sharpe≈{sharpe:.2f}")
        if dd is not None:
            focus_bits.append(f"|DD|≈{dd:.2f}")
        if trades is not None:
            focus_bits.append(f"trades≈{trades:.0f}")

    focus = ", ".join(focus_bits) if focus_bits else "baseline неизвестен (артефакты не найдены)"

    audits = state.get("audits") if isinstance(state.get("audits"), dict) else {}
    sharpe_audit = audits.get("sharpe_consistency") if isinstance(audits, dict) else None
    sharpe_audit_ok = bool(sharpe_audit.get("ok")) if isinstance(sharpe_audit, dict) else False

    best = last_results.get("best") if isinstance(last_results, dict) else None
    best_results_dir = str(best.get("results_dir") or "") if isinstance(best, dict) else ""
    if best_results_dir and not best_results_dir.startswith("coint4/"):
        best_results_dir = f"coint4/{best_results_dir}"

    tasks: list[dict[str, Any]] = [
        {
            "title": f"S{next_sprint}: Обновить optimization_state.md по текущему rollup",
            "priority": 1,
            "description": (
                "Цель: зафиксировать текущее состояние оптимизаций (best config + метрики).\n\n"
                f"Ожидаемый эффект: единая точка правды в docs/optimization_state.md (фокус: {focus}).\n\n"
                "Проверка:\n"
                "- [ ] Обновлён docs/optimization_state.md: текущий best + метрики (Sharpe, |DD|, trades)\n"
                "- [ ] Указаны пути к rollup: coint4/artifacts/wfa/aggregate/rollup/run_index.*\n"
            ),
        },
        {
            "title": f"S{next_sprint}: Подготовить remote-очередь tailguard/holdout на топ-кандидатов",
            "priority": 2,
            "description": (
                "Цель: сформировать следующую информативную серию прогонов на удалённом VPS.\n\n"
                "Ожидаемый эффект: новая очередь в coint4/artifacts/wfa/aggregate/<group>/run_queue.csv.\n\n"
                "Проверка:\n"
                "- [ ] Создан новый run_group (датированный) и run_queue.csv (6–20 задач)\n"
                "- [ ] Для queue-прогонов явно задан walk_forward.max_steps (<=5)\n"
                "- [ ] В docs/optimization_runs_YYYYMMDD.md добавлен план: что запускаем и критерии успеха\n"
                "\n"
                "Запуск: только через coint4/scripts/remote/run_server_job.sh на 85.198.90.128 (STOP_AFTER=1).\n"
            ),
        },
        {
            "title": f"S{next_sprint}: Синхронизировать статусы run_queue и пересобрать rollup (если надо)",
            "priority": 2,
            "description": (
                "Цель: чтобы ранкер/rollup видел актуальные результаты (и не было вечных planned).\n\n"
                "Ожидаемый эффект: корректные статусы в run_queue.csv и актуальный run_index.\n\n"
                "Проверка:\n"
                "- [ ] Для затронутых групп запущен scripts/optimization/sync_queue_status.py (best-effort)\n"
                "- [ ] Пересобран индекс: scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup\n"
                "- [ ] В ретро/дневнике отмечено, что было обновлено\n"
                "\n"
                "Ограничение: не трогать тяжёлые артефакты runs/ (только aggregate/rollup).\n"
            ),
        },
        {
            "title": f"S{next_sprint}: Улучшить критерий отбора кандидатов (DD/tail/trades) в одном месте",
            "priority": 3,
            "description": (
                "Цель: сделать единый ‘gate’ для кандидатов, чтобы не оптимизировать мусор.\n\n"
                "Ожидаемый эффект: документированный фильтр (в docs или утилите), который использует менеджер.\n\n"
                "Проверка:\n"
                "- [ ] Добавлен/обновлён короткий раздел в docs/optimization_state.md: какие фильтры применяем\n"
                "- [ ] В tools/sprint_manager.py (или отдельной утилите) эти фильтры отражены как параметры/константы\n"
            ),
        },
        {
            "title": f"S{next_sprint}: Sprint Retro + Plan Next Sprint",
            "priority": 4,
            "description": (
                "Цель: закрыть спринт осмысленным резюме и подготовить следующий цикл.\n\n"
                "Ожидаемый эффект: обновлён .ralph-tui/progress.md и согласованные гипотезы на следующий спринт.\n\n"
                "Проверка:\n"
                "- [ ] В .ralph-tui/progress.md добавлена секция по спринту (что сделали/выводы)\n"
                "- [ ] Явно записано: что запускаем дальше на remote VPS и зачем\n"
                "- [ ] Никаких git commit руками (autoCommit=true)\n"
            ),
        },
    ]

    if not sharpe_audit_ok:
        target_line = f"- target_run_dir: `{best_results_dir}`\n" if best_results_dir else ""
        tasks.insert(
            1,
            {
                "title": f"S{next_sprint}: Проверить ‘слишком высокий Sharpe’ на консистентность метрик",
                "priority": 1,
                "description": (
                    "Цель: исключить артефакты расчёта Sharpe/annualization/данных.\n\n"
                    "Ожидаемый эффект: понимание, можно ли доверять топовым Sharpe из run_index.\n\n"
                    "Контекст:\n"
                    f"{target_line}"
                    "\n"
                    "Проверка:\n"
                    "- [ ] Запущен tools/audit_sharpe.py (или validate_single_rerun.py) для 1–3 топовых прогонов\n"
                    + (f"  - например: `python3 tools/audit_sharpe.py --runs-glob '{best_results_dir}'`\n" if best_results_dir else "")
                    + "- [ ] Краткий вывод добавлен в docs/optimization_runs_YYYYMMDD.md\n"
                    "- [ ] Если найден баг/несовпадение — заведена отдельная задача/фикс (маленькая, в 1 итерацию)\n"
                    "\n"
                    "Ограничение: не запускать тяжёлые прогоны здесь; только аудит уже имеющихся артефактов.\n"
                ),
            },
        )

    return _ensure_min_tasks(tasks, min_count=6, warnings=warnings)


def _should_stop(
    *,
    state: dict[str, Any],
    last_results: Optional[dict[str, Any]],
    warnings: list[str],
) -> tuple[bool, str]:
    if bool(state.get("done")):
        return True, str(state.get("done_reason") or "done flag already set")
    if bool(state.get("force_done")):
        return True, "force_done=true"
    # Conservative: never stop on early sprints.
    sprint_n = int(state.get("sprint") or 1)
    if sprint_n < 5:
        return False, "sprint < 5 (conservative stop guardrail)"
    if not last_results or not isinstance(last_results.get("best"), dict):
        return False, "no parsed results"
    beads_stats = last_results.get("beads")
    if isinstance(beads_stats, dict):
        closed_cnt = int(beads_stats.get("closed") or 0)
        total_cnt = int(beads_stats.get("total") or 0)
        if total_cnt > 0 and closed_cnt == 0:
            return False, "no closed beads in reviewed sprint (stop disabled)"
    best = last_results["best"]
    sharpe = _to_float(best.get("sharpe"))
    dd = _to_float(best.get("dd_abs"))
    trades = _to_float(best.get("trades"))
    if sharpe is None or dd is None or trades is None:
        return False, "missing key metrics"
    if sharpe < 3.0:
        return False, "Sharpe < 3.0"
    if dd > 0.30:
        return False, "|DD| > 0.30"
    if trades < 1000:
        return False, "trades < 1000"
    # If we reached here: plausible candidate. Still allow continuation unless plateau.
    history = state.get("history")
    if isinstance(history, list):
        vals = [x for x in history if isinstance(x, (int, float))]
        if len(vals) >= 3:
            last3 = vals[-3:]
            if max(last3) - min(last3) < 0.01:
                return True, "goal met and best Sharpe plateaued for 3 sprints"
    warnings.append("goal met, but no plateau evidence in history; continuing")
    return False, "goal met but plateau not proven"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epic-id", required=True, help="Beads epic id")
    ap.add_argument("--goal", default=str(DEFAULT_GOAL), help="Path to goal.md")
    ap.add_argument("--state", default=str(DEFAULT_STATE), help="Path to state.json")
    ap.add_argument("--repo-root", default=str(PROJECT_ROOT), help="Repo root (default: inferred)")
    ap.add_argument("--dry-run", action="store_true", help="Do not create beads tasks")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    state_path = Path(args.state).resolve()
    goal_path = Path(args.goal).resolve()

    warnings: list[str] = []

    if not state_path.exists():
        raise SystemExit(f"state.json not found: {state_path}")

    state = _read_json(state_path)
    if not isinstance(state, dict):
        raise SystemExit("state.json must be an object")

    sprint_n = int(state.get("sprint") or 1)

    # Step A: best-effort results collection
    run_index_path = _find_run_index_json(repo_root)
    top_rows: list[RunIndexRow] = []
    if run_index_path and run_index_path.exists():
        try:
            payload = _read_json(run_index_path)
            if isinstance(payload, list):
                top_rows = _pick_top_candidates(_extract_run_index_rows(payload), limit=5)
            else:
                warnings.append(f"run_index.json unexpected root type: {type(payload).__name__}")
        except Exception as e:
            warnings.append(f"failed to read {run_index_path}: {e}")
    else:
        warnings.append("run_index.json not found (checked canonical paths)")

    best_row = top_rows[0] if top_rows else None
    last_results: dict[str, Any] = {
        "generatedAt": _now_iso(),
        "sprintReviewed": sprint_n,
        "run_index_path": str(run_index_path) if run_index_path else None,
        "best": (
            {
                "sharpe": best_row.sharpe,
                "dd_abs": best_row.dd_abs,
                "trades": best_row.trades,
                "run_group": best_row.run_group,
                "run_id": best_row.run_id,
                "config_path": best_row.config_path,
                "results_dir": best_row.results_dir,
            }
            if best_row
            else None
        ),
        "top5": [
            {
                "sharpe": r.sharpe,
                "dd_abs": r.dd_abs,
                "trades": r.trades,
                "run_group": r.run_group,
                "run_id": r.run_id,
                "config_path": r.config_path,
                "results_dir": r.results_dir,
            }
            for r in top_rows
        ],
    }

    # Beads sprint task summary (label-driven, best-effort)
    sprint_label = f"sprint-{sprint_n}"
    sprint_tasks = _bd_json(
        [
            "list",
            "--json",
            "--parent",
            args.epic_id,
            "--label",
            f"autopilot,{sprint_label}",
            "--limit",
            "0",
            "--all",
        ],
        cwd=repo_root,
        warnings=warnings,
    )
    closed = [t for t in sprint_tasks if str(t.get("status")) == "closed"]
    in_progress = [t for t in sprint_tasks if str(t.get("status")) == "in_progress"]
    open_ = [t for t in sprint_tasks if str(t.get("status")) == "open"]
    last_results["beads"] = {
        "label": sprint_label,
        "total": len(sprint_tasks),
        "closed": len(closed),
        "in_progress": len(in_progress),
        "open": len(open_),
        "closed_task_ids": [str(t.get("id")) for t in closed],
    }

    # Sticky "done once" audits: if the Sharpe-consistency audit task was closed in this sprint,
    # record it into state so it won't be re-scheduled every sprint.
    if any(_is_sharpe_audit_task(t) for t in closed):
        audits = state.get("audits")
        if not isinstance(audits, dict):
            audits = {}
            state["audits"] = audits
        sharpe_audit = audits.get("sharpe_consistency")
        if not isinstance(sharpe_audit, dict):
            sharpe_audit = {}
            audits["sharpe_consistency"] = sharpe_audit
        sharpe_audit.setdefault("ok", True)
        sharpe_audit.setdefault("at", _now_iso())
        sharpe_audit.setdefault("sprint", sprint_n)

    sprint_complete = len(sprint_tasks) > 0 and len(open_) == 0 and len(in_progress) == 0

    # Step B: retro report
    _ensure_reports_dir()
    retro_path = REPORTS_DIR / f"sprint_{sprint_n}_retro.md"
    template_path = DEFAULT_RETRO_TEMPLATE
    if not template_path.exists():
        warnings.append(f"retro template not found: {template_path}")
        template = "# Sprint {{SPRINT}} — Retro\n\n{{WARNINGS}}\n"
    else:
        template = template_path.read_text(encoding="utf-8")

    what_done_lines = []
    if closed:
        for t in closed:
            tid = str(t.get("id") or "")
            title = str(t.get("title") or "").strip()
            what_done_lines.append(f"- [{tid}] {title}".rstrip())
    else:
        what_done_lines.append("- (не найдено закрытых задач по label sprint-*; проверьте labels)")

    artifacts_lines = []
    if run_index_path:
        artifacts_lines.append(f"- {run_index_path}")
        md = run_index_path.with_suffix(".md")
        csv = run_index_path.with_suffix(".csv")
        if md.exists():
            artifacts_lines.append(f"- {md}")
        if csv.exists():
            artifacts_lines.append(f"- {csv}")
    else:
        artifacts_lines.append("- (run_index.* не найден)")

    decisions_lines = []
    if isinstance(state.get("decisions"), list) and state["decisions"]:
        decisions_lines.append("Предыдущие:")
        for d in state["decisions"][-5:]:
            if isinstance(d, dict):
                decisions_lines.append(f"- S{d.get('sprint')}: {d.get('summary')}")
    decisions_lines.append("Новые:")
    decisions_lines.append(f"- S{sprint_n}: зафиксировать результаты и запланировать следующий спринт")

    # Step C: next sprint planning
    next_sprint = sprint_n + 1
    planned_tasks = _generate_next_sprint_tasks(
        next_sprint=next_sprint,
        state=state,
        last_results=last_results,
        warnings=warnings,
    )
    planned_titles = "\n".join([f"- {t['title']}" for t in planned_tasks])

    rendered = _render_template(
        template,
        {
            "SPRINT": str(sprint_n),
            "DATE": _today_ymd(),
            "EPIC_ID": args.epic_id,
            "SPRINT_GOAL": "Стабилизировать качество и подготовить следующий информативный блок прогонов",
            "WHAT_DONE": "\n".join(what_done_lines) + "\n",
            "WHAT_RAN": (
                "- Парсер собрал типовые метрики (run_index.*) и статусы beads задач по label\n"
                "- Тяжёлые прогоны на этом сервере не выполнялись\n"
            ),
            "METRICS": _format_candidates_table(top_rows),
            "ARTIFACTS": "\n".join(artifacts_lines) + "\n",
            "CONCLUSIONS": (
                "- Текущий best-effort baseline зафиксирован из rollup.\n"
                "- Следующий шаг — дешёвые проверки доверия к метрикам + план remote-прогонов.\n"
            ),
            "DECISIONS": "\n".join(decisions_lines) + "\n",
            "NEXT_SPRINT": planned_titles + "\n",
            "WARNINGS": ("\n".join([f"- {w}" for w in warnings]) + "\n") if warnings else "- (нет)\n",
        },
    )
    retro_path.write_text(rendered, encoding="utf-8")

    if not sprint_complete:
        _log(
            f"Спринт S{sprint_n} НЕ завершён (open={len(open_)}, in_progress={len(in_progress)}, total={len(sprint_tasks)}); планирование следующего спринта пропущено"
        )
        _log(f"Ретро записано: {retro_path}")
        warnings.append(
            "current sprint tasks are not complete (open/in_progress remain); skipping next sprint creation and state increment"
        )
        state["last_results"] = last_results
        state.setdefault("decisions", [])
        if isinstance(state["decisions"], list):
            state["decisions"].append(
                {
                    "sprint": sprint_n,
                    "at": _now_iso(),
                    "summary": "retro generated, but sprint not complete; no planning",
                    "retro_path": str(retro_path.relative_to(repo_root)),
                    "best": last_results.get("best"),
                }
            )
        state["done"] = False
        state["done_reason"] = "sprint not complete"
        _write_json(state_path, state)
        print(
            json.dumps(
                {"retro": str(retro_path), "done": False, "done_reason": "sprint not complete", "createdTaskIds": []},
                ensure_ascii=False,
            )
        )
        for w in warnings:
            print(f"[sprint_manager][WARN] {w}")
        return 0

    # Update state
    state["last_results"] = last_results
    state.setdefault("decisions", [])
    if isinstance(state["decisions"], list):
        state["decisions"].append(
            {
                "sprint": sprint_n,
                "at": _now_iso(),
                "summary": "retro generated; next sprint planned",
                "retro_path": str(retro_path.relative_to(repo_root)),
                "best": last_results.get("best"),
                "next_sprint": next_sprint,
            }
        )
    state.setdefault("history", [])
    if isinstance(state["history"], list) and best_row is not None:
        state["history"].append(best_row.sharpe)

    done, done_reason = _should_stop(state=state, last_results=last_results, warnings=warnings)
    if done:
        state["done"] = True
        state["done_reason"] = done_reason
    else:
        state["done"] = False
        state["done_reason"] = done_reason
        state["sprint"] = next_sprint

    _write_json(state_path, state)

    # Create next sprint tasks (if not done)
    created_ids: list[str] = []
    if not done and not args.dry_run:
        # Idempotency: skip if tasks for next sprint label already exist.
        existing_next = _bd_json(
            [
                "list",
                "--json",
                "--parent",
                args.epic_id,
                "--label",
                f"autopilot,sprint-{next_sprint}",
                "--limit",
                "0",
                "--all",
            ],
            cwd=repo_root,
            warnings=warnings,
        )
        if existing_next:
            warnings.append(f"next sprint tasks already exist for sprint-{next_sprint}; skipping creation")
        else:
            labels = f"ralph,autopilot,sprint-{next_sprint}"
            for t in planned_tasks:
                title = str(t["title"])
                desc = str(t["description"])
                prio = int(t.get("priority") or 2)
                proc = _run(
                    [
                        "bd",
                        "create",
                        "--type",
                        "task",
                        "--title",
                        title,
                        "--description",
                        desc,
                        "--labels",
                        labels,
                        "--priority",
                        str(prio),
                        "--parent",
                        args.epic_id,
                        "--silent",
                    ],
                    cwd=repo_root,
                )
                if proc.returncode != 0:
                    warnings.append(f"bd create failed for '{title}': {proc.stderr.strip() or proc.stdout.strip()}")
                    continue
                created_ids.append(proc.stdout.strip())
            _run(["bd", "sync"], cwd=repo_root)

    if best_row is not None:
        _log(
            "Baseline (best-effort, из rollup): "
            f"Sharpe≈{best_row.sharpe:.3f}, |DD|≈{best_row.dd_abs:.3f}, trades≈{best_row.trades:.0f}; "
            f"run_group={best_row.run_group or '-'}; config={best_row.config_path or '-'}"
        )
    else:
        _log("Baseline не найден: run_index.* отсутствует или не распарсился (см. WARNINGS в ретро)")

    _log(f"Ретро записано: {retro_path}")

    if done:
        _log(f"Остановка: done=true ({done_reason})")
    else:
        _log("Почему такие задачи: сначала дешёвые проверки доверия к метрикам, затем план remote-прогонов и hygiene rollup/queues")
        _log(f"Следующий спринт S{next_sprint} (tasks={len(planned_tasks)}, created={len(created_ids)}):")
        for t in planned_tasks:
            title = str(t.get("title") or "").strip()
            prio = int(t.get("priority") or 3)
            _log(f"- P{prio}: {title}")

    # Machine-readable line (useful for scripts/CI)
    print(
        json.dumps(
            {"retro": str(retro_path), "done": done, "done_reason": done_reason, "createdTaskIds": created_ids},
            ensure_ascii=False,
        )
    )
    if warnings:
        for w in warnings:
            print(f"[sprint_manager][WARN] {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
