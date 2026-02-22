#!/usr/bin/env python3
"""Run WFA queue on a powered compute VPS via SSH."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
import textwrap
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional


class _PoweredFailure(RuntimeError):
    """Typed failure for powered runner control flow."""

    def __init__(self, message: str, *, error_class: str, fatal: bool) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.fatal = fatal


_SP_MODULE = None


def _find_repo_root(start_path: Path | None = None) -> Path:
    cursor = (start_path or Path(__file__).resolve()).resolve()
    if not cursor.is_dir():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(
        f"Cannot find repo root from {start_path or Path(__file__).resolve()}; expected parent with src/ and scripts/"
    )


def _project_root() -> Path:
    return _find_repo_root(Path(__file__).resolve())


def _load_serverspace_module(project_root: Path | None = None):
    global _SP_MODULE
    if _SP_MODULE is not None:
        return _SP_MODULE
    module_path = (
        (project_root or _find_repo_root(Path(__file__).resolve()))
        / "src"
        / "coint2"
        / "ops"
        / "serverspace_power.py"
    ).resolve()
    if not module_path.exists():
        raise RuntimeError(f"serverspace module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("_coint2_serverspace_power", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load serverspace module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    _SP_MODULE = module
    return module


def _safe_api_key(project_root: Path | None = None) -> str:
    return _load_serverspace_module(project_root).load_api_key()


def _safe_serverspace_error(project_root: Path | None = None):
    return _load_serverspace_module(project_root).ServerspaceError


def _safe_get_status(api_key: str, server_id: str, project_root: Path | None = None) -> str:
    return _load_serverspace_module(project_root).get_status(api_key, server_id)


def _safe_power_on(api_key: str, server_id: str, project_root: Path | None = None) -> None:
    return _load_serverspace_module(project_root).power_on(api_key, server_id)


def _safe_power_off(api_key: str, server_id: str, project_root: Path | None = None) -> None:
    return _load_serverspace_module(project_root).power_off(api_key, server_id)


def _safe_resolve_server_id_by_ip(api_key: str, ip: str, project_root: Path | None = None) -> str:
    return _load_serverspace_module(project_root).resolve_server_id_by_ip(api_key, ip)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _derive_run_group(queue_path: Path) -> str:
    if queue_path.parent.name and queue_path.parent.name != "aggregate":
        return queue_path.parent.name
    return "unknown"


def _log_file_path(project_root: Path, queue_path: Path) -> Path:
    run_group = _derive_run_group(queue_path)
    return (
        project_root
        / "artifacts"
        / "wfa"
        / "aggregate"
        / run_group
        / "logs"
        / f"powered_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    )


def _write_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now_stamp()} {message}\n")
        handle.flush()


def _emit(msg: str, log_path: Path, *, to_stderr: bool = True) -> None:
    if to_stderr:
        print(msg, file=sys.stderr, flush=True)
    _write_log(log_path, msg)


def _parse_bool_flag(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}. Use true/false.")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def compute_backoff_delay(attempt_index: int, base_seconds: float, cap_seconds: float = 120.0) -> float:
    return min(float(cap_seconds), float(base_seconds) * (2.0 ** max(0, attempt_index)))


def _resolve_queue_argument(queue_arg: str, project_root: Path) -> Path:
    queue_candidate = Path(queue_arg)
    if queue_candidate.is_absolute():
        return queue_candidate

    cwd_path = (Path.cwd() / queue_candidate).resolve()
    if cwd_path.exists():
        return cwd_path

    return (project_root / queue_candidate).resolve()


def _to_repo_relative(path: Path, project_root: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = project_root / p
    p = p.resolve()
    try:
        return p.relative_to(project_root.resolve())
    except ValueError:
        parts = p.parts
        if len(parts) > 1:
            return Path(*parts[1:])
        return Path(p.name)


def _is_busy_power_error(exc: Exception) -> bool:
    text = str(exc).strip().lower()
    return (
        "busy" in text
        or "conflict occurred during the competitive change of the object" in text
        or "wait until the end of the previous operation and try again" in text
    )


def _exec_ssh_command(
    host: str,
    user: str,
    remote_command: str,
    log_path: Path,
    *,
    port: int,
    command_purpose: str,
    log: Callable[[str], None],
) -> tuple[int, str]:
    remote_shell_cmd = f"bash -lc {shlex.quote(remote_command)}"
    ssh_cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-p",
        str(int(port)),
        f"{user}@{host}",
        remote_shell_cmd,
    ]
    log(f"powered: ssh_cmd={shlex.join(ssh_cmd)}")
    try:
        proc = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise _PoweredFailure(f"SSH executable missing: {exc}", error_class="SSH_BINARY_MISSING", fatal=True) from exc
    except Exception as exc:  # noqa: BLE001
        raise _PoweredFailure(
            f"SSH command failed to start: {type(exc).__name__}: {exc}",
            error_class="NETWORK",
            fatal=False,
        ) from exc

    output = f"{proc.stdout or ''}{proc.stderr or ''}"
    _write_log(log_path, f"ssh: purpose={command_purpose} rc={proc.returncode}")
    _write_log(log_path, f"remote_cmd={remote_command}")
    if output.strip():
        _write_log(log_path, f"output=\n{output}")
    return proc.returncode, output


def _run_scp_file(source: Path, destination_user: str, destination_host: str, destination: Path, *, port: int, log: Callable[[str], None]) -> None:
    target = f"{destination_user}@{destination_host}:{destination}"
    scp_cmd = [
        "scp",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-P",
        str(int(port)),
        str(source),
        target,
    ]
    log(f"powered: scp_cmd={shlex.join(scp_cmd[:-1])} {target}")
    try:
        proc = subprocess.run(scp_cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise _PoweredFailure(f"SCP executable missing: {exc}", error_class="SCP_BINARY_MISSING", fatal=True) from exc
    except Exception as exc:  # noqa: BLE001
        raise _PoweredFailure(
            f"SCP command failed: {type(exc).__name__}: {exc}",
            error_class="NETWORK",
            fatal=False,
        ) from exc

    output = f"{proc.stdout or ''}{proc.stderr or ''}"
    if proc.returncode != 0:
        raise _PoweredFailure(
            f"SCP failed for {target}",
            error_class="REMOTE_SYNC_FAILED",
            fatal=True,
        )
    log(f"powered: scp_ok target={target}")
    if output.strip():
        log(f"powered: scp_output=\n{output}")


def _fetch_remote_file(
    source_user: str,
    source_host: str,
    source: Path,
    destination: Path,
    *,
    port: int,
    log: Callable[[str], None],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_ref = f"{source_user}@{source_host}:{source}"
    scp_cmd = [
        "scp",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-P",
        str(int(port)),
        source_ref,
        str(destination),
    ]
    log(f"powered: scp_fetch={source_ref} -> {destination}")
    try:
        proc = subprocess.run(scp_cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise _PoweredFailure(f"SCP executable missing: {exc}", error_class="SCP_BINARY_MISSING", fatal=True) from exc
    except Exception as exc:  # noqa: BLE001
        raise _PoweredFailure(
            f"SCP fetch failed: {type(exc).__name__}: {exc}",
            error_class="NETWORK",
            fatal=False,
        ) from exc

    output = f"{proc.stdout or ''}{proc.stderr or ''}"
    if proc.returncode != 0:
        raise _PoweredFailure(
            f"SCP fetch failed for {source_ref}",
            error_class="REMOTE_SYNC_FAILED",
            fatal=True,
        )
    if output.strip():
        log(f"powered: scp_fetch_output=\n{output}")


def _sync_rollup_back(
    *,
    host: str,
    user: str,
    remote_repo: Path,
    local_project_root: Path,
    queue_path: Path,
    queue_relative: Path,
    port: int,
    log: Callable[[str], None],
) -> None:
    remote_queue = remote_repo / str(queue_relative).replace("\\", "/")
    _fetch_remote_file(
        source_user=user,
        source_host=host,
        source=remote_queue,
        destination=queue_path,
        port=port,
        log=log,
    )

    local_rollup_dir = local_project_root / "artifacts" / "wfa" / "aggregate" / "rollup"
    remote_rollup_dir = remote_repo / "artifacts" / "wfa" / "aggregate" / "rollup"
    _fetch_remote_file(
        source_user=user,
        source_host=host,
        source=remote_rollup_dir / "run_index.csv",
        destination=local_rollup_dir / "run_index.csv",
        port=port,
        log=log,
    )
    for name in ("run_index.json", "run_index.md"):
        try:
            _fetch_remote_file(
                source_user=user,
                source_host=host,
                source=remote_rollup_dir / name,
                destination=local_rollup_dir / name,
                port=port,
                log=log,
            )
        except _PoweredFailure:
            log(f"powered: optional rollup artifact missing on remote: {name}")


def _remote_rank_result_local_path(project_root: Path, run_group: str) -> Path:
    return (
        project_root
        / "artifacts"
        / "optimization_state"
        / "rank_results"
        / f"{run_group}_latest.json"
    )


def _remote_rank_and_sync(
    *,
    host: str,
    user: str,
    remote_repo: Path,
    remote_python: str,
    queue_relative: str,
    run_group: str,
    project_root: Path,
    log_path: Path,
    port: int,
    log: Callable[[str], None],
) -> Path:
    queue_rel = str(queue_relative).replace("\\", "/")
    rank_out_rel = f"artifacts/wfa/aggregate/{run_group}/rank_result.json"
    remote_py = textwrap.dedent(
        """
        import csv
        import json
        import os
        import pathlib
        import subprocess
        import sys
        import tempfile

        def _to_float(value):
            try:
                if value is None:
                    return None
                text = str(value).strip()
                if not text:
                    return None
                return float(text)
            except Exception:
                return None

        def _normalize(path):
            return str(path or "").strip().replace("\\\\", "/")

        repo = pathlib.Path.cwd()
        run_group = str(os.environ.get("RUN_GROUP") or "").strip()
        queue_rel = str(os.environ.get("QUEUE_REL") or "").strip()
        rank_out_rel = str(os.environ.get("RANK_OUT_REL") or "").strip()
        out_path = repo / rank_out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "ok": False,
            "run_group": run_group,
            "best_run_name": "",
            "best_config_path": "",
            "best_score": None,
            "worst_robust_sharpe": None,
            "worst_dd_pct": None,
            "details": "NO_COMPLETED_METRICS_YET",
        }

        try:
            queue_path = pathlib.Path(queue_rel)
            if not queue_path.is_absolute():
                queue_path = repo / queue_path

            run_ids = set()
            results_dirs = set()
            config_paths = set()
            if queue_path.exists():
                with queue_path.open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        run_id = str(row.get("run_name") or row.get("run_id") or "").strip()
                        if run_id:
                            run_ids.add(run_id)
                        results_dir = _normalize(row.get("results_dir"))
                        if results_dir:
                            results_dirs.add(results_dir)
                        config_path = _normalize(row.get("config_path"))
                        if config_path:
                            config_paths.add(config_path)

            if not (run_ids or results_dirs or config_paths):
                result["details"] = "QUEUE_EMPTY_OR_UNREADABLE"
            else:
                run_index = repo / "artifacts/wfa/aggregate/rollup/run_index.csv"
                if not run_index.exists():
                    result["details"] = "RUN_INDEX_MISSING"
                else:
                    tmp_fd, tmp_name = tempfile.mkstemp(
                        prefix="rank_scope_",
                        suffix=".csv",
                        dir=str(run_index.parent),
                    )
                    os.close(tmp_fd)
                    scoped_path = pathlib.Path(tmp_name)
                    scoped_rows = 0
                    try:
                        with run_index.open("r", encoding="utf-8", newline="") as src, scoped_path.open(
                            "w", encoding="utf-8", newline=""
                        ) as dst:
                            reader = csv.DictReader(src)
                            fieldnames = list(reader.fieldnames or [])
                            if not fieldnames:
                                result["details"] = "RUN_INDEX_INVALID_HEADER"
                            else:
                                writer = csv.DictWriter(dst, fieldnames=fieldnames)
                                writer.writeheader()
                                for row in reader:
                                    run_id = str(row.get("run_id") or "").strip()
                                    results_dir = _normalize(row.get("results_dir"))
                                    config_path = _normalize(row.get("config_path"))
                                    is_match = False
                                    if run_ids and run_id in run_ids:
                                        is_match = True
                                    if results_dirs and results_dir in results_dirs:
                                        is_match = True
                                    if config_paths and config_path in config_paths:
                                        is_match = True
                                    if not is_match:
                                        continue
                                    writer.writerow(row)
                                    scoped_rows += 1

                        if scoped_rows <= 0:
                            result["details"] = "NO_MATCHING_ROWS_IN_RUN_INDEX"
                        else:
                            rank_cmd = [
                                sys.executable,
                                "scripts/optimization/rank_multiwindow_robust_runs.py",
                                "--run-index",
                                str(scoped_path),
                                "--contains",
                                run_group,
                                "--top",
                                "1",
                            ]
                            proc = subprocess.run(rank_cmd, capture_output=True, text=True, check=False)
                            parsed = False
                            table_lines = [
                                line.strip()
                                for line in (proc.stdout or "").splitlines()
                                if line.strip().startswith("|")
                            ]
                            if len(table_lines) >= 3:
                                header = [chunk.strip() for chunk in table_lines[0].strip("|").split("|")]
                            else:
                                header = []
                            for line in table_lines[2:]:
                                if not header:
                                    continue
                                cols = [chunk.strip() for chunk in line.strip("|").split("|")]
                                if len(cols) != len(header):
                                    continue
                                row = dict(zip(header, cols))
                                if str(row.get("rank") or "").strip() != "1":
                                    continue
                                best_score = _to_float(row.get("score"))
                                worst_robust = _to_float(row.get("worst_robust_sh"))
                                if best_score is None:
                                    best_score = worst_robust
                                if worst_robust is None:
                                    worst_robust = best_score
                                worst_dd = _to_float(row.get("worst_dd_pct"))
                                best_run_name = str(row.get("variant_id") or "").strip()
                                best_config_path = str(row.get("sample_config") or "").strip()
                                if best_score is None or not best_run_name:
                                    continue
                                result.update(
                                    {
                                        "ok": True,
                                        "run_group": run_group,
                                        "best_run_name": best_run_name,
                                        "best_config_path": best_config_path,
                                        "best_score": best_score,
                                        "worst_robust_sharpe": worst_robust,
                                        "worst_dd_pct": worst_dd,
                                        "details": "REMOTE_RANK_OK",
                                    }
                                )
                                parsed = True
                                break
                            if not parsed:
                                result["details"] = "NO_COMPLETED_METRICS_YET"
                    finally:
                        try:
                            scoped_path.unlink()
                        except Exception:
                            pass
        except Exception as exc:  # noqa: BLE001
            result["ok"] = False
            result["details"] = f"REMOTE_RANK_EXCEPTION:{type(exc).__name__}:{exc}"

        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
        """
    ).strip()
    rank_cmd = (
        f"cd {shlex.quote(str(remote_repo))} "
        f"&& RUN_GROUP={shlex.quote(run_group)} "
        f"QUEUE_REL={shlex.quote(queue_rel)} "
        f"RANK_OUT_REL={shlex.quote(rank_out_rel)} "
        f"PYTHONPATH=src {shlex.quote(remote_python)} - <<'PY'\n"
        f"{remote_py}\n"
        "PY"
    )

    _run_remote_command(
        host,
        user,
        rank_cmd,
        log_path=log_path,
        port=port,
        log=log,
        command_purpose="remote-rank",
    )

    local_rank_path = _remote_rank_result_local_path(project_root, run_group)
    _fetch_remote_file(
        source_user=user,
        source_host=host,
        source=remote_repo / rank_out_rel,
        destination=local_rank_path,
        port=port,
        log=log,
    )
    ok = None
    details = ""
    try:
        payload = json.loads(local_rank_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            ok = bool(payload.get("ok"))
            details = str(payload.get("details") or "").strip()
        else:
            ok = None
    except Exception:  # noqa: BLE001
        ok = None
        details = "LOCAL_RANK_RESULT_PARSE_FAILED"
    log(f"powered: remote_rank ok={ok} path={remote_repo / rank_out_rel} details={details or 'n/a'}")
    log(f"powered: scp_fetch rank_result.json -> {local_rank_path}")
    return local_rank_path


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for item in paths:
        key = str(item.expanduser())
        if key in seen:
            continue
        seen.add(key)
        out.append(item.expanduser())
    return out


def _build_local_repo_bundle(
    *,
    project_root: Path,
    bundle_path: Path,
    log: Callable[[str], None],
) -> None:
    repo_root = project_root.parent
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists():
        bundle_path.unlink()

    excludes = [
        ".git",
        ".venv",
        ".codex",
        ".ralph-tui.bak_*",
        "artifacts",
        "logs",
        "coint4/.venv",
        "coint4/artifacts",
        "coint4/.codex",
        "coint4/.ralph-tui.bak_*",
        "coint4/data_downloaded",
        "coint4/logs",
    ]
    cmd = ["tar", "-czf", str(bundle_path)]
    for pattern in excludes:
        cmd.extend(["--exclude", pattern])
    cmd.extend(["-C", str(repo_root), "."])

    log(f"powered: bootstrap repo bundle create path={bundle_path}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise _PoweredFailure(
            f"tar executable missing: {exc}",
            error_class="LOCAL_BUNDLE_FAILED",
            fatal=True,
        ) from exc

    if proc.returncode != 0:
        raise _PoweredFailure(
            f"failed to build repo bundle rc={proc.returncode}",
            error_class="LOCAL_BUNDLE_FAILED",
            fatal=True,
        )

    size = bundle_path.stat().st_size if bundle_path.exists() else 0
    if size <= 0:
        raise _PoweredFailure(
            f"empty repo bundle: {bundle_path}",
            error_class="LOCAL_BUNDLE_FAILED",
            fatal=True,
        )
    log(f"powered: bootstrap repo bundle ready bytes={size}")


def _bootstrap_remote_repo(
    *,
    host: str,
    user: str,
    project_root: Path,
    bootstrap_remote_dir: Path,
    baseline_candidates: Iterable[Path],
    port: int,
    log_path: Path,
    log: Callable[[str], None],
) -> Path:
    local_bundle = Path("/tmp/coint4_repo_bundle.tar.gz")
    remote_bundle = Path("/tmp/coint4_repo_bundle.tar.gz")
    remote_root = bootstrap_remote_dir.expanduser()

    _build_local_repo_bundle(project_root=project_root, bundle_path=local_bundle, log=log)
    _run_remote_mkdir(host, user, remote_root, log_path, port=port, log=log)
    _run_scp_file(local_bundle, user, host, remote_bundle, port=port, log=log)

    unpack_cmd = (
        f"mkdir -p {shlex.quote(str(remote_root))} "
        f"&& tar -xzf {shlex.quote(str(remote_bundle))} -C {shlex.quote(str(remote_root))} "
        f"&& rm -f {shlex.quote(str(remote_bundle))}"
    )
    _run_remote_command(
        host,
        user,
        unpack_cmd,
        log_path=log_path,
        port=port,
        log=log,
        command_purpose="bootstrap-repo",
    )

    bootstrap_candidates = _unique_paths(
        [
            remote_root / "coint4",
            remote_root,
            *baseline_candidates,
        ]
    )
    detected = _detect_remote_repo(
        host=host,
        user=user,
        candidates=bootstrap_candidates,
        port=port,
        log_path=log_path,
        log=log,
    )
    log(f"powered: bootstrap repo complete remote_repo_detected={detected}")
    return detected


def _bootstrap_remote_venv(
    *,
    host: str,
    user: str,
    remote_repo: Path,
    port: int,
    log_path: Path,
    log: Callable[[str], None],
) -> None:
    quoted_repo = shlex.quote(str(remote_repo))
    quoted_venv_python = shlex.quote(str(remote_repo / ".venv" / "bin" / "python"))
    cmd = (
        f"if [ -x {quoted_venv_python} ]; then echo VENV_OK; exit 0; fi; "
        "if ! command -v python3 >/dev/null 2>&1; then echo PYTHON3_MISSING; exit 41; fi; "
        f"cd {quoted_repo} && python3 -m venv .venv && "
        ".venv/bin/python -m pip install --upgrade pip setuptools wheel && "
        "if [ -f requirements.txt ]; then "
        ".venv/bin/pip install -r requirements.txt; "
        "elif [ -f pyproject.toml ]; then "
        ".venv/bin/pip install -e .; "
        "else "
        "echo DEPS_FILE_MISSING; exit 42; "
        "fi"
    )
    rc, _output = _exec_ssh_command(
        host,
        user,
        cmd,
        log_path,
        port=port,
        command_purpose="bootstrap-venv",
        log=log,
    )
    if rc != 0:
        raise _PoweredFailure(
            "remote venv bootstrap failed",
            error_class="REMOTE_VENV_BOOTSTRAP_FAILED",
            fatal=True,
        )
    log(f"powered: bootstrap venv ready repo={remote_repo}")


def _classify_remote_error(remote_command: str, return_code: int, output: str) -> tuple[str, bool]:
    lowered = (output or "").lower()

    if return_code == 0:
        return "SUCCESS", False

    if "no such file or directory" in lowered:
        if "bash:" in lowered and "cd" in lowered:
            return "REMOTE_REPO_NOT_FOUND", True
        if "run_wfa_queue.py" in lowered:
            return "REMOTE_CMD_NOT_FOUND", True
        if ".csv" in lowered or "queue" in lowered:
            return "REMOTE_QUEUE_MISSING", True

    if "command not found" in lowered and "python" in lowered:
        return "REMOTE_PYTHON_NOT_FOUND", True

    if return_code == 127:
        if "python3" in lowered or "python" in lowered:
            return "REMOTE_PYTHON_NOT_FOUND", True
        return "REMOTE_CMD_NOT_FOUND", True

    if return_code == 255:
        return "NETWORK", False

    return "REMOTE_EXEC_FAILED", False


def _run_with_retries(
    operation: Callable[[], int],
    *,
    max_retries: int,
    backoff_seconds: float,
    log: Callable[[str], None],
) -> int:
    if max_retries < 1:
        raise ValueError("--max-retries must be >= 1")

    for attempt in range(1, int(max_retries) + 1):
        try:
            return int(operation())
        except _PoweredFailure as exc:
            log(f"powered: error_class={exc.error_class} fatal={str(exc.fatal).lower()}")
            if exc.fatal:
                raise
            if attempt >= max_retries:
                raise
            delay = compute_backoff_delay(attempt - 1, float(backoff_seconds), cap_seconds=120.0)
            log(f"retry: attempt={attempt}/{max_retries} error={exc.error_class} next_in={delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(f"run failed after {max_retries} attempts")


def _run_remote_command(
    host: str,
    user: str,
    remote_command: str,
    log_path: Path,
    *,
    port: int,
    log: Callable[[str], None],
    command_purpose: str,
) -> int:
    return_code, output = _exec_ssh_command(
        host,
        user,
        remote_command,
        log_path,
        port=port,
        command_purpose=command_purpose,
        log=log,
    )
    if return_code != 0:
        error_class, fatal = _classify_remote_error(remote_command, return_code, output)
        raise _PoweredFailure(
            f"Remote command failed: {remote_command!r} rc={return_code}",
            error_class=error_class,
            fatal=fatal,
        )
    return int(return_code)


def _run_remote_mkdir(
    host: str,
    user: str,
    directory: Path,
    log_path: Path,
    *,
    port: int,
    log: Callable[[str], None],
) -> None:
    cmd = f"mkdir -p {shlex.quote(str(directory))}"
    log(f"powered: run_remote_mkdir={directory}")
    _run_remote_command(
        host,
        user,
        cmd,
        log_path,
        port=port,
        log=log,
        command_purpose="mkdir",
    )


def _detect_remote_repo(
    host: str,
    user: str,
    candidates: Iterable[Path],
    *,
    port: int,
    log_path: Path,
    log: Callable[[str], None],
) -> Path:
    checked: list[str] = []
    last_network_exc: Optional[_PoweredFailure] = None
    for candidate in candidates:
        candidate = candidate.expanduser()
        candidate_str = str(candidate)
        checked.append(candidate_str)
        log(f"powered: remote_repo_check={candidate_str}")
        check_cmd = (
            f"[ -f {shlex.quote(candidate_str + '/scripts/optimization/run_wfa_queue.py')} ] "
            f"&& [ -d {shlex.quote(candidate_str + '/src')} ] "
            f"&& ( [ -f {shlex.quote(candidate_str + '/pyproject.toml')} ] || [ -f {shlex.quote(candidate_str + '/requirements.txt')} ] )"
        )
        try:
            rc = _run_remote_command(
                host=host,
                user=user,
                remote_command=check_cmd,
                log_path=log_path,
                port=port,
                log=log,
                command_purpose="repo-check",
            )
        except _PoweredFailure as exc:
            if exc.error_class == "NETWORK":
                last_network_exc = exc
                continue
            if exc.fatal:
                raise
            continue
        if rc == 0:
            log(f"powered: remote_repo_detected={candidate_str}")
            return candidate
    if last_network_exc is not None:
        raise last_network_exc
    raise _PoweredFailure(
        f"remote repo not found on compute; checked: {', '.join(checked)}",
        error_class="REMOTE_REPO_NOT_FOUND",
        fatal=True,
    )


def _resolve_remote_repo(
    remote_repo_arg: str,
    host: str,
    user: str,
    candidates: Iterable[Path],
    port: int,
    log_path: Path,
    log: Callable[[str], None],
) -> Path:
    if remote_repo_arg.lower() == "auto":
        return _detect_remote_repo(host, user, candidates, port=port, log_path=log_path, log=log)

    explicit = Path(remote_repo_arg).expanduser()
    return _detect_remote_repo(host, user, [explicit], port=port, log_path=log_path, log=log)


def _detect_remote_python(
    host: str,
    user: str,
    remote_repo: Path,
    *,
    port: int,
    log_path: Path,
    log: Callable[[str], None],
) -> str:
    remote_repo_str = str(remote_repo)
    quoted_venv_python = shlex.quote(str(remote_repo / ".venv" / "bin" / "python"))
    cmd = (
        f"if [ -x {quoted_venv_python} ]; then\n"
        f"  {quoted_venv_python} -V >/dev/null 2>&1 && echo {quoted_venv_python}\n"
        f"elif command -v python3 >/dev/null 2>&1; then\n"
        f"  python3 -V >/dev/null 2>&1 && echo python3\n"
        f"else\n"
        f"  echo __NOT_FOUND__\n"
        f"fi"
    )
    rc, output = _exec_ssh_command(
        host,
        user,
        cmd,
        log_path,
        port=port,
        command_purpose="detect-python",
        log=log,
    )
    if rc != 0:
        raise _PoweredFailure("Could not detect remote python", error_class="REMOTE_PYTHON_NOT_FOUND", fatal=True)
    chosen = (output or "").strip().splitlines()[-1] if output else ""
    if not chosen or chosen == "__NOT_FOUND__":
        raise _PoweredFailure("Remote python not found", error_class="REMOTE_PYTHON_NOT_FOUND", fatal=True)
    log(f"powered: remote_python={chosen}")
    return chosen


def _wait_for_status(
    api_key: str,
    server_id: str,
    project_root: Optional[Path] = None,
    *,
    timeout_sec: int = 600,
    poll_sec: float = 3.0,
    log: Callable[[str], None],
) -> None:
    deadline = time.time() + max(1, int(timeout_sec))
    attempts = 0
    last_status = ""
    while time.time() < deadline:
        attempts += 1
        status = (_safe_get_status(api_key, server_id, project_root=project_root) or "").strip().lower()
        if status:
            if status in {"on", "running", "active", "started", "up"}:
                log(f"powered: wait_status running reached in {attempts} attempts")
                return
            last_status = status
        log(f"powered: wait_status running ... attempt={attempts}, current={last_status or 'unknown'}")
        time.sleep(max(0.2, float(poll_sec)))
    raise _safe_serverspace_error(project_root)(
        f"Timeout waiting for server RUNNING. Last status={last_status or 'unknown'}"
    )


def ensure_server_ready(
    api_key: str,
    server_id: str,
    host: str,
    user: str,
    *,
    project_root: Optional[Path] = None,
    port: int = 22,
    log_path: Path,
    log: Callable[[str], None],
) -> None:
    log(f"powered: power_on requested")
    try:
        _safe_power_on(api_key, server_id, project_root=project_root)
    except Exception as exc:  # noqa: BLE001
        if _is_busy_power_error(exc):
            log("powered: power_on reported busy/conflict; continue with readiness checks")
        else:
            raise
    _wait_for_status(
        api_key,
        server_id,
        project_root=project_root,
        timeout_sec=600,
        poll_sec=3.0,
        log=log,
    )
    log(f"powered: wait_ssh host={host} port={port}")
    if _wait_ssh_ready(host, port, timeout_sec=300, poll_sec=2.0, log=log):
        return

    # Sometimes API reports ACTIVE while SSH remains unreachable; attempt one hard power-cycle.
    log("powered: wait_ssh failed after power_on; attempting recovery power cycle")
    try:
        _safe_power_off(api_key, server_id, project_root=project_root)
        log("powered: recovery power_off requested")
    except Exception as exc:  # noqa: BLE001
        log(f"powered: recovery power_off failed: {type(exc).__name__}")
    time.sleep(5.0)

    log("powered: recovery power_on requested")
    try:
        _safe_power_on(api_key, server_id, project_root=project_root)
    except Exception as exc:  # noqa: BLE001
        if _is_busy_power_error(exc):
            log("powered: recovery power_on reported busy/conflict; continue")
        else:
            raise

    _wait_for_status(
        api_key,
        server_id,
        project_root=project_root,
        timeout_sec=600,
        poll_sec=3.0,
        log=log,
    )
    log(f"powered: wait_ssh retry host={host} port={port}")
    if not _wait_ssh_ready(host, port, timeout_sec=300, poll_sec=2.0, log=log):
        raise _PoweredFailure("SSH readiness check failed after recovery cycle", error_class="NETWORK", fatal=True)


def _wait_ssh_ready(
    host: str,
    port: int,
    *,
    timeout_sec: int = 300,
    poll_sec: float = 2.0,
    log: Callable[[str], None],
) -> bool:
    deadline = time.time() + max(1, int(timeout_sec))
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with socket.create_connection((host, int(port)), timeout=3):
                log(f"powered: ssh ok host={host} port={port}")
                return True
        except OSError as exc:
            msg = str(exc)
            log(f"powered: wait_ssh host={host} attempt={attempt} err={msg}")
        time.sleep(max(0.2, float(poll_sec)))
    return False


def _read_queue_config_paths(queue_path: Path) -> list[tuple[str, Path]]:
    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return []

    header = rows[0]
    config_field = None
    for key in header:
        normalized = (key or "").strip().lower().replace(" ", "")
        if normalized in {"config_path", "configpath", "config", "configfile"}:
            config_field = key
            break
    if config_field is None:
        for key in header:
            if key and "config" in key.lower():
                config_field = key
                break
    if config_field is None:
        raise _PoweredFailure("Queue file does not contain config path column", error_class="QUEUE_FORMAT", fatal=True)

    paths = []
    seen = set()
    for row in rows:
        value = (row.get(config_field) or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        paths.append((value, Path(value)))
    return paths


def _sync_configs_bulk(
    *,
    host: str,
    user: str,
    config_paths: list[Path],
    remote_repo: Path,
    project_root: Path,
    port: int,
    log_path: Path,
    log: Callable[[str], None],
) -> int:
    if not config_paths:
        return 0

    bundle_name = f"powered_configs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.tar.gz"
    local_bundle = Path("/tmp") / bundle_name
    remote_bundle = Path("/tmp") / bundle_name
    rel_paths: list[str] = []
    for path in config_paths:
        rel = _to_repo_relative(path, project_root).as_posix()
        rel_paths.append(rel)

    cmd = ["tar", "-czf", str(local_bundle), "-C", str(project_root), *rel_paths]
    log(f"powered: bulk-config-sync bundle_create files={len(rel_paths)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise _PoweredFailure(
            f"tar executable missing: {exc}",
            error_class="LOCAL_BUNDLE_FAILED",
            fatal=True,
        ) from exc
    if proc.returncode != 0:
        raise _PoweredFailure(
            f"bulk config bundle failed rc={proc.returncode}",
            error_class="LOCAL_BUNDLE_FAILED",
            fatal=True,
        )

    try:
        _run_scp_file(local_bundle, user, host, remote_bundle, port=port, log=log)
        unpack_cmd = (
            f"mkdir -p {shlex.quote(str(remote_repo))} "
            f"&& tar -xzf {shlex.quote(str(remote_bundle))} -C {shlex.quote(str(remote_repo))} "
            f"&& rm -f {shlex.quote(str(remote_bundle))}"
        )
        _run_remote_command(
            host,
            user,
            unpack_cmd,
            log_path=log_path,
            port=port,
            log=log,
            command_purpose="bulk-sync-configs",
        )
    finally:
        try:
            local_bundle.unlink(missing_ok=True)
        except OSError:
            pass

    log(f"powered: synced configs bulk: {len(rel_paths)} files")
    return len(rel_paths)


def _sync_inputs(
    host: str,
    user: str,
    queue_path: Path,
    remote_repo: Path,
    remote_python: str,
    project_root: Path,
    *,
    bulk_configs: bool,
    force_remote_queue_overwrite: bool,
    port: int,
    log_path: Path,
    log: Callable[[str], None],
) -> None:
    queue_relative = _to_repo_relative(queue_path, project_root)
    queue_relative_str = str(queue_relative).replace("\\", "/")
    remote_queue = remote_repo / queue_relative
    _run_remote_mkdir(host, user, remote_queue.parent, log_path, port=port, log=log)

    queue_uploaded = 0
    should_upload_queue = True
    if not bool(force_remote_queue_overwrite):
        try:
            remote_probe = get_remote_queue_counts(
                host=host,
                user=user,
                remote_repo=remote_repo,
                remote_python=remote_python,
                queue_relative=queue_relative_str,
                port=int(port),
                log_path=log_path,
                log=log,
            )
            remote_counts = remote_probe.get("counts") if isinstance(remote_probe.get("counts"), dict) else {}
            has_progress = False
            for status, value in remote_counts.items():
                status_name = str(status or "").strip().lower()
                try:
                    count = int(value or 0)
                except (TypeError, ValueError):
                    count = 0
                if count <= 0:
                    continue
                if status_name not in {"planned", "stalled"}:
                    has_progress = True
                    break
            try:
                remote_completed = int(remote_counts.get("completed", 0) or 0)
            except (TypeError, ValueError):
                remote_completed = 0
            if remote_completed > 0:
                has_progress = True
            if has_progress:
                should_upload_queue = False
                log(
                    "powered: sync_inputs skip_remote_queue_overwrite "
                    "reason=REMOTE_QUEUE_HAS_PROGRESS counts={counts}".format(
                        counts=remote_counts,
                    )
                )
                _fetch_remote_file(
                    source_user=user,
                    source_host=host,
                    source=remote_queue,
                    destination=queue_path,
                    port=int(port),
                    log=log,
                )
        except _PoweredFailure as exc:
            if exc.error_class == "QUEUE_MISSING":
                should_upload_queue = True
            else:
                raise

    if should_upload_queue:
        _run_scp_file(queue_path, user, host, remote_queue, port=port, log=log)
        queue_uploaded = 1

    config_rows = _read_queue_config_paths(queue_path)
    unique_paths = [entry[1] for entry in config_rows]

    resolved_configs: list[Path] = []
    for cfg_path in unique_paths:
        local_cfg = cfg_path
        if not local_cfg.is_absolute():
            local_cfg = project_root / local_cfg
        local_cfg = local_cfg.resolve()
        if not local_cfg.exists():
            raise _PoweredFailure(
                f"Config file not found locally: {local_cfg}",
                error_class="LOCAL_CONFIG_MISSING",
                fatal=True,
            )
        resolved_configs.append(local_cfg)

    if bool(bulk_configs):
        copied = _sync_configs_bulk(
            host=host,
            user=user,
            config_paths=resolved_configs,
            remote_repo=remote_repo,
            project_root=project_root,
            port=port,
            log_path=log_path,
            log=log,
        )
    else:
        copied = 0
        for local_cfg in resolved_configs:
            remote_cfg = remote_repo / _to_repo_relative(local_cfg, project_root)
            _run_remote_mkdir(host, user, remote_cfg.parent, log_path, port=port, log=log)
            _run_scp_file(local_cfg, user, host, remote_cfg, port=port, log=log)
            copied += 1

    log(f"powered: synced queue={queue_uploaded} configs={copied}")
    _write_log(log_path, f"powered: synced queue={queue_uploaded} configs={copied}")


def _build_remote_command(
    remote_repo: Path,
    remote_python: str,
    queue_relative: str,
    *,
    statuses: str,
    parallel: int,
    postprocess: bool,
) -> str:
    _ = postprocess  # Postprocessing happens in this wrapper, not in run_wfa_queue.py.
    thread_exports = (
        "export OMP_NUM_THREADS=1; "
        "export MKL_NUM_THREADS=1; "
        "export OPENBLAS_NUM_THREADS=1; "
        "export NUMBA_NUM_THREADS=1; "
        "export NUMEXPR_NUM_THREADS=1;"
    )
    return (
        f"{thread_exports} "
        f"cd {shlex.quote(str(remote_repo))} "
        f"&& PYTHONPATH=src "
        f"{shlex.quote(remote_python)} "
        f"scripts/optimization/run_wfa_queue.py --queue {shlex.quote(queue_relative)} "
        f"--statuses {shlex.quote(statuses)} --parallel {int(parallel)}"
    )


def _detect_remote_nproc(
    *,
    host: str,
    user: str,
    port: int,
    log_path: Path,
    log: Callable[[str], None],
) -> int:
    rc, output = _exec_ssh_command(
        host,
        user,
        "nproc",
        log_path,
        port=int(port),
        command_purpose="detect-nproc",
        log=log,
    )
    if rc != 0:
        log(f"powered: parallel auto nproc probe failed rc={rc}; fallback=1")
        return 1
    for raw in (output or "").splitlines():
        token = raw.strip().split()
        if not token:
            continue
        try:
            value = int(token[0])
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    log("powered: parallel auto nproc parse failed; fallback=1")
    return 1


def _resolve_parallel(
    parallel_arg: object,
    *,
    host: str,
    user: str,
    port: int,
    log_path: Path,
    log: Callable[[str], None],
) -> int:
    text = str(parallel_arg).strip().lower()
    if text in {"auto", "0"}:
        nproc = _detect_remote_nproc(
            host=host,
            user=user,
            port=int(port),
            log_path=log_path,
            log=log,
        )
        chosen = max(1, int(nproc))
        log(f"powered: parallel auto detected nproc={nproc} chosen_parallel={chosen}")
        return chosen
    try:
        parsed = int(text)
    except (TypeError, ValueError) as exc:
        raise _PoweredFailure(
            f"invalid --parallel value: {parallel_arg!r}",
            error_class="PARALLEL_INVALID",
            fatal=True,
        ) from exc
    if parsed < 0:
        raise _PoweredFailure(
            f"invalid --parallel value: {parallel_arg!r}",
            error_class="PARALLEL_INVALID",
            fatal=True,
        )
    if parsed == 0:
        nproc = _detect_remote_nproc(
            host=host,
            user=user,
            port=int(port),
            log_path=log_path,
            log=log,
        )
        chosen = max(1, int(nproc))
        log(f"powered: parallel auto detected nproc={nproc} chosen_parallel={chosen}")
        return chosen
    return parsed


def _normalize_status_counts(raw_counts: object) -> dict[str, int]:
    normalized: dict[str, int] = {}
    if not isinstance(raw_counts, dict):
        return normalized
    for key, value in raw_counts.items():
        status = str(key or "").strip().lower()
        if not status:
            continue
        try:
            normalized[status] = int(value or 0)
        except (TypeError, ValueError):
            normalized[status] = 0
    return normalized


def _resolve_executor_statuses(
    statuses_arg: str,
    *,
    probe_payload: dict,
    log: Callable[[str], None],
) -> tuple[str, str, bool, bool]:
    requested = str(statuses_arg or "").strip()
    if requested.lower() != "auto":
        return requested, f"EXPLICIT_{requested or 'EMPTY'}", True, False

    counts = _normalize_status_counts(probe_payload.get("counts"))
    try:
        total = int(probe_payload.get("total") or 0)
    except (TypeError, ValueError):
        total = 0
    if total <= 0:
        total = int(sum(counts.values()))

    planned = int(counts.get("planned", 0))
    stalled = int(counts.get("stalled", 0))
    running = (
        int(counts.get("running", 0))
        + int(counts.get("active", 0))
        + int(counts.get("partial", 0))
        + int(counts.get("queued", 0))
        + int(counts.get("pending", 0))
    )
    completed = int(counts.get("completed", 0))
    failed = int(counts.get("failed", 0))
    error = int(counts.get("error", 0))

    chosen_statuses = ""
    rationale = "NOTHING_TO_RUN"
    should_wait = False
    failed_retry_mode = False

    if planned + stalled > 0:
        chosen_statuses = "planned,stalled"
        rationale = "HAS_PLANNED_OR_STALLED"
        should_wait = True
    elif running > 0:
        chosen_statuses = ""
        rationale = "REMOTE_RUNNING_IN_PROGRESS"
        should_wait = True
    elif failed + error > 0:
        failed_statuses: list[str] = []
        if failed > 0:
            failed_statuses.append("failed")
        if error > 0:
            failed_statuses.append("error")
        chosen_statuses = ",".join(failed_statuses)
        rationale = "RETRY_FAILED_OR_ERROR"
        should_wait = True
        failed_retry_mode = True
    elif total > 0 and completed >= total:
        chosen_statuses = ""
        rationale = "ALL_COMPLETED"
        should_wait = False

    log(
        "powered: auto_statuses counts={counts} chosen_statuses={chosen} rationale={rationale}".format(
            counts=counts,
            chosen=(chosen_statuses or "<none>"),
            rationale=rationale,
        )
    )
    return chosen_statuses, rationale, should_wait, failed_retry_mode


def _build_watchdog_command(
    *,
    remote_repo: Path,
    remote_python: str,
    queue_relative: str,
    stale_sec: int,
) -> str:
    queue_rel = str(queue_relative).replace("\\", "/")
    stale = max(60, int(stale_sec))
    remote_py = textwrap.dedent(
        """
        import csv
        import json
        import os
        import pathlib
        import subprocess
        import tempfile
        import time

        def _safe_int(value, default):
            try:
                return int(value)
            except Exception:
                return int(default)

        def _infer_run_dir(row, cwd, run_group):
            results_dir = str(row.get("results_dir") or "").strip()
            if results_dir:
                path = pathlib.Path(results_dir)
                if not path.is_absolute():
                    path = cwd / path
                return path
            run_id = str(row.get("run_name") or row.get("run_id") or "").strip()
            if not run_id:
                return None
            candidates = [
                cwd / "artifacts" / "wfa" / "runs" / run_group / run_id,
                cwd / "artifacts" / "wfa" / "runs_clean" / run_group / run_id,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return candidates[-1]

        def _latest_mtime(run_dir):
            if run_dir is None:
                return None
            try:
                if not run_dir.exists():
                    return None
            except Exception:
                return None

            mtimes = []
            tracked_names = (
                "run.log",
                "stdout.log",
                "stderr.log",
                "progress.log",
                "progress.json",
                "status.json",
                "strategy_metrics.csv",
                "equity_curve.csv",
                "canonical_metrics.json",
            )
            for name in tracked_names:
                candidate = run_dir / name
                if candidate.exists():
                    try:
                        mtimes.append(candidate.stat().st_mtime)
                    except Exception:
                        pass

            for pattern in ("*.log", "*.csv", "*.json"):
                for candidate in run_dir.glob(pattern):
                    if not candidate.is_file():
                        continue
                    try:
                        mtimes.append(candidate.stat().st_mtime)
                    except Exception:
                        pass

            if not mtimes:
                try:
                    return run_dir.stat().st_mtime
                except Exception:
                    return None
            return max(mtimes)

        queue_rel = str(os.environ.get("QUEUE_REL") or "").strip()
        stale_sec = _safe_int(os.environ.get("STALE_SEC"), 900)
        stale_sec = max(60, stale_sec)
        if not queue_rel:
            print(json.dumps({"ok": False, "error": "QUEUE_REL_EMPTY"}))
            raise SystemExit(2)

        cwd = pathlib.Path.cwd()
        queue_path = pathlib.Path(queue_rel)
        if not queue_path.is_absolute():
            queue_path = cwd / queue_path
        if not queue_path.exists():
            print(json.dumps({"ok": False, "error": "QUEUE_MISSING", "queue": str(queue_path)}))
            raise SystemExit(2)

        try:
            with queue_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = list(reader.fieldnames or [])
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": "QUEUE_PARSE_ERROR", "detail": str(exc)}))
            raise SystemExit(3)

        if not fieldnames:
            print(json.dumps({"ok": False, "error": "QUEUE_PARSE_ERROR", "detail": "missing_header"}))
            raise SystemExit(3)

        run_group = queue_path.parent.name
        now_ts = time.time()
        changed = 0
        stale_marked = 0
        running = 0
        sample = []

        ps_lines = []
        try:
            ps_proc = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, check=False)
            ps_lines = [line.strip().lower() for line in (ps_proc.stdout or "").splitlines() if line.strip()]
        except Exception:
            ps_lines = []

        for row in rows:
            status = str(row.get("status") or "").strip().lower()
            if status != "running":
                continue
            running += 1
            run_id = str(row.get("run_name") or row.get("run_id") or "").strip()
            run_dir = _infer_run_dir(row, cwd, run_group)
            run_dir_str = str(run_dir) if run_dir is not None else ""
            run_id_l = run_id.lower()
            run_dir_l = run_dir_str.lower()

            process_alive = False
            for line in ps_lines:
                if (
                    "run_wfa_queue.py" not in line
                    and "watch_wfa_queue.sh" not in line
                    and "run_wfa" not in line
                    and "backtest" not in line
                ):
                    continue
                if run_id_l and run_id_l in line:
                    process_alive = True
                    break
                if run_dir_l and run_dir_l in line:
                    process_alive = True
                    break
            if process_alive:
                continue

            latest_mtime = _latest_mtime(run_dir)
            if latest_mtime is None:
                stale = True
                age_sec = None
            else:
                age_sec = int(max(0.0, now_ts - latest_mtime))
                stale = bool(age_sec >= stale_sec)
            if not stale:
                continue

            row["status"] = "stalled"
            changed += 1
            stale_marked += 1
            if len(sample) < 10:
                sample.append(
                    {
                        "run_id": run_id,
                        "results_dir": str(row.get("results_dir") or "").strip(),
                        "age_sec": age_sec,
                    }
                )

        if changed > 0:
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix="queue_watchdog_",
                suffix=".csv",
                dir=str(queue_path.parent),
            )
            os.close(tmp_fd)
            tmp_path = pathlib.Path(tmp_name)
            try:
                with tmp_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
                os.replace(tmp_path, queue_path)
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

        payload = {
            "ok": True,
            "queue": str(queue_path),
            "stale_sec": stale_sec,
            "running": running,
            "changed": changed,
            "stale_marked": stale_marked,
            "rationale": "STALE_RUNNING_MARKED" if changed > 0 else "NO_STALE_RUNNING",
            "sample": sample,
        }
        print(json.dumps(payload, sort_keys=True))
        """
    ).strip()
    return (
        f"cd {shlex.quote(str(remote_repo))} "
        f"&& QUEUE_REL={shlex.quote(queue_rel)} "
        f"STALE_SEC={shlex.quote(str(stale))} "
        f"{shlex.quote(remote_python)} - <<'PY'\n"
        f"{remote_py}\n"
        "PY"
    )


def _remote_watchdog_stale_running(
    *,
    host: str,
    user: str,
    remote_repo: Path,
    remote_python: str,
    queue_relative: str,
    port: int,
    stale_sec: int,
    log_path: Path,
    log: Callable[[str], None],
) -> dict:
    cmd = _build_watchdog_command(
        remote_repo=remote_repo,
        remote_python=remote_python,
        queue_relative=queue_relative,
        stale_sec=stale_sec,
    )
    rc, output = _exec_ssh_command(
        host,
        user,
        cmd,
        log_path,
        port=int(port),
        command_purpose="queue-watchdog",
        log=log,
    )
    payload = _extract_last_json_line(output)
    if not isinstance(payload, dict):
        payload = {
            "ok": False,
            "error": "WATCHDOG_INVALID_OUTPUT",
            "rc": int(rc),
            "changed": 0,
            "stale_marked": 0,
            "rationale": "WATCHDOG_INVALID_OUTPUT",
            "sample": [],
        }
    payload.setdefault("ok", rc == 0)
    payload.setdefault("changed", 0)
    payload.setdefault("stale_marked", 0)
    payload.setdefault("sample", [])
    payload.setdefault("rationale", "WATCHDOG_OK" if rc == 0 else "WATCHDOG_EXEC_FAILED")
    payload["rc"] = int(rc)
    return payload


def _extract_last_json_line(text: str) -> Optional[dict]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _build_wait_probe_command(
    *,
    remote_repo: Path,
    remote_python: str,
    queue_relative: str,
) -> str:
    queue_rel = str(queue_relative).replace("\\", "/")
    return (
        f"cd {shlex.quote(str(remote_repo))} "
        f"&& QUEUE_REL={shlex.quote(queue_rel)} "
        f"{shlex.quote(remote_python)} - <<'PY'\n"
        "import csv\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "queue_rel = os.environ.get('QUEUE_REL', '').strip()\n"
        "if not queue_rel:\n"
        "    print(json.dumps({'ok': False, 'error': 'QUEUE_REL_EMPTY'}))\n"
        "    raise SystemExit(2)\n"
        "cwd = pathlib.Path.cwd()\n"
        "queue_path = pathlib.Path(queue_rel)\n"
        "if not queue_path.is_absolute():\n"
        "    queue_path = cwd / queue_path\n"
        "if not queue_path.exists():\n"
        "    print(json.dumps({'ok': False, 'error': 'QUEUE_MISSING', 'queue': str(queue_path)}))\n"
        "    raise SystemExit(2)\n"
        "try:\n"
        "    with queue_path.open('r', encoding='utf-8', newline='') as handle:\n"
        "        rows = list(csv.DictReader(handle))\n"
        "except Exception as exc:  # noqa: BLE001\n"
        "    print(json.dumps({'ok': False, 'error': 'QUEUE_PARSE_ERROR', 'detail': str(exc)}))\n"
        "    raise SystemExit(3)\n"
        "pending = {'planned', 'running', 'active', 'partial', 'queued', 'pending'}\n"
        "terminal = {'completed', 'stalled', 'failed', 'error', 'cancelled', 'skipped'}\n"
        "counts = {}\n"
        "all_done = True\n"
        "has_metrics = False\n"
        "for row in rows:\n"
        "    status = str(row.get('status') or 'planned').strip().lower()\n"
        "    counts[status] = counts.get(status, 0) + 1\n"
        "    if status in pending:\n"
        "        all_done = False\n"
        "    elif status not in terminal:\n"
        "        all_done = False\n"
        "    results_dir = str(row.get('results_dir') or '').strip()\n"
        "    if not results_dir:\n"
        "        continue\n"
        "    run_dir = pathlib.Path(results_dir)\n"
        "    if not run_dir.is_absolute():\n"
        "        run_dir = cwd / run_dir\n"
        "    if (\n"
        "        (run_dir / 'strategy_metrics.csv').exists()\n"
        "        and (run_dir / 'equity_curve.csv').exists()\n"
        "        and (run_dir / 'canonical_metrics.json').exists()\n"
        "    ):\n"
        "        has_metrics = True\n"
        "payload = {\n"
        "    'ok': True,\n"
        "    'queue': str(queue_path),\n"
        "    'total': len(rows),\n"
        "    'counts': counts,\n"
        "    'all_done': all_done,\n"
        "    'has_metrics': has_metrics,\n"
        "}\n"
        "print(json.dumps(payload, sort_keys=True))\n"
        "PY"
    )


def get_remote_queue_counts(
    host: str,
    user: str,
    remote_repo: Path,
    remote_python: str,
    queue_relative: str,
    port: int,
    *,
    log_path: Optional[Path] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Return remote queue status counters without waiting for completion."""
    probe_log_path = log_path or (Path("/tmp") / "powered_probe.log")
    if log_path is None:
        probe_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not probe_log_path.exists():
            probe_log_path.write_text("", encoding="utf-8")

    log_fn = log or (lambda _msg: None)
    probe_cmd = _build_wait_probe_command(
        remote_repo=remote_repo,
        remote_python=remote_python,
        queue_relative=queue_relative,
    )
    rc, output = _exec_ssh_command(
        host,
        user,
        probe_cmd,
        probe_log_path,
        port=int(port),
        command_purpose="queue-counts-probe",
        log=log_fn,
    )
    payload = _extract_last_json_line(output)
    if rc != 0:
        if isinstance(payload, dict):
            err = str(payload.get("error") or "QUEUE_PROBE_FAILED").strip() or "QUEUE_PROBE_FAILED"
            fatal = err in {"QUEUE_MISSING", "QUEUE_PARSE_ERROR", "QUEUE_REL_EMPTY"}
            raise _PoweredFailure(
                f"queue counts probe failed rc={rc}",
                error_class=err,
                fatal=fatal,
            )
        err_cls, fatal = _classify_remote_error(probe_cmd, rc, output)
        if err_cls == "REMOTE_EXEC_FAILED":
            err_cls = "QUEUE_PROBE_FAILED"
        raise _PoweredFailure(
            f"queue counts probe failed rc={rc}",
            error_class=err_cls,
            fatal=fatal,
        )

    if not payload:
        raise _PoweredFailure(
            "queue counts probe returned no JSON payload",
            error_class="QUEUE_PROBE_INVALID",
            fatal=False,
        )
    if not bool(payload.get("ok")):
        err = str(payload.get("error") or "QUEUE_PROBE_FAILED")
        fatal = err in {"QUEUE_MISSING", "QUEUE_PARSE_ERROR", "QUEUE_REL_EMPTY"}
        raise _PoweredFailure(
            f"queue counts probe error: {err}",
            error_class=err,
            fatal=fatal,
        )

    raw_counts = payload.get("counts")
    counts: dict[str, int] = {}
    if isinstance(raw_counts, dict):
        for key, value in raw_counts.items():
            norm = str(key or "").strip().lower()
            if not norm:
                continue
            try:
                counts[norm] = int(value or 0)
            except (TypeError, ValueError):
                counts[norm] = 0
    try:
        total = int(payload.get("total") or 0)
    except (TypeError, ValueError):
        total = 0
    if total <= 0:
        total = int(sum(counts.values()))
    return {
        "ok": True,
        "queue_exists": True,
        "queue": str(payload.get("queue") or ""),
        "counts": counts,
        "total": int(total),
        "all_done": bool(payload.get("all_done")),
        "has_metrics": bool(payload.get("has_metrics")),
    }


def _wait_for_completion(
    *,
    host: str,
    user: str,
    remote_repo: Path,
    remote_python: str,
    queue_relative: str,
    log_path: Path,
    port: int,
    timeout_sec: int,
    poll_sec: int,
    log: Callable[[str], None],
) -> None:
    deadline = time.time() + max(1, int(timeout_sec))
    attempt = 0
    all_done_without_metrics_streak = 0
    all_done_without_metrics_grace = 2

    while True:
        attempt += 1
        payload = get_remote_queue_counts(
            host=host,
            user=user,
            remote_repo=remote_repo,
            remote_python=remote_python,
            queue_relative=queue_relative,
            port=int(port),
            log_path=log_path,
            log=log,
        )

        all_done = bool(payload.get("all_done"))
        has_metrics = bool(payload.get("has_metrics"))
        counts = payload.get("counts")
        total = payload.get("total")
        elapsed = int(max(0, timeout_sec - max(0.0, deadline - time.time())))
        log(
            f"powered: wait_completion attempt={attempt} elapsed={elapsed}s total={total} "
            f"counts={counts} has_metrics={has_metrics}"
        )

        if all_done and has_metrics:
            log("powered: wait_completion completion detected")
            return
        if all_done and not has_metrics:
            all_done_without_metrics_streak += 1
            log(
                "powered: wait_completion all_done_without_metrics streak={streak}/{grace}".format(
                    streak=all_done_without_metrics_streak,
                    grace=all_done_without_metrics_grace,
                )
            )
            if all_done_without_metrics_streak >= all_done_without_metrics_grace:
                log("powered: wait_completion completion detected without metrics; continue to postprocess")
                return
        else:
            all_done_without_metrics_streak = 0

        if time.time() >= deadline:
            raise _PoweredFailure(
                "queue completion timeout",
                error_class="QUEUE_TIMEOUT",
                fatal=True,
            )
        time.sleep(max(1, int(poll_sec)))


def _remote_rebuild_rollup(
    *,
    host: str,
    user: str,
    remote_repo: Path,
    remote_python: str,
    queue_relative: str,
    log_path: Path,
    port: int,
    log: Callable[[str], None],
) -> None:
    cmd = (
        f"cd {shlex.quote(str(remote_repo))} "
        f"&& PYTHONPATH=src {shlex.quote(remote_python)} "
        "scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup"
    )
    _run_remote_command(
        host,
        user,
        cmd,
        log_path=log_path,
        port=port,
        log=log,
        command_purpose="remote-build-run-index",
    )


def run_powered_queue(
    args: argparse.Namespace,
    *,
    project_root: Path,
    log_path: Path,
) -> int:
    queue_path = _resolve_queue_argument(args.queue, project_root)
    queue_relative = _to_repo_relative(queue_path, project_root)
    remote_candidates = [Path(p) for p in args.remote_repo_candidates]

    started_on_compute = False
    remote_repo: Optional[Path] = None
    remote_python = None
    log = lambda msg: _emit(msg, log_path)
    skip_power = bool(getattr(args, "skip_power", False))
    api_key = ""
    server_id = ""
    try:
        if skip_power:
            log("powered: skip_power=true (Serverspace API bypass)")
            if args.poweroff:
                log("powered: skip_power=true so API power_off is disabled for this run")
            log(f"powered: skip_power ssh preflight host={args.compute_host} port={int(args.ssh_port)}")
            _run_remote_command(
                args.compute_host,
                args.ssh_user,
                "echo POWER_OK",
                log_path=log_path,
                port=int(args.ssh_port),
                log=log,
                command_purpose="skip-power-preflight",
            )
            started_on_compute = True
        else:
            api_key = _safe_api_key(project_root)
            server_id = (args.serverspace_server_id or os.getenv("SERVERSPACE_SERVER_ID") or "").strip()
            if server_id:
                log(f"powered: resolve server_id from env override {server_id}")
            else:
                log(f"powered: resolve server_id by ip={args.compute_host}")
                server_id = _safe_resolve_server_id_by_ip(api_key, args.compute_host, project_root=project_root)

            ensure_server_ready(
                api_key,
                server_id,
                args.compute_host,
                args.ssh_user,
                project_root=project_root,
                port=int(args.ssh_port),
                log_path=log_path,
                log=log,
            )
            started_on_compute = True

        if bool(args.preflight):
            _run_remote_command(
                args.compute_host,
                args.ssh_user,
                "echo POWER_OK",
                log_path=log_path,
                port=int(args.ssh_port),
                log=log,
                command_purpose="preflight",
            )

        try:
            remote_repo = _resolve_remote_repo(
                str(args.remote_repo),
                host=args.compute_host,
                user=args.ssh_user,
                candidates=remote_candidates,
                port=int(args.ssh_port),
                log_path=log_path,
                log=log,
            )
        except _PoweredFailure as exc:
            if exc.error_class != "REMOTE_REPO_NOT_FOUND" or not bool(args.bootstrap_repo):
                raise
            log("powered: remote repo not found; bootstrap repo start")
            remote_repo = _bootstrap_remote_repo(
                host=args.compute_host,
                user=args.ssh_user,
                project_root=project_root,
                bootstrap_remote_dir=Path(str(args.bootstrap_remote_dir)),
                baseline_candidates=remote_candidates,
                port=int(args.ssh_port),
                log_path=log_path,
                log=log,
            )

        if bool(args.bootstrap_venv):
            _bootstrap_remote_venv(
                host=args.compute_host,
                user=args.ssh_user,
                remote_repo=remote_repo,
                port=int(args.ssh_port),
                log_path=log_path,
                log=log,
            )

        remote_python = _detect_remote_python(
            args.compute_host,
            args.ssh_user,
            remote_repo,
            port=int(args.ssh_port),
            log_path=log_path,
            log=log,
        )

        remote_queue_name = str(queue_relative).replace("\\", "/")

        if bool(getattr(args, "probe_queue", False)):
            probe = get_remote_queue_counts(
                host=args.compute_host,
                user=args.ssh_user,
                remote_repo=remote_repo,
                remote_python=remote_python,
                queue_relative=remote_queue_name,
                port=int(args.ssh_port),
                log_path=log_path,
                log=log,
            )
            log(
                "powered: probe_queue total={total} counts={counts} has_metrics={has_metrics}".format(
                    total=int(probe.get("total") or 0),
                    counts=probe.get("counts") or {},
                    has_metrics=bool(probe.get("has_metrics")),
                )
            )
            print(json.dumps(probe, ensure_ascii=False, sort_keys=True), flush=True)
            return 0

        chosen_parallel = _resolve_parallel(
            args.parallel,
            host=args.compute_host,
            user=args.ssh_user,
            port=int(args.ssh_port),
            log_path=log_path,
            log=log,
        )

        if bool(args.sync_inputs):
            _sync_inputs(
                host=args.compute_host,
                user=args.ssh_user,
                queue_path=queue_path,
                remote_repo=remote_repo,
                remote_python=remote_python,
                project_root=project_root,
                bulk_configs=bool(args.sync_configs_bulk),
                force_remote_queue_overwrite=bool(args.force_remote_queue_overwrite),
                port=int(args.ssh_port),
                log_path=log_path,
                log=log,
            )

        if bool(args.dry_run):
            _emit("powered: DRY_RUN_SUCCESS", log_path)
            return 0

        assert remote_repo is not None
        assert remote_python is not None
        selected_statuses = str(args.statuses or "").strip()
        auto_statuses_rationale = "EXPLICIT"
        should_wait_completion = bool(args.wait_completion)
        failed_retry_mode = False
        if selected_statuses.lower() == "auto":
            auto_probe_payload: Optional[dict] = None
            try:
                auto_probe_payload = get_remote_queue_counts(
                    host=args.compute_host,
                    user=args.ssh_user,
                    remote_repo=remote_repo,
                    remote_python=remote_python,
                    queue_relative=remote_queue_name,
                    port=int(args.ssh_port),
                    log_path=log_path,
                    log=log,
                )
            except _PoweredFailure as exc:
                if exc.error_class in {"QUEUE_MISSING", "QUEUE_PARSE_ERROR", "QUEUE_REL_EMPTY", "QUEUE_PROBE_FAILED", "QUEUE_PROBE_INVALID"}:
                    selected_statuses = "planned,stalled"
                    auto_statuses_rationale = f"PROBE_FAILED_{exc.error_class}"
                    should_wait_completion = bool(args.wait_completion)
                    log(
                        "powered: auto_statuses counts={} chosen_statuses=planned,stalled "
                        "rationale={rationale}".format(
                            auto_probe_payload.get("counts") if isinstance(auto_probe_payload, dict) else {},
                            rationale=auto_statuses_rationale,
                        )
                    )
                else:
                    raise
            if auto_probe_payload is not None:
                (
                    selected_statuses,
                    auto_statuses_rationale,
                    auto_should_wait,
                    failed_retry_mode,
                ) = _resolve_executor_statuses(
                    selected_statuses,
                    probe_payload=auto_probe_payload,
                    log=log,
                )
                should_wait_completion = bool(args.wait_completion) and bool(auto_should_wait)

                if (
                    bool(getattr(args, "watchdog", False))
                    and not selected_statuses
                    and auto_statuses_rationale == "REMOTE_RUNNING_IN_PROGRESS"
                ):
                    watchdog_payload = _remote_watchdog_stale_running(
                        host=args.compute_host,
                        user=args.ssh_user,
                        remote_repo=remote_repo,
                        remote_python=remote_python,
                        queue_relative=remote_queue_name,
                        port=int(args.ssh_port),
                        stale_sec=int(getattr(args, "watchdog_stale_sec", 900)),
                        log_path=log_path,
                        log=log,
                    )
                    watchdog_changed = int(watchdog_payload.get("changed") or 0)
                    watchdog_running = int(
                        watchdog_payload.get("running")
                        or _normalize_status_counts(auto_probe_payload.get("counts")).get("running", 0)
                    )
                    watchdog_rationale = str(watchdog_payload.get("rationale") or "WATCHDOG_RESULT").strip()
                    log(
                        "powered: watchdog checked stale_sec={stale_sec} running={running} changed={changed} rationale={rationale}".format(
                            stale_sec=int(getattr(args, "watchdog_stale_sec", 900)),
                            running=watchdog_running,
                            changed=watchdog_changed,
                            rationale=watchdog_rationale,
                        )
                    )

                    if watchdog_changed > 0:
                        try:
                            auto_probe_payload = get_remote_queue_counts(
                                host=args.compute_host,
                                user=args.ssh_user,
                                remote_repo=remote_repo,
                                remote_python=remote_python,
                                queue_relative=remote_queue_name,
                                port=int(args.ssh_port),
                                log_path=log_path,
                                log=log,
                            )
                            (
                                selected_statuses,
                                auto_statuses_rationale,
                                auto_should_wait,
                                failed_retry_mode,
                            ) = _resolve_executor_statuses(
                                "auto",
                                probe_payload=auto_probe_payload,
                                log=log,
                            )
                            should_wait_completion = bool(args.wait_completion) and bool(auto_should_wait)
                        except _PoweredFailure as exc:
                            log(f"powered: watchdog reprobe failed error_class={exc.error_class}")

        remote_rc = 0
        if selected_statuses:
            log("powered: remote_env pinned_threads=1")
            full_remote_cmd = _build_remote_command(
                remote_repo,
                remote_python,
                remote_queue_name,
                statuses=selected_statuses,
                parallel=int(chosen_parallel),
                postprocess=bool(args.postprocess),
            )

            def _run_once() -> int:
                _emit(
                    f"powered: remote run start queue={remote_queue_name} statuses={selected_statuses}",
                    log_path,
                )
                return _run_remote_command(
                    args.compute_host,
                    args.ssh_user,
                    full_remote_cmd,
                    log_path=log_path,
                    port=int(args.ssh_port),
                    log=log,
                    command_purpose="queue-run",
                )

            queue_max_retries = int(args.max_retries)
            if failed_retry_mode:
                queue_max_retries = max(1, min(queue_max_retries, 2))
                log(
                    f"powered: auto_statuses retry_limit statuses={selected_statuses} max_retries={queue_max_retries}"
                )

            remote_rc = _run_with_retries(
                _run_once,
                max_retries=queue_max_retries,
                backoff_seconds=float(args.backoff_seconds),
                log=lambda msg: _emit(msg, log_path),
            )
        else:
            log(
                "powered: queue-run skipped chosen_statuses=<none> rationale={rationale}".format(
                    rationale=auto_statuses_rationale,
                )
            )

        if should_wait_completion:
            _emit(
                "powered: wait_completion start timeout_sec={timeout} poll_sec={poll}".format(
                    timeout=int(args.wait_timeout_sec),
                    poll=int(args.wait_poll_sec),
                ),
                log_path,
            )
            _wait_for_completion(
                host=args.compute_host,
                user=args.ssh_user,
                remote_repo=remote_repo,
                remote_python=remote_python,
                queue_relative=remote_queue_name,
                log_path=log_path,
                port=int(args.ssh_port),
                timeout_sec=int(args.wait_timeout_sec),
                poll_sec=int(args.wait_poll_sec),
                log=log,
            )

        _remote_rebuild_rollup(
            host=args.compute_host,
            user=args.ssh_user,
            remote_repo=remote_repo,
            remote_python=remote_python,
            queue_relative=remote_queue_name,
            log_path=log_path,
            port=int(args.ssh_port),
            log=log,
        )
        _sync_rollup_back(
            host=args.compute_host,
            user=args.ssh_user,
            remote_repo=remote_repo,
            local_project_root=project_root,
            queue_path=queue_path,
            queue_relative=queue_relative,
            port=int(args.ssh_port),
            log=log,
        )
        _remote_rank_and_sync(
            host=args.compute_host,
            user=args.ssh_user,
            remote_repo=remote_repo,
            remote_python=remote_python,
            queue_relative=remote_queue_name,
            run_group=_derive_run_group(queue_path),
            project_root=project_root,
            log_path=log_path,
            port=int(args.ssh_port),
            log=log,
        )

        return int(remote_rc)
    finally:
        if args.poweroff and started_on_compute and not skip_power:
            _emit("powered: power_off requested (finally)", log_path)
            try:
                _safe_power_off(api_key, server_id, project_root=project_root)
            except Exception as exc:  # noqa: BLE001
                _emit(f"powered: power_off failed: {type(exc).__name__}", log_path)
        elif args.poweroff and skip_power:
            _emit("powered: power_off skipped (skip_power=true)", log_path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WFA queue on compute VPS")
    parser.add_argument("--queue", required=True, help="Path to run_queue.csv")
    parser.add_argument("--compute-host", default="85.198.90.128", help="Compute VPS public IP")
    parser.add_argument(
        "--remote-repo",
        default="auto",
        help="Remote repo path on compute or 'auto' to detect.",
    )
    parser.add_argument(
        "--remote-repo-candidates",
        nargs="*",
        default=(
            [
                "/home/claudeuser/coint4/coint4",
                "/home/claudeuser/coint4",
                "/root/coint4/coint4",
                "/root/coint4",
                "/opt/coint4/coint4",
                "/opt/coint4",
            ]
        ),
    )
    parser.add_argument(
        "--statuses",
        default=None,
        help="Statuses for remote queue runner. Use 'auto' to choose statuses from remote queue progress.",
    )
    parser.add_argument(
        "--parallel",
        default="1",
        help="Parallel workers for remote run_wfa_queue.py (int, 0, or auto).",
    )
    parser.add_argument("--postprocess", type=_parse_bool_flag, default=True)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--poweroff", type=_parse_bool_flag, default=True)
    parser.add_argument(
        "--skip-power",
        type=_parse_bool_flag,
        default=_env_bool("COINT4_SKIP_POWER", False),
        help="Skip Serverspace API power actions and use direct SSH readiness check.",
    )
    parser.add_argument("--max-retries", type=int, default=30)
    parser.add_argument("--backoff-seconds", type=float, default=10.0)
    parser.add_argument("--serverspace-server-id", default=None)
    parser.add_argument("--preflight", type=_parse_bool_flag, default=True)
    parser.add_argument("--sync-inputs", type=_parse_bool_flag, default=True)
    parser.add_argument(
        "--force-remote-queue-overwrite",
        type=_parse_bool_flag,
        default=False,
        help="Force upload local queue over remote queue even when remote has progress.",
    )
    parser.add_argument(
        "--sync-configs-bulk",
        type=_parse_bool_flag,
        default=None,
        help=(
            "Sync config files as one tarball/scp batch. "
            "Default: true in AUTONOMOUS_MODE=1, otherwise false."
        ),
    )
    parser.add_argument(
        "--wait-completion",
        type=_parse_bool_flag,
        default=None,
        help=(
            "Wait for queue terminal statuses and at least one completed metrics set "
            "(strategy_metrics.csv + equity_curve.csv + canonical_metrics.json). "
            "Default: true in AUTONOMOUS_MODE=1, otherwise false."
        ),
    )
    parser.add_argument(
        "--wait-timeout-sec",
        type=int,
        default=21600,
        help="Queue completion wait timeout in seconds (default: 21600).",
    )
    parser.add_argument(
        "--wait-poll-sec",
        type=int,
        default=60,
        help="Queue completion polling interval in seconds (default: 60).",
    )
    parser.add_argument(
        "--watchdog",
        type=_parse_bool_flag,
        default=None,
        help="Mark stale remote running rows as stalled before auto-statuses decision.",
    )
    parser.add_argument(
        "--watchdog-stale-sec",
        type=int,
        default=900,
        help="Stale timeout for running rows in watchdog mode (default: 900).",
    )
    parser.add_argument(
        "--bootstrap-repo",
        type=_parse_bool_flag,
        default=None,
        help="Bootstrap remote repo bundle when repo is missing.",
    )
    parser.add_argument(
        "--bootstrap-remote-dir",
        default="/opt/coint4",
        help="Remote dir used by bootstrap repo flow.",
    )
    parser.add_argument(
        "--bootstrap-venv",
        type=_parse_bool_flag,
        default=None,
        help="Bootstrap remote .venv when missing.",
    )
    parser.add_argument(
        "--probe-queue",
        type=_parse_bool_flag,
        default=False,
        help="Only probe remote queue counts (planned/running/stalled/completed) and exit.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    autonomous_mode = str(os.environ.get("AUTONOMOUS_MODE") or "").strip() == "1"
    non_interactive = not sys.stdin.isatty()
    if args.bootstrap_repo is None:
        args.bootstrap_repo = bool(autonomous_mode or non_interactive)
    if args.bootstrap_venv is None:
        args.bootstrap_venv = bool(autonomous_mode)
    if args.wait_completion is None:
        args.wait_completion = bool(autonomous_mode)
    if args.sync_configs_bulk is None:
        args.sync_configs_bulk = bool(autonomous_mode)
    if args.watchdog is None:
        args.watchdog = bool(autonomous_mode)
    if int(args.watchdog_stale_sec) <= 0:
        args.watchdog_stale_sec = 900
    raw_statuses = str(args.statuses or "").strip()
    if not raw_statuses:
        args.statuses = "auto" if autonomous_mode else "planned,stalled"
    else:
        args.statuses = raw_statuses
    return args


def _ensure_log_file(project_root: Path, queue_path: Path) -> Path:
    queue_for_log = queue_path.resolve()
    if queue_for_log.suffix.lower() != ".csv":
        queue_for_log = queue_for_log.parent / "run_queue.csv"
    preferred = _log_file_path(project_root, queue_for_log)
    preferred.parent.mkdir(parents=True, exist_ok=True)
    preferred.write_text("", encoding="utf-8")
    return preferred


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    project_root = _project_root()
    queue_path = _resolve_queue_argument(args.queue, project_root)

    try:
        log_path = _ensure_log_file(project_root, queue_path)
    except OSError:
        log_path = Path("/tmp") / "artifacts" / "wfa" / "aggregate" / _derive_run_group(queue_path.resolve()) / "logs" / f"powered_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    _emit(f"powered: log_file={log_path}", log_path)
    _emit(
        f"powered: start dry_run={bool(args.dry_run)}",
        log_path,
    )
    _emit(
        "powered: mode bootstrap_repo={repo} bootstrap_venv={venv} bootstrap_remote_dir={root} "
        "wait_completion={wait} wait_timeout_sec={timeout} wait_poll_sec={poll} "
        "sync_configs_bulk={bulk} watchdog={watchdog} watchdog_stale_sec={stale}".format(
            repo=bool(args.bootstrap_repo),
            venv=bool(args.bootstrap_venv),
            root=str(args.bootstrap_remote_dir),
            wait=bool(args.wait_completion),
            timeout=int(args.wait_timeout_sec),
            poll=int(args.wait_poll_sec),
            bulk=bool(args.sync_configs_bulk),
            watchdog=bool(getattr(args, "watchdog", False)),
            stale=int(getattr(args, "watchdog_stale_sec", 900)),
        ),
        log_path,
    )

    if not queue_path.exists():
        reason = f"Queue file not found: {queue_path}"
        _emit(f"powered: FAIL reason={reason}", log_path)
        return 2

    if queue_path.suffix.lower() != ".csv":
        reason = f"Queue file must be CSV, got: {queue_path}"
        _emit(f"powered: FAIL reason={reason}", log_path)
        return 2

    try:
        rc = run_powered_queue(args, project_root=project_root, log_path=log_path)
        _emit("powered: done", log_path)
        return int(rc)
    except _PoweredFailure as exc:
        _emit(f"powered: FAIL reason={exc.error_class}", log_path)
        _emit(f"powered: {type(exc).__name__}: {exc}", log_path)
        return 3
    except Exception as exc:  # noqa: BLE001
        _emit(f"powered: FAIL reason={type(exc).__name__}", log_path)
        with log_path.open("a", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
