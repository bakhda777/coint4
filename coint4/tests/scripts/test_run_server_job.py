from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "remote" / "run_server_job.sh"


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=repo)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)


def _prepare_repos(tmp_path: Path, initial_files: dict[str, str], removed_after_clone: list[str]) -> tuple[Path, Path, str]:
    local_repo = tmp_path / "local_repo"
    remote_repo = tmp_path / "remote_repo"
    local_repo.mkdir(parents=True, exist_ok=True)

    _git(local_repo, "init")
    _git(local_repo, "config", "user.email", "test@example.com")
    _git(local_repo, "config", "user.name", "Test User")

    for rel_path, content in initial_files.items():
        _write(local_repo / rel_path, content)
    _commit_all(local_repo, "initial")

    _run(["git", "clone", str(local_repo), str(remote_repo)], cwd=tmp_path)

    for rel_path in removed_after_clone:
        (local_repo / rel_path).unlink()
    _commit_all(local_repo, "remove stale files")
    head_sha = _git(local_repo, "rev-parse", "HEAD").stdout.strip()
    return local_repo, remote_repo, head_sha


def _write_stubs(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)

    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|-o)
      shift 2
      ;;
    *)
      break
      ;;
  esac
done
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -eq 0 ]]; then
  exit 0
fi
cmd="$1"
shift || true
if [[ $# -gt 0 ]]; then
  cmd="${cmd} $*"
fi
exec bash -lc "$cmd"
""",
        encoding="utf-8",
    )
    ssh_stub.chmod(0o755)

    rsync_stub = bin_dir / "rsync"
    rsync_stub.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def unwrap(spec: str) -> tuple[Path, bool]:
    if ":" in spec and "@" in spec.split(":", 1)[0]:
        return Path(spec.split(":", 1)[1]), True
    return Path(spec), False


args = sys.argv[1:]
files_from = None
from0 = False
positionals: list[str] = []
i = 0
while i < len(args):
    arg = args[i]
    if arg in {"-a", "-z", "-az", "-za"}:
        i += 1
        continue
    if arg == "-e":
        i += 2
        continue
    if arg == "--from0":
        from0 = True
        i += 1
        continue
    if arg.startswith("--files-from="):
        files_from = arg.split("=", 1)[1]
        i += 1
        continue
    if arg == "--files-from":
        files_from = args[i + 1]
        i += 2
        continue
    if arg.startswith("-"):
        i += 1
        continue
    positionals.append(arg)
    i += 1

if len(positionals) != 2:
    raise SystemExit(f"unexpected rsync args: {sys.argv[1:]!r}")

src, _ = unwrap(positionals[0])
dst, _ = unwrap(positionals[1])

if files_from is not None:
    raw = sys.stdin.buffer.read() if files_from == "-" else Path(files_from).read_bytes()
    items = [chunk.decode("utf-8") for chunk in raw.split(b"\\0" if from0 else b"\\n") if chunk]
    for rel_path in items:
        src_path = src / rel_path
        dst_path = dst / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.is_symlink():
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            os.symlink(os.readlink(src_path), dst_path)
        elif src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
else:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
""",
        encoding="utf-8",
    )
    rsync_stub.chmod(0o755)


def _run_sync_up(tmp_path: Path, local_repo: Path, remote_repo: Path, mode: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    _write_stubs(bin_dir)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["SKIP_POWER"] = "1"
    env["STOP_AFTER"] = "0"
    env["SYNC_BACK"] = "0"
    env["UPDATE_CODE"] = "0"
    env["SYNC_UP"] = "1"
    env["SYNC_UP_MODE"] = mode
    env["SERVER_IP"] = "127.0.0.1"
    env["SERVER_USER"] = "root"
    env["SSH_KEY"] = str(tmp_path / "dummy_ed25519")
    env["LOCAL_REPO_DIR"] = str(local_repo)
    env["SERVER_REPO_DIR"] = str(remote_repo)
    env["SERVER_WORK_DIR"] = str(remote_repo)
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), "true"],
        cwd=local_repo,
        env=env,
        capture_output=True,
        text=True,
    )


def test_sync_up_tracked_removes_stale_remote_tracked_files(tmp_path: Path) -> None:
    local_repo, remote_repo, head_sha = _prepare_repos(
        tmp_path,
        initial_files={
            "README.md": "demo\n",
            "scripts/data/current.py": "print('current')\n",
            "scripts/data/stale.py": "print('stale')\n",
        },
        removed_after_clone=["scripts/data/stale.py"],
    )
    _write(remote_repo / "coint4/outputs/runtime.log", "runtime\n")

    proc = _run_sync_up(tmp_path, local_repo, remote_repo, "tracked")

    assert proc.returncode == 0, proc.stderr
    assert not (remote_repo / "scripts/data/stale.py").exists()
    assert (remote_repo / "scripts/data/current.py").exists()
    assert (remote_repo / "coint4/outputs/runtime.log").exists()
    assert (remote_repo / "SYNCED_FROM_COMMIT.txt").read_text(encoding="utf-8").strip() == head_sha
    assert "sync_up cleanup removed 1 stale scope files (mode=tracked)" in proc.stdout


def test_sync_up_code_cleans_only_in_scope_tracked_paths(tmp_path: Path) -> None:
    local_repo, remote_repo, _ = _prepare_repos(
        tmp_path,
        initial_files={
            "README.md": "demo\n",
            "scripts/data/current.py": "print('current')\n",
            "scripts/data/stale.py": "print('stale')\n",
            "coint4/artifacts/keep/tracked.txt": "keep me on code sync\n",
        },
        removed_after_clone=[
            "scripts/data/stale.py",
            "coint4/artifacts/keep/tracked.txt",
        ],
    )
    _write(remote_repo / "scripts/data/untracked_orphan.py", "print('orphan')\n")
    _write(remote_repo / "coint4/outputs/runtime.log", "runtime\n")

    proc = _run_sync_up(tmp_path, local_repo, remote_repo, "code")

    assert proc.returncode == 0, proc.stderr
    assert not (remote_repo / "scripts/data/stale.py").exists()
    assert not (remote_repo / "scripts/data/untracked_orphan.py").exists()
    assert (remote_repo / "scripts/data/current.py").exists()
    assert (remote_repo / "coint4/artifacts/keep/tracked.txt").exists()
    assert (remote_repo / "coint4/outputs/runtime.log").exists()
    assert "sync_up cleanup removed 2 stale scope files (mode=code)" in proc.stdout
