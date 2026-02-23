#!/usr/bin/env bash
set -euo pipefail

CANONICAL_COMPUTE_HOST="85.198.90.128"
SSH_TARGET="root@${CANONICAL_COMPUTE_HOST}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

ok() {
  echo "[ok] $*"
}

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${repo_root}" || ! -d "${repo_root}/.git" ]]; then
  fail "must be run inside the git repository"
fi
cd "${repo_root}"

echo "[preflight-loop] repo_root=${repo_root}"

python3 - "${repo_root}" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

checks = [
    (
        "loop compute-host pinned to canonical VPS",
        root / "coint4/scripts/optimization/autonomous_optimize.py",
        re.compile(r'"--compute-host"\s*,\s*"85\.198\.90\.128"'),
    ),
    (
        "loop enables remote poweroff",
        root / "coint4/scripts/optimization/autonomous_optimize.py",
        re.compile(r'"--poweroff"\s*,\s*"true"'),
    ),
    (
        "powered runner default compute-host is canonical VPS",
        root / "coint4/scripts/optimization/run_wfa_queue_powered.py",
        re.compile(r'parser\.add_argument\("--compute-host",\s*default="85\.198\.90\.128"'),
    ),
    (
        "powered runner default poweroff=true",
        root / "coint4/scripts/optimization/run_wfa_queue_powered.py",
        re.compile(r'parser\.add_argument\("--poweroff",\s*type=_parse_bool_flag,\s*default=True\)'),
    ),
    (
        "loop uses powered runner (no local heavy runner wiring)",
        root / "coint4/scripts/optimization/autonomous_optimize.py",
        re.compile(r'self\.powered_runner\s*=\s*self\.app_root\s*/\s*"scripts"\s*/\s*"optimization"\s*/\s*"run_wfa_queue_powered\.py"'),
    ),
    (
        "remote job wrapper defaults to STOP_AFTER=1",
        root / "coint4/scripts/remote/run_server_job.sh",
        re.compile(r'STOP_AFTER=\$\{STOP_AFTER:-"1"\}'),
    ),
    (
        "remote job wrapper defaults to canonical VPS",
        root / "coint4/scripts/remote/run_server_job.sh",
        re.compile(r'SERVER_IP=\$\{SERVER_IP:-"85\.198\.90\.128"\}'),
    ),
]

errors: list[str] = []
for name, path, pattern in checks:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append(f"{name}: missing file {path}")
        continue
    if not pattern.search(text):
        errors.append(f"{name}: pattern not found in {path}")

if errors:
    for err in errors:
        print(err, file=sys.stderr)
    raise SystemExit(1)
PY
ok "remote-heavy policy guardrails are pinned to ${CANONICAL_COMPUTE_HOST}"

key_rel=".secrets/serverspace_api_key"
key_repo_file="${repo_root}/${key_rel}"
key_home_file="${HOME}/.serverspace_api_key"
key_etc_file="/etc/serverspace_api_key"

has_env_key=0
if [[ -n "${SERVSPACE_API_KEY:-}" || -n "${SERVERSPACE_API_KEY:-}" ]]; then
  has_env_key=1
fi

if git ls-files --error-unmatch -- "${key_rel}" >/dev/null 2>&1; then
  fail "${key_rel} must stay untracked (secret file)"
fi

has_file_key=0
key_source=""
for key_file in "${key_home_file}" "${key_etc_file}" "${key_repo_file}"; do
  if [[ ! -f "${key_file}" ]]; then
    continue
  fi
  if [[ ! -r "${key_file}" ]]; then
    continue
  fi
  if [[ ! -s "${key_file}" ]]; then
    fail "Serverspace key file exists but is empty: ${key_file}"
  fi
  mode="$(stat -c '%a' "${key_file}" 2>/dev/null || true)"
  if [[ "${mode}" != "600" ]]; then
    fail "${key_file} must have chmod 600 (current: ${mode:-unknown})"
  fi
  has_file_key=1
  key_source="${key_file}"
  break
done

if [[ "${has_env_key}" == "0" && "${has_file_key}" == "0" ]]; then
  fail "Serverspace API key is missing: set SERVSPACE_API_KEY/SERVERSPACE_API_KEY or create ~/.serverspace_api_key or /etc/serverspace_api_key (or legacy ${key_repo_file})"
fi

if [[ "${has_env_key}" == "1" ]]; then
  ok "SERVSPACE_API_KEY/SERVERSPACE_API_KEY is present in env (value not printed)"
else
  ok "Serverspace key file exists with chmod 600 (value not printed): ${key_source}"
fi

set +e
ssh_output="$(
  ssh \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=8 \
    "${SSH_TARGET}" \
    "echo ok" 2>&1
)"
ssh_rc=$?
set -e
ssh_last_line="$(printf '%s\n' "${ssh_output}" | tail -n 1 | tr -d '\r')"
if [[ "${ssh_rc}" -ne 0 || "${ssh_last_line}" != "ok" ]]; then
  fail "passwordless SSH check failed for ${SSH_TARGET} (rc=${ssh_rc}, last_line='${ssh_last_line}')"
fi
ok "passwordless SSH access to ${SSH_TARGET} works"

echo "[preflight-loop] make hygiene"
make hygiene
echo "[preflight-loop] make lint"
make lint
echo "[preflight-loop] make test"
make test

ok "quality gates passed (make hygiene + make lint + make test)"
echo "[preflight-loop] OK"
