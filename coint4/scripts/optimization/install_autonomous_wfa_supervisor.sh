#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SYSTEMD_SRC_DIR="$ROOT_DIR/scripts/optimization/systemd"
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

mkdir -p "$SYSTEMD_USER_DIR"

for name in autonomous-wfa-driver.service autonomous-wfa-watchdog.service autonomous-wfa-watchdog.timer; do
  src="$SYSTEMD_SRC_DIR/$name"
  dst="$SYSTEMD_USER_DIR/$name"
  sed "s|__ROOT_DIR__|$ROOT_DIR|g" "$src" >"$dst"
  echo "installed $dst"
done

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl not found; units installed but not started."
  exit 0
fi

set +e
systemctl --user daemon-reload
rc_reload=$?
systemctl --user enable --now autonomous-wfa-driver.service
rc_driver=$?
systemctl --user enable --now autonomous-wfa-watchdog.timer
rc_timer=$?
set -e

if [[ "$rc_reload" -ne 0 || "$rc_driver" -ne 0 || "$rc_timer" -ne 0 ]]; then
  echo "warning: failed to fully activate user systemd units (reload=$rc_reload driver=$rc_driver timer=$rc_timer)."
  echo "If running in non-interactive/no-user-systemd session, run these manually in a login shell:"
  echo "  systemctl --user daemon-reload"
  echo "  systemctl --user enable --now autonomous-wfa-driver.service"
  echo "  systemctl --user enable --now autonomous-wfa-watchdog.timer"
  exit 1
fi

echo "autonomous WFA supervisor enabled (driver + watchdog timer)."
