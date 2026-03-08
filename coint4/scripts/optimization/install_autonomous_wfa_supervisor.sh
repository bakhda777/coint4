#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SYSTEMD_SRC_DIR="$ROOT_DIR/scripts/optimization/systemd"
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
LEGACY_SIDECAR_UNITS=(
  autonomous-search-director-agent.service
  autonomous-search-director-agent.timer
  autonomous-gate-surrogate-agent.service
  autonomous-gate-surrogate-agent.timer
  autonomous-surrogate-calibrator-agent.service
  autonomous-surrogate-calibrator-agent.timer
  autonomous-promotion-gatekeeper-agent.service
  autonomous-promotion-gatekeeper-agent.timer
  autonomous-contract-auditor-agent.service
  autonomous-contract-auditor-agent.timer
  autonomous-vps-capacity-controller-agent.service
  autonomous-vps-capacity-controller-agent.timer
  autonomous-queue-seeder.service
  autonomous-queue-seeder.timer
  autonomous-confirm-dispatch-agent.service
  autonomous-confirm-dispatch-agent.timer
  autonomous-confirm-diversity-guard-agent.service
  autonomous-confirm-diversity-guard-agent.timer
  autonomous-confirm-sla-escalator-agent.service
  autonomous-confirm-sla-escalator-agent.timer
  autonomous-deterministic-error-blacklist-agent.service
  autonomous-deterministic-error-blacklist-agent.timer
  autonomous-single-writer-guard-agent.service
  autonomous-single-writer-guard-agent.timer
  autonomous-promotion-ledger-compactor.service
  autonomous-promotion-ledger-compactor.timer
  autonomous-backlog-janitor-agent.service
  autonomous-backlog-janitor-agent.timer
  autonomous-progress-guard-agent.service
  autonomous-progress-guard-agent.timer
  autonomous-process-slo-guard-agent.service
  autonomous-process-slo-guard-agent.timer
  ops-sentinel-agent.service
  ops-sentinel-agent.timer
)

mkdir -p "$SYSTEMD_USER_DIR"

for name in \
  autonomous-wfa-driver.service \
  autonomous-wfa-watchdog.service \
  autonomous-wfa-watchdog.timer; do
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
rc_disable_legacy=0
for legacy_unit in "${LEGACY_SIDECAR_UNITS[@]}"; do
  systemctl --user disable --now "$legacy_unit" >/dev/null 2>&1 || rc_disable_legacy=1
done
systemctl --user enable --now autonomous-wfa-driver.service
rc_driver=$?
systemctl --user enable --now autonomous-wfa-watchdog.timer
rc_timer=$?

# Apply updated unit definitions immediately even when unit is already active.
systemctl --user restart autonomous-wfa-driver.service
rc_restart_driver=$?
rc_restart_timers=0
for timer_unit in \
  autonomous-wfa-watchdog.timer; do
  systemctl --user restart "$timer_unit" || rc_restart_timers=1
done
set -e

if [[ "$rc_reload" -ne 0 || "$rc_driver" -ne 0 || "$rc_timer" -ne 0 || "$rc_restart_driver" -ne 0 || "$rc_restart_timers" -ne 0 ]]; then
  echo "warning: failed to fully activate user systemd units (reload=$rc_reload disable_legacy=$rc_disable_legacy driver=$rc_driver watchdog=$rc_timer restart_driver=$rc_restart_driver restart_timers=$rc_restart_timers)."
  echo "If running in non-interactive/no-user-systemd session, run these manually in a login shell:"
  echo "  systemctl --user daemon-reload"
  echo "  systemctl --user disable --now ${LEGACY_SIDECAR_UNITS[*]}"
  echo "  systemctl --user enable --now autonomous-wfa-driver.service"
  echo "  systemctl --user enable --now autonomous-wfa-watchdog.timer"
  exit 1
fi

echo "autonomous WFA supervisor enabled (driver + watchdog timer; legacy sidecars disabled)."
