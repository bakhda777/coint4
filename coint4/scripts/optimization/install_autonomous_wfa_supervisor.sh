#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SYSTEMD_SRC_DIR="$ROOT_DIR/scripts/optimization/systemd"
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

mkdir -p "$SYSTEMD_USER_DIR"

for name in \
  autonomous-wfa-driver.service \
  autonomous-wfa-watchdog.service \
  autonomous-wfa-watchdog.timer \
  autonomous-queue-seeder.service \
  autonomous-queue-seeder.timer \
  autonomous-confirm-dispatch-agent.service \
  autonomous-confirm-dispatch-agent.timer \
  autonomous-search-director-agent.service \
  autonomous-search-director-agent.timer \
  autonomous-gate-surrogate-agent.service \
  autonomous-gate-surrogate-agent.timer \
  autonomous-surrogate-calibrator-agent.service \
  autonomous-surrogate-calibrator-agent.timer \
  autonomous-promotion-gatekeeper-agent.service \
  autonomous-promotion-gatekeeper-agent.timer \
  autonomous-contract-auditor-agent.service \
  autonomous-contract-auditor-agent.timer \
  autonomous-vps-capacity-controller-agent.service \
  autonomous-vps-capacity-controller-agent.timer \
  autonomous-confirm-diversity-guard-agent.service \
  autonomous-confirm-diversity-guard-agent.timer \
  autonomous-deterministic-error-blacklist-agent.service \
  autonomous-deterministic-error-blacklist-agent.timer \
  autonomous-confirm-sla-escalator-agent.service \
  autonomous-confirm-sla-escalator-agent.timer \
  autonomous-single-writer-guard-agent.service \
  autonomous-single-writer-guard-agent.timer \
  autonomous-promotion-ledger-compactor.service \
  autonomous-promotion-ledger-compactor.timer \
  autonomous-progress-guard-agent.service \
  autonomous-progress-guard-agent.timer \
  autonomous-backlog-janitor-agent.service \
  autonomous-backlog-janitor-agent.timer \
  autonomous-process-slo-guard-agent.service \
  autonomous-process-slo-guard-agent.timer \
  ops-sentinel-agent.service \
  ops-sentinel-agent.timer; do
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
systemctl --user enable --now autonomous-queue-seeder.timer
rc_queue=$?
systemctl --user enable --now autonomous-confirm-dispatch-agent.timer
rc_confirm=$?
systemctl --user enable --now autonomous-search-director-agent.timer
rc_search_director=$?
systemctl --user enable --now autonomous-gate-surrogate-agent.timer
rc_gate_surrogate=$?
systemctl --user enable --now autonomous-surrogate-calibrator-agent.timer
rc_surrogate_calibrator=$?
systemctl --user enable --now autonomous-promotion-gatekeeper-agent.timer
rc_gatekeeper=$?
systemctl --user enable --now autonomous-contract-auditor-agent.timer
rc_contract_auditor=$?
systemctl --user enable --now autonomous-vps-capacity-controller-agent.timer
rc_capacity=$?
systemctl --user enable --now autonomous-confirm-diversity-guard-agent.timer
rc_confirm_diversity=$?
systemctl --user enable --now autonomous-deterministic-error-blacklist-agent.timer
rc_error_blacklist=$?
systemctl --user enable --now autonomous-confirm-sla-escalator-agent.timer
rc_confirm_sla=$?
systemctl --user enable --now autonomous-single-writer-guard-agent.timer
rc_single_writer=$?
systemctl --user enable --now autonomous-promotion-ledger-compactor.timer
rc_ledger_compactor=$?
systemctl --user enable --now autonomous-progress-guard-agent.timer
rc_progress_guard=$?
systemctl --user enable --now autonomous-backlog-janitor-agent.timer
rc_backlog_janitor=$?
systemctl --user enable --now autonomous-process-slo-guard-agent.timer
rc_process_slo=$?
systemctl --user enable --now ops-sentinel-agent.timer
rc_sentinel=$?

# Apply updated unit definitions immediately even when unit is already active.
systemctl --user restart autonomous-wfa-driver.service
rc_restart_driver=$?
rc_restart_timers=0
for timer_unit in \
  autonomous-wfa-watchdog.timer \
  autonomous-queue-seeder.timer \
  autonomous-confirm-dispatch-agent.timer \
  autonomous-search-director-agent.timer \
  autonomous-gate-surrogate-agent.timer \
  autonomous-surrogate-calibrator-agent.timer \
  autonomous-promotion-gatekeeper-agent.timer \
  autonomous-contract-auditor-agent.timer \
  autonomous-vps-capacity-controller-agent.timer \
  autonomous-confirm-diversity-guard-agent.timer \
  autonomous-deterministic-error-blacklist-agent.timer \
  autonomous-confirm-sla-escalator-agent.timer \
  autonomous-single-writer-guard-agent.timer \
  autonomous-promotion-ledger-compactor.timer \
  autonomous-progress-guard-agent.timer \
  autonomous-backlog-janitor-agent.timer \
  autonomous-process-slo-guard-agent.timer \
  ops-sentinel-agent.timer; do
  systemctl --user restart "$timer_unit" || rc_restart_timers=1
done
set -e

if [[ "$rc_reload" -ne 0 || "$rc_driver" -ne 0 || "$rc_timer" -ne 0 || "$rc_queue" -ne 0 || "$rc_confirm" -ne 0 || "$rc_search_director" -ne 0 || "$rc_gate_surrogate" -ne 0 || "$rc_surrogate_calibrator" -ne 0 || "$rc_gatekeeper" -ne 0 || "$rc_contract_auditor" -ne 0 || "$rc_capacity" -ne 0 || "$rc_confirm_diversity" -ne 0 || "$rc_error_blacklist" -ne 0 || "$rc_confirm_sla" -ne 0 || "$rc_single_writer" -ne 0 || "$rc_ledger_compactor" -ne 0 || "$rc_progress_guard" -ne 0 || "$rc_backlog_janitor" -ne 0 || "$rc_process_slo" -ne 0 || "$rc_sentinel" -ne 0 || "$rc_restart_driver" -ne 0 || "$rc_restart_timers" -ne 0 ]]; then
  echo "warning: failed to fully activate user systemd units (reload=$rc_reload driver=$rc_driver watchdog=$rc_timer queue_seeder=$rc_queue confirm_dispatch=$rc_confirm search_director=$rc_search_director gate_surrogate=$rc_gate_surrogate surrogate_calibrator=$rc_surrogate_calibrator gatekeeper=$rc_gatekeeper contract_auditor=$rc_contract_auditor capacity_controller=$rc_capacity confirm_diversity=$rc_confirm_diversity error_blacklist=$rc_error_blacklist confirm_sla=$rc_confirm_sla single_writer=$rc_single_writer ledger_compactor=$rc_ledger_compactor progress_guard=$rc_progress_guard backlog_janitor=$rc_backlog_janitor process_slo=$rc_process_slo sentinel=$rc_sentinel restart_driver=$rc_restart_driver restart_timers=$rc_restart_timers)."
  echo "If running in non-interactive/no-user-systemd session, run these manually in a login shell:"
  echo "  systemctl --user daemon-reload"
  echo "  systemctl --user enable --now autonomous-wfa-driver.service"
  echo "  systemctl --user enable --now autonomous-wfa-watchdog.timer"
  echo "  systemctl --user enable --now autonomous-queue-seeder.timer"
  echo "  systemctl --user enable --now autonomous-confirm-dispatch-agent.timer"
  echo "  systemctl --user enable --now autonomous-search-director-agent.timer"
  echo "  systemctl --user enable --now autonomous-gate-surrogate-agent.timer"
  echo "  systemctl --user enable --now autonomous-surrogate-calibrator-agent.timer"
  echo "  systemctl --user enable --now autonomous-promotion-gatekeeper-agent.timer"
  echo "  systemctl --user enable --now autonomous-contract-auditor-agent.timer"
  echo "  systemctl --user enable --now autonomous-vps-capacity-controller-agent.timer"
  echo "  systemctl --user enable --now autonomous-confirm-diversity-guard-agent.timer"
  echo "  systemctl --user enable --now autonomous-deterministic-error-blacklist-agent.timer"
  echo "  systemctl --user enable --now autonomous-confirm-sla-escalator-agent.timer"
  echo "  systemctl --user enable --now autonomous-single-writer-guard-agent.timer"
  echo "  systemctl --user enable --now autonomous-promotion-ledger-compactor.timer"
  echo "  systemctl --user enable --now autonomous-progress-guard-agent.timer"
  echo "  systemctl --user enable --now autonomous-backlog-janitor-agent.timer"
  echo "  systemctl --user enable --now autonomous-process-slo-guard-agent.timer"
  echo "  systemctl --user enable --now ops-sentinel-agent.timer"
  exit 1
fi

echo "autonomous WFA supervisor enabled (driver + watchdog + queue seeder + search director + gate surrogate + surrogate calibrator + confirm dispatch + promotion gatekeeper + contract auditor + capacity controller + confirm diversity + error blacklist + confirm SLA + single writer guard + ledger compactor + progress guard + backlog janitor + process SLO guard + sentinel timers)."
