SHELL := /bin/bash
.DEFAULT_GOAL := help

COINT4_DIR := coint4
COINT4_VENV_BIN := $(COINT4_DIR)/.venv/bin

# Queue path is relative to app root (coint4/). Override if needed.
VPS_WFA_QUEUE ?= artifacts/wfa/aggregate/20260215_baseline_queue10/run_queue.csv

define _ensure_venv
	@test -x "$(COINT4_VENV_BIN)/$(1)" || ( \
		echo "Missing $(COINT4_VENV_BIN)/$(1). Run: make setup" >&2; \
		exit 1; \
	)
endef

.PHONY: help
help:
	@echo "Usage:"
	@echo "  make setup       Install Python dependencies via Poetry into coint4/.venv"
	@echo "  make test        Run pytest (default markers from coint4/pytest.ini)"
	@echo "  make test-serial Run pytest -m serial"
	@echo "  make test-slow   Run pytest -m slow"
	@echo "  make lint        Run minimal ruff lint (syntax/undefined names)"
	@echo "  make ci          Run lint + test (local CI parity)"
	@echo "  make preflight-loop Run loop preflight (remote policy + secrets + SSH + hygiene/lint/test)"
	@echo "  make hygiene     Fail if heavy/generated files are accidentally tracked in Git"
	@echo "  make secret-scan Run staged secret scan (forbidden paths + sensitive patterns)"
	@echo "  make install-hooks Configure git pre-commit hook for staged secret scan"
	@echo "  make vps-baseline Run baseline WFA queue on VPS (uses run_server_job.sh + STOP_AFTER=1 + sync_back)"
	@echo ""
	@echo "Notes:"
	@echo "  - Most commands use coint4/.venv/bin/* directly (no need for 'poetry run')."
	@echo "  - 'make setup' requires Poetry to be installed."

.PHONY: setup
setup:
	@command -v poetry >/dev/null 2>&1 || (echo "poetry not found. Install Poetry, then re-run: make setup" && exit 1)
	@cd $(COINT4_DIR) && poetry install --with dev

.PHONY: test
test:
	@$(call _ensure_venv,pytest)
	@cd $(COINT4_DIR) && ./.venv/bin/pytest -q -p no:rerunfailures

.PHONY: test-serial
test-serial:
	@$(call _ensure_venv,pytest)
	@cd $(COINT4_DIR) && ./.venv/bin/pytest -q -p no:rerunfailures -m serial

.PHONY: test-slow
test-slow:
	@$(call _ensure_venv,pytest)
	@cd $(COINT4_DIR) && ./.venv/bin/pytest -q -p no:rerunfailures -m slow

.PHONY: lint
lint:
	@$(call _ensure_venv,ruff)
	@cd $(COINT4_DIR) && ./.venv/bin/ruff check src tests

.PHONY: ci
ci: lint test

.PHONY: preflight-loop
preflight-loop:
	@bash coint4/scripts/optimization/preflight_loop_ops.sh

.PHONY: hygiene
hygiene:
	@python3 coint4/scripts/check_repo_hygiene.py

.PHONY: secret-scan
secret-scan:
	@bash coint4/scripts/dev/secret_scan_staged.sh

.PHONY: install-hooks
install-hooks:
	@git config core.hooksPath .githooks
	@chmod +x .githooks/pre-commit
	@echo "Configured git hooksPath=.githooks"

.PHONY: vps-baseline
vps-baseline:
	@set -euo pipefail; \
	test -f "coint4/$(VPS_WFA_QUEUE)" || { echo "Queue not found: coint4/$(VPS_WFA_QUEUE)" >&2; exit 1; }; \
	if [[ -z "$${SERVSPACE_API_KEY:-}" ]]; then \
		test -f .secrets/serverspace_api_key || { echo "Missing SERVSPACE_API_KEY and .secrets/serverspace_api_key" >&2; exit 1; }; \
		export SERVSPACE_API_KEY="$$(tr -d '\\n' < .secrets/serverspace_api_key)"; \
	fi; \
	SYNC_UP="$${SYNC_UP:-1}" STOP_AFTER="$${STOP_AFTER:-1}" bash coint4/scripts/remote/run_server_job.sh \
		bash scripts/optimization/watch_wfa_queue.sh --queue "$(VPS_WFA_QUEUE)"
