SHELL := /bin/bash
.DEFAULT_GOAL := help

COINT4_DIR := coint4
COINT4_VENV_BIN := $(COINT4_DIR)/.venv/bin
LOOP_HOME ?= $(HOME)

# Queue path is relative to app root (coint4/). Override if needed.
VPS_WFA_QUEUE ?= artifacts/wfa/aggregate/20260215_baseline_queue10/run_queue.csv
LOOP_USE_CODEX_EXEC ?= 1
LOOP_PLANNER_MODE ?= evolution
LOOP_RESUME ?= 1
LOOP_PLAN_REPEAT ?= 0
LOOP_EVOLUTION_NUM_VARIANTS ?= 12
CODEX_ENV_FILE ?= /etc/coint4/codex.env
# codex auth mode for loop targets:
#   subscription (default): use existing `codex login` session (recommended; no API key usage).
#   api-key: require OPENAI_API_KEY and bootstrap via `codex login --with-api-key`.
#   auto: try session first, then OPENAI_API_KEY bootstrap if key is present.
LOOP_CODEX_AUTH_MODE ?= subscription

ifeq ($(LOOP_USE_CODEX_EXEC),1)
LOOP_CODEX_FLAG := --use-codex-exec
else
LOOP_CODEX_FLAG :=
endif

ifeq ($(LOOP_RESUME),1)
LOOP_RESUME_FLAG := --resume
else
LOOP_RESUME_FLAG :=
endif

ifeq ($(LOOP_PLAN_REPEAT),1)
LOOP_PLAN_REPEAT_FLAG := --until-done
else
LOOP_PLAN_REPEAT_FLAG :=
endif

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
	@echo "  make loop        Online WFA driver (canonical autonomous runtime via autonomous_wfa_driver.sh)"
	@echo "  make loop-plan   Offline planner loop (single-pass by default; autonomous_optimize.py)"
	@echo "  make loop-once   Deprecated alias for the single-pass planner run"
	@echo "  make loop-api-power Legacy one-shot baseline via API power-cycle wrapper"
	@echo "  make preflight-loop Run loop preflight (remote policy + secrets + SSH + hygiene/lint/test)"
	@echo "  make hygiene     Fail if heavy/generated files are accidentally tracked in Git"
	@echo "  make secret-scan Run staged secret scan (forbidden paths + sensitive patterns)"
	@echo "  make install-hooks Configure git pre-commit hook for staged secret scan"
	@echo "  make vps-baseline Run baseline WFA queue on VPS (uses run_server_job.sh + STOP_AFTER=1 + sync_back)"
	@echo ""
	@echo "Notes:"
	@echo "  - Most commands use coint4/.venv/bin/* directly (no need for 'poetry run')."
	@echo "  - 'make setup' requires Poetry to be installed."
	@echo "  - `make loop` is the canonical online entrypoint; `make loop-plan` is planner-only."
	@echo "  - loop-plan auth mode: LOOP_CODEX_AUTH_MODE=subscription|api-key|auto (default: subscription)."
	@echo "  - planner mode: LOOP_PLANNER_MODE=evolution|legacy (default: evolution)."
	@echo "  - planner resume: LOOP_RESUME=1|0 (default: 1; resume if state=done)."
	@echo "  - planner repeat: LOOP_PLAN_REPEAT=1 for repeated in-process planner passes."
	@echo "  - evolution planner: LOOP_EVOLUTION_NUM_VARIANTS=<n> (default: 12; variants per batch)."

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

.PHONY: loop
loop: loop-driver

.PHONY: loop-driver
loop-driver:
	@$(call _ensure_venv,python)
	@cd $(COINT4_DIR) && set -euo pipefail; \
		if [[ -r "$(CODEX_ENV_FILE)" ]]; then set -a; . "$(CODEX_ENV_FILE)"; set +a; fi; \
		HOME="$(LOOP_HOME)" PYTHONPATH=src bash scripts/optimization/autonomous_wfa_driver.sh

.PHONY: loop-plan
loop-plan: loop-closed

.PHONY: loop-closed
loop-closed:
	@$(call _ensure_venv,python)
	@cd $(COINT4_DIR) && set -euo pipefail; \
		if [[ -r "$(CODEX_ENV_FILE)" ]]; then set -a; . "$(CODEX_ENV_FILE)"; set +a; fi; \
		if [[ "$(LOOP_CODEX_AUTH_MODE)" == "subscription" ]]; then unset OPENAI_API_KEY; fi; \
		HOME="$(LOOP_HOME)" \
		COINT4_CODEX_AUTH_MODE="$(LOOP_CODEX_AUTH_MODE)" \
		PYTHONPATH=src ./.venv/bin/python scripts/optimization/autonomous_optimize.py $(LOOP_PLAN_REPEAT_FLAG) $(LOOP_RESUME_FLAG) --planner-mode "$(LOOP_PLANNER_MODE)" --evolution-num-variants "$(LOOP_EVOLUTION_NUM_VARIANTS)" $(LOOP_CODEX_FLAG)

.PHONY: loop-once
loop-once:
	@echo "loop-once is deprecated; use 'make loop-plan' for the default single-pass planner run or 'make loop-plan LOOP_PLAN_REPEAT=1' for repeated planner passes." >&2
	@$(MAKE) --no-print-directory loop-plan

.PHONY: loop-api-power
loop-api-power:
	@python3 coint4/scripts/optimization/run_loop_with_api_power.py

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
