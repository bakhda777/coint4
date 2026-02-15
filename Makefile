SHELL := /bin/bash
.DEFAULT_GOAL := help

COINT4_DIR := coint4
COINT4_VENV_BIN := $(COINT4_DIR)/.venv/bin

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
