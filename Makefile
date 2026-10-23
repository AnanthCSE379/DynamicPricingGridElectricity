# Run from the repository root. Override paths with make NAME=value.
PYTHON ?= python3
PACKAGE := norway_energy_crisis
RAW_DIR ?= $(PACKAGE)
CLEANED_DIR ?= $(PACKAGE)/cleaned
WEATHER_CACHE ?= $(PACKAGE)/weather_cache
DATA_DIR ?= $(PACKAGE)/prepared

# New fits use separate folders; existing checkpoints remain available for evaluation.
RESPONSE_OUTPUT ?= $(PACKAGE)/runs/comparison_v2
RESPONSE_ROOT ?= $(PACKAGE)/runs/comparison_v1
DQN_OUTPUT ?= $(PACKAGE)/runs/dqn_v2
EVALUATION_RUN ?= $(PACKAGE)/runs/dqn_v1
RESPONSE_CHECKPOINT ?= $(RESPONSE_ROOT)/transformer_24h_seed42_price/best.pt
CHECKPOINT ?= $(EVALUATION_RUN)/cap2_seed2026/best.zip
DEVICE ?= mps
BUDGET_HOURS ?= 10
SPLIT ?= validation
ENGINE ?= ensemble

.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format format-check test check clean-data \
	preprocess prepare check-response check-mps train-response train-response-awake \
	train-dqn train-dqn-awake evaluate-response evaluate-dqn

help: ## List commands and common overrides.
	@awk 'BEGIN {FS = ":.*## "; print "Usage: make <target> [NAME=value]\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-23s %s\n", $$1, $$2}' Makefile
	@printf '%s\n' '' \
		'Examples:' \
		'  make install-dev PYTHON=python3' \
		'  make check' \
		'  make train-dqn-awake BUDGET_HOURS=10 DEVICE=mps' \
		'  make evaluate-dqn SPLIT=test ENGINE=reserved' \
		'Override DATA_DIR, RESPONSE_ROOT, RESPONSE_OUTPUT, DQN_OUTPUT or CHECKPOINT as needed.'

install: ## Install runtime dependencies into the selected Python environment.
	"$(PYTHON)" -m pip install -r requirements.txt

install-dev: ## Install runtime dependencies and pinned formatting/lint tools.
	"$(PYTHON)" -m pip install -r requirements-dev.txt

lint: ## Check Python errors and style without changing files.
	"$(PYTHON)" -m ruff check $(PACKAGE) --select E,F

format: ## Format active Python source and tests.
	"$(PYTHON)" -m black $(PACKAGE)

format-check: ## Verify formatting without changing files.
	"$(PYTHON)" -m black --check $(PACKAGE)

test: ## Run self-contained tests; MPS coverage runs when available.
	"$(PYTHON)" -m unittest $(PACKAGE).test_pipeline $(PACKAGE).dqn.test_dqn -v

check: format-check lint test ## Run the same code checks used by GitHub Actions.

clean-data: ## Validate local raw downloads and write a new cleaned directory.
	"$(PYTHON)" -m $(PACKAGE).clean_data --raw "$(RAW_DIR)" \
		--output "$(CLEANED_DIR)" --weather-cache "$(WEATHER_CACHE)"

preprocess: ## Fit preprocessing on training data and write a new prepared directory.
	"$(PYTHON)" -m $(PACKAGE).preprocess --cleaned "$(CLEANED_DIR)" --output "$(DATA_DIR)"

prepare: ## Run initial cleaning then preprocessing; existing outputs are protected.
	$(MAKE) clean-data
	$(MAKE) preprocess

check-response: ## Check prepared data and all four response architectures locally.
	"$(PYTHON)" -m $(PACKAGE).train_response --data "$(DATA_DIR)" --check-only

check-mps: ## Check full-household MPS parity and DQN updates using local checkpoints.
	"$(PYTHON)" -m $(PACKAGE).dqn.train --data "$(DATA_DIR)" \
		--response-root "$(RESPONSE_ROOT)" --output "$(DQN_OUTPUT)" --check-only --device mps

train-response: ## Train/resume the response comparison using local prepared data.
	"$(PYTHON)" -u -m $(PACKAGE).train_response --data "$(DATA_DIR)" --output "$(RESPONSE_OUTPUT)"

train-response-awake: ## Train response models with macOS display/sleep prevention.
	caffeinate -d -i -s -m $(MAKE) train-response

train-dqn: ## Train/resume all six DQN fits with the configured compute budget.
	"$(PYTHON)" -u -m $(PACKAGE).dqn.train --data "$(DATA_DIR)" \
		--response-root "$(RESPONSE_ROOT)" --output "$(DQN_OUTPUT)" \
		--budget-hours "$(BUDGET_HOURS)" --device "$(DEVICE)"

train-dqn-awake: ## Train DQN with macOS display/sleep prevention.
	caffeinate -d -i -s -m $(MAKE) train-dqn

evaluate-response: ## Score RESPONSE_CHECKPOINT on the chosen chronological split.
	"$(PYTHON)" -m $(PACKAGE).train_response --data "$(DATA_DIR)" \
		--evaluate "$(RESPONSE_CHECKPOINT)" --split "$(SPLIT)"

evaluate-dqn: ## Score CHECKPOINT using EVALUATION_RUN's frozen selection and outputs.
	"$(PYTHON)" -m $(PACKAGE).dqn.train --data "$(DATA_DIR)" \
		--response-root "$(RESPONSE_ROOT)" --output "$(EVALUATION_RUN)" \
		--evaluate "$(CHECKPOINT)" --split "$(SPLIT)" --engine "$(ENGINE)" --device "$(DEVICE)"
