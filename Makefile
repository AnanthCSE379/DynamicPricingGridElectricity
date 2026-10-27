# Run from the repository root. Override paths with make NAME=value.
PYTHON ?= python3
PACKAGE := norway_energy_crisis
RAW_DIR ?= $(PACKAGE)
CLEANED_DIR ?= $(PACKAGE)/cleaned
WEATHER_CACHE ?= $(PACKAGE)/weather_cache
DATA_DIR ?= $(PACKAGE)/prepared
RESPONSE_ROOT ?= $(PACKAGE)/runs/comparison_v1
REFINEMENT_OUTPUT ?= $(PACKAGE)/runs/response_refinement_v2
RESPONSE_SLOT_SECONDS ?= 3000
PRICING_OUTPUT ?= $(PACKAGE)/runs/pricing_checkpoint_v1
BUDGET_HOURS ?= 10
DASHBOARD_PORT ?= 8501

.DEFAULT_GOAL := help
.PHONY: help install install-dev install-dashboard format format-check lint test \
	check clean-data preprocess prepare response-v2-preflight train-response-v2 \
	train-response-v2-awake evaluate-response-v2 pricing-preflight pricing-setup \
	train-pricing train-pricing-awake evaluate-pricing dashboard help-legacy \
	pricing-status

help: ## List the main data, consumer, pricing and dashboard commands.
	@awk 'BEGIN {FS = ":.*## "; print "Usage: make <target> [NAME=value]\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-25s %s\n", $$1, $$2}' Makefile
	@printf '%s\n' '' 'Use PYTHON=python to select your environment.' \
		'PRICING_OUTPUT and REFINEMENT_OUTPUT select experiment folders.' \
		'Use a fresh output after changing fingerprinted training code.' \
		'Run make help-legacy for earlier experiments and diagnostics.'

install: ## Install runtime dependencies into the selected Python environment.
	"$(PYTHON)" -m pip install -r requirements.txt

install-dev: ## Install runtime dependencies and pinned formatting/lint tools.
	"$(PYTHON)" -m pip install -r requirements-dev.txt

install-dashboard: ## Install the optional Streamlit viewer dependencies.
	"$(PYTHON)" -m pip install -r requirements-dashboard.txt

format: ## Format active Python source and tests.
	"$(PYTHON)" -m black $(PACKAGE)

format-check: ## Verify formatting without changing files.
	"$(PYTHON)" -m black --check $(PACKAGE)

lint: ## Check Python errors and style without changing files.
	"$(PYTHON)" -m ruff check $(PACKAGE) --select E,F

test: ## Discover all fixture tests; GPU/UI checks run when available.
	"$(PYTHON)" -m unittest discover -s $(PACKAGE) -t . -p 'test_*.py' -v

check: format-check lint test ## Run the same code checks used by GitHub Actions.

clean-data: ## Validate local raw downloads and write a new cleaned directory.
	"$(PYTHON)" -m $(PACKAGE).clean_data --raw "$(RAW_DIR)" \
		--output "$(CLEANED_DIR)" --weather-cache "$(WEATHER_CACHE)"

preprocess: ## Fit preprocessing on training data and write a new prepared directory.
	"$(PYTHON)" -m $(PACKAGE).preprocess --cleaned "$(CLEANED_DIR)" --output "$(DATA_DIR)"

prepare: ## Run initial cleaning then preprocessing; existing outputs are protected.
	$(MAKE) clean-data
	$(MAKE) preprocess

response-v2-preflight: ## Verify full-cohort MPS forecasts, gradients and saved-model parity.
	PYTORCH_ENABLE_MPS_FALLBACK=0 "$(PYTHON)" -u -m $(PACKAGE).response_v2.train --cleaned "$(CLEANED_DIR)" \
		--data "$(DATA_DIR)" --output "$(REFINEMENT_OUTPUT)" \
		--incumbent "$(RESPONSE_ROOT)" --device mps --stage audit \
		--budget-hours "$(BUDGET_HOURS)" --slot-seconds "$(RESPONSE_SLOT_SECONDS)"

train-response-v2: ## Train/resume the eight-fit consumer comparison and final diagnostics.
	PYTORCH_ENABLE_MPS_FALLBACK=0 "$(PYTHON)" -u -m $(PACKAGE).response_v2.train --cleaned "$(CLEANED_DIR)" \
		--data "$(DATA_DIR)" --output "$(REFINEMENT_OUTPUT)" \
		--incumbent "$(RESPONSE_ROOT)" --device mps --stage all \
		--budget-hours "$(BUDGET_HOURS)" --slot-seconds "$(RESPONSE_SLOT_SECONDS)"

train-response-v2-awake: ## Run consumer refinement with display and sleep prevention.
	caffeinate -d -i -s -m $(MAKE) train-response-v2

evaluate-response-v2: ## Evaluate the frozen consumer selection and write its report.
	PYTORCH_ENABLE_MPS_FALLBACK=0 "$(PYTHON)" -u -m $(PACKAGE).response_v2.train --cleaned "$(CLEANED_DIR)" \
		--data "$(DATA_DIR)" --output "$(REFINEMENT_OUTPUT)" \
		--incumbent "$(RESPONSE_ROOT)" --device mps --stage evaluate \
		--budget-hours "$(BUDGET_HOURS)" --slot-seconds "$(RESPONSE_SLOT_SECONDS)"

pricing-preflight: ## Verify full-cohort MPS inference, updates and recovery.
	PYTORCH_ENABLE_MPS_FALLBACK=0 "$(PYTHON)" -u -m $(PACKAGE).pricing_v2.train \
		--stage preflight --device mps --budget-hours "$(BUDGET_HOURS)" \
		--output "$(PRICING_OUTPUT)"

pricing-setup: ## Audit rewards and prepare references/baselines before timed fits.
	PYTORCH_ENABLE_MPS_FALLBACK=0 "$(PYTHON)" -u -m $(PACKAGE).pricing_v2.train \
		--stage setup --device mps --budget-hours "$(BUDGET_HOURS)" \
		--output "$(PRICING_OUTPUT)"

train-pricing: ## Run/resume the six-fit matched pricing comparison and final report.
	PYTORCH_ENABLE_MPS_FALLBACK=0 "$(PYTHON)" -u -m $(PACKAGE).pricing_v2.train \
		--device mps --budget-hours "$(BUDGET_HOURS)" --output "$(PRICING_OUTPUT)"

train-pricing-awake: ## Keep display and Mac awake throughout setup, fits and evaluation.
	caffeinate -d -i -s -m $(MAKE) train-pricing

evaluate-pricing: ## Evaluate already-frozen pricing policies without training new fits.
	PYTORCH_ENABLE_MPS_FALLBACK=0 "$(PYTHON)" -u -m $(PACKAGE).pricing_v2.train \
		--stage evaluate --device mps --budget-hours "$(BUDGET_HOURS)" \
		--output "$(PRICING_OUTPUT)"

dashboard: ## Open the local training and household episode inspector.
	"$(PYTHON)" -m streamlit run $(PACKAGE)/inspection/app.py \
		--server.address 127.0.0.1 --server.port $(DASHBOARD_PORT) \
		--browser.gatherUsageStats false

pricing-status: ## Read the current pricing stage without starting a process.
	@if test -f "$(PRICING_OUTPUT)/status.json"; then \
		"$(PYTHON)" -m json.tool "$(PRICING_OUTPUT)/status.json"; \
	else \
		printf '%s\n' 'No pricing run at this path.'; \
	fi

help-legacy: ## List earlier experiment and diagnostic commands.
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-25s %s\n", $$1, $$2}' make/legacy.mk

# Retain earlier commands without crowding the main workflow.
include make/legacy.mk
