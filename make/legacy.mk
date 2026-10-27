# Earlier studies and standalone investigations. Listed by make help-legacy.
RESPONSE_OUTPUT ?= $(PACKAGE)/runs/comparison_v2
DQN_OUTPUT ?= $(PACKAGE)/runs/dqn_v2
CAPACITY_OUTPUT ?= $(PACKAGE)/runs/capacity_v1
EVALUATION_RUN ?= $(PACKAGE)/runs/dqn_v1
RESPONSE_CHECKPOINT ?= $(RESPONSE_ROOT)/transformer_24h_seed42_price/best.pt
CHECKPOINT ?= $(EVALUATION_RUN)/cap2_seed2026/best.zip
DEVICE ?= mps
SPLIT ?= validation
ENGINE ?= ensemble
MINIMUM_SUPPLY_OUTPUT ?= $(PACKAGE)/runs/minimum_supply_v1
RECOVERY_OUTPUT ?= $(PACKAGE)/runs/consumer_recovery_v1
ROLLOUT_OUTPUT ?= $(PACKAGE)/runs/sliding_72h_investigation_v1
REPLAY_RUN ?= $(PACKAGE)/runs/dqn_v1/cap2_seed42
REPLAY_DAYS ?= 3

.PHONY: dashboard-replay check-response check-mps train-response train-response-awake \
	train-dqn train-dqn-awake evaluate-response evaluate-dqn capacity-preflight \
	capacity-audit capacity-baselines train-capacity train-capacity-awake \
	evaluate-capacity minimum-supply-stress consumer-recovery response-rollout-audit

dashboard-replay: ## Export labelled validation checkpoint replays; no retraining.
	"$(PYTHON)" -m $(PACKAGE).inspection.replay --run "$(REPLAY_RUN)" \
		--data "$(DATA_DIR)" --response-root "$(RESPONSE_ROOT)" \
		--days $(REPLAY_DAYS) --device "$(DEVICE)"

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

capacity-preflight: ## Check full-cohort MPS inference, gradients, parity and recovery.
	"$(PYTHON)" -u -m $(PACKAGE).capacity.train --data "$(DATA_DIR)" \
		--response-root "$(RESPONSE_ROOT)" --output "$(CAPACITY_OUTPUT)" --device mps --check-only

capacity-audit: ## Audit provenance, MPS correctness and safe reference schedules.
	"$(PYTHON)" -u -m $(PACKAGE).capacity.train --data "$(DATA_DIR)" \
		--response-root "$(RESPONSE_ROOT)" --output "$(CAPACITY_OUTPUT)" --stage audit --device mps

capacity-baselines: ## Enumerate validation prices and compare allocation methods.
	"$(PYTHON)" -u -m $(PACKAGE).capacity.train --data "$(DATA_DIR)" \
		--response-root "$(RESPONSE_ROOT)" --output "$(CAPACITY_OUTPUT)" --stage baselines --device mps

train-capacity: ## Run/resume setup, nine bounded fits, evaluation and final reports.
	"$(PYTHON)" -u -m $(PACKAGE).capacity.train --data "$(DATA_DIR)" \
		--response-root "$(RESPONSE_ROOT)" --output "$(CAPACITY_OUTPUT)" \
		--budget-hours "$(BUDGET_HOURS)" --device mps --stage all

train-capacity-awake: ## Run the capacity study with macOS display/sleep prevention.
	caffeinate -d -i -s -m $(MAKE) train-capacity

evaluate-capacity: ## Evaluate all completed capacity fits using frozen selection.
	"$(PYTHON)" -u -m $(PACKAGE).capacity.train --data "$(DATA_DIR)" \
		--response-root "$(RESPONSE_ROOT)" --output "$(CAPACITY_OUTPUT)" --stage evaluate --device mps

minimum-supply-stress: ## Fit supply reservations and stress low loads without retraining.
	"$(PYTHON)" -u -m $(PACKAGE).analysis.stress_minimum_supply --data "$(DATA_DIR)" \
		--responses "$(RESPONSE_ROOT)" --output "$(MINIMUM_SUPPLY_OUTPUT)" --device "$(DEVICE)"

consumer-recovery: ## Simulate sudden cuts, verify repeated forecasts and report recovery.
	"$(PYTHON)" -u -m $(PACKAGE).analysis.consumer_recovery --data "$(DATA_DIR)" \
		--responses "$(RESPONSE_ROOT)" --output "$(RECOVERY_OUTPUT)" --device "$(DEVICE)"
	"$(PYTHON)" -m $(PACKAGE).analysis.report_consumer_recovery --data "$(DATA_DIR)" \
		--output "$(RECOVERY_OUTPUT)"

response-rollout-audit: ## Compare 72h predicted-load and measured-load sliding histories.
	PYTORCH_ENABLE_MPS_FALLBACK=0 caffeinate -d -i -s -m \
		"$(PYTHON)" -u -m $(PACKAGE).analysis.investigate_rollout --device "$(DEVICE)" \
		--cleaned "$(CLEANED_DIR)" --data "$(DATA_DIR)" \
		--responses "$(REFINEMENT_OUTPUT)" --output "$(ROLLOUT_OUTPUT)"
	"$(PYTHON)" -u -m $(PACKAGE).analysis.diagnose_price_channels \
		--responses "$(REFINEMENT_OUTPUT)" --output "$(ROLLOUT_OUTPUT)/price_channels"
	"$(PYTHON)" -m $(PACKAGE).analysis.report_rollout_households --output "$(ROLLOUT_OUTPUT)"
