## hs-code-llm — Trade-compliance HS-code classifier
##
## Phased training pipeline per project_details.md §5:
##   Phase 0   Qwen 0.5B smoke test on Mac (~15 min)
##   Phase A   Qwen 7B SFT then GRPO RL on Mac M4 / RunPod (1-2 days)
##   Phase B   Per-country LoRA adapter (USA first)
##
## Mac-local training runs on MLX via mlx-tune. Same script + same args
## carry over to CUDA Unsloth on RunPod when we promote to bigger runs.

# ---------- Paths / config -------------------------------------------------

DATA_DIR     ?= data/current
TRAIN_FILE   ?= $(DATA_DIR)/train.jsonl
EVAL_FILE    ?= $(DATA_DIR)/eval.jsonl

# Phase 0 — small model, 1K examples, free
PHASE0_MODEL ?= mlx-community/Qwen2.5-0.5B-Instruct-4bit
PHASE0_OUT   ?= checkpoints/phase0
PHASE0_ROWS  ?= 1000

# Phase A — 7B base, full dataset. M4 with 36+ GB unified can handle
# Qwen 7B 4-bit; bump to bf16 if you have 64 GB+.
PHASEA_MODEL ?= mlx-community/Qwen2.5-7B-Instruct-4bit
PHASEA_OUT   ?= checkpoints/phaseA-sft
PHASEA_ROWS  ?= 50000
PHASEA_BATCH ?= 1
PHASEA_GA    ?= 8

# RL output (Phase A3)
PHASEA_RL_OUT ?= checkpoints/phaseA-rl

# Where the dataset comes from — refresh target shells out to the
# us-ingest sibling repo and re-points the symlink.
US_INGEST_DIR ?= ../shelley-data-ingest/us-ingest
DATASET_DATE  ?= $(shell date +%F)

UV ?= uv

.PHONY: help install test data data-refresh phase0-crash phase0 \
        phaseA-sft phaseA-rl predict eval clean clean-deep

.DEFAULT_GOAL := help

# ---------- Help -----------------------------------------------------------

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
		/^[a-zA-Z0-9_-]+:.*##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------- Setup / quality -----------------------------------------------

install: ## uv sync — install all deps including mlx-tune
	$(UV) sync

test: ## Run reward unit tests (52 cases)
	$(UV) run pytest

# ---------- Data -----------------------------------------------------------

data: $(TRAIN_FILE) ## Make sure the dataset symlink exists
	@ls -la data/current >/dev/null

data-refresh: ## Re-sweep CROSS + rebuild dataset, repoint data/current → today's snapshot
	@echo "==> refreshing dataset via $(US_INGEST_DIR)"
	cd $(US_INGEST_DIR) && DATABASE_URL=$$DATABASE_URL .venv/bin/python -m src.cross.run sweep
	cd $(US_INGEST_DIR) && DATABASE_URL=$$DATABASE_URL .venv/bin/python -m src.cross.run ingest
	cd $(US_INGEST_DIR) && DATABASE_URL=$$DATABASE_URL .venv/bin/python -m src.dataset.build
	ln -snf ../../regulations/dataset/us/$(DATASET_DATE) data/current
	@echo "==> data/current →" && readlink data/current

# ---------- Phase 0 — Qwen 0.5B, smoke test --------------------------------

phase0-crash: ## 50-step crash test on 100 rows (~3 min, verifies the loop)
	$(UV) run python -m hs_code_llm.sft_phase0 \
		--train $(TRAIN_FILE) --eval $(EVAL_FILE) \
		--model $(PHASE0_MODEL) --out-dir $(PHASE0_OUT) \
		--max-train-rows 100 --max-eval-rows 50 --max-steps 50

phase0: ## Phase 0 SFT (~15 min): 1000 rows, 1 epoch, doc §5 scale
	$(UV) run python -m hs_code_llm.sft_phase0 \
		--train $(TRAIN_FILE) --eval $(EVAL_FILE) \
		--model $(PHASE0_MODEL) --out-dir $(PHASE0_OUT) \
		--max-train-rows $(PHASE0_ROWS) --epochs 1

# ---------- Phase A — Qwen 7B SFT then RL ----------------------------------

phaseA-sft: ## Phase A SFT (~1-2 days on M4 36 GB): Qwen 7B 4-bit, 50K rows
	$(UV) run python -m hs_code_llm.sft_phase0 \
		--train $(TRAIN_FILE) --eval $(EVAL_FILE) \
		--model $(PHASEA_MODEL) --out-dir $(PHASEA_OUT) \
		--max-train-rows $(PHASEA_ROWS) --epochs 3 \
		--batch-size $(PHASEA_BATCH) --grad-accum $(PHASEA_GA) \
		--max-seq-length 2048

phaseA-rl: ## Phase A RL with GRPO (~7-10 days on Mac; consider RunPod)
	@echo "Phase A RL not yet implemented — uses mlx_tune.GRPOTrainer + hs_code_llm.reward"
	@echo "See src/hs_code_llm/rl_phaseA.py (next step). On CUDA the same script runs"
	@echo "under trl.GRPOTrainer with no other code changes."
	@exit 1

# ---------- Hand-test / eval -----------------------------------------------

predict: ## Interactive prediction — type product descriptions
	$(UV) run python -m hs_code_llm.predict --adapter $(PHASE0_OUT)

eval: ## Score 200 held-out eval examples per tier (full / 8 / 6 / 4 / 2)
	$(UV) run python -m hs_code_llm.predict \
		--adapter $(PHASE0_OUT) --eval $(EVAL_FILE) --limit 200

# ---------- Cleanup --------------------------------------------------------

clean: ## Remove checkpoints + outputs (keeps .venv)
	rm -rf checkpoints/ outputs/

clean-deep: clean ## Also drop the venv — forces a full uv sync next install
	rm -rf .venv/
