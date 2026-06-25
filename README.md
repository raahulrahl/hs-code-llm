# hs-code-llm

> RL-fine-tuned LLM that predicts HTSUS HS codes from product descriptions, with
> legal reasoning grounded in CROSS rulings and HTSUS chapter notes.

Plan & background: [project_details.md](project_details.md).

## Where everything lives

This is the **training** project. The data plumbing it consumes lives in
two sibling repos:

| Concern | Repo / path |
|---|---|
| HTSUS schema + migrations | [`../shelley/db/migrations/htsus/`](../shelley/db/migrations/htsus/) |
| CROSS rulings schema | `../shelley/db/migrations/precedents/` |
| Postgres data | `htsus.*` and `precedents.rulings` in `bindu_db` |
| HTSUS + CROSS scrapers / loaders | [`../shelley-data-ingest/us-ingest/`](../shelley-data-ingest/us-ingest/) |
| Generated dataset (JSONL) | `../regulations/dataset/us/<date>/` |

The current dataset snapshot is symlinked into [`data/`](data/) — see [data/README.md](data/README.md).

## What's in this repo

```
hs-code-llm/
├── data/                       symlinks to the latest dataset JSONL
├── notebooks/
│   └── phase0_smoke_test.ipynb Kaggle-ready Phase 0 (Qwen 0.5B, free T4)
├── src/hs_code_llm/
│   ├── reward.py               §6 hierarchical reward function
│   ├── data.py                 JSONL loaders + code extraction
│   └── sft_phase0.py           Phase 0 SFT script (TRL + LoRA)
├── tests/
│   └── test_reward.py          50+ reward unit tests (pre-flight checklist §10)
├── project_details.md
├── pyproject.toml
└── README.md
```

## The phased path

Mirrors [project_details.md §5](project_details.md):

1. **Phase 0** — Qwen 2.5 0.5B SFT on Kaggle, ~1K examples, smoke test.
   * `notebooks/phase0_smoke_test.ipynb`
   * Expected accuracy 20–40% (this is just "does the loop run?").
2. **Phase A** — Qwen 2.5 7B SFT then RL with PRIME-RL on the full 50K+
   dataset for the international 6-digit base.
3. **Phase B** — per-country LoRA adapters, starting with USA HTSUS.

## Quick start

```bash
make install        # uv sync + scaffolds .env from .env.example
# Edit .env — drop in your HF_TOKEN + WANDB_API_KEY (both optional).
# .env is gitignored. WANDB_API_KEY unset = wandb logging is a no-op.
make test           # 52 reward unit tests
make phase0-crash   # ~3 min: 50-step crash test on 100 rows
make phase0         # ~15 min: 1000 rows, 1 epoch (doc §5 scale)
make predict        # interactive — type product descriptions
make eval           # tier breakdown on 200 held-out examples

# For Phase A on a beefier Mac (M4):
make phaseA-sft     # Qwen3-4B-Instruct-2507 SFT, 50K rows, ~6-12 h
make phaseA-rl      # GRPO RL — wire-up pending
```

### Base model choice

Phase A defaults to **`mlx-community/Qwen3-4B-Instruct-2507-4bit`** (the
doc originally specced Qwen 2.5 7B; the upgrade is documented in the
Makefile comment for `PHASEA_MODEL`). distil labs benchmarked 12 small
LMs for fine-tuning quality — Qwen3-4B-Instruct-2507 ranked #1, beating
its 8B sibling. It hit 0.89 on Banking77 (77-class intent
classification, the closest public analog to our 98-chapter HS
classification), is Apache 2.0, and trains ~2× faster than 7B on M4
unified memory.

A/B alternatives, one-line override:

```bash
make phaseA-sft PHASEA_MODEL=mlx-community/Qwen3-4B-Thinking-2507-4bit   # CoT sibling
make phaseA-sft PHASEA_MODEL=mlx-community/Qwen3-8B-4bit                  # doc's size class
make phaseA-sft PHASEA_MODEL=mlx-community/Qwen3.5-9B-4bit                # newest gen
```

All training runs on MLX (Apple Silicon GPU) via mlx-tune. Same script
re-runs unmodified on RunPod CUDA by swapping the import to
`from unsloth import …`.

## Disclaimer

Decision support only. Final HTSUS classification authority rests with US Customs
and Border Protection.
