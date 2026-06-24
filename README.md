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
make install        # uv sync (mlx-tune + datasets)
make test           # 52 reward unit tests
make phase0-crash   # ~3 min: 50-step crash test on 100 rows
make phase0         # ~15 min: 1000 rows, 1 epoch (doc §5 scale)
make predict        # interactive — type product descriptions
make eval           # tier breakdown on 200 held-out examples

# For Phase A on a beefier Mac (M4):
make phaseA-sft     # Qwen 7B SFT, 50K rows, ~1-2 days
make phaseA-rl      # GRPO RL — wire-up pending
```

All training runs on MLX (Apple Silicon GPU) via mlx-tune. Same script
re-runs unmodified on RunPod CUDA by swapping the import to
`from unsloth import …`.

## Disclaimer

Decision support only. Final HTSUS classification authority rests with US Customs
and Border Protection.
