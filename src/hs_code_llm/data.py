"""Dataset loaders for the JSONL files produced by
[``shelley-data-ingest/us-ingest/src/dataset/build.py``](../../../shelley-data-ingest/us-ingest/src/dataset/build.py).

Two formats live in ``data/current/`` (the symlink to the latest
snapshot):

  * ``train.jsonl`` / ``eval.jsonl``  — chat-format, one
    ``{"messages": [...], "meta": {...}}`` per line.
  * ``flat.jsonl``                    — legacy SFT shape, one
    ``{"prompt", "completion", "code", "chapter", "source"}`` per line.

This module gives you back ``datasets.Dataset`` objects ready to feed
into TRL's ``SFTTrainer`` (chat) or GRPO trainer (flat / for
reward-based RL).

The reward function (:mod:`hs_code_llm.reward`) consumes the rollout
text + the gold code; ``meta.code`` carries the gold during RL so
batched rollouts can be scored without a second DB query.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# `datasets` is optional — keep the import lazy so the reward tests
# don't require it.
try:
    from datasets import Dataset  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover — only hit before deps installed
    Dataset = None  # type: ignore[assignment, misc]


DATASET_ROOT = Path(__file__).resolve().parents[2] / "data" / "current"


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Chat format (SFT)
# ---------------------------------------------------------------------------


def load_chat(split: str = "train", path: Path | None = None) -> "Dataset":
    """Load ``train.jsonl`` / ``eval.jsonl`` as a chat-format
    ``datasets.Dataset``.

    Each row has:

      * ``messages`` — list of ``{role, content}`` dicts
      * ``meta``     — {code, chapter, source, ruling_id}
    """
    if Dataset is None:
        raise RuntimeError("`datasets` package not installed — `uv sync` first")
    p = path or DATASET_ROOT / f"{split}.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {p}\n"
            f"Run the dataset builder in shelley-data-ingest/us-ingest then symlink "
            f"data/current → regulations/dataset/us/<date>/"
        )
    return Dataset.from_list(list(_read_jsonl(p)))


# ---------------------------------------------------------------------------
# Flat format (legacy SFT / RL prompts)
# ---------------------------------------------------------------------------


def load_flat(path: Path | None = None) -> "Dataset":
    """Load ``flat.jsonl`` — the legacy ``(prompt, completion, ...)``
    shape. Most useful for:

      * GRPO / PRIME-RL where you want raw prompts to roll out from.
      * Quick baseline accuracy checks before any training.
    """
    if Dataset is None:
        raise RuntimeError("`datasets` package not installed — `uv sync` first")
    p = path or DATASET_ROOT / "flat.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    return Dataset.from_list(list(_read_jsonl(p)))


# ---------------------------------------------------------------------------
# Sampling helpers — useful for Phase 0 smoke tests
# ---------------------------------------------------------------------------


def take_balanced(ds: "Dataset", n: int, *, seed: int = 42) -> "Dataset":
    """Take a chapter-balanced subset of size ``n``.

    Used by Phase 0 to drop from ~67K rows down to ~1K without losing
    chapter coverage. Looks up the chapter from ``meta.chapter`` (chat
    format) or ``chapter`` (flat). Returns a shuffled subset.
    """
    if Dataset is None:
        raise RuntimeError("`datasets` package not installed — `uv sync` first")
    if len(ds) <= n:
        return ds.shuffle(seed=seed)

    def get_chapter(row: dict) -> str:
        if "meta" in row:
            return (row.get("meta") or {}).get("chapter", "00") or "00"
        return row.get("chapter") or "00"

    from collections import defaultdict
    import random
    rng = random.Random(seed)
    by_ch: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(ds):
        by_ch[get_chapter(row)].append(i)
    chapters = list(by_ch.keys())
    per = max(1, n // len(chapters))
    picked: list[int] = []
    for ch in chapters:
        idxs = by_ch[ch][:]
        rng.shuffle(idxs)
        picked.extend(idxs[:per])
    rng.shuffle(picked)
    return ds.select(picked[:n])


__all__ = ["DATASET_ROOT", "load_chat", "load_flat", "take_balanced"]
