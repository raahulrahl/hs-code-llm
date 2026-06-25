"""Hand-test a trained adapter on product descriptions.

Per [project_details.md §14 step 5](../../project_details.md): "Hand-test
with 20 product descriptions you write yourself." This CLI loads a
saved LoRA adapter, prompts you for product descriptions (or reads a
JSONL file), generates a code, and scores it against an optional gold
code using the same hierarchical reward we'll train against.

Three modes:

  * **interactive** — type descriptions at a prompt, see predictions
    live. Useful for the doc's 20-product hand-test.
  * **batch** — feed a JSONL file with ``{"description": ..., "gold": ...}``
    rows, get an accuracy summary across tiers (full / 8 / 6 / 4 / 2).
  * **eval** — point at ``data/current/eval.jsonl`` (already gold-
    labelled by ``meta.code``) and get the full per-tier breakdown.

Example:

    # interactive
    uv run python -m hs_code_llm.predict \\
        --adapter checkpoints/phase0

    # eval on 200 held-out examples
    uv run python -m hs_code_llm.predict \\
        --adapter checkpoints/phase0 \\
        --eval data/current/eval.jsonl \\
        --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from hs_code_llm.reward import compute_reward, RewardBreakdown


DEFAULT_BASE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_ADAPTER    = Path(__file__).resolve().parents[2] / "checkpoints" / "phase0"
SYSTEM_PROMPT = (
    "You are an expert US customs classifier specialising in the "
    "Harmonized Tariff Schedule of the United States (HTSUS). For "
    "every product you receive, return the most specific 10-digit "
    "HTSUS code and a short legal-style reasoning grounded in chapter "
    "notes or the General Rules of Interpretation (GRIs). This is "
    "decision support — final classification authority rests with US "
    "Customs and Border Protection."
)


def _import_heavy():
    from mlx_tune import FastLanguageModel
    from mlx_lm import generate
    return FastLanguageModel, generate


def _ensure_adapter_config(adapter_dir: Path) -> None:
    """If a half-trained adapter is missing ``adapter_config.json``
    (Ctrl-C before mlx-tune's final `_save_adapter_config()` runs),
    write a default one that matches the LoRA shape sft_phase0 trains
    with. Lets us load *any* intermediate checkpoint without a stack
    trace.
    """
    cfg = adapter_dir / "adapter_config.json"
    if cfg.exists():
        return
    if not (adapter_dir / "adapters.safetensors").exists():
        return
    import json as _json
    # Layer count varies by model (Qwen3-4B = 36, Qwen2.5-0.5B = 24,
    # Qwen2.5-7B = 28). mlx-lm's load_adapters uses num_layers to
    # decide HOW MANY layers to swap LoRA into; passing a value larger
    # than the model itself is a no-op for the missing layers, so
    # over-specifying (36) is the safe default.
    _json.dump({
        "fine_tune_type": "lora",
        "num_layers": 36,
        "lora_parameters": {
            "rank": 16,
            "scale": 2.0,
            "dropout": 0.05,
            "keys": [
                "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj",
                "self_attn.k_proj", "self_attn.o_proj",
                "self_attn.q_proj", "self_attn.v_proj",
            ],
        },
    }, cfg.open("w"), indent=2)
    print(f"[predict] wrote missing {cfg} (matched sft_phase0 LoRA defaults)")


def _load(adapter_dir: Path, base_model: str, max_seq_length: int):
    FastLanguageModel, generate = _import_heavy()
    print(f"[predict] loading base model {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    adapter_path = adapter_dir / "adapters" / "adapters.safetensors"
    if adapter_path.exists():
        _ensure_adapter_config(adapter_path.parent)
        print(f"[predict] loading adapter {adapter_path}")
        try:
            from mlx_lm.tuner.utils import load_adapters
            model.model = load_adapters(model.model, str(adapter_path.parent))
        except ImportError:
            # Older mlx-lm names — fall back silently. The adapter is
            # already part of the model object on save in some versions.
            pass
    else:
        print(f"[predict] WARNING: no adapter at {adapter_path} — using base model")
    FastLanguageModel.for_inference(model)
    return model, tokenizer, generate


def _predict_one(model, tokenizer, generate, description: str, *,
                 max_tokens: int = 200, temperature: float = 0.0) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Country: USA\nProduct: {description}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(
        model.model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


def interactive(model, tokenizer, generate) -> int:
    print("\n[predict] interactive mode — type a product description, Ctrl-D to quit.\n")
    n = 0
    while True:
        try:
            desc = input(f"product #{n + 1}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not desc:
            continue
        rollout = _predict_one(model, tokenizer, generate, desc)
        print(f"\n--- model output ---\n{rollout}\n")
        gold = input("  (optional) gold HTSUS code, or enter to skip: ").strip()
        if gold:
            score, bd = compute_reward(rollout, gold)
            print(f"  → predicted code: {bd.predicted_code or '(none)'}")
            print(f"  → match level:    {bd.match_level}")
            print(f"  → reward:         {score:+.2f}")
        print()
        n += 1
    return 0


# ---------------------------------------------------------------------------
# Batch / eval mode
# ---------------------------------------------------------------------------


def _iter_eval_rows(path: Path):
    """Yield (description, gold) from data/current/eval.jsonl (chat-format)
    or a custom JSONL of ``{description, gold}`` rows."""
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # eval.jsonl shape (from dataset.build): messages + meta.code
            if "messages" in row and "meta" in row:
                desc = ""
                for m in row["messages"]:
                    if m["role"] == "user":
                        # Strip the "Country: USA\nProduct: " prefix
                        desc = m["content"].split("Product:", 1)[-1].strip()
                        break
                gold = (row.get("meta") or {}).get("code") or ""
            else:
                desc = row.get("description") or row.get("product") or ""
                gold = row.get("gold") or row.get("code") or ""
            if desc and gold:
                yield desc, gold


def run_eval(model, tokenizer, generate, path: Path, limit: int | None) -> int:
    print(f"[predict] evaluating against {path}")
    rows = list(_iter_eval_rows(path))
    if limit:
        rows = rows[:limit]
    print(f"[predict] {len(rows)} rows")

    tier_counts: Counter[str] = Counter()
    score_sum = 0.0
    for i, (desc, gold) in enumerate(rows, 1):
        rollout = _predict_one(model, tokenizer, generate, desc)
        score, bd = compute_reward(rollout, gold)
        tier_counts[bd.match_level] += 1
        score_sum += score
        if i % 10 == 0 or i == len(rows):
            mean = score_sum / i
            print(f"  [{i}/{len(rows)}] running mean reward = {mean:+.3f}; "
                  f"tier counts so far = {dict(tier_counts)}")

    n = len(rows)
    print()
    print(f"=== eval summary ({n} examples) ===")
    print(f"  mean reward:         {score_sum / max(1, n):+.3f}")
    for tier in ("full", "8", "6", "4", "2", "wrong", "none"):
        c = tier_counts.get(tier, 0)
        pct = (c / n * 100) if n else 0
        print(f"  {tier:>5}: {c:>5}  ({pct:5.1f}%)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER,
                   help="Adapter directory (saved by sft_phase0).")
    p.add_argument("--model", default=DEFAULT_BASE_MODEL,
                   help="Base model — must match the one used during SFT.")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--eval", type=Path,
                   help="Eval JSONL — runs batch eval and prints a tier breakdown.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap eval rows (default: all).")
    args = p.parse_args()

    model, tokenizer, generate = _load(args.adapter, args.model, args.max_seq_length)

    if args.eval:
        return run_eval(model, tokenizer, generate, args.eval, args.limit)
    return interactive(model, tokenizer, generate)


if __name__ == "__main__":
    sys.exit(main())
