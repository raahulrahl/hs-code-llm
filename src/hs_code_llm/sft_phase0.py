"""Phase 0 — smoke-test SFT of Qwen 2.5 0.5B on the US dataset, MLX-native.

Per [project_details.md §5 — Phase 0](../../project_details.md): "does
the loop run end-to-end?" pass. ~1K examples, 1 epoch, free hardware,
$0 cost. Expected post-SFT accuracy 20–40% on a hand-test — this is
not a production model.

Why MLX (not PyTorch+MPS):

* MLX is purpose-built for Apple Silicon — order-of-magnitude faster
  than ``torch.device('mps')`` for LLM training.
* mlx-tune wraps MLX with an Unsloth-compatible ``FastLanguageModel``
  + ``SFTTrainer`` API. The SAME script runs against ``unsloth`` on
  RunPod in Phase A — just swap one import line.
* Phase A's GRPO algorithm is already supported (``mlx_tune.GRPOTrainer``)
  so we can smoke-test the RL loop locally before paying for A100 time.

Run:

    uv run python -m hs_code_llm.sft_phase0 \\
        --train data/current/train.jsonl \\
        --eval  data/current/eval.jsonl  \\
        --max-train-rows 1000

Output: a LoRA adapter at ``checkpoints/phase0/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_OUT   = Path(__file__).resolve().parents[2] / "checkpoints" / "phase0"


def _load_env() -> None:
    """Load ``.env`` (gitignored) from the repo root, if present.

    Sets HF_TOKEN, WANDB_API_KEY, etc. Idempotent — silent no-op if no
    .env exists, so CI / contributors without secrets keep working.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=False)


def _maybe_init_wandb(args) -> bool:
    """If WANDB_API_KEY is in the env, init a wandb run and return True.

    Otherwise no-op. Returns False so callers know whether to bother
    constructing the callback / monkey-patch.
    """
    if not os.environ.get("WANDB_API_KEY"):
        return False
    import wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "hs-code-llm"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=f"phase0-{args.model.rsplit('/', 1)[-1]}-{args.max_train_rows or 'all'}rows",
        config={
            "phase": "phase0",
            "model": args.model,
            "max_train_rows": args.max_train_rows,
            "max_eval_rows":  args.max_eval_rows,
            "epochs":         args.epochs,
            "max_steps":      args.max_steps,
            "batch_size":     args.batch_size,
            "grad_accum":     args.grad_accum,
            "lr":             args.lr,
            "max_seq_length": args.max_seq_length,
            "lora_r":         16,
            "lora_alpha":     32,
        },
    )
    return True


def _install_wandb_callback() -> None:
    """Monkey-patch ``mlx_lm.tuner.trainer.train`` to inject a wandb
    ``TrainingCallback``.

    mlx-tune's ``SFTTrainer`` calls ``mlx_lm.tuner.trainer.train()``
    directly without exposing the ``training_callback`` parameter, so
    wandb integration has to slip in at the mlx-lm boundary. The
    callback receives ``train_info`` per logging step (iter, loss, LR,
    tokens/s, peak memory) and ``val_info`` per eval (iter, val_loss,
    val_time) — both go straight to ``wandb.log()`` so the wandb UI
    can chart everything live.
    """
    import wandb
    import mlx_lm.tuner.trainer as _tr
    from mlx_lm.tuner.callbacks import TrainingCallback

    class _WandbCallback(TrainingCallback):
        def on_train_loss_report(self, train_info: dict) -> None:
            it = train_info.get("iteration")
            wandb.log({
                "train/loss":         train_info.get("train_loss"),
                "train/learning_rate": train_info.get("learning_rate"),
                "train/it_per_sec":   train_info.get("iterations_per_second"),
                "train/tokens_per_sec": train_info.get("tokens_per_second"),
                "train/trained_tokens": train_info.get("trained_tokens"),
                "train/peak_memory_gb": train_info.get("peak_memory"),
            }, step=it)

        def on_val_loss_report(self, val_info: dict) -> None:
            it = val_info.get("iteration")
            wandb.log({
                "val/loss":     val_info.get("val_loss"),
                "val/time_sec": val_info.get("val_time"),
            }, step=it)

    cb = _WandbCallback()
    _orig_train = _tr.train

    def _train_with_cb(*pargs, training_callback=None, **kwargs):
        return _orig_train(*pargs, training_callback=training_callback or cb, **kwargs)

    _tr.train = _train_with_cb


def _import_heavy():
    """Defer heavy imports so ``--help`` stays snappy and the reward
    tests don't pull MLX onto a non-Apple-Silicon box."""
    from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
    return FastLanguageModel, SFTTrainer, SFTConfig


def _format_chat(row: dict, tokenizer) -> dict:
    """Convert our chat-format JSONL row → mlx-tune's expected ``text`` field.

    The dataset builder emits ``{"messages": [...], "meta": {...}}``. mlx-tune's
    SFTTrainer wants a flat ``text`` column with the chat template already
    applied (matching Unsloth + TRL convention).
    """
    text = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def _load_dataset(path: Path, max_rows: int | None, tokenizer):
    """Load chat-format JSONL, take a chapter-balanced subset, apply the
    chat template."""
    from hs_code_llm.data import load_chat, take_balanced
    ds = load_chat(path=path)
    if max_rows and max_rows < len(ds):
        ds = take_balanced(ds, max_rows)
    return ds.map(lambda r: _format_chat(r, tokenizer))


def run(args: argparse.Namespace) -> int:
    _load_env()
    wandb_on = _maybe_init_wandb(args)
    if wandb_on:
        _install_wandb_callback()
        print("[phase0] wandb run started — see https://wandb.ai/")

    FastLanguageModel, SFTTrainer, SFTConfig = _import_heavy()

    print(f"[phase0] model={args.model} max_seq_length={args.max_seq_length}")

    # ---- Step 1: model + tokenizer ----------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    # ---- Step 2: LoRA adapter ---------------------------------------------
    # Same target modules + rank as the Unsloth tutorial — keeps the
    # script portable when we move to CUDA Unsloth in Phase A.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        random_state=42,
    )

    # ---- Step 3: datasets --------------------------------------------------
    print(f"[phase0] loading train ← {args.train}")
    train_ds = _load_dataset(Path(args.train), args.max_train_rows, tokenizer)
    print(f"[phase0] train rows: {len(train_ds)}")

    eval_ds = None
    if args.eval:
        print(f"[phase0] loading eval ← {args.eval}")
        eval_ds = _load_dataset(Path(args.eval), args.max_eval_rows, tokenizer)
        print(f"[phase0] eval rows:  {len(eval_ds)}")

    # ---- Step 4: training config ------------------------------------------
    cfg = SFTConfig(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=5,
        logging_steps=10,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
    )

    # ---- Step 5: trainer + go ---------------------------------------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        args=cfg,
        max_seq_length=args.max_seq_length,
    )

    trainer.train()

    # ---- Step 6: save LoRA adapter ----------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out_dir))
    print(f"[phase0] adapter saved → {args.out_dir}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default="data/current/train.jsonl")
    p.add_argument("--eval",  default="data/current/eval.jsonl")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-train-rows", type=int, default=1000,
                   help="Cap train rows (Phase 0 default 1000). 0 = use all.")
    p.add_argument("--max-eval-rows", type=int, default=200,
                   help="Cap eval rows. 0 = use all.")
    p.add_argument("--epochs",         type=int,   default=1)
    p.add_argument("--max-steps",      type=int,   default=-1,
                   help="If >0, overrides --epochs. Useful for crash testing.")
    p.add_argument("--batch-size",     type=int,   default=2)
    p.add_argument("--grad-accum",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int,   default=1024)
    args = p.parse_args()
    if args.max_train_rows == 0:
        args.max_train_rows = None
    if args.max_eval_rows == 0:
        args.max_eval_rows = None
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
