"""End-to-end key + wiring scan.

Six checks, designed to be the first thing you run on a fresh checkout
(local Mac or M4) after pasting your tokens into ``.env``:

  1. ``.env`` actually loaded → HF_TOKEN + WANDB_API_KEY present.
  2. HF token authenticates (``whoami``).
  3. HF token can pull a model file (rate-limit applies the *auth'd*
     bucket, not the anonymous one).
  4. Wandb key authenticates (``wandb.login`` + ``api.viewer``).
  5. Wandb can actually init/log/finish a real online run.
  6. The ``sft_phase0`` monkey-patch successfully wraps
     ``mlx_lm.tuner.trainer.train``.

Exit code 0 means: next ``make phase0`` will stream live to wandb and
download from HF authenticated. Anything ≥1 means a setup gap; the
output points at which step.

Run via:  ``make scan``  or  ``uv run python -m hs_code_llm.scan``
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def _err(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")


def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    fails: list[str] = []

    # 1) .env
    _section("1) .env loading")
    _load_env()
    hf = os.environ.get("HF_TOKEN", "")
    wb = os.environ.get("WANDB_API_KEY", "")
    print(f"  HF_TOKEN       : {'set' if hf else 'MISSING'} ({len(hf)} chars)")
    print(f"  WANDB_API_KEY  : {'set' if wb else 'MISSING'} ({len(wb)} chars)")
    print(f"  WANDB_PROJECT  : {os.environ.get('WANDB_PROJECT','<none>')}")
    print(f"  WANDB_ENTITY   : {os.environ.get('WANDB_ENTITY') or '<default>'}")
    if not hf:
        fails.append(".env missing HF_TOKEN")
    if not wb:
        fails.append(".env missing WANDB_API_KEY")
    if not (hf and wb):
        # Don't bail — the user might want to see what works without HF
        # before pasting the rest. Just record the gap.
        pass

    # 2) HF whoami
    _section("2) HF_TOKEN — read auth")
    if hf:
        try:
            from huggingface_hub import whoami
            me = whoami(token=hf)
            _ok(f"whoami: {me['name']} ({me['type']}, email={me.get('email','-')})")
        except Exception as e:
            _err(f"whoami FAILED: {e}")
            fails.append("HF whoami")
    else:
        print("  (skipped — no HF_TOKEN)")

    # 3) HF download
    _section("3) HF_TOKEN — actual download")
    if hf:
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(
                repo_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                filename="config.json",
                token=hf,
            )
            sz = os.path.getsize(p)
            _ok(f"downloaded config.json ({sz} bytes) — authenticated rate-limit OK")
        except Exception as e:
            _err(f"download FAILED: {e}")
            fails.append("HF download")
    else:
        print("  (skipped — no HF_TOKEN)")

    # 4) Wandb login
    _section("4) WANDB_API_KEY — auth")
    wandb = None
    if wb:
        try:
            import wandb as _wb
            wandb = _wb
            ok = wandb.login(key=wb, relogin=True, verify=True, anonymous="never")
            _ok(f"login: {ok}")
            api = wandb.Api()
            v = api.viewer
            _ok(f"viewer: {v.username} ({v.email})")
        except Exception as e:
            _err(f"wandb auth FAILED: {e}")
            fails.append("wandb auth")
    else:
        print("  (skipped — no WANDB_API_KEY)")

    # 5) Real wandb run
    _section("5) wandb — init/log/finish a real online run")
    if wandb is not None:
        try:
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "hs-code-llm"),
                name="env-key-scan",
                job_type="key-scan",
                config={"scan": "end-to-end", "n": 3},
                mode="online",
            )
            for i, loss in enumerate([0.9, 0.7, 0.55]):
                wandb.log({"scan/loss": loss}, step=i)
            _ok(f"run url: {run.url}")
            wandb.finish()
        except Exception as e:
            _err(f"wandb run FAILED: {e}")
            traceback.print_exc()
            fails.append("wandb run")
    else:
        print("  (skipped — no wandb)")

    # 6) monkey-patch
    _section("6) wandb callback monkey-patch reachable")
    try:
        from hs_code_llm.sft_phase0 import _install_wandb_callback
        import mlx_lm.tuner.trainer as t
        orig = t.train
        _install_wandb_callback()
        if t.train is orig:
            raise RuntimeError("monkey-patch did NOT replace mlx_lm.tuner.trainer.train")
        _ok("mlx_lm.tuner.trainer.train was wrapped")
    except Exception as e:
        _err(f"monkey-patch test FAILED: {e}")
        traceback.print_exc()
        fails.append("monkey-patch")

    # Summary
    _section("SUMMARY")
    if not fails:
        print("  \033[32mAll checks passed.\033[0m Next `make phase0` will:")
        print("    - download Qwen with HF rate-limit headroom")
        print("    - stream loss / lr / tok-s / memory live to wandb")
        return 0
    print(f"  \033[31m{len(fails)} check(s) failed:\033[0m")
    for f in fails:
        print(f"    - {f}")
    print("  Fix the rows above and re-run `make scan`.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
