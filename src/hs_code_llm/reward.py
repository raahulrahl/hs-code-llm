"""Hierarchical reward function for HTSUS classification RL.

Implements the reward table from
[project_details.md §6](../../project_details.md):

  Full 10-digit match              +1.0
  8-digit match                    +0.7
  6-digit match (WCO subheading)   +0.5
  4-digit match (WCO heading)      +0.3
  2-digit match (chapter only)     +0.1
  Wrong format / garbage           −0.1

  Bonus signals (additive, capped):
    +0.2  cites a CROSS ruling number (Nxxxxx / Hxxxxxx / 8-digit legacy)
    +0.1  references HTS chapter notes or GRI text
    −0.3  "I'm not sure" with no attempt
    −0.5  hallucinated chapter (00, 77, or 100+)

The reward is computed by :func:`compute_reward` which takes the model's
text rollout plus the gold HTSUS code, returns a (score, breakdown)
tuple where ``breakdown`` is a small dict you can log for debugging.

Designed to be:

* **Verifiable** — every signal is a programmatic check, no LLM-judge.
* **Idempotent** — same input ⇒ same output ⇒ replayable for tests.
* **Surfaceable** — caller gets the breakdown to debug mode collapse,
  reward hacking, etc. (the failure modes called out in §9).

Why this shape: the doc emphasises that a naive 0/1 reward gives no
gradient. Hierarchical partial credit lets the model climb wrong→close→
right; the bonus signals stop it from gaming by emitting garbage with
the right chapter. See ``tests/test_reward.py`` for the 50+ cases that
exercise every tier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field

# ---------------------------------------------------------------------------
# Constants (tuneable; tests pin the table to these defaults)
# ---------------------------------------------------------------------------

R_FULL_10 = 1.0
R_8DIGIT  = 0.7
R_6DIGIT  = 0.5
R_4DIGIT  = 0.3
R_2DIGIT  = 0.1
R_WRONG   = -0.1

B_CROSS_CITED  = 0.2
B_CHAPTER_NOTE = 0.1
P_REFUSAL      = -0.3
P_HALLUCINATED = -0.5

# A chapter must be 01..97 inclusive (chapter 77 is WCO-reserved/unused;
# 98 and 99 are valid US special provisions). Anything else is hallucinated.
_VALID_CHAPTER_RE = re.compile(r"^(0[1-9]|[1-6][0-9]|7[0-689]|8[0-9]|9[0-9])$")
# CROSS ruling numbers (per CBP): legacy 6-digit, modern "Nxxxxxx" / "HQ Hxxxxxx" /
# "HQ H..." / single-letter prefixes like G/K/L/M/N. Be permissive — any
# letter prefix followed by 5–7 digits, or a bare 6-digit code. Always
# uppercased before matching.
_CROSS_CITATION_RE = re.compile(
    r"(?:(?:HQ\s+)?[A-Z]\d{5,7}|(?<!\d)\d{6}(?!\d))"
)
# Detects references to HTS legal text. Conservative — we want to reward
# real legal-style reasoning, not coincidental word use. Requires a
# chapter / section / note / GRI / "rule" keyword paired with a number.
_LEGAL_REF_RE = re.compile(
    r"\b("
    r"chapter\s+notes?\s*\d+|"
    r"section\s+notes?\s*[IVX]+|"
    r"note\s+\d+|"
    r"additional\s+u\.?s\.?\s+(?:rule|note)|"
    r"general\s+rule\s+of\s+interpretation|"
    r"gri\s*\d+|"
    r"hts(?:us)?\s+(?:heading|subheading)\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)
# A refusal — model says "I don't know" without producing any code attempt.
_REFUSAL_RE = re.compile(
    r"\b(?:i(?:'m| am)\s+not\s+sure|i\s+(?:don'?t|do not)\s+know|cannot\s+determine|unable\s+to\s+classify)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_CODE_PATTERNS = (
    # 10-digit with dots: 6109.10.00.27  or  6109.10.0027
    re.compile(r"\b(\d{4}\.\d{2}\.\d{2}\.\d{2})\b"),
    re.compile(r"\b(\d{4}\.\d{2}\.\d{4})\b"),
    # 8-digit dotted:  6109.10.00
    re.compile(r"\b(\d{4}\.\d{2}\.\d{2})\b"),
    # 6-digit dotted (international WCO): 6109.10
    re.compile(r"\b(\d{4}\.\d{2})\b"),
    # bare digits: 6109100027 / 61091000 / 610910 — longer matches win
    re.compile(r"\b(\d{10})\b"),
    re.compile(r"\b(\d{8})\b"),
    re.compile(r"\b(\d{6})\b"),
    re.compile(r"\b(\d{4})\b"),
)


def normalize_code(code: str | None) -> str:
    """Return digits-only HTSUS code (no dots, no spaces)."""
    if not code:
        return ""
    return re.sub(r"[^0-9]", "", code)


def extract_first_code(text: str) -> str:
    """Pull the FIRST HTSUS-shaped token out of ``text`` and return it
    normalised (digits only).

    Prefers longer codes when they overlap on the same span (e.g.
    "6109.10.00.27" wins over "6109.10").
    """
    if not text:
        return ""
    best: tuple[int, int, str] = (-1, -1, "")  # (start, -length-pref, digits)
    for pat in _CODE_PATTERNS:
        m = pat.search(text)
        if m:
            digits = normalize_code(m.group(1))
            # Earlier-in-text wins; ties broken by longer
            start = m.start()
            if best[0] == -1 or start < best[0] or (start == best[0] and len(digits) > len(best[2])):
                best = (start, -len(digits), digits)
    return best[2]


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


@dataclass
class RewardBreakdown:
    """Per-signal log of how a reward was assembled, for debugging."""

    match_score: float = 0.0
    match_level: str = "none"          # full | 8 | 6 | 4 | 2 | wrong | none
    cross_cited: bool = False
    legal_ref: bool = False
    refusal: bool = False
    hallucinated_chapter: bool = False
    predicted_code: str = ""
    gold_code: str = ""
    total: float = 0.0

    def asdict(self) -> dict:
        return asdict(self)


def _match_level_and_score(predicted_digits: str, gold_digits: str) -> tuple[str, float]:
    """Return ('full'|'8'|'6'|'4'|'2'|'wrong'|'none', score).

    Compares prefixes of the digit-only strings. If predicted is empty,
    returns ('none', 0.0) — that case is handled separately because we
    don't want to layer it on top of the refusal penalty.
    """
    if not predicted_digits:
        return ("none", 0.0)
    g = gold_digits
    p = predicted_digits
    if len(p) >= 10 and len(g) >= 10 and p[:10] == g[:10]:
        return ("full", R_FULL_10)
    if len(p) >= 8  and len(g) >= 8  and p[:8]  == g[:8]:
        return ("8",   R_8DIGIT)
    if len(p) >= 6  and len(g) >= 6  and p[:6]  == g[:6]:
        return ("6",   R_6DIGIT)
    if len(p) >= 4  and len(g) >= 4  and p[:4]  == g[:4]:
        return ("4",   R_4DIGIT)
    if len(p) >= 2  and len(g) >= 2  and p[:2]  == g[:2]:
        return ("2",   R_2DIGIT)
    return ("wrong", R_WRONG)


def _chapter_is_hallucinated(predicted_digits: str) -> bool:
    """A chapter is hallucinated if its first two digits aren't a real
    HTSUS chapter. Real chapters: 01..97 (excluding the reserved 77),
    plus the US special provisions 98 and 99.
    """
    if len(predicted_digits) < 2:
        return False
    ch = predicted_digits[:2]
    return _VALID_CHAPTER_RE.match(ch) is None


def compute_reward(rollout: str, gold_code: str) -> tuple[float, RewardBreakdown]:
    """Score a model rollout against the gold HTSUS code.

    Parameters
    ----------
    rollout:
        Raw text the model emitted — the full assistant turn including
        any reasoning. We extract the first HTSUS-shaped code and the
        textual signals from the same string.
    gold_code:
        The reference HTSUS code in any common form ("6109.10.00.27",
        "6109100027", "6109.10.0027"). Normalised to digits internally.

    Returns
    -------
    (score, breakdown):
        * ``score`` is the combined reward (no min/max clip — caller
          decides whether to clip).
        * ``breakdown`` is a :class:`RewardBreakdown` with each signal
          so you can log it.
    """
    gold_digits = normalize_code(gold_code)
    predicted_digits = extract_first_code(rollout)

    bd = RewardBreakdown(predicted_code=predicted_digits, gold_code=gold_digits)

    # Hallucinated chapter is checked BEFORE match — a model that
    # confidently emits "Chapter 00" or "Chapter 77" gets the penalty
    # even if some other tier might score it elsewhere.
    if predicted_digits and _chapter_is_hallucinated(predicted_digits):
        bd.hallucinated_chapter = True
        bd.match_level = "wrong"
        bd.match_score = P_HALLUCINATED
        bd.total = bd.match_score
        # Bonuses still apply — model gets credit for legal reasoning
        # even when the code is bogus, that's still useful behaviour.
        _apply_text_signals(rollout, bd)
        bd.total = bd.match_score + (B_CROSS_CITED if bd.cross_cited else 0.0) \
                                  + (B_CHAPTER_NOTE if bd.legal_ref else 0.0)
        return bd.total, bd

    # Refusal — text says "I don't know" with no code at all.
    if not predicted_digits and _REFUSAL_RE.search(rollout or ""):
        bd.refusal = True
        bd.match_level = "none"
        bd.match_score = P_REFUSAL
        bd.total = bd.match_score
        return bd.total, bd

    # Standard hierarchical match.
    level, score = _match_level_and_score(predicted_digits, gold_digits)
    bd.match_level = level
    bd.match_score = score

    _apply_text_signals(rollout, bd)

    bd.total = (
        bd.match_score
        + (B_CROSS_CITED  if bd.cross_cited else 0.0)
        + (B_CHAPTER_NOTE if bd.legal_ref   else 0.0)
    )
    return bd.total, bd


def _apply_text_signals(rollout: str, bd: RewardBreakdown) -> None:
    """Mutate ``bd`` with cross_cited / legal_ref flags."""
    if not rollout:
        return
    text = rollout.upper()
    # CROSS citation: requires the keyword "CROSS" OR "RULING" near a
    # ruling-shaped token. Bare 6-digit codes would otherwise fire on
    # any random HTSUS code mention.
    if _CROSS_CITATION_RE.search(text) and re.search(r"\b(CROSS|RULING)\b", text):
        bd.cross_cited = True
    if _LEGAL_REF_RE.search(rollout):
        bd.legal_ref = True


# ---------------------------------------------------------------------------
# Batch helper — what TRL / PRIME-RL will call from a GRPO loop
# ---------------------------------------------------------------------------


def reward_fn(rollouts: list[str], gold_codes: list[str]) -> list[float]:
    """Return the float reward for each (rollout, gold) pair.

    Convenience wrapper so the trainer can pass parallel lists. See
    :func:`compute_reward` for the per-item breakdown.
    """
    assert len(rollouts) == len(gold_codes), "rollouts/gold_codes length mismatch"
    return [compute_reward(r, g)[0] for r, g in zip(rollouts, gold_codes, strict=True)]


def reward_with_breakdowns(
    rollouts: list[str], gold_codes: list[str]
) -> list[tuple[float, RewardBreakdown]]:
    """Same as :func:`reward_fn` but returns the per-rollout breakdown
    too, for logging during eval / debugging."""
    return [compute_reward(r, g) for r, g in zip(rollouts, gold_codes, strict=True)]


__all__ = [
    "RewardBreakdown",
    "compute_reward",
    "reward_fn",
    "reward_with_breakdowns",
    "extract_first_code",
    "normalize_code",
    # Constants exported so tests can pin against them, not against literals.
    "R_FULL_10", "R_8DIGIT", "R_6DIGIT", "R_4DIGIT", "R_2DIGIT", "R_WRONG",
    "B_CROSS_CITED", "B_CHAPTER_NOTE", "P_REFUSAL", "P_HALLUCINATED",
]
