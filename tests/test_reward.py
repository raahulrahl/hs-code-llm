"""Pin the reward function to the table in project_details.md §6.

Covers every reward tier and bonus signal, plus the edge cases that
§9 (Common Pitfalls) warned about: hallucinated chapters, refusals,
mode collapse via citing-but-wrong, etc.

Run: ``uv run pytest`` (or ``pytest`` from a venv with project deps).
"""

from __future__ import annotations

import pytest

from hs_code_llm.reward import (
    B_CHAPTER_NOTE,
    B_CROSS_CITED,
    P_HALLUCINATED,
    P_REFUSAL,
    R_2DIGIT,
    R_4DIGIT,
    R_6DIGIT,
    R_8DIGIT,
    R_FULL_10,
    R_WRONG,
    compute_reward,
    extract_first_code,
    normalize_code,
    reward_fn,
)

GOLD = "6109.10.00.27"   # men's cotton T-shirt, knitted (the canonical example from §B1)
GOLD_DIGITS = "6109100027"


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("6109.10.00.27", "6109100027"),
        ("6109.10.0027",  "6109100027"),
        ("6109100027",    "6109100027"),
        ("6109.10.00",    "61091000"),
        ("6109.10",       "610910"),
        ("",              ""),
        (None,            ""),
    ],
)
def test_normalize_code(raw, expected):
    assert normalize_code(raw) == expected


@pytest.mark.parametrize(
    "rollout,expected",
    [
        ("HTSUS Code: 6109.10.00.27\n\nReasoning: ...",          "6109100027"),
        ("The classification is 6109.10.0027.",                  "6109100027"),
        ("Try 6109100027.",                                      "6109100027"),
        ("Falls under 6109.10.00 — Of cotton.",                  "61091000"),
        ("The international subheading is 6109.10.",             "610910"),
        ("Heading 6109 covers t-shirts.",                        "6109"),
        ("No code given, just rambling text here.",              ""),
    ],
)
def test_extract_first_code(rollout, expected):
    assert extract_first_code(rollout) == expected


# ---------------------------------------------------------------------------
# Hierarchical match tiers (§6 table)
# ---------------------------------------------------------------------------


def test_full_10_digit_match():
    score, bd = compute_reward("HTSUS Code: 6109.10.00.27", GOLD)
    assert bd.match_level == "full"
    assert bd.match_score == R_FULL_10
    assert score == R_FULL_10


def test_8_digit_match():
    score, bd = compute_reward("Classification: 6109.10.00.99", GOLD)
    assert bd.match_level == "8"
    assert score == R_8DIGIT


def test_6_digit_match_wrong_8th_digit():
    score, bd = compute_reward("Code 6109.10.40.20 applies.", GOLD)
    assert bd.match_level == "6"
    assert score == R_6DIGIT


def test_4_digit_match_wrong_heading_suffix():
    score, bd = compute_reward("My answer: 6109.20.10.00", GOLD)
    assert bd.match_level == "4"
    assert score == R_4DIGIT


def test_2_digit_chapter_only_match():
    score, bd = compute_reward("The HTSUS code is 6110.20.20.00.", GOLD)
    assert bd.match_level == "2"
    assert score == R_2DIGIT


def test_wrong_chapter_entirely():
    # Chapter 02 (meat) — wrong but valid chapter, not hallucinated.
    score, bd = compute_reward("HTSUS 0207.14.00.20", GOLD)
    assert bd.match_level == "wrong"
    assert score == R_WRONG


# ---------------------------------------------------------------------------
# Hallucinated chapters (§6 bonus penalty)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bogus_chapter", ["00", "77"])
def test_hallucinated_chapter_penalty(bogus_chapter):
    rollout = f"Probably {bogus_chapter}10.20.30.40 applies."
    score, bd = compute_reward(rollout, GOLD)
    assert bd.hallucinated_chapter is True
    assert score == P_HALLUCINATED


def test_chapter_99_is_valid_not_hallucinated():
    """Chapter 99 is a real US special-provision chapter (Section 301
    etc.) per §3 of the doc — must NOT trigger the penalty."""
    score, bd = compute_reward("Code: 9903.88.15.00", GOLD)
    assert bd.hallucinated_chapter is False
    assert bd.match_level == "wrong"


def test_chapter_98_is_valid_not_hallucinated():
    score, bd = compute_reward("Code: 9801.00.10.00", GOLD)
    assert bd.hallucinated_chapter is False
    assert bd.match_level == "wrong"


def test_three_digit_chapter_caught():
    """100+ chapters are impossible — must trip the hallucination check."""
    # The extractor will match the 8-digit "10120304" — chapter "10" (real, cereals)
    # so this test instead picks a pattern that yields chapter "00".
    score, bd = compute_reward("Code: 0017.30.20.10", GOLD)
    assert bd.hallucinated_chapter is True
    assert score == P_HALLUCINATED


# ---------------------------------------------------------------------------
# Refusal (§6 −0.3 signal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rollout",
    [
        "I'm not sure how to classify this product.",
        "I do not know what HTSUS chapter applies.",
        "Unable to classify without more information.",
        "Cannot determine the right tariff line.",
    ],
)
def test_refusal_with_no_code(rollout):
    score, bd = compute_reward(rollout, GOLD)
    assert bd.refusal is True
    assert score == P_REFUSAL


def test_refusal_with_code_does_not_penalize():
    """Saying 'I'm not sure' while still emitting a code = not a refusal,
    it's an honest hedge. The model gets the match score it earned, not
    the −0.3."""
    rollout = "I'm not sure, but my best guess is 6109.10.00.27."
    score, bd = compute_reward(rollout, GOLD)
    assert bd.refusal is False
    assert bd.match_level == "full"


# ---------------------------------------------------------------------------
# Bonus: CROSS citation (+0.2)
# ---------------------------------------------------------------------------


def test_cross_citation_with_correct_code():
    rollout = "Per CROSS ruling N123456, classification is 6109.10.00.27."
    score, bd = compute_reward(rollout, GOLD)
    assert bd.cross_cited is True
    assert score == pytest.approx(R_FULL_10 + B_CROSS_CITED)


def test_cross_citation_word_required():
    """A 6-digit code in text alone shouldn't count as a CROSS citation —
    only when keyed by 'CROSS' or 'RULING'."""
    rollout = "Code 6109.10.00.27 — confidence high."
    score, bd = compute_reward(rollout, GOLD)
    assert bd.cross_cited is False
    assert score == R_FULL_10


def test_cross_citation_with_hq_prefix():
    rollout = "Per HQ H308123 and CROSS ruling, code is 6109.10.00.27."
    score, bd = compute_reward(rollout, GOLD)
    assert bd.cross_cited is True


def test_cross_citation_applies_even_to_wrong_code():
    """§9 'reward hacking' surface: bonuses in the §6 table are ADDITIVE,
    so a wrong code + CROSS citation totals R_WRONG + B_CROSS_CITED =
    -0.1 + 0.2 = +0.1 — i.e. citing a CROSS ruling can net-positive a
    flat-wrong answer. The reward function follows the spec; this test
    documents the behaviour so it's visible and we can decide later
    whether to cap bonuses below zero. See §9 'Reward hacking' for the
    failure mode this enables."""
    rollout = "Per CROSS ruling N123456, code is 0207.14.00.20."
    score, bd = compute_reward(rollout, GOLD)
    assert bd.cross_cited is True
    assert score == pytest.approx(R_WRONG + B_CROSS_CITED)
    assert score == pytest.approx(0.1)   # net positive — a known §9 risk


# ---------------------------------------------------------------------------
# Bonus: legal text reference (+0.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rollout",
    [
        "Per Chapter Notes 1 and Section Notes II, code is 6109.10.00.27.",
        "GRI 3(b) applies; the code is 6109.10.00.27.",
        "Note 1 to chapter 61 says: 6109.10.00.27.",
        "Additional U.S. Rule 1 sends us to 6109.10.00.27.",
        "General Rule of Interpretation 1 — code 6109.10.00.27.",
    ],
)
def test_legal_text_reference_bonus(rollout):
    score, bd = compute_reward(rollout, GOLD)
    assert bd.legal_ref is True
    assert score == pytest.approx(R_FULL_10 + B_CHAPTER_NOTE)


def test_no_legal_ref_no_bonus():
    rollout = "HTSUS Code: 6109.10.00.27. Done."
    score, bd = compute_reward(rollout, GOLD)
    assert bd.legal_ref is False
    assert score == R_FULL_10


# ---------------------------------------------------------------------------
# Bonus stacking
# ---------------------------------------------------------------------------


def test_all_bonuses_stack_on_full_match():
    rollout = (
        "Per Chapter Notes 1 to chapter 61 and CROSS ruling N123456, "
        "the men's cotton t-shirt is classified under 6109.10.00.27."
    )
    score, bd = compute_reward(rollout, GOLD)
    assert bd.match_level == "full"
    assert bd.cross_cited is True
    assert bd.legal_ref is True
    assert score == pytest.approx(R_FULL_10 + B_CROSS_CITED + B_CHAPTER_NOTE)


# ---------------------------------------------------------------------------
# Boundary / form variations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code_form",
    [
        "6109.10.00.27",
        "6109.10.0027",
        "6109100027",
        "the code is 6109.10.00.27 (men's cotton t-shirts)",
    ],
)
def test_full_match_regardless_of_code_form(code_form):
    score, bd = compute_reward(code_form, GOLD)
    assert bd.match_level == "full"


@pytest.mark.parametrize("gold_form", [GOLD, GOLD_DIGITS, "6109.10.0027"])
def test_gold_code_form_does_not_matter(gold_form):
    score, bd = compute_reward("Code: 6109.10.00.27", gold_form)
    assert bd.match_level == "full"
    assert score == R_FULL_10


def test_empty_rollout():
    score, bd = compute_reward("", GOLD)
    assert bd.match_level == "none"
    assert bd.refusal is False
    assert score == 0.0


def test_no_code_no_refusal_keyword():
    rollout = "This looks like apparel of some kind."
    score, bd = compute_reward(rollout, GOLD)
    assert bd.match_level == "none"
    assert score == 0.0


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


def test_reward_fn_batch():
    rollouts = [
        "HTSUS Code: 6109.10.00.27",                          # full
        "Code 6109.10.40.20",                                  # 6-digit
        "I don't know.",                                       # refusal
        "Maybe 0017.30.20.10?",                                # hallucinated
    ]
    golds = [GOLD] * 4
    scores = reward_fn(rollouts, golds)
    assert scores == [R_FULL_10, R_6DIGIT, P_REFUSAL, P_HALLUCINATED]


def test_reward_fn_length_mismatch_raises():
    with pytest.raises(AssertionError):
        reward_fn(["a"], ["x", "y"])
