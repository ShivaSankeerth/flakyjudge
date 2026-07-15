"""Score extraction: direct digit, logprob-weighted expectation, k-sample mean.

The logprob-weighted estimator is ported from LMUnit's
score_from_top_logprobs (ContextualAI/LMUnit, lmunit/lmunit.py), restricted
to the valid 1-5 scale (LMUnit's includes out-of-scale digits 0 and 6).
"""

import math

DIGIT_VALUES = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0}


def parse_direct_score(text: str) -> float | None:
    """Extract the single-digit score from judge output text.

    Returns None on parse failure — failures are recorded and reported as a
    per-judge robustness metric, never silently imputed.
    """
    stripped = text.strip()
    if stripped and stripped[0] in DIGIT_VALUES:
        return DIGIT_VALUES[stripped[0]]
    for char in stripped:
        if char in DIGIT_VALUES:
            return DIGIT_VALUES[char]
    return None


def logprob_expected_score(top_logprobs: dict[str, float]) -> float | None:
    """Probability-weighted expected score over digit tokens 1-5.

    top_logprobs maps candidate token strings to logprobs at the digit
    position. Tokens are stripped so " 3" and "3" both count.
    """
    weighted_sum = 0.0
    total_prob = 0.0
    for token, logprob in top_logprobs.items():
        value = DIGIT_VALUES.get(token.strip())
        if value is not None:
            p = math.exp(logprob)
            weighted_sum += p * value
            total_prob += p
    return weighted_sum / total_prob if total_prob > 0 else None


def find_digit_position(tokens: list[str]) -> int | None:
    """Index of the first token that parses as a 1-5 digit."""
    for i, token in enumerate(tokens):
        if token.strip() in DIGIT_VALUES:
            return i
    return None


def k_sample_mean(scores: list[float | None]) -> float | None:
    """Monte-Carlo analog of the logprob expectation for providers without
    logprobs (e.g. Anthropic): mean of repeated direct scores."""
    valid = [s for s in scores if s is not None]
    return sum(valid) / len(valid) if valid else None
