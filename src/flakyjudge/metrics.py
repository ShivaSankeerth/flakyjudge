"""Reliability and bias metrics.

Implemented from the ANOVA definitions rather than pulled from a stats
package so the math is auditable; tests/test_metrics.py checks them against
published worked examples (Shrout & Fleiss 1979).
"""

import numpy as np

PASS_THRESHOLD = 2.5  # LMUnit's direct-scoring decision threshold (lmunit/tasks.py)


def icc_2_1(matrix: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater.

    matrix: items x raters (raters = paraphrases or resamples), no NaNs.
    """
    data = np.asarray(matrix, dtype=float)
    n, k = data.shape
    grand = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_total = ((data - grand) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return float((ms_rows - ms_error) / denom)


def flip_rate(matrix: np.ndarray, threshold: float = PASS_THRESHOLD) -> float:
    """Fraction of items whose pass/fail verdict is not unanimous across
    raters (paraphrases/resamples) at the given threshold."""
    passes = np.asarray(matrix, dtype=float) > threshold
    return float((passes.any(axis=1) & ~passes.all(axis=1)).mean())


def excess_sd(perturbation_matrix: np.ndarray, noise_matrix: np.ndarray) -> float:
    """Mean per-item SD under perturbation minus mean per-item SD under
    identical-input resampling (the noise floor). The headline sensitivity
    statistic: what the perturbation adds beyond inherent judge noise."""
    perturb_sd = np.asarray(perturbation_matrix, dtype=float).std(axis=1, ddof=1).mean()
    noise_sd = np.asarray(noise_matrix, dtype=float).std(axis=1, ddof=1).mean()
    return float(perturb_sd - noise_sd)


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    statistic,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for statistic(a) - statistic(b), resampling items jointly
    (paired design). Returns point estimate, 95% CI, and P(diff > 0)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape[0] == b.shape[0], "paired design requires equal item counts"
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = statistic(a[idx]) - statistic(b[idx])
    point = statistic(a) - statistic(b)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_positive": float((diffs > 0).mean()),
    }
