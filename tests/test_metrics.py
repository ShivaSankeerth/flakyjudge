import numpy as np

from flakyjudge.metrics import excess_sd, flip_rate, icc_2_1, paired_bootstrap_diff

# Shrout & Fleiss (1979), Table 2: 6 targets rated by 4 judges.
SHROUT_FLEISS = np.array(
    [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ],
    dtype=float,
)


class TestICC:
    def test_shrout_fleiss_published_value(self):
        # Published ICC(2,1) for this dataset is 0.29.
        assert abs(icc_2_1(SHROUT_FLEISS) - 0.29) < 0.005

    def test_perfect_agreement(self):
        matrix = np.tile([[1.0], [3.0], [5.0]], (1, 4))
        assert icc_2_1(matrix) > 0.999


class TestFlipRate:
    def test_hand_computed(self):
        # Item 1: all pass; item 2: split -> flip; item 3: all fail.
        matrix = np.array([[3, 4], [2, 3], [1, 2]], dtype=float)
        assert flip_rate(matrix, threshold=2.5) == 1 / 3

    def test_unanimous_never_flips(self):
        matrix = np.array([[5, 5, 5], [1, 1, 1]], dtype=float)
        assert flip_rate(matrix) == 0.0


class TestExcessSD:
    def test_perturbation_noise_subtraction(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(3.0, 0.1, size=(500, 5))
        perturbed = rng.normal(3.0, 0.5, size=(500, 5))
        result = excess_sd(perturbed, noise)
        # Expect roughly 0.5 - 0.1 = 0.4
        assert 0.3 < result < 0.5

    def test_zero_when_same_distribution(self):
        rng = np.random.default_rng(1)
        a = rng.normal(3.0, 0.3, size=(1000, 5))
        b = rng.normal(3.0, 0.3, size=(1000, 5))
        assert abs(excess_sd(a, b)) < 0.02


class TestPairedBootstrap:
    def test_detects_known_shift(self):
        rng = np.random.default_rng(2)
        b = rng.normal(0.0, 1.0, size=200)
        a = b + 0.5
        result = paired_bootstrap_diff(a, b, statistic=np.mean, n_resamples=2000)
        assert abs(result["diff"] - 0.5) < 1e-9
        assert result["ci_low"] > 0
        assert result["p_positive"] == 1.0

    def test_null_diff_covers_zero(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0.0, 1.0, size=300)
        b = rng.normal(0.0, 1.0, size=300)
        result = paired_bootstrap_diff(a, b, statistic=np.mean, n_resamples=2000)
        assert result["ci_low"] < 0 < result["ci_high"]
