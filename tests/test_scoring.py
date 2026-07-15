import math

from flakyjudge.scoring import (
    find_digit_position,
    k_sample_mean,
    logprob_expected_score,
    parse_direct_score,
)


class TestParseDirectScore:
    def test_bare_digit(self):
        assert parse_direct_score("4") == 4.0

    def test_whitespace_and_trailing_text(self):
        assert parse_direct_score("  3\n") == 3.0
        assert parse_direct_score("Score: 5") == 5.0

    def test_out_of_scale_digits_rejected(self):
        assert parse_direct_score("0") is None
        assert parse_direct_score("6") is None

    def test_garbage_returns_none(self):
        assert parse_direct_score("The response is good.") is None
        assert parse_direct_score("") is None


class TestLogprobExpectedScore:
    def test_hand_computed_two_candidates(self):
        # p(4) = 0.7, p(3) = 0.3 -> E = (0.7*4 + 0.3*3) / 1.0 = 3.7
        top = {"4": math.log(0.7), "3": math.log(0.3)}
        assert abs(logprob_expected_score(top) - 3.7) < 1e-9

    def test_renormalizes_over_digit_mass_only(self):
        # Non-digit tokens are excluded and probabilities renormalized:
        # p(4)=0.5, p(3)=0.25 -> E = (0.5*4 + 0.25*3) / 0.75 = 3.666...
        top = {"4": math.log(0.5), "3": math.log(0.25), "The": math.log(0.25)}
        assert abs(logprob_expected_score(top) - 11 / 3) < 1e-9

    def test_tokens_with_leading_space(self):
        top = {" 5": math.log(0.9), " 4": math.log(0.1)}
        assert abs(logprob_expected_score(top) - 4.9) < 1e-9

    def test_out_of_scale_digits_excluded(self):
        # LMUnit includes 0 and 6; we restrict to the valid 1-5 scale.
        top = {"6": math.log(0.5), "5": math.log(0.5)}
        assert logprob_expected_score(top) == 5.0

    def test_no_digit_mass_returns_none(self):
        assert logprob_expected_score({"a": -0.1, "b": -3.0}) is None


class TestHelpers:
    def test_find_digit_position(self):
        assert find_digit_position(["", " ", "4", "."]) == 2
        assert find_digit_position(["no", "digits"]) is None

    def test_k_sample_mean_skips_parse_failures(self):
        assert k_sample_mean([4.0, None, 5.0]) == 4.5
        assert k_sample_mean([None, None]) is None
