from unittest.mock import AsyncMock, patch

import pytest

from flakyjudge.ensemble import (
    BUILTIN_JUDGES,
    EnsembleResult,
    _resolve_judge,
    ensemble_score_async,
)
from flakyjudge.providers.base import NormalizedResponse


class TestResolveJudge:
    def test_builtin_by_name(self):
        assert _resolve_judge("gpt-4o-mini").model == "gpt-4o-mini-2024-07-18"

    def test_spec_passthrough(self):
        spec = BUILTIN_JUDGES["claude-haiku"]
        assert _resolve_judge(spec) is spec

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown judge"):
            _resolve_judge("gpt-99")


def normalized(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        text=text, input_tokens=10, output_tokens=1, tokens=[], top_logprobs=[]
    )


@pytest.fixture
def mocked_calls(tmp_path):
    """Patch generation + judge calls; yields the judge-score queue."""
    scores: list[str] = []
    para_json = '["Was the refund window mentioned?", "Verify the refund window is cited."]'

    async def fake_cached_call(client, cache, spec, system, prompt, max_tokens,
                               temperature=0.0):
        if max_tokens == 800:
            return normalized(para_json)
        return normalized(scores.pop(0))

    with patch("flakyjudge.ensemble._cached_call", new=AsyncMock(side_effect=fake_cached_call)):
        yield scores, tmp_path / "cache.db"


async def run(scores_queue, cache_path, judge_scores):
    scores_queue.extend(judge_scores)
    return await ensemble_score_async(
        "q", "r", "Does the reply cite the refund window?",
        judge="gpt-4o-mini", n_paraphrases=2, cache_path=cache_path,
    )


class TestEnsembleAggregation:
    async def test_stable_pass(self, mocked_calls):
        scores, cache_path = mocked_calls
        result = await run(scores, cache_path, ["4", "4", "5"])
        assert isinstance(result, EnsembleResult)
        assert result.verdict == "pass"
        assert result.stable is True
        assert result.flip_fraction == 0.0
        assert result.score == pytest.approx(13 / 3)
        assert len(result.variants) == 3

    async def test_unstable_when_wordings_disagree(self, mocked_calls):
        scores, cache_path = mocked_calls
        result = await run(scores, cache_path, ["3", "2", "3"])
        assert result.verdict == "pass"  # mean 2.67 > 2.5
        assert result.stable is False
        assert result.flip_fraction == pytest.approx(1 / 3)
        assert result.margin == pytest.approx(2.67 - 2.5, abs=0.01)

    async def test_parse_failures_excluded(self, mocked_calls):
        scores, cache_path = mocked_calls
        result = await run(scores, cache_path, ["garbage", "4", "4"])
        assert result.score == 4.0
        assert result.stable is True
        assert sum(v.score is None for v in result.variants) == 1

    async def test_all_failures_raise(self, mocked_calls):
        scores, cache_path = mocked_calls
        with pytest.raises(RuntimeError, match="parseable"):
            await run(scores, cache_path, ["x", "y", "z"])
