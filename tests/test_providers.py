import pytest

from flakyjudge.providers.base import JudgeSpec, compute_cost, normalize
from flakyjudge.providers.client import build_request
from flakyjudge.runner import extract_scores

OPENAI_SPEC = JudgeSpec(
    name="gpt-4o",
    provider="openai-compat",
    model="gpt-4o-2024-11-20",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    price_in_per_mtok=2.50,
    price_out_per_mtok=10.00,
    supports_logprobs=True,
)

ANTHROPIC_SPEC = JudgeSpec(
    name="claude-haiku",
    provider="anthropic",
    model="claude-haiku-4-5",
    base_url="https://api.anthropic.com/v1",
    api_key_env="ANTHROPIC_API_KEY",
    price_in_per_mtok=1.00,
    price_out_per_mtok=5.00,
    supports_logprobs=False,
)

OPENAI_RAW = {
    "choices": [
        {
            "message": {"content": "4"},
            "logprobs": {
                "content": [
                    {
                        "token": "4",
                        "logprob": -0.35667494,  # ln(0.7)
                        "top_logprobs": [
                            {"token": "4", "logprob": -0.35667494},
                            {"token": "3", "logprob": -1.2039728},  # ln(0.3)
                        ],
                    }
                ]
            },
        }
    ],
    "usage": {"prompt_tokens": 750, "completion_tokens": 1},
}

ANTHROPIC_RAW = {
    "content": [{"type": "text", "text": "3"}],
    "usage": {"input_tokens": 800, "output_tokens": 2},
}


class TestNormalize:
    def test_openai_shape(self):
        norm = normalize("openai-compat", OPENAI_RAW)
        assert norm.text == "4"
        assert norm.input_tokens == 750
        assert norm.tokens == ["4"]
        assert norm.top_logprobs[0]["3"] == pytest.approx(-1.2039728)

    def test_anthropic_shape(self):
        norm = normalize("anthropic", ANTHROPIC_RAW)
        assert norm.text == "3"
        assert norm.input_tokens == 800
        assert norm.tokens == []


class TestExtractScores:
    def test_openai_yields_both_estimators(self):
        direct, logprob = extract_scores(normalize("openai-compat", OPENAI_RAW))
        assert direct == 4.0
        assert logprob == pytest.approx(3.7, abs=1e-6)

    def test_anthropic_yields_direct_only(self):
        direct, logprob = extract_scores(normalize("anthropic", ANTHROPIC_RAW))
        assert direct == 3.0
        assert logprob is None


class TestBuildRequest:
    def test_openai_request(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        url, headers, body = build_request(OPENAI_SPEC, "sys", "prompt", 0.0, 8, logprobs=True)
        assert url.endswith("/chat/completions")
        assert headers["Authorization"] == "Bearer sk-test"
        assert body["logprobs"] is True and body["top_logprobs"] == 20
        assert body["temperature"] == 0.0
        assert body["messages"][0] == {"role": "system", "content": "sys"}

    def test_anthropic_request(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        url, headers, body = build_request(ANTHROPIC_SPEC, "sys", "prompt", 0.0, 8, logprobs=True)
        assert url.endswith("/messages")
        assert headers["x-api-key"] == "sk-ant-test"
        assert body["system"] == "sys"
        assert "logprobs" not in body  # unsupported -> never sent

    def test_temperature_omitted_when_unsupported(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        spec = JudgeSpec(**{**ANTHROPIC_SPEC.__dict__, "supports_temperature": False})
        _, _, body = build_request(spec, "sys", "prompt", 0.0, 8, logprobs=False)
        assert "temperature" not in body


class TestCost:
    def test_hand_computed(self):
        # 750 in @ $2.50/M + 1 out @ $10/M
        expected = 750 / 1e6 * 2.50 + 1 / 1e6 * 10.00
        assert compute_cost(OPENAI_SPEC, 750, 1) == pytest.approx(expected)
