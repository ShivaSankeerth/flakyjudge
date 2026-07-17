"""Judge specifications and provider response normalization.

Judges are hard-coded in config/judges.yaml (see scope guardrails: no plugin
system). Two wire protocols cover all six judges: OpenAI-compatible chat
completions (OpenAI, OpenRouter, Gemini's compat endpoint) and the Anthropic
Messages API.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class JudgeSpec:
    name: str
    provider: str  # "openai-compat" | "anthropic"
    model: str
    base_url: str
    api_key_env: str
    price_in_per_mtok: float
    price_out_per_mtok: float
    supports_logprobs: bool
    supports_temperature: bool = True
    extra_headers: dict | None = None
    # Extra request-body fields (e.g. OpenRouter provider pinning). Not part
    # of the cache key: it changes serving, not the request semantics.
    extra_body: dict | None = None


def load_judges(config_path: str | Path) -> dict[str, JudgeSpec]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    judges = {}
    for name, cfg in raw["judges"].items():
        judges[name] = JudgeSpec(name=name, **cfg)
    return judges


@dataclass
class NormalizedResponse:
    """Provider-agnostic view of one judge call, derived from the raw JSON."""

    text: str
    input_tokens: int | None
    output_tokens: int | None
    # Per output-token-position: token string and {candidate_token: logprob}.
    # Empty lists when the provider does not return logprobs.
    tokens: list[str]
    top_logprobs: list[dict[str, float]]


def normalize(provider: str, raw: dict) -> NormalizedResponse:
    if provider == "openai-compat":
        return _normalize_openai(raw)
    if provider == "anthropic":
        return _normalize_anthropic(raw)
    raise ValueError(f"Unknown provider: {provider}")


def _normalize_openai(raw: dict) -> NormalizedResponse:
    choice = raw["choices"][0]
    text = choice["message"]["content"] or ""
    usage = raw.get("usage") or {}
    tokens: list[str] = []
    top_logprobs: list[dict[str, float]] = []
    content_logprobs = (choice.get("logprobs") or {}).get("content") or []
    for position in content_logprobs:
        tokens.append(position["token"])
        top_logprobs.append(
            {cand["token"]: cand["logprob"] for cand in position.get("top_logprobs", [])}
        )
    return NormalizedResponse(
        text=text,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        tokens=tokens,
        top_logprobs=top_logprobs,
    )


def _normalize_anthropic(raw: dict) -> NormalizedResponse:
    text = "".join(block["text"] for block in raw.get("content", []) if block["type"] == "text")
    usage = raw.get("usage") or {}
    return NormalizedResponse(
        text=text,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        tokens=[],
        top_logprobs=[],
    )


def compute_cost(spec: JudgeSpec, input_tokens: int | None, output_tokens: int | None) -> float:
    return (input_tokens or 0) / 1e6 * spec.price_in_per_mtok + (
        output_tokens or 0
    ) / 1e6 * spec.price_out_per_mtok
