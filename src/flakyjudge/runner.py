"""Cache-first async fan-out over judge calls.

Every call is identified by a RequestKey; cache hits never touch the network,
so interrupted runs resume for free and completed experiments replay at zero
cost. The budget hard-stop is checked before each network call.
"""

import asyncio
from dataclasses import dataclass

import httpx

from .cache import RequestKey, ResponseCache
from .prompts import JUDGE_SYSTEM_PROMPT
from .providers.base import JudgeSpec, NormalizedResponse, compute_cost, normalize
from .providers.client import call_judge
from .scoring import find_digit_position, logprob_expected_score, parse_direct_score

DEFAULT_MAX_TOKENS = 8
DEFAULT_CONCURRENCY = 8


@dataclass
class ScoredCall:
    key: RequestKey
    text: str
    score_direct: float | None
    score_logprob: float | None
    input_tokens: int | None
    output_tokens: int | None
    from_cache: bool


def extract_scores(normalized: NormalizedResponse) -> tuple[float | None, float | None]:
    score_direct = parse_direct_score(normalized.text)
    score_logprob = None
    position = find_digit_position(normalized.tokens)
    if position is not None and position < len(normalized.top_logprobs):
        score_logprob = logprob_expected_score(normalized.top_logprobs[position])
    return score_direct, score_logprob


async def run_calls(
    cache: ResponseCache,
    calls: list[tuple[JudgeSpec, RequestKey, str]],
    concurrency: int = DEFAULT_CONCURRENCY,
    system: str = JUDGE_SYSTEM_PROMPT,
) -> list[ScoredCall]:
    """Execute (spec, key, prompt) triples, cache-first. Order-preserving."""
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:

        async def one(spec: JudgeSpec, key: RequestKey, prompt: str) -> ScoredCall:
            raw = cache.get(key)
            from_cache = raw is not None
            if raw is None:
                cache.check_budget()
                async with semaphore:
                    raw = await call_judge(
                        client,
                        spec,
                        system=key.system,
                        prompt=prompt,
                        temperature=key.temperature,
                        max_tokens=key.max_tokens,
                        logprobs=key.logprobs,
                    )
                normalized = normalize(spec.provider, raw)
                cache.put(
                    key,
                    raw,
                    normalized.input_tokens,
                    normalized.output_tokens,
                    compute_cost(spec, normalized.input_tokens, normalized.output_tokens),
                )
            normalized = normalize(spec.provider, raw)
            score_direct, score_logprob = extract_scores(normalized)
            return ScoredCall(
                key=key,
                text=normalized.text,
                score_direct=score_direct,
                score_logprob=score_logprob,
                input_tokens=normalized.input_tokens,
                output_tokens=normalized.output_tokens,
                from_cache=from_cache,
            )

        return list(await asyncio.gather(*(one(s, k, p) for s, k, p in calls)))


def make_key(
    spec: JudgeSpec,
    prompt: str,
    temperature: float | None = 0.0,
    repeat_idx: int = 0,
    system: str = JUDGE_SYSTEM_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> RequestKey:
    return RequestKey(
        provider=spec.provider,
        model=spec.model,
        system=system,
        prompt=prompt,
        temperature=temperature if spec.supports_temperature else None,
        max_tokens=max_tokens,
        logprobs=spec.supports_logprobs,
        repeat_idx=repeat_idx,
    )
