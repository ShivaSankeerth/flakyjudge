"""Async HTTP client for judge calls with retry/backoff.

Raw JSON responses are returned untouched — they are cached verbatim so any
estimator (direct, logprob-weighted, alternatives) can be re-derived later
without re-spending.
"""

import asyncio
import os

import httpx

from .base import JudgeSpec

RETRYABLE_STATUS = {429, 500, 502, 503, 529}
MAX_RETRIES = 6
ANTHROPIC_VERSION = "2023-06-01"


class ProviderError(RuntimeError):
    pass


def _api_key(spec: JudgeSpec) -> str:
    key = os.environ.get(spec.api_key_env)
    if not key:
        raise ProviderError(f"Missing API key: set {spec.api_key_env}")
    return key


def build_request(
    spec: JudgeSpec,
    system: str,
    prompt: str,
    temperature: float | None,
    max_tokens: int,
    logprobs: bool,
) -> tuple[str, dict, dict]:
    """Returns (url, headers, json_body) for one judge call."""
    if spec.provider == "openai-compat":
        headers = {"Authorization": f"Bearer {_api_key(spec)}"}
        if spec.extra_headers:
            headers.update(spec.extra_headers)
        body: dict = {
            "model": spec.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
        }
        if temperature is not None and spec.supports_temperature:
            body["temperature"] = temperature
        if logprobs and spec.supports_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 20
        return f"{spec.base_url}/chat/completions", headers, body

    if spec.provider == "anthropic":
        headers = {
            "x-api-key": _api_key(spec),
            "anthropic-version": ANTHROPIC_VERSION,
        }
        body = {
            "model": spec.model,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if temperature is not None and spec.supports_temperature:
            body["temperature"] = temperature
        return f"{spec.base_url}/messages", headers, body

    raise ProviderError(f"Unknown provider: {spec.provider}")


async def call_judge(
    client: httpx.AsyncClient,
    spec: JudgeSpec,
    system: str,
    prompt: str,
    temperature: float | None,
    max_tokens: int,
    logprobs: bool,
) -> dict:
    url, headers, body = build_request(spec, system, prompt, temperature, max_tokens, logprobs)
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(url, headers=headers, json=body, timeout=120.0)
        except httpx.TransportError:
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
            retry_after = resp.headers.get("retry-after")
            await asyncio.sleep(float(retry_after) if retry_after else delay)
            delay *= 2
            continue
        raise ProviderError(f"{spec.name}: HTTP {resp.status_code}: {resp.text[:500]}")
    raise ProviderError(f"{spec.name}: exhausted retries")
