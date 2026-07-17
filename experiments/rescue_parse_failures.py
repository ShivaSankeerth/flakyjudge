"""Salvage pass for judge format non-compliance (preregistered amendment).

Claude Sonnet ignores the bare-digit instruction on ~10% of calls: it starts
free-text analysis and truncates at max_tokens before emitting any digit.
For every cached call whose text contains no digit score, this pass reissues
the IDENTICAL request with max_tokens=256 so the judge can finish its
analysis and emit a concluding score, parsed as the LAST "score: N" /
standalone digit pattern in the text. (An assistant-prefill rescue was
preregistered first but claude-sonnet-4-6's API rejects prefill.)

Salvaged scores are written to results/rescue_scores.parquet keyed by
(model, prompt_sha) with tier='extended'. Analyses run with and without
them.
"""

import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flakyjudge.cache import RequestKey, ResponseCache
from flakyjudge.providers.base import compute_cost, load_judges, normalize
from flakyjudge.providers.client import ANTHROPIC_VERSION, ProviderError, _api_key
from flakyjudge.scoring import parse_direct_score

CACHE_PATH = ROOT / "data" / "cache" / "responses.db"
RESULTS = ROOT / "results"
RESCUE_MAX_TOKENS = 256
SCORE_PATTERNS = [
    re.compile(r"score\s*[:=]?\s*\**\s*([1-5])\b", re.IGNORECASE),
    re.compile(r"\b([1-5])\s*(?:/\s*5)?\s*\.?\s*$"),
]


def parse_concluding_score(text: str) -> float | None:
    """Last explicit score mention wins; fall back to a trailing digit."""
    for pattern in SCORE_PATTERNS:
        matches = pattern.findall(text.strip())
        if matches:
            return float(matches[-1])
    return None


def failed_anthropic_calls(cache: ResponseCache, model: str) -> list[dict]:
    """Cached calls for `model` whose output contains no parseable score."""
    rows = cache.conn.execute(
        "SELECT request_json, response_json FROM responses WHERE model = ?",
        (model,),
    ).fetchall()
    failures = []
    for request_json, response_json in rows:
        request = json.loads(request_json)
        raw = json.loads(response_json)
        text = "".join(
            b["text"] for b in raw.get("content", []) if b["type"] == "text"
        )
        if parse_direct_score(text) is None:
            failures.append(request)
    return failures


async def rescue_one(
    client: httpx.AsyncClient, cache: ResponseCache, spec, request: dict
) -> dict | None:
    key = RequestKey(**{**request, "max_tokens": RESCUE_MAX_TOKENS})
    raw = cache.get(key)
    if raw is None:
        cache.check_budget()
        body = {
            "model": spec.model,
            "system": request["system"],
            "messages": [{"role": "user", "content": request["prompt"]}],
            "max_tokens": RESCUE_MAX_TOKENS,
        }
        if request["temperature"] is not None:
            body["temperature"] = request["temperature"]
        resp = await client.post(
            f"{spec.base_url}/messages",
            headers={"x-api-key": _api_key(spec),
                     "anthropic-version": ANTHROPIC_VERSION},
            json=body,
            timeout=120.0,
        )
        if resp.status_code != 200:
            raise ProviderError(f"rescue: HTTP {resp.status_code}: {resp.text[:200]}")
        raw = resp.json()
        norm = normalize("anthropic", raw)
        cache.put(key, raw, norm.input_tokens, norm.output_tokens,
                  compute_cost(spec, norm.input_tokens, norm.output_tokens))
    norm = normalize("anthropic", raw)
    score = parse_direct_score(norm.text[:8]) or parse_concluding_score(norm.text)
    return {
        "model": request["model"],
        "prompt_sha": hashlib.sha256(request["prompt"].encode()).hexdigest(),
        "temperature": request["temperature"],
        "repeat_idx": request["repeat_idx"],
        "tier": "extended",
        "score_direct": score,
        "raw_text": norm.text[-120:],
    }


async def main() -> None:
    judges = load_judges(ROOT / "config" / "judges.yaml")
    cache = ResponseCache(CACHE_PATH)
    out_rows = []
    async with httpx.AsyncClient() as client:
        semaphore = asyncio.Semaphore(8)
        for judge_name in ("claude-sonnet", "claude-haiku"):
            spec = judges[judge_name]
            failures = failed_anthropic_calls(cache, spec.model)
            print(f"{judge_name}: {len(failures)} format failures to rescue")

            async def one(request, spec=spec):
                async with semaphore:
                    return await rescue_one(client, cache, spec, request)

            results = await asyncio.gather(*(one(f) for f in failures))
            rescued = [r for r in results if r and r["score_direct"] is not None]
            out_rows.extend(r for r in results if r)
            print(f"{judge_name}: rescued {len(rescued)}/{len(failures)}; "
                  f"spend ${cache.total_spend():.2f}")

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(out_rows).to_parquet(RESULTS / "rescue_scores.parquet", index=False)
    print(f"wrote {len(out_rows)} rows -> results/rescue_scores.parquet")


if __name__ == "__main__":
    asyncio.run(main())
