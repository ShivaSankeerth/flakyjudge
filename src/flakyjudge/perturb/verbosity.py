"""Content-matched verbosity variants: padded (~1.8x) and condensed (~0.6x).

The treatment must be length, not content: padding may add hedging,
restatement, politeness, and sign-posting but not one new factual claim;
condensing must preserve every claim. Gates (PREREGISTRATION control #4):
  1. realized token-length ratio in the intended direction
  2. bidirectional LLM claim audit (gpt-4o-mini): claims in A absent from B
     and vice versa must both be empty
Analysis regresses drift on the REALIZED log length ratio, not the target.
"""

import json
import re

import httpx

from ..cache import RequestKey, ResponseCache
from ..providers.base import JudgeSpec, compute_cost, normalize
from ..providers.client import call_judge

GENERATOR = JudgeSpec(
    name="verbosity-generator",
    provider="openai-compat",
    model="gpt-4o-2024-11-20",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    price_in_per_mtok=2.50,
    price_out_per_mtok=10.00,
    supports_logprobs=False,
)

AUDITOR = JudgeSpec(
    name="claim-auditor",
    provider="openai-compat",
    model="gpt-4o-mini-2024-07-18",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    price_in_per_mtok=0.15,
    price_out_per_mtok=0.60,
    supports_logprobs=False,
)

PAD_SYSTEM = (
    "You rewrite texts to be longer without changing their substance. You may "
    "add hedging, polite framing, restatements of points already made, "
    "sign-posting ('First...', 'To summarize...'), and a closing summary. You "
    "must NOT add a single new factual claim, example, or recommendation, and "
    "must not remove or alter any existing claim. Reply with the rewritten "
    "text only."
)

CONDENSE_SYSTEM = (
    "You rewrite texts to be much more concise. Preserve every factual claim, "
    "example, and recommendation - remove only redundancy, hedging, filler, "
    "and repetition. Reply with the rewritten text only."
)

AUDIT_SYSTEM = (
    "You compare two texts. List every factual claim, example, or "
    "recommendation present in TEXT B but absent from TEXT A. Reply with a "
    'JSON object {"new_claims": ["..."]}. If there are none, reply '
    '{"new_claims": []}. JSON only.'
)


def word_ratio(original: str, variant: str) -> float:
    return len(variant.split()) / max(1, len(original.split()))


async def _cached_call(
    client: httpx.AsyncClient,
    cache: ResponseCache,
    spec: JudgeSpec,
    system: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    key = RequestKey(
        provider=spec.provider, model=spec.model, system=system, prompt=prompt,
        temperature=temperature, max_tokens=max_tokens, logprobs=False,
    )
    raw = cache.get(key)
    if raw is None:
        cache.check_budget()
        raw = await call_judge(client, spec, system, prompt,
                               temperature=temperature, max_tokens=max_tokens,
                               logprobs=False)
        norm = normalize(spec.provider, raw)
        cache.put(key, raw, norm.input_tokens, norm.output_tokens,
                  compute_cost(spec, norm.input_tokens, norm.output_tokens))
    return normalize(spec.provider, raw).text.strip()


async def new_claims(
    client: httpx.AsyncClient, cache: ResponseCache, text_a: str, text_b: str
) -> list[str]:
    """Claims in B absent from A."""
    prompt = f"TEXT A:\n{text_a}\n\nTEXT B:\n{text_b}"
    text = await _cached_call(client, cache, AUDITOR, AUDIT_SYSTEM, prompt, 500)
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text).get("new_claims", [])
    except json.JSONDecodeError:
        return ["<audit parse failure>"]


async def make_variant(
    client: httpx.AsyncClient,
    cache: ResponseCache,
    response: str,
    kind: str,  # "padded" | "condensed"
) -> dict:
    """Generate one variant and run the gates. Returns dict with the variant,
    realized ratio, audit results, and pass/fail per gate."""
    n_words = len(response.split())
    if kind == "padded":
        system = PAD_SYSTEM
        prompt = (f"Rewrite this to roughly {int(n_words * 1.8)} words "
                  f"(it is {n_words} now):\n\n{response}")
        max_tokens = min(4000, int(n_words * 1.8 * 1.6) + 200)
        ratio_ok = lambda r: r >= 1.3  # noqa: E731
    else:
        system = CONDENSE_SYSTEM
        prompt = (f"Rewrite this to roughly {int(n_words * 0.6)} words "
                  f"(it is {n_words} now):\n\n{response}")
        max_tokens = min(4000, int(n_words * 1.2) + 200)
        ratio_ok = lambda r: r <= 0.8  # noqa: E731

    variant = await _cached_call(client, cache, GENERATOR, system, prompt,
                                 max_tokens)
    ratio = word_ratio(response, variant)
    added = await new_claims(client, cache, response, variant)
    dropped = await new_claims(client, cache, variant, response)
    gates = {
        "ratio": ratio_ok(ratio),
        "no_added_claims": len(added) == 0,
        "no_dropped_claims": len(dropped) == 0,
    }
    return {
        "kind": kind,
        "variant": variant,
        "realized_ratio": round(ratio, 3),
        "added_claims": added,
        "dropped_claims": dropped,
        "gates": gates,
        "passed": all(gates.values()),
    }
