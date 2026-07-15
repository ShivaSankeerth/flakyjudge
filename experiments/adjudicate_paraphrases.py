"""Second-tier adjudication for cosine-rejected paraphrases.

The strict cosine >= 0.80 gate systematically rejects lexical and casual-
register paraphrases (synonym swaps move embeddings even when meaning is
preserved), which would bias E3 toward finding stability. Candidates in the
0.70-0.80 band that pass polarity+length are adjudicated by a bidirectional
semantic-equivalence check (gpt-4o-mini, cached); admitted variants are
labeled gate_tier='adjudicated' vs 'strict' so all E3 analyses run both
with and without them (preregistered sensitivity analysis).
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flakyjudge.cache import RequestKey, ResponseCache
from flakyjudge.data import DATA_DIR, read_jsonl, write_jsonl
from flakyjudge.perturb.paraphrase import (
    cosine,
    embed,
    length_ok,
    polarity_ok,
)
from flakyjudge.providers.base import JudgeSpec, compute_cost, normalize
from flakyjudge.providers.client import call_judge

CACHE_PATH = ROOT / "data" / "cache" / "responses.db"
COSINE_FLOOR = 0.70

ADJUDICATOR = JudgeSpec(
    name="equivalence-adjudicator",
    provider="openai-compat",
    model="gpt-4o-mini-2024-07-18",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    price_in_per_mtok=0.15,
    price_out_per_mtok=0.60,
    supports_logprobs=False,
)

ADJUDICATION_SYSTEM = (
    "You compare two evaluation criteria for LLM unit tests. Answer YES only "
    "if they test exactly the same property of a response with the same "
    "polarity, such that any response passing one must pass the other. "
    "Answer with YES or NO only."
)


def adjudication_prompt(original: str, variant: str) -> str:
    return f"Criterion A: {original}\n\nCriterion B: {variant}\n\nEquivalent?"


async def adjudicate(client, cache, original: str, variant: str) -> bool:
    key = RequestKey(
        provider=ADJUDICATOR.provider,
        model=ADJUDICATOR.model,
        system=ADJUDICATION_SYSTEM,
        prompt=adjudication_prompt(original, variant),
        temperature=0.0,
        max_tokens=4,
        logprobs=False,
    )
    raw = cache.get(key)
    if raw is None:
        cache.check_budget()
        raw = await call_judge(
            client, ADJUDICATOR, ADJUDICATION_SYSTEM,
            adjudication_prompt(original, variant),
            temperature=0.0, max_tokens=4, logprobs=False,
        )
        norm = normalize(ADJUDICATOR.provider, raw)
        cache.put(key, raw, norm.input_tokens, norm.output_tokens,
                  compute_cost(ADJUDICATOR, norm.input_tokens, norm.output_tokens))
    return normalize(ADJUDICATOR.provider, raw).text.strip().upper().startswith("YES")


async def main() -> None:
    items = {i["item_id"]: i for i in read_jsonl(DATA_DIR / "items_perturb.jsonl")}
    manifest = read_jsonl(DATA_DIR / "paraphrases.jsonl")
    report = json.load(open(DATA_DIR / "paraphrase_gate_report.json"))

    have = {(r["item_id"], r["variant_type"]) for r in manifest}
    for row in manifest:
        row.setdefault("gate_tier", "strict" if row["variant_type"] != "original" else None)

    admitted, examined = 0, 0
    cache = ResponseCache(CACHE_PATH)
    async with httpx.AsyncClient() as client:
        for item_id, kinds in report.items():
            original = items[item_id]["natural_unit_test"]
            original_vec = (await embed(client, cache, [original]))[0]
            for kind, entry in kinds.items():
                if (item_id, kind) in have or "rejected" not in entry:
                    continue
                variant = entry["rejected"]
                if not (polarity_ok(original, variant) and length_ok(original, variant)):
                    continue
                variant_vec = (await embed(client, cache, [variant]))[0]
                cos = cosine(original_vec, variant_vec)
                if cos < COSINE_FLOOR:
                    continue
                examined += 1
                if await adjudicate(client, cache, original, variant):
                    manifest.append(
                        {
                            "item_id": item_id,
                            "variant_type": kind,
                            "unit_test": variant,
                            "gate_tier": "adjudicated",
                            "cosine": round(cos, 4),
                        }
                    )
                    admitted += 1

    manifest.sort(key=lambda r: (r["item_id"], r["variant_type"]))
    write_jsonl(DATA_DIR / "paraphrases.jsonl", manifest)
    n_variants = sum(1 for r in manifest if r["variant_type"] != "original")
    print(f"examined {examined} borderline candidates, admitted {admitted} as tier-2")
    print(f"manifest now {n_variants}/1200 variants; spend: ${cache.total_spend():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
