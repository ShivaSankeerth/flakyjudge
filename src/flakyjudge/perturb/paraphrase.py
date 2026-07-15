"""Typed paraphrase generation for unit-test criteria, with validity gates.

Each of the 150 perturbation-manifest unit tests gets:
  - 6 meaning-preserving paraphrases, one per frozen type (lexical,
    syntactic, formal/casual register, question/imperative form)
  - 2 positive controls that deliberately CHANGE the criterion (negated
    polarity, swapped criterion) — judges SHOULD flip on these; they prove
    the instrument has resolution (PREREGISTRATION control #3)

Gates (rejected variants are regenerated up to 3 times; pass rates reported):
  1. embedding cosine >= 0.80 vs original (text-embedding-3-small)
  2. polarity rule check: no negators introduced or removed
  3. length within 0.5-2.0x of original

Generation is cache-first through the same ResponseCache as scoring, so the
frozen paraphrase manifest is regenerable byte-for-byte at zero cost.
"""

import json
import re

import httpx
import numpy as np

from ..cache import RequestKey, ResponseCache
from ..providers.base import JudgeSpec, compute_cost, normalize
from ..providers.client import call_judge

PARAPHRASE_TYPES = {
    "lexical": "Replace content words with synonyms; keep the sentence structure.",
    "syntactic": "Restructure the sentence (voice, clause order); keep the same words "
    "where possible.",
    "register_formal": "Rewrite in a more formal, precise register.",
    "register_casual": "Rewrite in a more casual, conversational register.",
    "form_question": "Express as a direct question (if it already is one, rephrase it "
    "as a different question).",
    "form_imperative": "Express as an imperative instruction to check something, e.g. "
    "'Verify that ...'.",
}

CONTROL_TYPES = {
    "control_negated": "Negate the criterion so it tests the OPPOSITE property.",
    "control_swapped": "Replace the criterion with a plausible but DIFFERENT criterion "
    "about another quality of the response.",
}

GENERATION_SYSTEM = (
    "You rewrite evaluation criteria for LLM unit tests. Preserve the exact "
    "evaluative meaning and polarity unless the instruction says otherwise. "
    "Never add or drop sub-criteria. Reply with JSON only."
)

NEGATORS = re.compile(
    r"\b(not|never|no|avoids?|without|fails?|lacks?|missing|excludes?|omits?|"
    r"free (?:from|of)|devoid|refrains?|absent|steers? clear)\b",
    re.IGNORECASE,
)

GENERATOR = JudgeSpec(
    name="paraphrase-generator",
    provider="openai-compat",
    model="gpt-4o-2024-11-20",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    price_in_per_mtok=2.50,
    price_out_per_mtok=10.00,
    supports_logprobs=False,
)

EMBED_MODEL = "text-embedding-3-small"
EMBED_PRICE_PER_MTOK = 0.02
COSINE_THRESHOLD = 0.80


def generation_prompt(unit_test: str, attempt: int) -> str:
    kinds = {**PARAPHRASE_TYPES, **CONTROL_TYPES}
    spec = "\n".join(f'- "{k}": {v}' for k, v in kinds.items())
    retry_note = (
        f"\nThis is regeneration attempt {attempt}; produce different wordings "
        "than a typical first attempt." if attempt else ""
    )
    return (
        f"Original unit test criterion:\n{unit_test}\n\n"
        f"Produce one rewrite per key:\n{spec}\n{retry_note}\n"
        'Reply with a JSON object mapping each key to its rewrite, e.g. '
        '{"lexical": "...", ...}. JSON only, no code fences.'
    )


def polarity_ok(original: str, variant: str) -> bool:
    return bool(NEGATORS.search(original)) == bool(NEGATORS.search(variant))


def length_ok(original: str, variant: str) -> bool:
    ratio = len(variant.split()) / max(1, len(original.split()))
    return 0.5 <= ratio <= 2.0


async def embed(
    client: httpx.AsyncClient, cache: ResponseCache, texts: list[str]
) -> np.ndarray:
    """Cached OpenAI embeddings, one text per cache entry."""
    import os

    vectors = []
    for text in texts:
        key = RequestKey(
            provider="openai-embed",
            model=EMBED_MODEL,
            system="",
            prompt=text,
            temperature=None,
            max_tokens=0,
            logprobs=False,
        )
        raw = cache.get(key)
        if raw is None:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={"model": EMBED_MODEL, "input": text},
                timeout=60.0,
            )
            resp.raise_for_status()
            raw = resp.json()
            tokens = raw.get("usage", {}).get("prompt_tokens", 0)
            cache.put(key, raw, tokens, 0, tokens / 1e6 * EMBED_PRICE_PER_MTOK)
        vectors.append(raw["data"][0]["embedding"])
    return np.array(vectors, dtype=float)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


async def generate_for_item(
    client: httpx.AsyncClient,
    cache: ResponseCache,
    unit_test: str,
    max_attempts: int = 5,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Returns (accepted variants by type, gate report by type)."""
    accepted: dict[str, str] = {}
    report: dict[str, dict] = {}
    original_vec = (await embed(client, cache, [unit_test]))[0]

    for attempt in range(max_attempts):
        missing = [k for k in {**PARAPHRASE_TYPES, **CONTROL_TYPES} if k not in accepted]
        if not missing:
            break
        key = RequestKey(
            provider=GENERATOR.provider,
            model=GENERATOR.model,
            system=GENERATION_SYSTEM,
            prompt=generation_prompt(unit_test, attempt),
            temperature=1.0,
            max_tokens=1200,
            logprobs=False,
            repeat_idx=attempt,
        )
        raw = cache.get(key)
        if raw is None:
            cache.check_budget()
            raw = await call_judge(
                client, GENERATOR, GENERATION_SYSTEM,
                generation_prompt(unit_test, attempt),
                temperature=1.0, max_tokens=1200, logprobs=False,
            )
            norm = normalize(GENERATOR.provider, raw)
            cache.put(key, raw, norm.input_tokens, norm.output_tokens,
                      compute_cost(GENERATOR, norm.input_tokens, norm.output_tokens))
        text = normalize(GENERATOR.provider, raw).text.strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            variants = json.loads(text)
        except json.JSONDecodeError:
            continue

        for kind in missing:
            variant = variants.get(kind, "").strip()
            if not variant:
                continue
            is_control = kind in CONTROL_TYPES
            gates = {"length": length_ok(unit_test, variant)}
            if not is_control:
                gates["polarity"] = polarity_ok(unit_test, variant)
                vec = (await embed(client, cache, [variant]))[0]
                gates["cosine"] = cosine(original_vec, vec) >= COSINE_THRESHOLD
            if all(gates.values()):
                accepted[kind] = variant
                report[kind] = {"attempt": attempt, "gates": gates}
            else:
                report[kind] = {"attempt": attempt, "gates": gates, "rejected": variant}
    return accepted, report
