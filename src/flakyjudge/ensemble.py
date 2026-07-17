"""Paraphrase-ensemble scoring: the mitigation this study's findings motivate.

Rewording a unit test flips single-shot verdicts on 14-25% of items (see
report/). ensemble_score() scores the SAME (query, response) against the
original criterion plus n auto-generated meaning-preserving paraphrases and
reports the spread, so a wording-dependent verdict is visible instead of
silent.

    from flakyjudge import ensemble_score
    r = ensemble_score(query, response,
                       "Does the reply cite the refund window?",
                       judge="gpt-4o-mini")
    r.score      # 3.4  (mean across wordings)
    r.sd         # 0.62
    r.verdict    # "pass"
    r.stable     # False -> 2 of 5 wordings disagree with the majority
    r.margin     # 0.9   (distance from the 2.5 threshold; <1 = borderline)

Costs n_paraphrases+2 judge calls plus one generation call. Use it for
gating/CI evals where a silent flip matters; single-shot is fine for
exploration. Paraphrases pass polarity and length gates (the study's
embedding gate needs an extra embedding call and is skipped here; for
audit-grade use, review r.variants).
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .cache import RequestKey, ResponseCache
from .metrics import PASS_THRESHOLD
from .perturb.paraphrase import length_ok, polarity_ok
from .prompts import JUDGE_SYSTEM_PROMPT, build_prompt
from .providers.base import JudgeSpec, compute_cost, normalize
from .providers.client import call_judge
from .runner import extract_scores

DEFAULT_CACHE = Path.home() / ".flakyjudge" / "cache.db"

# Built-in judge specs so pip users need no config file. Mirrors
# config/judges.yaml; pass a JudgeSpec for anything else.
BUILTIN_JUDGES = {
    "gpt-4o": JudgeSpec(
        name="gpt-4o", provider="openai-compat", model="gpt-4o-2024-11-20",
        base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY",
        price_in_per_mtok=2.50, price_out_per_mtok=10.00, supports_logprobs=True,
    ),
    "gpt-4o-mini": JudgeSpec(
        name="gpt-4o-mini", provider="openai-compat",
        model="gpt-4o-mini-2024-07-18",
        base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY",
        price_in_per_mtok=0.15, price_out_per_mtok=0.60, supports_logprobs=True,
    ),
    "claude-sonnet": JudgeSpec(
        name="claude-sonnet", provider="anthropic", model="claude-sonnet-4-6",
        base_url="https://api.anthropic.com/v1", api_key_env="ANTHROPIC_API_KEY",
        price_in_per_mtok=3.00, price_out_per_mtok=15.00, supports_logprobs=False,
    ),
    "claude-haiku": JudgeSpec(
        name="claude-haiku", provider="anthropic",
        model="claude-haiku-4-5-20251001",
        base_url="https://api.anthropic.com/v1", api_key_env="ANTHROPIC_API_KEY",
        price_in_per_mtok=1.00, price_out_per_mtok=5.00, supports_logprobs=False,
    ),
}

PARAPHRASE_SYSTEM = (
    "You rewrite evaluation criteria for LLM unit tests. Each rewrite must "
    "preserve the exact evaluative meaning and polarity: any response that "
    "passes one wording must pass all of them. Vary syntax, word choice, and "
    "form (question / imperative / declarative). Reply with a JSON array of "
    "strings only."
)


@dataclass
class VariantScore:
    unit_test: str
    kind: str  # "original" | "paraphrase"
    score: float | None
    passed: bool | None


@dataclass
class EnsembleResult:
    score: float
    sd: float
    verdict: str  # "pass" | "fail"
    stable: bool
    flip_fraction: float
    margin: float
    threshold: float
    variants: list[VariantScore] = field(default_factory=list)

    def __repr__(self) -> str:
        state = "STABLE" if self.stable else "UNSTABLE"
        return (f"EnsembleResult(score={self.score:.2f}±{self.sd:.2f}, "
                f"verdict={self.verdict} [{state}], margin={self.margin:.2f}, "
                f"{len(self.variants)} wordings)")


def _resolve_judge(judge: str | JudgeSpec) -> JudgeSpec:
    if isinstance(judge, JudgeSpec):
        return judge
    if judge in BUILTIN_JUDGES:
        return BUILTIN_JUDGES[judge]
    raise ValueError(
        f"Unknown judge '{judge}'. Built-ins: {list(BUILTIN_JUDGES)}; "
        "or pass a flakyjudge.JudgeSpec."
    )


async def _cached_call(client, cache, spec, system, prompt, max_tokens,
                       temperature=0.0):
    key = RequestKey(
        provider=spec.provider, model=spec.model, system=system, prompt=prompt,
        temperature=temperature if spec.supports_temperature else None,
        max_tokens=max_tokens, logprobs=spec.supports_logprobs,
    )
    raw = cache.get(key)
    if raw is None:
        cache.check_budget()
        raw = await call_judge(client, spec, system, prompt,
                               temperature=key.temperature,
                               max_tokens=max_tokens,
                               logprobs=spec.supports_logprobs)
        norm = normalize(spec.provider, raw)
        cache.put(key, raw, norm.input_tokens, norm.output_tokens,
                  compute_cost(spec, norm.input_tokens, norm.output_tokens))
    return normalize(spec.provider, raw)


async def generate_paraphrases(
    client, cache, generator: JudgeSpec, unit_test: str, n: int
) -> list[str]:
    prompt = (f"Original criterion:\n{unit_test}\n\n"
              f"Produce {n + 2} distinct rewrites. JSON array only.")
    norm = await _cached_call(client, cache, generator, PARAPHRASE_SYSTEM,
                              prompt, max_tokens=800, temperature=1.0)
    text = re.sub(r"^```(json)?|```$", "", norm.text.strip(),
                  flags=re.MULTILINE).strip()
    try:
        candidates = json.loads(text)
    except json.JSONDecodeError:
        return []
    accepted = [c.strip() for c in candidates
                if isinstance(c, str) and c.strip()
                and polarity_ok(unit_test, c) and length_ok(unit_test, c)]
    return accepted[:n]


async def ensemble_score_async(
    query: str,
    response: str,
    unit_test: str,
    judge: str | JudgeSpec = "gpt-4o-mini",
    n_paraphrases: int = 4,
    threshold: float = PASS_THRESHOLD,
    generator: str | JudgeSpec | None = None,
    cache_path: str | Path | None = None,
) -> EnsembleResult:
    spec = _resolve_judge(judge)
    generator_spec = (
        _resolve_judge(generator) if generator is not None
        else (spec if spec.provider == "openai-compat"
              else BUILTIN_JUDGES["gpt-4o-mini"])
    )
    cache = ResponseCache(cache_path or DEFAULT_CACHE)
    try:
        async with httpx.AsyncClient() as client:
            paraphrases = await generate_paraphrases(
                client, cache, generator_spec, unit_test, n_paraphrases)
            wordings = [("original", unit_test)] + [
                ("paraphrase", p) for p in paraphrases]

            async def score_one(kind, wording):
                prompt = build_prompt(query, response, wording)
                norm = await _cached_call(client, cache, spec,
                                          JUDGE_SYSTEM_PROMPT, prompt,
                                          max_tokens=8)
                direct, logprob = extract_scores(norm)
                score = logprob if logprob is not None else direct
                return VariantScore(
                    unit_test=wording, kind=kind, score=score,
                    passed=None if score is None else score > threshold)

            variants = list(await asyncio.gather(
                *(score_one(k, w) for k, w in wordings)))
    finally:
        cache.close()

    scored = [v for v in variants if v.score is not None]
    if not scored:
        raise RuntimeError("No wording produced a parseable score.")
    scores = [v.score for v in scored]
    mean = sum(scores) / len(scores)
    sd = (sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)) ** 0.5
    verdict = "pass" if mean > threshold else "fail"
    majority_pass = mean > threshold
    flips = sum(1 for v in scored if v.passed != majority_pass)
    return EnsembleResult(
        score=mean, sd=sd, verdict=verdict,
        stable=flips == 0,
        flip_fraction=flips / len(scored),
        margin=abs(mean - threshold),
        threshold=threshold,
        variants=variants,
    )


def ensemble_score(*args, **kwargs) -> EnsembleResult:
    """Synchronous wrapper around ensemble_score_async()."""
    return asyncio.run(ensemble_score_async(*args, **kwargs))
