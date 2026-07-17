# How Stable Are Natural-Language Unit Tests for LLM Evaluation?

**Shiva Sankeerth Reddy Yarradla** · July 2026 · [github.com/ShivaSankeerth/flakyjudge](https://github.com/ShivaSankeerth/flakyjudge)

## Abstract

Natural-language unit tests — discrete assertions like *"Does the response
cite the refund window?"* scored by a judge model — are an increasingly
popular interface for evaluating LLM outputs (LMUnit, TICK, checklist evals).
The paradigm's implicit promise is that the criterion's *meaning*, not its
*wording*, determines the score. We test that promise. Across 150 FLASK items
with six typed, validity-gated paraphrases per unit test and five judges in
three model families, **rewording a criterion flips the pass/fail verdict
on 14–25% of items** (claude-sonnet 25.0%, gpt-4o and claude-haiku 20.5%,
gpt-4o-mini 14.1%) — 5–20× each judge's identical-input resampling noise
floor (0.8–4.0%). Flips concentrate ~3–7× on items whose scores sit near
the decision threshold: criterion wording acts as a hidden decision
threshold. The positive controls expose a subtler failure: llama-3.1-8b
appears most "stable" (5.1% flips) while barely reacting to deliberately
meaning-changed criteria (1–3% vs Claude's 47–54%) — stability without
criterion sensitivity is just not reading the test. In contrast, the well-documented verbosity bias of
holistic and pairwise judging **largely disappears** under criterion-anchored
judging in every family: content-matched 1.8× padding produces no
significant positive drift for any judge; gpt-4o *rewards concision*
(+0.23 for 0.54× condensed variants, p = 0.03) and Claude trends mildly
negative in both directions. A scoring-mode ablation adds a practical
result: logprob-weighted expectation scoring barely helps GPT-4o-class
judges (Δρ +0.01–0.02, CIs crossing zero) but transforms the small open
judge (llama-8b: Δρ +0.23, CI [+0.17, +0.29], from ρ≈0.13 to ρ≈0.43).
Together: decomposed judging shields judges from the length halo but
introduces a wording-sensitivity failure mode that practitioners should
mitigate — we discuss paraphrase-ensemble scoring and threshold-margin
reporting. All results reproduce from a committed API-response cache with
zero API keys (`make figures`).

## 1. Motivation

Teams increasingly evaluate LLM systems with fine-grained, criteria-based
judging rather than holistic 1–10 scores: LMUnit trains judges to score
(query, response, unit test) triples [1]; TICK [5] and RocketEval [6] use
generated checklists; production eval suites embed such assertions in CI.
Judge *robustness*, meanwhile, has been studied at the **response** level
(perturbing the judged text; Judge Reliability Harness [2]) and the
**template** level (perturbing the scoring prompt scaffold [3]) — but not at
the **criterion** level: if two engineers write the same assertion in
different words, do they get the same eval? Since free-form criterion
authorship is the paradigm's core interface, this is its load-bearing
assumption, and [2] explicitly names it as unmeasured. We measure it.

## 2. Study design

Preregistered in [`PREREGISTRATION.md`](../PREREGISTRATION.md) (hypotheses,
sample sizes, metrics frozen before data collection; all amendments are
pre-scoring and documented inline). Four experiments, all using LMUnit's
exact prompt template and its 2.5 pass/fail threshold [1] so results are
comparable to the paper's:

- **E1 (anchor):** 998 items (FLASK + BiGGen-Bench, seeded stratified
  sample) × 2 prompt modes (bare assertion vs. LMUnit-parity
  rubric+reference). Sanity gate: judge–human correlation must land in a
  plausible band vs. published numbers before any perturbation runs.
- **E2 (noise floor):** 100 items × 5 identical-input repeats at T=0 and
  T=1, plus a field-reordering variant. Every perturbation effect is
  reported as **excess over this floor** — without it, paraphrase variance
  is uninterpretable.
- **E3 (headline):** 150 items × 6 typed meaning-preserving paraphrases of
  the unit test (lexical, syntactic, formal/casual register,
  question/imperative form) + 2 **positive controls** per item (negated and
  swapped criteria) that *should* flip — proving the instrument
  distinguishes rewording from real semantic change.
- **E4 (verbosity):** content-matched padded (realized 1.84×) and condensed
  (0.54×) response variants, scored against the original criterion.

**Perturbation validity.** Paraphrases pass automated gates (embedding
cosine ≥ 0.80, polarity lexicon check, length ratio 0.5–2.0×); borderline
candidates (cosine 0.70–0.80) are admitted only via a bidirectional
semantic-equivalence adjudication and tracked as a second tier — all E3
analyses run strict-only and combined (they agree: 20.8% vs 20.5% flip). A
10% manual sample was reviewed by the author with no meaning changes
flagged. Verbosity variants pass a strict bidirectional claim audit (nothing
added, nothing dropped); only 82/150 per kind survived, and rejects were
verified to be genuine content changes — the surviving pairs isolate length.

**Judges.** gpt-4o (2024-11-20), gpt-4o-mini (2024-07-18),
claude-sonnet-4-6, claude-haiku-4-5 (20251001), llama-3.1-8b-instruct
(OpenRouter, provider pinned) — T=0, single frozen system prompt,
direct-digit plus logprob-weighted scores where the API exposes logprobs.
Gemini was excluded: Google's free tier for new accounts allows 15
requests/minute on the only accessible alias (documented amendment).
Format compliance is itself a result: claude-sonnet ignores the bare-digit
instruction on ~10% of calls (haiku ~1.5%, OpenAI and Llama ~0%); an
extended-token reissue salvaged 61% of Sonnet's failures, and all Claude
analyses run with and without salvage.

## 3. Results

### 3.1 Rewording flips verdicts far above the noise floor (E3 vs E2)

| | gpt-4o | gpt-4o-mini | claude-sonnet | claude-haiku | llama-8b |
|---|---|---|---|---|---|
| Identical-input flip rate (T=0) | 4.0% | 2.0% | 1.1% | 1.0% | 1.0% |
| Paraphrase flip rate (7 wordings) | **20.5%** | **14.1%** | **25.0%** | **20.5%** | 5.1%* |
| Excess over floor | +16.5 | +12.1 | +23.9 | +19.5 | +4.1 |
| Meaning-changed control flip (vs orig.) | 21% | 14% | 35% | 39% | 2%* |
| ICC(2,1): resamples → paraphrases | 0.98→0.91 | 0.98→0.88 | 0.99→0.89 | 0.97→0.86 | 0.99→0.74 |

\* llama-8b fails the instrument-resolution check: its control flip rate
is *below* its paraphrase flip rate — its apparent stability reflects
insensitivity to the criterion, not robustness. Its per-item score SD
still triples under paraphrase.

Per-variant verdict disagreement with the original wording: 0.8–2.5%
(resampled identical input), 8.8–12.1% (meaning-preserving paraphrase),
14–39% (meaning-*changed* controls, real judges). Controls shift scores
3–4× more than true paraphrases for every judge that passes the resolution
check.

**Mechanism: wording is a hidden threshold.** Flip rates split 29–40% on
near-threshold items (|score − 2.5| ≤ 1) vs 5–13% on clear verdicts,
consistently across families. When a response is borderline, *which
paraphrase you happened to write* decides the eval. Casual-register
rewrites are the most destabilizing single type for gpt-4o-mini (|Δ| 0.42)
and claude-haiku (0.35); form changes (question/imperative) lead for
gpt-4o and claude-sonnet.

### 3.2 Verbosity bias largely disappears under criterion anchoring (E4)

| drift vs original | gpt-4o | gpt-4o-mini | claude-sonnet | claude-haiku | llama-8b |
|---|---|---|---|---|---|
| Padded 1.84× | +0.01 | −0.13 | −0.12 | −0.11 | +0.09 |
| Condensed 0.54× | **+0.23 (p=.03)** | −0.05 | −0.11 | +0.05 | +0.00 |
| Elasticity (Δ/log ratio) | −0.16 | −0.08 | −0.02 | −0.10 | +0.06 |

No judge in any family shows significant positive length drift. The length
halo documented for holistic and pairwise judging [4, 7] does not appear;
gpt-4o significantly *rewards concision*, and prior reports of Claude's
concision preference [7] shrink to non-significance here.
This supports the decomposition hypothesis — a narrow criterion appears to
shield the judge from global length cues — and means the paradigm's
trade-off is real: **unit-test judging fixes verbosity bias and introduces
wording sensitivity.**

### 3.3 Anchors and scoring-mode ablations (E1)

Judge–human Spearman ρ: 0.54–0.70 for the four large judges (best:
claude-sonnet at 0.69–0.70 on FLASK, near the 0.56 leave-one-annotator-out
human ceiling), against llama-8b's 0.06–0.23 direct-scored — sanity gate
passed for all large judges. Notably, the most human-aligned judge
(sonnet) is also the most paraphrase-flippy and least format-compliant:
validity and stability do not come together. Two secondary findings:

- **Logprob-weighted scoring is a small-model story:** Δρ = +0.013
  (gpt-4o) / +0.024 (gpt-4o-mini), CIs crossing zero — but **+0.232**
  (CI [+0.169, +0.295]) for llama-8b, lifting it from unusable (ρ≈0.13)
  to usable (ρ≈0.43). G-Eval-style expectation scoring [8] is what makes
  small judges viable, and is nearly free where logprobs are exposed.
- **Rubrics make judges harsher, not more valid:** appending FLASK's rubric
  and reference answer shifts scores −0.25 to −0.29 (CIs exclude zero) while
  buying ~0 correlation gain (Δρ +0.001 / +0.034, CIs crossing zero).

## 4. Practical guidance

1. **Report threshold margins, not just verdicts.** Most flips are
   near-threshold; a verdict with |score − threshold| > 1 is ~3× more
   stable. Treat borderline passes as borderline.
2. **Ensemble over wordings for load-bearing criteria.** Scoring k
   paraphrases and averaging directly attacks the dominant variance
   component (wording, not sampling). At 5 paraphrases this costs 5× — use
   it for gating evals, not exploratory ones.
3. **Prefer cheap judges with repeats over expensive judges one-shot for
   stability:** gpt-4o-mini's paraphrase flip rate (14.1%) is lower than
   gpt-4o's (20.5%) in our data, and its noise floor is half — though its
   human correlation is also lower; pick by whether validity or stability
   binds.
4. **Don't pay for rubrics reflexively** — in these datasets they shifted
   the operating point without improving agreement with humans.

## 5. Limitations

Five judges across three families (Gemini excluded by free-tier rate
limits — documented amendment); FLASK items and
human labels may be in judges' training data (perturbation experiments are
contamination-resistant by construction, the E1 correlations less so);
paraphrase generation and claim auditing use LLMs (mitigated by typed gates,
positive controls, tier sensitivity analysis, and manual review, but not
eliminated); human "ground truth" itself has a ceiling of ρ ≈ 0.56; n = 78
complete-matrix items for the headline (CIs in
[`analysis/`](../analysis)); single frozen prompt per judge — prompt-tuning
could shift absolute levels, though the *within-judge* contrasts are what
the claims rest on.

## References

[1] Saad-Falcon et al., *LMUnit: Fine-grained Evaluation with Natural
Language Unit Tests*, arXiv:2412.13091.
[2] *Judge Reliability Harness*, arXiv:2603.05399.
[3] *All Prompts Are Created Equal?*, ACL 2026 Findings.
[4] Zheng et al., *Judging LLM-as-a-Judge with MT-Bench*, NeurIPS 2023.
[5] *TICK: Targeted Instruct-evaluation with ChecKlists*, arXiv:2410.03608.
[6] *RocketEval*, ICLR 2025, arXiv:2503.05142.
[7] *Reliability without Validity*, arXiv:2606.19544.
[8] Liu et al., *G-Eval*, EMNLP 2023.
