# How Stable Are Natural-Language Unit Tests for LLM Evaluation?

**Shiva Sankeerth Reddy Yarradla** · July 2026 · [github.com/ShivaSankeerth/flakyjudge](https://github.com/ShivaSankeerth/flakyjudge)

## Abstract

Natural-language unit tests — discrete assertions like *"Does the response
cite the refund window?"* scored by a judge model — are an increasingly
popular interface for evaluating LLM outputs (LMUnit, TICK, checklist evals).
The paradigm's implicit promise is that the criterion's *meaning*, not its
*wording*, determines the score. We test that promise with six judges in
four model families on FLASK items (n = 68–78 complete-matrix items per
judge, drawn from a 150-item preregistered sample). **Rewording a criterion
flips the pass/fail verdict on 13–25% of items** (Wilson 95% CIs roughly
±8–11 points; judge-vs-judge differences are not distinguishable at this n).
Under matched-k definitions this is **4–18× each judge's identical-input
resampling flip rate** — and one judge (gemini flash-lite) is perfectly
deterministic under T=0 resampling (0 flips in 500 repeat pairs) yet still
flips 12.8% of verdicts under rewording, cleanly separating wording
sensitivity from sampling noise. Flips concentrate on items near the decision
threshold (29–40% vs 4–13% on clear verdicts, a 2–7× concentration):
criterion wording acts as a hidden decision threshold. Deliberately
meaning-changed control criteria flip 3.6–6.4× more than true paraphrases
for all five large judges — and expose that the apparently most "stable"
judge (llama-3.1-8b, control/paraphrase ratio 1.1) is stable because it
barely reads the criterion at all. In contrast, we observe **no verbosity
bias that survives multiple-comparison correction in any family**:
with Holm correction applied symmetrically, equivalence within ±0.25
points is established in 2 of 12 judge×condition cells (8 of 12
uncorrected), the remaining cells inconclusive, and no drift in any cell
survives correction (smallest corrected p = 0.07 — gemini's *negative*
drift under padding, a padding penalty, not a halo). A
scoring-mode ablation shows logprob-weighted expectation scoring barely
moves GPT-4o-class judges (Δρ +0.01–0.02, CIs crossing zero) but transforms
the small open judge (llama-8b: Δρ +0.23, CI [+0.17, +0.29]); contrary to
our preregistered hypothesis, k-sample averaging recovers less than half of
that benefit. All results reproduce from a committed API-response cache
with zero API keys (`make cache-unpack && make figures`).

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

Preregistered in [`PREREGISTRATION.md`](../PREREGISTRATION.md) — hypotheses,
sample sizes, and metrics were frozen before data collection; **all
amendments predate the analysis they affect and are documented inline**,
and a dated deviations addendum lists every difference between the plan and
what was realized. Four experiments, all using LMUnit's exact prompt
template and its 2.5 pass/fail threshold [1]:

- **E1 (anchor):** 998 items (FLASK + BiGGen-Bench, seeded stratified
  sample; 998 rather than 1,000 due to proportional-allocation rounding) ×
  2 prompt modes (bare assertion vs. LMUnit-parity rubric+reference).
  Sanity gate: judge–human correlation must land in a plausible band vs.
  published numbers before any perturbation runs.
- **E2 (noise floor):** 100 items × 5 identical-input repeats at T=0 and
  T=1, plus a field-reordering variant. Every perturbation effect is
  reported against this floor.
- **E3 (headline, FLASK only):** 150 items × 6 typed meaning-preserving
  paraphrases of the unit test + 2 **positive controls** per item (negated
  and swapped criteria) that *should* flip. Complete-matrix analysis
  retains 78 items (68 for claude-sonnet; see §5 on the selection this
  induces).
- **E4 (verbosity, FLASK only):** content-matched padded (realized 1.84×)
  and condensed (0.54×) response variants; 82 of 150 pairs per kind
  survived the claim audit, so the realized design detects |dz| ≥ ~0.31.

**Perturbation validity.** Paraphrases pass automated gates (embedding
cosine ≥ 0.80, polarity lexicon, length ratio 0.5–2.0×); borderline
candidates (cosine 0.70–0.80) are admitted via a bidirectional
semantic-equivalence adjudication and tracked as a second tier, with all
headline analyses run both ways (§3.3). The preregistered 10% spot-check
(112 variants) was performed by an LLM reviewer (a Claude model that is
not one of the judges, though it shares a vendor family with two of
them — see §5) with per-row verdicts recorded in
`data/spot_check_sample.jsonl`: 79/83 true paraphrases judged equivalent
(4.8% failure), and all 29 controls in the sample correctly judged
non-equivalent. The dominant failure (a "spatial"→"physical reasoning"
scope shift) affects 22/136 lexical variants population-wide; excluding
them moves headline flip rates by at most 4 points with conclusions
unchanged (§3.3). Verbosity variants pass a strict bidirectional claim
audit; rejects were verified to be genuine content changes.

**Judges.** gpt-4o (2024-11-20), gpt-4o-mini (2024-07-18),
claude-sonnet-4-6, claude-haiku-4-5 (20251001), gemini-flash-lite (rolling
alias — no pinned snapshot exists for this account tier; served model
recorded per call), llama-3.1-8b-instruct (OpenRouter, provider pinned) —
T=0, single frozen system prompt, direct-digit plus logprob-weighted
scores where the API exposes logprobs. Gemini was initially excluded on
free-tier rate limits and re-included after a paid-tier upgrade
(documented amendment); it was added after the spot-check review, whose
reviewer is not any judge under test. Format
compliance is itself a result: claude-sonnet ignores the bare-digit
instruction on ~10% of calls (haiku ~1.5%, OpenAI and Llama ~0%); an
extended-token reissue salvaged 61% of Sonnet's failures, and §3.3 shows
the headline with and without salvage.

## 3. Results

### 3.1 Rewording flips verdicts far above the noise floor (E3 vs E2)

Two flip-rate definitions are used and always labeled: **any-of-k** = the
fraction of items whose verdict is not unanimous across k wordings
(mechanically grows with k, so cross-condition comparisons use matched k);
**per-variant** = mean disagreement between one variant and the original.

| | gpt-4o | gpt-4o-mini | claude-sonnet | claude-haiku | gemini-flash | llama-8b |
|---|---|---|---|---|---|---|
| n (complete matrix) | 78 | 78 | 68 | 78 | 78 | 78 |
| Any-of-7 paraphrase flip [95% CI] | 20.5% [13, 31] | 14.1% [8, 24] | 25.0% [16, 36] | 20.5% [13, 31] | 12.8% [7, 22] | 5.1% [2, 12] |
| Any-of-5 paraphrase flip (matched k) | 16.2% | 10.8% | 20.6% | 16.9% | 11.4% | 4.6% |
| Any-of-5 resample floor (T=0) | 4.0% | 2.0% | 1.1% | 1.0% | 0.0% | 1.0% |
| **Matched-k ratio** | **4.1×** | **5.4×** | **18.3×** | **16.9×** | undefined* | 4.6× |
| Per-variant: paraphrase / floor | 5.6 / 2.5% | 3.9 / 1.2% | 8.2 / 0.8% | 6.2 / 0.8% | 5.3 / 0.0% | 2.2 / 0.8% |

\* gemini produced zero flips across all T=0 resample pairs (floor Wilson
CI [0, 3.7%]) — the ratio is undefined; at the floor CI's upper bound it
would be ≥3.1×. Its 11.4% paraphrase flip rate on a 0.0% measured floor is
the cleanest wording-vs-sampling separation in the study.

At these sample sizes the judge-vs-judge ordering is **not** established —
the CIs overlap heavily; the supported claim is that all five large judges
flip in the 13–25% range, far above their floors.

**Instrument resolution (preregistered validity control #3), evaluated
formally with matched per-variant definitions** — control flip vs
paraphrase flip, both per-variant vs original:

| | gpt-4o | gpt-4o-mini | claude-sonnet | claude-haiku | gemini-flash | llama-8b |
|---|---|---|---|---|---|---|
| Paraphrase (per-variant) | 5.6% | 3.9% | 8.2% | 6.2% | 5.3% | 2.2% |
| Meaning-changed control (per-variant) | 21.3% | 14.0% | 34.6% | 39.5% | 19.3% | 2.3% |
| Ratio | 3.8 | 3.6 | 4.2 | 6.4 | 3.6 | **1.1** |

The pass criterion (control > 1.5× paraphrase) is a post-hoc formalization
of the preregistered "substantially exceeds" — but the pattern is not
subtle: all five large judges clear 3.6×, and llama-8b sits at 1.1×. Its low
paraphrase flip rate reflects insensitivity to the criterion, not
robustness. (Controls also shift raw scores 3–4× more than paraphrases for
the large judges.)

**Mechanism: wording is a hidden threshold.** Flip rates split 29–40% on
near-threshold items (|score − 2.5| ≤ 1) vs 4–13% on clear verdicts — a
2–7× concentration, consistent across families. When a response is
borderline, *which paraphrase you happened to write* decides the eval.
Casual-register rewrites are the most destabilizing single type for
gpt-4o-mini (|Δ| 0.42) and claude-haiku (0.35); form changes
(question/imperative) lead for gpt-4o and claude-sonnet.

**The T=1 noise floor is a finding of its own.** All scoring in E1/E3/E4
ran at temperature 0, so the T=0 floor is the matched baseline above. But
the preregistered T=1 floor deserves its own line: identical-input
resampling at T=1 flips 7–24% of verdicts (gpt-4o 19%, llama 24%) —
rivaling paraphrase sensitivity itself. If you run a judge at nonzero
temperature, sampling noise alone can dominate everything this study
measures. (This also means the "4–18×" multiplier is specific to T=0
operation; at T=1 wording sensitivity and sampling noise are comparable.)

**Field order matters as much as wording (E2b, previously unreported):**
mechanically reordering the prompt fields (Unit Test / Query / Response
instead of Query / Response / Unit Test) flips 7–14% of verdicts and moves
scores |Δ| = 0.27–0.51 — format sensitivity on par with paraphrase
sensitivity, echoing [2]'s response-level finding.

### 3.2 No detectable verbosity bias under criterion anchoring (E4)

Per-judge paired drifts (variant − original; per-cell n = 71–82 after the
claim audit), with 95% CIs and Holm step-down correction applied
symmetrically to BOTH twelve-test families — the Wilcoxon difference tests
and the TOST equivalence tests (±0.25-point bound, chosen post-hoc; the
preregistered design detects ~0.16 at n=150, the realized n detects
~0.22–0.24):

| drift | gpt-4o | gpt-4o-mini | claude-sonnet | claude-haiku | gemini-flash | llama-8b |
|---|---|---|---|---|---|---|
| Padded 1.84× | +0.01 [−0.20, +0.23] | −0.13 [−0.40, +0.13] | −0.12 [−0.25, +0.01] | −0.11 [−0.32, +0.10] | −0.27 [−0.46, −0.08] | +0.09 [−0.04, +0.21] |
| Condensed 0.54× | +0.23 [+0.04, +0.43] | −0.05 [−0.18, +0.08]ᴱ | −0.11 [−0.24, +0.02] | +0.05 [−0.17, +0.26] | +0.05 [−0.17, +0.26] | +0.00 [−0.11, +0.11]ᴱ |

ᴱ = equivalence within ±0.25 established after Holm correction (2 of 12
cells; 8 of 12 before correction). **No drift in any cell survives Holm
correction either** (smallest corrected p = 0.071 — and that near-miss is
gemini's *negative* padded drift: the only marginal length effect in the
study is a padding penalty, not a halo). The
honest summary: we detect no verbosity bias in any family, but with
corrected equivalence established in only 3 cells, most cells are
*inconclusive* — consistent with no bias, insufficient to prove its
absence. A larger-n replication is the fix (the audit-surviving pool, not
API cost, is the binding constraint). The preregistered H2 (drift correlates
with log length ratio, direction family-specific) is **not supported**:
elasticity slopes are −0.16 to +0.06, all p > 0.14. Because this is a
cross-paper comparison (we did not run a holistic-judging arm), the claim
is *"no detectable length halo in this criterion-anchored setting"* — not
that decomposition causally removes it. The gpt-4o concision effect
(+0.23, uncorrected p = .03) is suggestive at best.

### 3.3 Robustness and supplementary analyses

All numbers regenerable via `analysis/supplementary.py` →
`results/supplementary.json`.

- **Two-tier paraphrase gate:** strict-only vs strict+adjudicated any-of-7
  flip rates agree within 2.3 points for every judge (e.g. gpt-4o 20.8% vs
  20.5%; claude-sonnet 22.7% vs 25.0%).
- **Salvage sensitivity:** claude-sonnet 25.0% without vs 24.6% with
  extended-token salvage; claude-haiku unchanged (20.5%). No conclusion
  depends on the rescue tier.
- **Spot-check drift exclusion:** removing the 22 lexical variants with the
  "spatial→physical" scope shift moves flip rates ≤4 points (e.g. haiku
  20.5%→16.1%, gpt-4o 20.5%→21.4%); conclusions unchanged.
- **Logprob scoring does not reduce flip rates for large judges** (gpt-4o
  20.5%→19.2%, mini 14.1%→15.4%) — H3's flip-rate prediction is not
  supported there. llama-8b's logprob flips drop to 0% because expectation
  scoring compresses its scores away from the threshold — stability again
  without evidence of criterion sensitivity.
- **k-sample averaging (H3) is refuted in its strong form:** on the E2
  items, mean-of-5 sampling lifts llama-8b's ρ vs human from 0.12 to 0.26,
  less than half of what logprob weighting achieves (0.43); for the four
  large judges mean-of-5 changes ρ by <0.02.
- **Variance decomposition and ICC:** within-item wording variance is
  0.10–0.27 (per-item SD ≈ 0.32–0.52 points) vs a T=0 sampling floor of
  0.003–0.054. The preregistered ICC(2,1) across paraphrases is 0.74–0.91
  (resampling: 0.97–0.99) — absolute scores move modestly under
  rewording; verdicts flip as often as they do because so many items sit
  near the 2.5 threshold. "Flaky verdicts atop fairly stable scores" is
  the accurate summary, and it is exactly why margin reporting (§4)
  matters more than the flip rate alone.

### 3.4 Anchors and scoring-mode ablations (E1)

Judge–human Spearman ρ: 0.54–0.70 for the five large judges (best:
claude-sonnet 0.69–0.70 on FLASK) vs llama-8b's 0.06–0.23 direct-scored.
For context, FLASK's leave-one-annotator-out agreement is ρ ≈ 0.56 — note
this is a *different statistic* (single annotator vs the other two) than
judge-vs-consensus (a less noisy target), so judges exceeding 0.56 are not
"better than human"; the number contextualizes how noisy the gold labels
are. Sanity gate passed for all large judges. Notably, the most
human-aligned judge (sonnet) is also among the most paraphrase-sensitive
and the least format-compliant: validity and stability do not come
together. Secondary findings:

- **Logprob-weighted scoring is a small-model story:** Δρ = +0.013
  (gpt-4o) / +0.024 (gpt-4o-mini), CIs crossing zero — but **+0.232**
  (CI [+0.169, +0.295]) for llama-8b, lifting it from unusable (ρ≈0.13)
  to usable (ρ≈0.43). It does not, however, reduce paraphrase flip rates
  for the large judges (§3.3).
- **Rubrics make every judge harsher, but improve validity only for one:**
  appending the FLASK rubric+reference shifts scores down for all six
  judges (−0.11 to −0.36) while the correlation gain is
  ~zero for the OpenAI judges (Δρ +0.001 / +0.034, CIs crossing zero),
  marginal for claude-haiku (+0.039 [−0.001, +0.080]), and real for
  claude-sonnet (+0.060 [+0.027, +0.095]). "Rubrics don't help" is
  judge-dependent; budget them for judges that use them.

## 4. Practical guidance

1. **Report threshold margins, not just verdicts.** Flips concentrate 2–8×
   on near-threshold items; a verdict with |score − threshold| > 1 is far
   more trustworthy. Treat borderline passes as borderline.
2. **Make wording sensitivity visible for load-bearing criteria.** Scoring
   k paraphrases (the `flakyjudge.ensemble_score()` API implements this)
   reveals when a verdict depends on phrasing. Note: whether ensembling
   *reduces* error vs humans is untested here — what it demonstrably does
   is surface instability that single-shot scoring hides.
3. **Don't buy stability without checking sensitivity.** Always run
   meaning-changed controls: the "most stable" judge in this study was the
   one not reading the criterion.
4. **If you must use a small judge, use logprob scoring, not repeat
   sampling** — averaging 5 samples recovered less than half of the
   logprob benefit.
5. **Run judges at temperature 0.** At T=1, identical-input sampling noise
   alone flips 10–24% of verdicts — as large as the wording effect this
   study is about.
6. **Don't pay for rubrics reflexively** — they made every judge harsher
   but improved human agreement only for claude-sonnet here.

## 5. Limitations

Six judges across four families (gemini via a rolling alias with no
pinned snapshot — served model recorded per call); FLASK-only for the headline experiments (E1 also covers
BiGGen-Bench); FLASK items and human labels may be in judges' training
data (the perturbation experiments are contamination-resistant by
construction, the E1 correlations less so). n = 68–78 complete-matrix
items for the headline — half the preregistered 150, so the prereg's power
targets were not met, and CIs are wide (±8–11 points). Matrix completeness
is partly conditioned on claude-sonnet's own format compliance, a
selection effect whose direction is unknown (though the salvage
sensitivity bounds it). Paraphrase generation, the borderline-gate
adjudication, and the spot-check all use LLMs; the adjudicator
(gpt-4o-mini) is itself a judge under test — a circularity affecting 96 of
1,200 variants, bounded by the strict-only sensitivity analysis; the
spot-check reviewer shares a vendor family with two judges (Claude), so
the validity chain is LLM-on-LLM throughout; the spot-check found a 4.8%
paraphrase failure rate whose dominant mode was excluded in sensitivity
analysis. E4's 82/150 audit survivors are a
selected subpopulation (responses paddable without adding claims). Human
"gold" itself has ρ ≈ 0.56 single-annotator agreement. Single frozen
prompt per judge: absolute levels could shift under prompt tuning, though
the within-judge contrasts are what the claims rest on.

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
