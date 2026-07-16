# Pre-registration

Committed before any experiment runs. Hypotheses, designs, sample sizes, and
metrics below are frozen; any deviation will be reported as such in the final
report. Judge prompt (`src/flakyjudge/prompts.py`) and judge lineup
(`config/judges.yaml`) are frozen as of this commit.

## Research questions and hypotheses

**RQ1 (headline): How stable is a natural-language unit test under paraphrase?**
Same (query, response); the unit-test criterion is reworded in 6
meaning-preserving ways. H1: paraphrase variance exceeds the identical-input
resampling noise floor; flip rates concentrate on items whose scores are near
the decision threshold and on register/compression paraphrase types.

**RQ2: Does verbosity bias survive criterion-anchored judging?**
Content-preserving padded (~1.8x) and condensed (~0.6x) response variants.
H2: score drift correlates with realized log length ratio; direction is
judge-family-specific (per Reliability without Validity, arXiv:2606.19544).

**RQ3: Does distribution-aware scoring improve reliability?**
Logprob-weighted expectation (where available) and k-sample averaging vs.
single direct digit. H3: distribution-aware scores have higher test-retest
reliability and lower paraphrase flip rates; k-sample averaging recovers most
of the logprob benefit on providers without logprobs.

**Anchor (context, not headline): judge-human correlation.**
Spearman rho vs. FLASK/BiGGen-Bench human scores per judge, interpreted
against the human-human agreement ceiling reported by those datasets. Serves
as the sanity gate: judges must land in a plausible band relative to LMUnit's
published numbers before any perturbation experiment is run.

**Explicitly out of scope:** self-preference, pairwise judging,
rationale-mode ablations, non-English robustness, multi-turn/RAG evaluation.

## Designs and sample sizes

**Prompt modes (amendment, pre-data-collection):** E1 runs in two prompt
modes: (a) LMUnit-parity — rubric and reference answer appended to the unit
test, matching the paper's FLASK/BiGGen setup, used for the sanity gate; and
(b) bare-assertion — unit test only, the deployed-practice setting. E2-E4 use
bare-assertion mode: paraphrase sensitivity is measured on the criterion
practitioners actually write, unanchored by a rubric. Comparing modes in E1
additionally quantifies what the rubric buys.

| Exp | Design | n |
|---|---|---|
| E1 anchor | seeded stratified sample, both scoring modes from one call, 2 prompt modes | 500 FLASK + 500 BiGGen x 6 judges x 2 modes |
| E2 noise floor | identical-input resamples + field-reorder variant | 100 items x 5 repeats x 6 judges; 100 x 2 orderings x 6 |
| E3 paraphrase | typed paraphrases (lexical/syntactic/register/form) + positive controls | 150 FLASK items (50 low / 50 mid / 50 high human score) x 6 paraphrases x 6 judges |
| E4 verbosity | NLI-verified matched-content pairs | 150 items x 2 variants x 6 judges |

Power: E1 paired bootstrap over items resolves judge-vs-judge and
logprob-vs-direct differences of roughly delta-rho 0.04-0.05. E3 at n=150,
k=6 gives ICC(2,1) CI half-width ~0.05-0.06. E4 paired design detects
standardized drift dz >= 0.23 at 80% power (~0.16 points on the 1-5 scale).

## Validity controls (frozen)

1. E2 runs before E3/E4; all perturbation effects are reported as excess over
   the identical-input noise floor.
2. Paraphrases pass automatic gates (embedding cosine >= 0.80, polarity rule
   check, length 0.5-2.0x) plus a 10% manual spot-check; gate pass rates are
   reported.
3. E3 includes deliberately meaning-changing positive controls; the
   instrument is valid only if their flip rate substantially exceeds the true
   paraphrase flip rate.
4. Verbosity variants pass bidirectional NLI entailment and an LLM claim
   audit; analysis regresses drift on realized (not nominal) length ratio.
5. Parse failures are recorded as missing and reported per judge, never
   imputed.
6. Model snapshot IDs pinned in `config/judges.yaml`; the `model` field of
   every raw response is stored as provenance; each judge's calls run within
   a tight time window.

**Gate calibration amendment (pre-E3-scoring, post-generation QA):** the
polarity gate's negator lexicon falsely flagged meaning-preserving phrasings
("free from", "steers clear") and was expanded; regeneration attempts raised
from 3 to 5. The cosine >= 0.80 and length gates are unchanged. No judge had
scored any paraphrase when this change was made; first-pass gate failure
rates (223 cosine, 73 polarity of 1,200) are preserved in
data/paraphrase_gate_report.json history.

**Two-tier gate amendment (pre-E3-scoring):** the strict cosine gate
systematically rejects lexical and casual-register paraphrases (synonym
swaps move embeddings even when meaning is preserved), a selection bias
TOWARD finding stability. Candidates with cosine in [0.70, 0.80) passing
polarity+length are admitted if a bidirectional semantic-equivalence check
(gpt-4o-mini, cached) answers equivalent, labeled gate_tier='adjudicated'
(96 of 1,200 variants). All E3 analyses are run twice — strict-only and
strict+adjudicated — as a sensitivity analysis.

**Spot-check completion note (2026-07-16):** the preregistered 10% manual
review (data/spot_check_sample.jsonl, 112 variants) was completed by the
author; no meaning-changing paraphrases were flagged among the admitted
variants.

## Primary metrics

- Decision flip rate at the 2.5 pass/fail threshold (LMUnit's), reported as
  excess over noise floor — the headline number.
- ICC(2,1) across paraphrases/resamples; per-item SD distributions; variance
  decomposition (item / paraphrase / residual).
- Verbosity: paired mean drift, Cohen's dz, slope vs. log length ratio.
- Correlation: Spearman rho vs. human scores, paired bootstrap CIs (10,000
  resamples) for all within-judge and between-judge comparisons.

## Budget

Hard cap $50 (enforced in code via FLAKYJUDGE_MAX_SPEND_USD); estimated
~19,500 calls, ~$23 raw.
