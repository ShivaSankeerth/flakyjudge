# flakyjudge

[![PyPI](https://img.shields.io/pypi/v/flakyjudge)](https://pypi.org/project/flakyjudge/) [![CI](https://github.com/ShivaSankeerth/flakyjudge/actions/workflows/ci.yml/badge.svg)](https://github.com/ShivaSankeerth/flakyjudge/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/ShivaSankeerth/flakyjudge/blob/main/LICENSE)

**How stable are natural-language unit tests for LLM evals?**

Teams increasingly evaluate LLM outputs with natural-language unit tests —
discrete assertions like *"Does the response cite the refund window?"* scored
by a judge model (LMUnit, TICK, checklist evals). This project measures the
reliability of that paradigm: if you reword the assertion without changing
its meaning, does the verdict change?

![Rewording a unit test flips 16 of 78 verdicts (gpt-4o)](https://raw.githubusercontent.com/ShivaSankeerth/flakyjudge/main/figures/fig1_hero_flips.png)

**📄 Tech report: [report/report.md](https://github.com/ShivaSankeerth/flakyjudge/blob/main/report/report.md)** — methods, all
results, practical guidance, limitations.

> Experimental design frozen in [PREREGISTRATION.md](https://github.com/ShivaSankeerth/flakyjudge/blob/main/PREREGISTRATION.md).
> Five judges across three families: gpt-4o, gpt-4o-mini, claude-sonnet-4-6,
> claude-haiku-4-5, llama-3.1-8b. ~24,000 scored triples, $19 total spend.

## Findings

- **Rewording a unit test flips its pass/fail verdict on 14–25% of items**
  (n=68–78 per judge; Wilson 95% CIs ±8–11 points, so the judge-vs-judge
  ordering is not established) — **4–18× each judge's identical-input
  resampling flip rate** under matched-k definitions (T=0; at T=1,
  sampling noise alone rivals the wording effect).
- **Criterion wording acts as a hidden decision threshold:** flips
  concentrate ~2–7× on items whose scores sit near the pass/fail cut
  (29–40%) vs clear verdicts (4–13%).

  ![Flip rates vs noise floor and controls](https://raw.githubusercontent.com/ShivaSankeerth/flakyjudge/main/figures/fig2_flip_rates.png)
- **"Stability" without validity is insensitivity — the controls catch it:**
  llama-8b looks most stable under paraphrase, but its meaning-*changed*
  control flip rate is only 1.1× its paraphrase rate (the four large
  judges: 3.6–6.4×) — it isn't robust, it isn't reading the criterion.

  ![Which rewordings move the score](https://raw.githubusercontent.com/ShivaSankeerth/flakyjudge/main/figures/fig3_mechanism.png)
- **No detectable verbosity bias in any family:** content-matched 1.8×
  padding produces no drift that survives multiple-comparison correction;
  with the same correction applied to the equivalence tests, formal
  equivalence within ±0.25 points holds in 3 of 10 judge×condition cells
  (7 of 10 uncorrected) — the rest are inconclusive, not positive. The
  length halo documented for holistic/pairwise judging does not appear in
  this criterion-anchored setting (no within-study holistic arm, so no
  causal claim).
- **Field order rivals wording:** mechanically reordering the prompt
  fields flips 7–14% of verdicts — format sensitivity on par with
  paraphrase sensitivity.
- **The logprob-weighted scoring trick is a small-model story:** it buys
  +0.01–0.02 Spearman for GPT-4o-class judges (CIs cross zero) but
  **+0.23** for llama-8b (CI [+0.17, +0.29]) — rescuing it from ρ≈0.13 to
  ρ≈0.43. Repeat-sampling recovers less than half of that benefit, and
  logprob scoring does not reduce large-judge flip rates.
- **Validity and stability don't come together:** claude-sonnet is the
  most human-aligned judge (ρ=0.69–0.70; for context, single-annotator
  agreement on FLASK is ρ≈0.56 — a different statistic, not a ceiling
  judges "beat") and also among the most paraphrase-sensitive — and the
  least format-compliant (~10% of calls ignore the bare-digit
  instruction; OpenAI judges: 0%).
- **Rubrics make every judge harsher (−0.11 to −0.36) but improve
  validity only for claude-sonnet** (Δρ +0.06, CI excludes zero; ~zero
  for the OpenAI judges).

## What this is

An empirical robustness study across 5 judge models in 3 families (GPT-4o,
GPT-4o-mini, Claude Sonnet 4.6, Claude Haiku 4.5, Llama-3.1-8B) on FLASK
and BiGGen-Bench items, plus a small library implementing the measurement
tools and mitigations:

- **E1** — judge-human correlation anchor; logprob-weighted vs. direct scoring
- **E2** — identical-input noise floor (resampling + field-order stability)
- **E3** — *headline:* paraphrase sensitivity of the unit-test criterion itself
- **E4** — verbosity bias under criterion-anchored judging

Related work measures judge robustness at the *response* level
([Judge Reliability Harness](https://arxiv.org/abs/2603.05399)) and *template*
level; this study targets the *criterion* level — the load-bearing assumption
of the unit-test paradigm ([LMUnit](https://arxiv.org/abs/2412.13091)).

## Use the mitigation: `pip install flakyjudge`

Our headline finding is that single-shot verdicts silently depend on how
you worded the assertion. `ensemble_score()` makes that instability
visible: it scores your criterion plus n auto-generated meaning-preserving
rewordings and reports the spread. (Whether ensembling also improves
agreement with humans is untested here — what it demonstrably does is
surface wording-dependence that single-shot scoring hides.)

```python
from flakyjudge import ensemble_score

r = ensemble_score(
    query="What is your refund policy?",
    response="You can return items within 30 days for a full refund.",
    unit_test="Does the response state the refund time window?",
    judge="gpt-4o-mini",        # or gpt-4o / claude-sonnet / claude-haiku,
    n_paraphrases=4,            #    or any flakyjudge.JudgeSpec
)
r.score    # 5.00 (mean across wordings; logprob-weighted where available)
r.stable   # True -> no wording flips the verdict
r.margin   # 2.50 (distance from threshold; <1 means borderline: distrust it)
```

Use it for gating/CI evals where a silent wording flip matters; responses
are cached in ~/.flakyjudge/ so repeat runs are free.

## Reproducibility

Every raw API response (including logprobs) is committed in a
content-addressed cache, so all results and figures regenerate with **zero
API keys and zero dollars**:

```bash
git clone https://github.com/ShivaSankeerth/flakyjudge && cd flakyjudge
uv sync
make cache-unpack   # inflate the committed API-response cache (19MB gz)
make figures
```

## Development

```bash
uv sync --extra dev
make test    # pytest: metrics vs published worked examples, cache, providers
make lint    # ruff
```

## License

MIT
