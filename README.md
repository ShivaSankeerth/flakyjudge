# flakyjudge

**How stable are natural-language unit tests for LLM evals?**

Teams increasingly evaluate LLM outputs with natural-language unit tests —
discrete assertions like *"Does the response cite the refund window?"* scored
by a judge model (LMUnit, TICK, checklist evals). This project measures the
reliability of that paradigm: if you reword the assertion without changing
its meaning, does the verdict change?

> 🚧 **Study in progress.** Experimental design is frozen in
> [PREREGISTRATION.md](PREREGISTRATION.md). Results, figures, and the tech
> report land here as the experiments complete.

## What this is

An empirical robustness study across 6 judge models (Claude, GPT-4o, Gemini,
Qwen families) on FLASK and BiGGen-Bench items, plus a small library
implementing the measurement tools and mitigations:

- **E1** — judge-human correlation anchor; logprob-weighted vs. direct scoring
- **E2** — identical-input noise floor (resampling + field-order stability)
- **E3** — *headline:* paraphrase sensitivity of the unit-test criterion itself
- **E4** — verbosity bias under criterion-anchored judging

Related work measures judge robustness at the *response* level
([Judge Reliability Harness](https://arxiv.org/abs/2603.05399)) and *template*
level; this study targets the *criterion* level — the load-bearing assumption
of the unit-test paradigm ([LMUnit](https://arxiv.org/abs/2412.13091)).

## Reproducibility

Every raw API response (including logprobs) is committed in a
content-addressed cache, so all results and figures regenerate with **zero
API keys and zero dollars**:

```bash
git clone https://github.com/shivasankeerth/flakyjudge && cd flakyjudge
uv sync
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
