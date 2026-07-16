"""E1 anchor: judge-human correlation, direct vs logprob scoring, rubric vs bare.

Scores every manifest item with every judge in two prompt modes:
  - parity: rubric + reference appended to the unit test (LMUnit's FLASK/
    BiGGen setup; sanity gate against the paper's published numbers)
  - bare:   the unit test alone (deployed practice; baseline for E3)

Writes one flat row per call to results/e1_scores.parquet. Idempotent and
resumable: all calls go through the content-addressed cache.

Usage: python experiments/e1_anchor.py [--judges gpt-4o-mini,...] [--limit N]
"""

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flakyjudge.cache import ResponseCache
from flakyjudge.data import DATA_DIR, read_jsonl
from flakyjudge.prompts import build_prompt
from flakyjudge.providers.base import load_judges
from flakyjudge.runner import make_key, run_calls

RESULTS = ROOT / "results"
CACHE_PATH = ROOT / "data" / "cache" / "responses.db"


def unit_test_text(item: dict, mode: str) -> str:
    if mode == "bare":
        return item["natural_unit_test"]
    # LMUnit parity: eval.py appends rubric then reference answer.
    return (
        f"{item['natural_unit_test']}\n\nRubric: {item['rubric']}"
        f"\n\nReference Answer: {item['reference_answer']}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", default=None, help="comma-separated subset")
    parser.add_argument("--limit", type=int, default=None, help="first N items (smoke test)")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    judges = load_judges(ROOT / "config" / "judges.yaml")
    if args.judges:
        judges = {name: judges[name] for name in args.judges.split(",")}
    items = read_jsonl(DATA_DIR / "items_e1.jsonl")
    if args.limit:
        items = items[: args.limit]

    cache = ResponseCache(CACHE_PATH)
    rows = []
    for judge_name, spec in judges.items():
        for mode in ("parity", "bare"):
            calls = []
            for item in items:
                prompt = build_prompt(
                    item["query"], item["response"], unit_test_text(item, mode)
                )
                calls.append((spec, make_key(spec, prompt), prompt))
            scored = await run_calls(cache, calls, concurrency=args.concurrency)
            for item, result in zip(items, scored, strict=True):
                rows.append(
                    {
                        "experiment": "e1",
                        "item_id": item["item_id"],
                        "dataset": item["dataset"],
                        "judge": judge_name,
                        "model": spec.model,
                        "prompt_mode": mode,
                        "score_direct": result.score_direct,
                        "score_logprob": result.score_logprob,
                        "raw_text": result.text,
                        "human_label": item["label"],
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                    }
                )
            hits = sum(r.from_cache for r in scored)
            print(
                f"{judge_name}/{mode}: {len(scored)} calls ({hits} cached), "
                f"parse failures: {sum(r.score_direct is None for r in scored)}"
            )
        print(f"cumulative spend: ${cache.total_spend():.2f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "e1_scores.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
