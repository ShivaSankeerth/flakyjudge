"""E4: verbosity bias under criterion-anchored judging.

Scores gate-passing padded/condensed response variants against the ORIGINAL
unit test (bare mode, T=0). Originals ride along from E1's bare-mode calls
via the cache, so this only pays for the variants.

Usage: python experiments/e4_verbosity.py [--judges a,b]
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


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    judges = load_judges(ROOT / "config" / "judges.yaml")
    if args.judges:
        judges = {name: judges[name] for name in args.judges.split(",")}

    items = {i["item_id"]: i for i in read_jsonl(DATA_DIR / "items_perturb.jsonl")}
    variants = [v for v in read_jsonl(DATA_DIR / "verbosity.jsonl") if v["passed"]]

    cache = ResponseCache(CACHE_PATH)
    rows = []
    for judge_name, spec in judges.items():
        conditions = []
        for item in items.values():
            conditions.append((item, "original", item["response"], 1.0))
        for v in variants:
            item = items[v["item_id"]]
            conditions.append((item, v["kind"], v["variant"], v["realized_ratio"]))

        calls = []
        for item, _, response_text, _ in conditions:
            prompt = build_prompt(item["query"], response_text,
                                  item["natural_unit_test"])
            calls.append((spec, make_key(spec, prompt), prompt))
        scored = await run_calls(cache, calls, concurrency=args.concurrency)
        for (item, kind, _, ratio), result in zip(conditions, scored, strict=True):
            rows.append(
                {
                    "experiment": "e4",
                    "item_id": item["item_id"],
                    "judge": judge_name,
                    "model": spec.model,
                    "kind": kind,
                    "realized_ratio": ratio,
                    "score_direct": result.score_direct,
                    "score_logprob": result.score_logprob,
                    "human_label": item["label"],
                }
            )
        n_fail = sum(r.score_direct is None for r in scored)
        print(f"{judge_name}: {len(scored)} calls, parse failures: {n_fail}, "
              f"spend: ${cache.total_spend():.2f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "e4_scores.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
