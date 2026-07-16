"""E3 (headline): paraphrase sensitivity of the unit-test criterion.

Scores every (item, unit-test variant) from the frozen paraphrase manifest
with every judge, bare-assertion mode, temperature 0. Positive controls
(negated / swapped criteria) ride along — their flip rate validates the
instrument's resolution.

Usage: python experiments/e3_paraphrase.py [--judges a,b] [--limit N]
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
    parser.add_argument("--limit", type=int, default=None, help="first N items")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    judges = load_judges(ROOT / "config" / "judges.yaml")
    if args.judges:
        judges = {name: judges[name] for name in args.judges.split(",")}

    items = {i["item_id"]: i for i in read_jsonl(DATA_DIR / "items_perturb.jsonl")}
    variants = read_jsonl(DATA_DIR / "paraphrases.jsonl")
    if args.limit:
        keep = set(list(items)[: args.limit])
        variants = [v for v in variants if v["item_id"] in keep]

    cache = ResponseCache(CACHE_PATH)
    rows = []
    for judge_name, spec in judges.items():
        calls = []
        for variant in variants:
            item = items[variant["item_id"]]
            prompt = build_prompt(item["query"], item["response"], variant["unit_test"])
            calls.append((spec, make_key(spec, prompt), prompt))
        scored = await run_calls(cache, calls, concurrency=args.concurrency)
        for variant, result in zip(variants, scored, strict=True):
            item = items[variant["item_id"]]
            rows.append(
                {
                    "experiment": "e3",
                    "item_id": variant["item_id"],
                    "judge": judge_name,
                    "model": spec.model,
                    "variant_type": variant["variant_type"],
                    "unit_test": variant["unit_test"],
                    "score_direct": result.score_direct,
                    "score_logprob": result.score_logprob,
                    "human_label": item["label"],
                }
            )
        n_fail = sum(r.score_direct is None for r in scored)
        print(f"{judge_name}: {len(scored)} calls, parse failures: {n_fail}, "
              f"spend: ${cache.total_spend():.2f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "e3_scores.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
