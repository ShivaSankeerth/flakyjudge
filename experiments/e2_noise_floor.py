"""E2: identical-input noise floor + field-order stability.

Per PREREGISTRATION this runs BEFORE E3/E4: every perturbation effect is
reported as excess over the resampling noise measured here.

Design (bare-assertion mode):
  - 100 items x 5 repeats at temperature 1.0 (sampling noise)
  - 100 items x 5 repeats at temperature 0.0 (serving nondeterminism)
  - 100 items x 2 field orderings at temperature 0.0 (position stability)

Usage: python experiments/e2_noise_floor.py [--judges a,b] [--limit N]
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
N_ITEMS = 100
N_REPEATS = 5


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    judges = load_judges(ROOT / "config" / "judges.yaml")
    if args.judges:
        judges = {name: judges[name] for name in args.judges.split(",")}
    items = read_jsonl(DATA_DIR / "items_perturb.jsonl")[: args.limit or N_ITEMS]

    cache = ResponseCache(CACHE_PATH)
    rows = []
    for judge_name, spec in judges.items():
        conditions = []
        for item in items:
            prompt = build_prompt(item["query"], item["response"], item["natural_unit_test"])
            reordered = build_prompt(
                item["query"], item["response"], item["natural_unit_test"], reordered=True
            )
            for repeat in range(N_REPEATS):
                conditions.append((item, "repeat_t1", repeat, prompt, 1.0))
                conditions.append((item, "repeat_t0", repeat, prompt, 0.0))
            conditions.append((item, "order_standard", 0, prompt, 0.0))
            conditions.append((item, "order_reversed", 0, reordered, 0.0))

        calls = [
            (spec, make_key(spec, prompt, temperature=temp, repeat_idx=repeat), prompt)
            for (_, _, repeat, prompt, temp) in conditions
        ]
        scored = await run_calls(cache, calls, concurrency=args.concurrency)
        for (item, condition, repeat, _, temp), result in zip(conditions, scored, strict=True):
            rows.append(
                {
                    "experiment": "e2",
                    "item_id": item["item_id"],
                    "judge": judge_name,
                    "model": spec.model,
                    "condition": condition,
                    "repeat_idx": repeat,
                    "temperature": temp,
                    "score_direct": result.score_direct,
                    "score_logprob": result.score_logprob,
                    "human_label": item["label"],
                }
            )
        n_fail = sum(r.score_direct is None for r in scored)
        print(f"{judge_name}: {len(scored)} calls, parse failures: {n_fail}, "
              f"spend: ${cache.total_spend():.2f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "e2_scores.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
