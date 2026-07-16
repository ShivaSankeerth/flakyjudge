"""Generate and freeze verbosity variants for E4.

Writes data/verbosity.jsonl (one row per item x kind with gate results;
only gate-passing variants are used by e4) and prints gate pass rates.
"""

import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flakyjudge.cache import ResponseCache
from flakyjudge.data import DATA_DIR, read_jsonl, write_jsonl
from flakyjudge.perturb.verbosity import make_variant

CACHE_PATH = ROOT / "data" / "cache" / "responses.db"


async def main() -> None:
    items = read_jsonl(DATA_DIR / "items_perturb.jsonl")
    cache = ResponseCache(CACHE_PATH)
    rows = []

    async with httpx.AsyncClient() as client:
        semaphore = asyncio.Semaphore(6)

        async def one(item, kind):
            async with semaphore:
                result = await make_variant(client, cache, item["response"], kind)
            result["item_id"] = item["item_id"]
            return result

        tasks = [one(item, kind) for item in items for kind in ("padded", "condensed")]
        rows = list(await asyncio.gather(*tasks))

    write_jsonl(DATA_DIR / "verbosity.jsonl", rows)
    for kind in ("padded", "condensed"):
        subset = [r for r in rows if r["kind"] == kind]
        passed = sum(r["passed"] for r in subset)
        ratios = [r["realized_ratio"] for r in subset if r["passed"]]
        mean_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        print(f"{kind}: {passed}/{len(subset)} passed gates, "
              f"mean realized ratio {mean_ratio:.2f}")
    print(f"cumulative spend: ${cache.total_spend():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
