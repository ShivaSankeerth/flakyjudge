"""Generate and freeze the paraphrase manifest for E3.

Writes data/paraphrases.jsonl (one row per item x variant, originals included
as variant_type='original') and data/paraphrase_gate_report.json (pass rates,
per PREREGISTRATION validity control #2).
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flakyjudge.cache import ResponseCache
from flakyjudge.data import DATA_DIR, read_jsonl, write_jsonl
from flakyjudge.perturb.paraphrase import CONTROL_TYPES, PARAPHRASE_TYPES, generate_for_item

CACHE_PATH = ROOT / "data" / "cache" / "responses.db"


async def main() -> None:
    items = read_jsonl(DATA_DIR / "items_perturb.jsonl")
    cache = ResponseCache(CACHE_PATH)
    rows, reports = [], {}
    all_types = list(PARAPHRASE_TYPES) + list(CONTROL_TYPES)

    async with httpx.AsyncClient() as client:
        semaphore = asyncio.Semaphore(6)

        async def one(item):
            async with semaphore:
                accepted, report = await generate_for_item(
                    client, cache, item["natural_unit_test"]
                )
            return item, accepted, report

        results = await asyncio.gather(*(one(item) for item in items))

    incomplete = 0
    for item, accepted, report in results:
        reports[item["item_id"]] = report
        rows.append(
            {
                "item_id": item["item_id"],
                "variant_type": "original",
                "unit_test": item["natural_unit_test"],
            }
        )
        for kind in all_types:
            if kind in accepted:
                rows.append(
                    {
                        "item_id": item["item_id"],
                        "variant_type": kind,
                        "unit_test": accepted[kind],
                    }
                )
        if len(accepted) < len(all_types):
            incomplete += 1

    write_jsonl(DATA_DIR / "paraphrases.jsonl", rows)
    with open(DATA_DIR / "paraphrase_gate_report.json", "w") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    n_variants = sum(1 for r in rows if r["variant_type"] != "original")
    print(f"items: {len(items)}, variants accepted: {n_variants} "
          f"(target {len(items) * len(all_types)}), items incomplete: {incomplete}")
    print(f"cumulative spend: ${cache.total_spend():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
