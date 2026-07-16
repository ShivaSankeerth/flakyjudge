"""Seeded, stratified sampling of evaluation items into frozen manifests.

Manifests are committed to data/ and never regenerated; all experiments read
items from the manifests, so the sample is fixed independently of upstream
dataset changes.

FLASK human_score is a list of 3 annotators — kept intact so analysis can
report the human-human agreement ceiling alongside judge-human correlation.
"""

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

SEED = 20260715
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

E1_PER_DATASET = 500
PERTURB_PER_BAND = 50  # low / mid / high human-score bands


def _label_band(label: float) -> str:
    if label <= 2:
        return "low"
    if label < 4:
        return "mid"
    return "high"


def _stratified_sample(indices_by_stratum: dict, n_total: int, rng: np.random.Generator):
    """Proportional allocation with at least one item per non-empty stratum."""
    strata = {k: np.array(v) for k, v in indices_by_stratum.items() if len(v) > 0}
    total = sum(len(v) for v in strata.values())
    chosen: list[int] = []
    for _, indices in sorted(strata.items()):
        take = max(1, round(n_total * len(indices) / total))
        take = min(take, len(indices))
        chosen.extend(rng.choice(indices, size=take, replace=False).tolist())
    # Trim overshoot deterministically.
    rng.shuffle(chosen)
    return sorted(chosen[:n_total])


def build_e1_manifest() -> list[dict]:
    rng = np.random.default_rng(SEED)
    items = []
    for dataset_name, short in [
        ("ContextualAI/Flask", "flask"),
        ("ContextualAI/BigGenBench", "biggen"),
    ]:
        ds = load_dataset(dataset_name, split="test")
        by_stratum: dict[str, list[int]] = {}
        for i, row in enumerate(ds):
            stratum = str(int(round(float(row["label"]))))
            if short == "biggen":
                stratum = f"{row['capability']}/{stratum}"
            by_stratum.setdefault(stratum, []).append(i)
        for i in _stratified_sample(by_stratum, E1_PER_DATASET, rng):
            row = ds[i]
            items.append(
                {
                    "item_id": f"{short}-{i}",
                    "dataset": short,
                    "query": row["query"],
                    "response": row["response"],
                    "natural_unit_test": row["natural_unit_test"],
                    "rubric": row["rubric"],
                    "reference_answer": row["reference_answer"],
                    "human_score": row["human_score"],
                    "label": float(row["label"]),
                }
            )
    return items


def build_perturb_manifest(e1_items: list[dict]) -> list[dict]:
    """150 FLASK items from the E1 sample, 50 per human-score band."""
    rng = np.random.default_rng(SEED + 1)
    flask_items = [item for item in e1_items if item["dataset"] == "flask"]
    by_band: dict[str, list[dict]] = {"low": [], "mid": [], "high": []}
    for item in flask_items:
        by_band[_label_band(item["label"])].append(item)
    chosen = []
    for _band, members in sorted(by_band.items()):
        take = min(PERTURB_PER_BAND, len(members))
        idx = rng.choice(len(members), size=take, replace=False)
        chosen.extend(members[i] for i in idx)
    return sorted(chosen, key=lambda item: item["item_id"])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    e1 = build_e1_manifest()
    write_jsonl(DATA_DIR / "items_e1.jsonl", e1)
    perturb = build_perturb_manifest(e1)
    write_jsonl(DATA_DIR / "items_perturb.jsonl", perturb)
    bands = [_label_band(item["label"]) for item in perturb]
    print(f"E1 manifest: {len(e1)} items "
          f"(flask={sum(1 for i in e1 if i['dataset'] == 'flask')}, "
          f"biggen={sum(1 for i in e1 if i['dataset'] == 'biggen')})")
    print(f"Perturbation manifest: {len(perturb)} items "
          f"(low={bands.count('low')}, mid={bands.count('mid')}, high={bands.count('high')})")
