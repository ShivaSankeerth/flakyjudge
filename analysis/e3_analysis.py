"""E3 analysis: paraphrase sensitivity of unit-test criteria.

Headline numbers per judge:
  - decision flip rate across paraphrases at the 2.5 threshold, reported
    raw AND as excess over the E2 identical-input noise floor
  - positive-control flip rate (instrument resolution check: must be much
    higher than the true-paraphrase rate)
  - ICC(2,1) across paraphrases vs across resamples
  - per-paraphrase-type mean |score shift| vs original
  - strict-only vs strict+adjudicated sensitivity comparison
  - flip concentration: near-threshold vs clear-verdict items
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flakyjudge.metrics import PASS_THRESHOLD, flip_rate, icc_2_1

PARA_TYPES = [
    "lexical", "syntactic", "register_formal", "register_casual",
    "form_question", "form_imperative",
]
CONTROL_TYPES = ["control_negated", "control_swapped"]


def matrix_for(df: pd.DataFrame, types: list[str], min_variants: int = 4) -> np.ndarray:
    """items x variants score matrix (NaN-free rows only)."""
    wide = df[df.variant_type.isin(types)].pivot_table(
        index="item_id", columns="variant_type", values="score_direct"
    )
    wide = wide.dropna(thresh=min_variants).dropna(axis=0)
    return wide.to_numpy()


def noise_floor(e2: pd.DataFrame, judge: str, condition: str = "repeat_t0") -> dict:
    grp = e2[(e2.judge == judge) & (e2.condition == condition)]
    wide = grp.pivot_table(index="item_id", columns="repeat_idx", values="score_direct").dropna()
    matrix = wide.to_numpy()
    return {
        "flip": flip_rate(matrix),
        "sd": float(np.std(matrix, axis=1, ddof=1).mean()),
        "icc": icc_2_1(matrix),
    }


def main() -> None:
    e3 = pd.read_parquet(ROOT / "results" / "e3_scores.parquet")
    e2 = pd.read_parquet(ROOT / "results" / "e2_scores.parquet")
    manifest = pd.read_json(ROOT / "data" / "paraphrases.jsonl", lines=True)
    tiers = manifest.set_index(["item_id", "variant_type"])["gate_tier"].to_dict()
    e3["gate_tier"] = [
        tiers.get((row.item_id, row.variant_type)) for row in e3.itertuples()
    ]

    for judge in sorted(e3.judge.unique()):
        sub = e3[e3.judge == judge]
        floor_t0 = noise_floor(e2, judge, "repeat_t0")
        floor_t1 = noise_floor(e2, judge, "repeat_t1")
        print(f"\n=== {judge} ===")
        print(f"noise floor: T=0 flip={floor_t0['flip']:.3f} sd={floor_t0['sd']:.3f} "
              f"icc={floor_t0['icc']:.3f} | T=1 flip={floor_t1['flip']:.3f} "
              f"sd={floor_t1['sd']:.3f}")

        for label, frame in [
            ("strict", sub[sub.gate_tier != "adjudicated"]),
            ("strict+adjudicated", sub),
        ]:
            matrix = matrix_for(frame, ["original", *PARA_TYPES])
            if matrix.size == 0:
                continue
            para_flip = flip_rate(matrix)
            para_sd = float(np.std(matrix, axis=1, ddof=1).mean())
            print(f"[{label}] n={matrix.shape[0]} items x {matrix.shape[1]} variants: "
                  f"flip={para_flip:.3f} (excess {para_flip - floor_t0['flip']:+.3f}) "
                  f"sd={para_sd:.3f} (excess {para_sd - floor_t0['sd']:+.3f}) "
                  f"icc={icc_2_1(matrix):.3f}")

        # Positive controls: score shift vs original, and verdict flip vs original.
        wide = sub.pivot_table(index="item_id", columns="variant_type",
                               values="score_direct")
        if "original" in wide:
            orig = wide["original"]
            for ctrl in CONTROL_TYPES:
                if ctrl not in wide:
                    continue
                pair = wide[[ctrl]].join(orig.rename("orig")).dropna()
                flips = (
                    (pair[ctrl] > PASS_THRESHOLD) != (pair["orig"] > PASS_THRESHOLD)
                ).mean()
                shift = (pair[ctrl] - pair["orig"]).abs().mean()
                print(f"control {ctrl}: verdict-flip vs original={flips:.3f}, "
                      f"mean |shift|={shift:.2f}")
            # Per-type shifts for true paraphrases.
            print("per-type mean |shift| vs original:")
            for ptype in PARA_TYPES:
                if ptype not in wide:
                    continue
                pair = wide[[ptype]].join(orig.rename("orig")).dropna()
                shift = (pair[ptype] - pair["orig"]).abs().mean()
                type_flip = (
                    (pair[ptype] > PASS_THRESHOLD) != (pair["orig"] > PASS_THRESHOLD)
                ).mean()
                print(f"  {ptype:<18} |shift|={shift:.3f}  flip-vs-orig={type_flip:.3f}")

            # Flip concentration: near-threshold (|orig - 2.5| <= 1) vs clear.
            para = wide[[c for c in PARA_TYPES if c in wide]]
            complete = para.join(orig.rename("orig")).dropna()
            near = complete[(complete["orig"] - PASS_THRESHOLD).abs() <= 1.0]
            clear = complete[(complete["orig"] - PASS_THRESHOLD).abs() > 1.0]
            for name, grp in [("near-threshold", near), ("clear-verdict", clear)]:
                if len(grp) == 0:
                    continue
                m = grp[[c for c in PARA_TYPES if c in grp]].to_numpy()
                passes = m > PASS_THRESHOLD
                fr = float((passes.any(axis=1) & ~passes.all(axis=1)).mean())
                print(f"  flip rate on {name} items (n={len(grp)}): {fr:.3f}")


if __name__ == "__main__":
    main()
