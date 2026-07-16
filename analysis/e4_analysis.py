"""E4 analysis: verbosity drift under criterion-anchored judging.

Per judge and variant kind: paired mean drift (variant - original), Cohen's
dz, Wilcoxon signed-rank p, threshold flip rate, and the verbosity
elasticity (OLS slope of drift on realized log length ratio).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flakyjudge.metrics import PASS_THRESHOLD


def main() -> None:
    df = pd.read_parquet(ROOT / "results" / "e4_scores.parquet")

    for judge in sorted(df.judge.unique()):
        sub = df[df.judge == judge]
        wide = sub.pivot_table(index="item_id", columns="kind",
                               values="score_direct")
        ratios = sub[sub.kind != "original"].set_index(["item_id", "kind"])[
            "realized_ratio"]
        print(f"\n=== {judge} ===")
        drift_frames = []
        for kind in ("padded", "condensed"):
            if kind not in wide:
                continue
            pair = wide[[kind, "original"]].dropna()
            drift = pair[kind] - pair["original"]
            dz = drift.mean() / drift.std(ddof=1) if drift.std(ddof=1) > 0 else 0
            wilcoxon_p = (stats.wilcoxon(drift).pvalue
                          if (drift != 0).any() else float("nan"))
            flips = ((pair[kind] > PASS_THRESHOLD)
                     != (pair["original"] > PASS_THRESHOLD)).mean()
            print(f"{kind:>10}: n={len(pair)}, mean drift {drift.mean():+.3f} "
                  f"(dz={dz:+.2f}, Wilcoxon p={wilcoxon_p:.4f}), "
                  f"flip rate {flips:.3f}")
            frame = drift.rename("drift").to_frame()
            kind_ratios = ratios.xs(kind, level="kind")
            frame["log_ratio"] = np.log(kind_ratios.reindex(frame.index))
            drift_frames.append(frame)

        both = pd.concat(drift_frames).replace([np.inf, -np.inf], np.nan).dropna()
        if len(both) > 10:
            slope, intercept, r, p, se = stats.linregress(
                both["log_ratio"], both["drift"])
            print(f"verbosity elasticity: {slope:+.3f} score-points per log "
                  f"length-ratio (r={r:+.2f}, p={p:.4f}, n={len(both)})")


if __name__ == "__main__":
    main()
