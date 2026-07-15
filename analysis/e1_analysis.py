"""E1 analysis: judge-human correlation, scoring-mode and prompt-mode effects.

Reads results/e1_scores.parquet, prints per-judge tables, and runs the
preregistered paired-bootstrap comparisons:
  - logprob-weighted vs direct scoring (within judge, same calls)
  - parity (rubric+reference) vs bare prompt mode (within judge)
Reports the FLASK human-human agreement ceiling alongside judge numbers.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flakyjudge.data import DATA_DIR, read_jsonl
from flakyjudge.metrics import paired_bootstrap_diff


def human_ceiling() -> float:
    """Mean leave-one-annotator-out Spearman rho on FLASK (3 annotators):
    each annotator vs the mean of the other two — the noise ceiling any
    judge-vs-consensus correlation should be read against."""
    items = read_jsonl(DATA_DIR / "items_e1.jsonl")
    triples = np.array(
        [i["human_score"] for i in items if i["dataset"] == "flask"], dtype=float
    )
    rhos = []
    for a in range(3):
        others = triples[:, [b for b in range(3) if b != a]].mean(axis=1)
        rhos.append(stats.spearmanr(triples[:, a], others).statistic)
    return float(np.mean(rhos))


def spearman(scores: np.ndarray, humans: np.ndarray) -> float:
    mask = ~np.isnan(scores)
    return float(stats.spearmanr(scores[mask], humans[mask]).statistic)


def main() -> None:
    df = pd.read_parquet(ROOT / "results" / "e1_scores.parquet")
    print(f"rows: {len(df)}, judges: {sorted(df.judge.unique())}")
    ceiling = human_ceiling()
    print(f"\nFLASK human-human ceiling (leave-one-out Spearman): {ceiling:.3f}\n")

    print(f"{'judge':<14}{'dataset':<9}{'mode':<8}{'rho_direct':>11}{'rho_logprob':>12}"
          f"{'parse_fail%':>12}")
    for (judge, dataset, mode), grp in df.groupby(["judge", "dataset", "prompt_mode"]):
        humans = grp["human_label"].to_numpy()
        rho_d = spearman(grp["score_direct"].to_numpy(dtype=float), humans)
        has_lp = grp["score_logprob"].notna().any()
        rho_l = (
            spearman(grp["score_logprob"].to_numpy(dtype=float), humans) if has_lp else np.nan
        )
        fail = grp["score_direct"].isna().mean() * 100
        print(f"{judge:<14}{dataset:<9}{mode:<8}{rho_d:>11.3f}{rho_l:>12.3f}{fail:>11.1f}%")

    print("\nPaired bootstrap: logprob vs direct (Spearman rho diff, bare mode)")
    for judge, grp in df[df.prompt_mode == "bare"].groupby("judge"):
        if grp["score_logprob"].isna().all():
            continue
        mask = grp["score_direct"].notna() & grp["score_logprob"].notna()
        sub = grp[mask]
        humans = sub["human_label"].to_numpy()

        def rho_of(col_values, humans=humans):
            return stats.spearmanr(col_values, humans).statistic

        # Paired over items: resample rows jointly.
        direct = sub["score_direct"].to_numpy(dtype=float)
        logprob = sub["score_logprob"].to_numpy(dtype=float)
        idx = np.arange(len(sub))
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(2000):
            boot = rng.integers(0, len(idx), len(idx))
            diffs.append(
                stats.spearmanr(logprob[boot], humans[boot]).statistic
                - stats.spearmanr(direct[boot], humans[boot]).statistic
            )
        low, high = np.percentile(diffs, [2.5, 97.5])
        point = rho_of(logprob) - rho_of(direct)
        print(f"  {judge}: delta_rho={point:+.4f}  95% CI [{low:+.4f}, {high:+.4f}]")

    print("\nPaired bootstrap: parity vs bare prompt mode (direct scores)")
    for judge, grp in df.groupby("judge"):
        wide = grp.pivot_table(
            index="item_id", columns="prompt_mode", values="score_direct"
        ).dropna()
        if not {"parity", "bare"} <= set(wide.columns):
            continue
        humans = (
            grp.drop_duplicates("item_id").set_index("item_id")["human_label"].loc[wide.index]
        ).to_numpy()
        parity = wide["parity"].to_numpy()
        bare = wide["bare"].to_numpy()
        rng = np.random.default_rng(1)
        diffs = []
        for _ in range(2000):
            boot = rng.integers(0, len(wide), len(wide))
            diffs.append(
                stats.spearmanr(parity[boot], humans[boot]).statistic
                - stats.spearmanr(bare[boot], humans[boot]).statistic
            )
        low, high = np.percentile(diffs, [2.5, 97.5])
        point = (
            stats.spearmanr(parity, humans).statistic - stats.spearmanr(bare, humans).statistic
        )
        drift = paired_bootstrap_diff(parity, bare, statistic=np.mean, n_resamples=2000)
        print(
            f"  {judge}: rubric buys delta_rho={point:+.4f} CI [{low:+.4f}, {high:+.4f}]; "
            f"mean score shift {drift['diff']:+.3f} CI [{drift['ci_low']:+.3f}, "
            f"{drift['ci_high']:+.3f}]"
        )


if __name__ == "__main__":
    main()
