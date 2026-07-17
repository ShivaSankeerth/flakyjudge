"""Supplementary analyses answering the external review.

Produces every number the report was promising but not showing:
  A. Headline flip rates with Wilson 95% CIs, plus matched-k comparison
     (any-of-5 wordings vs any-of-5 resamples, subset-averaged)
  B. Formal instrument-resolution check per judge, matched per-variant
     definitions (preregistration validity control #3)
  C. Strict-only vs strict+adjudicated per judge (two-tier sensitivity)
  D. Claude analyses with and without extended-token salvage
  E. Logprob-weighted vs direct flip rates (judges with logprobs)
  F. k-sample averaging vs single-shot vs logprob: rho vs human (E2 items)
  G. Field-reorder stability (E2b, never previously reported)
  H. Variance decomposition: item / wording / sampling
  I. E4 with CIs, Cohen's dz, Holm correction, and TOST equivalence
     (bound +/-0.25 points, post-hoc, labeled as such)

Writes results/supplementary.json and prints a readable summary.
"""

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flakyjudge.data import DATA_DIR, read_jsonl
from flakyjudge.metrics import PASS_THRESHOLD, icc_2_1
from flakyjudge.prompts import build_prompt

PARA_TYPES = ["lexical", "syntactic", "register_formal", "register_casual",
              "form_question", "form_imperative"]
CONTROLS = ["control_negated", "control_swapped"]
JUDGES = ["gpt-4o", "gpt-4o-mini", "claude-sonnet", "claude-haiku", "llama-8b"]
MODEL_OF = {"claude-sonnet": "claude-sonnet-4-6",
            "claude-haiku": "claude-haiku-4-5-20251001"}
TOST_BOUND = 0.25

out: dict = {}


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959964
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def any_flip(matrix: np.ndarray) -> tuple[int, int]:
    passes = matrix > PASS_THRESHOLD
    flips = passes.any(axis=1) & ~passes.all(axis=1)
    return int(flips.sum()), matrix.shape[0]


def any_flip_at_k(matrix: np.ndarray, k: int, n_subsets: int = 200,
                  seed: int = 0) -> float:
    """Mean any-flip rate over random size-k column subsets (matched-k)."""
    cols = matrix.shape[1]
    if cols < k:
        return float("nan")
    subsets = list(itertools.combinations(range(cols), k))
    rng = np.random.default_rng(seed)
    if len(subsets) > n_subsets:
        subsets = [subsets[i] for i in
                   rng.choice(len(subsets), n_subsets, replace=False)]
    rates = [any_flip(matrix[:, list(s)])[0] / matrix.shape[0] for s in subsets]
    return float(np.mean(rates))


def wide_scores(e3: pd.DataFrame, judge: str, value: str = "score_direct",
                tiers: set | None = None) -> pd.DataFrame:
    sub = e3[e3.judge == judge]
    if tiers is not None:
        sub = sub[sub.gate_tier.isin(tiers) | (sub.variant_type == "original")]
    return sub.pivot_table(index="item_id", columns="variant_type", values=value)


def per_variant_flip(wide: pd.DataFrame, cols: list[str]) -> float:
    orig_pass = wide["original"] > PASS_THRESHOLD
    rates = [((wide[c] > PASS_THRESHOLD) != orig_pass)[wide[c].notna()].mean()
             for c in cols if c in wide]
    return float(np.mean(rates))


def main() -> None:
    e2 = pd.read_parquet(ROOT / "results" / "e2_scores.parquet")
    e3 = pd.read_parquet(ROOT / "results" / "e3_scores.parquet")
    e4 = pd.read_parquet(ROOT / "results" / "e4_scores.parquet")
    manifest = pd.read_json(DATA_DIR / "paraphrases.jsonl", lines=True)
    tiers = manifest.set_index(["item_id", "variant_type"])["gate_tier"].to_dict()
    e3 = e3.assign(gate_tier=[tiers.get((r.item_id, r.variant_type))
                              for r in e3.itertuples()])

    # ---------- A. headline with CIs + matched-k ----------
    out["A_headline"] = {}
    for judge in JUDGES:
        wide = wide_scores(e3, judge)[["original", *PARA_TYPES]].dropna()
        k7, n = any_flip(wide.to_numpy())
        low, high = wilson_ci(k7, n)
        rep = e2[(e2.judge == judge) & (e2.condition == "repeat_t0")].pivot_table(
            index="item_id", columns="repeat_idx", values="score_direct").dropna()
        kf, nf = any_flip(rep.to_numpy())
        flow, fhigh = wilson_ci(kf, nf)
        out["A_headline"][judge] = {
            "n_complete": n,
            "any7_flip": k7 / n, "any7_ci": [low, high],
            "any5_matched_paraphrase": any_flip_at_k(wide.to_numpy(), 5),
            "any5_noise_floor": kf / nf, "floor_ci": [flow, fhigh],
            "per_variant_paraphrase": per_variant_flip(
                wide_scores(e3, judge), PARA_TYPES),
            "per_variant_floor": float(np.mean([
                ((rep[k] > PASS_THRESHOLD) != (rep[0] > PASS_THRESHOLD)).mean()
                for k in range(1, rep.shape[1])])),
        }

    # ---------- B. resolution check, matched per-variant ----------
    out["B_resolution"] = {}
    for judge in JUDGES:
        wide = wide_scores(e3, judge)
        para = per_variant_flip(wide, PARA_TYPES)
        ctrl = per_variant_flip(wide, CONTROLS)
        any2 = wide[["original", *[c for c in CONTROLS if c in wide]]].dropna()
        any2_rate = any_flip(any2.to_numpy())[0] / len(any2) if len(any2) else np.nan
        out["B_resolution"][judge] = {
            "per_variant_paraphrase": para, "per_variant_control": ctrl,
            "any2_control": float(any2_rate),
            "ratio_ctrl_over_para": ctrl / para if para else np.nan,
            "passes": bool(ctrl > para * 1.5),  # post-hoc formalization
        }

    # ---------- C. tier sensitivity ----------
    out["C_tiers"] = {}
    for judge in JUDGES:
        row = {}
        for label, tset in [("strict", {"strict", None}),
                            ("combined", {"strict", "adjudicated", None})]:
            wide = wide_scores(e3, judge, tiers=tset)[
                ["original", *PARA_TYPES]].dropna()
            k, n = any_flip(wide.to_numpy())
            row[label] = {"flip": k / n if n else np.nan, "n": n}
        out["C_tiers"][judge] = row

    # ---------- D. salvage on/off ----------
    rescue = pd.read_parquet(ROOT / "results" / "rescue_scores.parquet")
    rescue_map = {(r.model, r.prompt_sha): r.score_direct
                  for r in rescue.itertuples() if r.score_direct is not None}
    items = {i["item_id"]: i for i in read_jsonl(DATA_DIR / "items_perturb.jsonl")}
    out["D_salvage"] = {}
    for judge in ("claude-sonnet", "claude-haiku"):
        sub = e3[e3.judge == judge].copy()
        shas = [hashlib.sha256(build_prompt(
            items[r.item_id]["query"], items[r.item_id]["response"],
            r.unit_test).encode()).hexdigest() for r in sub.itertuples()]
        salvaged = [rescue_map.get((MODEL_OF[judge], sha)) for sha in shas]
        # Fill NaN direct scores with salvage-tier scores where available.
        sub["score_salvaged"] = sub.score_direct.where(
            sub.score_direct.notna(), pd.Series(salvaged, index=sub.index))
        row = {}
        for label, col in [("without_salvage", "score_direct"),
                           ("with_salvage", "score_salvaged")]:
            wide = sub.pivot_table(index="item_id", columns="variant_type",
                                   values=col)[["original", *PARA_TYPES]].dropna()
            k, n = any_flip(wide.to_numpy())
            row[label] = {"flip": k / n if n else np.nan, "n": n}
        out["D_salvage"][judge] = row

    # ---------- E. logprob vs direct flip (logprob judges) ----------
    out["E_logprob_flip"] = {}
    for judge in ("gpt-4o", "gpt-4o-mini", "llama-8b"):
        row = {}
        for label, col in [("direct", "score_direct"),
                           ("logprob", "score_logprob")]:
            wide = wide_scores(e3, judge, value=col)[
                ["original", *PARA_TYPES]].dropna()
            k, n = any_flip(wide.to_numpy())
            row[label] = {"flip": k / n if n else np.nan, "n": n}
        out["E_logprob_flip"][judge] = row

    # ---------- F. k-sample averaging vs single vs logprob (rho vs human) --
    out["F_ksample"] = {}
    for judge in JUDGES:
        rep1 = e2[(e2.judge == judge) & (e2.condition == "repeat_t1")].pivot_table(
            index="item_id", columns="repeat_idx", values="score_direct")
        single = e2[(e2.judge == judge) & (e2.condition == "repeat_t0")
                    & (e2.repeat_idx == 0)].set_index("item_id")
        humans = single["human_label"]
        idx = rep1.dropna().index.intersection(humans.index)
        r_single = stats.spearmanr(
            single.loc[idx, "score_direct"], humans.loc[idx]).statistic
        r_mean5 = stats.spearmanr(
            rep1.loc[idx].mean(axis=1), humans.loc[idx]).statistic
        lp = single.loc[idx, "score_logprob"]
        r_logprob = (stats.spearmanr(lp.dropna(),
                                     humans.loc[lp.dropna().index]).statistic
                     if lp.notna().sum() > 10 else None)
        out["F_ksample"][judge] = {
            "n": int(len(idx)), "rho_single_t0": float(r_single),
            "rho_mean5_t1": float(r_mean5),
            "rho_logprob_t0": None if r_logprob is None else float(r_logprob)}

    # ---------- G. field reorder ----------
    out["G_reorder"] = {}
    for judge in JUDGES:
        sub = e2[(e2.judge == judge) & e2.condition.str.startswith("order")]
        wide = sub.pivot_table(index="item_id", columns="condition",
                               values="score_direct").dropna()
        delta = wide["order_reversed"] - wide["order_standard"]
        flips = ((wide["order_reversed"] > PASS_THRESHOLD)
                 != (wide["order_standard"] > PASS_THRESHOLD)).mean()
        out["G_reorder"][judge] = {
            "n": len(wide), "mean_delta": float(delta.mean()),
            "mean_abs_delta": float(delta.abs().mean()),
            "flip_rate": float(flips)}

    # ---------- H. variance decomposition ----------
    out["H_variance"] = {}
    for judge in JUDGES:
        wide = wide_scores(e3, judge)[["original", *PARA_TYPES]].dropna()
        matrix = wide.to_numpy()
        var_item = float(np.var(matrix.mean(axis=1), ddof=1))
        var_within = float(np.mean(np.var(matrix, axis=1, ddof=1)))
        rep = e2[(e2.judge == judge) & (e2.condition == "repeat_t0")].pivot_table(
            index="item_id", columns="repeat_idx", values="score_direct").dropna()
        var_sampling = float(np.mean(np.var(rep.to_numpy(), axis=1, ddof=1)))
        out["H_variance"][judge] = {
            "between_item": var_item,
            "within_item_wording": var_within,
            "sampling_floor": var_sampling,
            "wording_minus_sampling": var_within - var_sampling,
            "icc_paraphrase": float(icc_2_1(matrix))}

    # ---------- I. E4 with CIs, dz, Holm, TOST ----------
    tests = []
    for judge in JUDGES:
        wide = e4[e4.judge == judge].pivot_table(
            index="item_id", columns="kind", values="score_direct")
        for kind in ("padded", "condensed"):
            pair = wide[[kind, "original"]].dropna()
            drift = pair[kind] - pair["original"]
            n = len(drift)
            mean = drift.mean()
            se = drift.std(ddof=1) / np.sqrt(n)
            ci = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
            dz = mean / drift.std(ddof=1)
            p = stats.wilcoxon(drift).pvalue if (drift != 0).any() else 1.0
            # TOST vs +/-BOUND
            t1 = (mean - (-TOST_BOUND)) / se
            t2 = (mean - TOST_BOUND) / se
            p_tost = max(1 - stats.t.cdf(t1, n - 1), stats.t.cdf(t2, n - 1))
            tests.append({"judge": judge, "kind": kind, "n": n,
                          "mean_drift": float(mean),
                          "ci": [float(ci[0]), float(ci[1])],
                          "dz": float(dz), "p_wilcoxon": float(p),
                          "p_tost_equiv": float(p_tost)})
    # Holm step-down applied SYMMETRICALLY: to the 10 difference tests and
    # to the 10 equivalence tests (an uncorrected TOST family next to a
    # corrected difference family would bias toward the null story).
    def holm(key: str, out_key: str) -> None:
        order = sorted(range(len(tests)), key=lambda i: tests[i][key])
        m = len(tests)
        running_max = 0.0
        for rank, i in enumerate(order):
            adj = min(1.0, (m - rank) * tests[i][key])
            running_max = max(running_max, adj)
            tests[i][out_key] = running_max

    holm("p_wilcoxon", "p_holm")
    holm("p_tost_equiv", "p_tost_holm")
    out["I_verbosity"] = tests

    (ROOT / "results" / "supplementary.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
