"""Generate all report figures from committed results. Zero API calls.

Palette/spec follow the validated reference dataviz palette (light surface):
accent blue #2a78d6, ordinal blue ramp 250/450/650, de-emphasis gray,
hairline solid gridlines, thin marks, selective direct labels.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flakyjudge.metrics import PASS_THRESHOLD

FIGURES = ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ACCENT = "#2a78d6"       # categorical slot 1
AQUA = "#1baf7a"         # categorical slot 2
DEEMPH = "#c3c2b7"
# Fixed judge -> color mapping (color follows the entity, never rank/order).
JUDGE_COLORS = {
    "gpt-4o": "#2a78d6",        # slot 1 blue
    "gpt-4o-mini": "#1baf7a",   # slot 2 aqua
    "claude-sonnet": "#eda100", # slot 3 yellow
    "claude-haiku": "#4a3aa7",  # slot 5 violet
    "llama-8b": "#e34948",      # slot 6 red
}
JUDGE_ORDER = list(JUDGE_COLORS)
ORDINAL = ["#86b6ef", "#2a78d6", "#0d366b"]  # validated --ordinal

PARA_TYPES = ["lexical", "syntactic", "register_formal", "register_casual",
              "form_question", "form_imperative"]
CONTROLS = ["control_negated", "control_swapped"]
TYPE_LABELS = {
    "lexical": "Synonym swap", "syntactic": "Restructured sentence",
    "register_formal": "Formal register", "register_casual": "Casual register",
    "form_question": "As a question", "form_imperative": "As an instruction",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
    "svg.fonttype": "none",
})


def load():
    e3 = pd.read_parquet(ROOT / "results" / "e3_scores.parquet")
    e2 = pd.read_parquet(ROOT / "results" / "e2_scores.parquet")
    return e3, e2


def variant_matrix(e3: pd.DataFrame, judge: str) -> pd.DataFrame:
    wide = e3[e3.judge == judge].pivot_table(
        index="item_id", columns="variant_type", values="score_direct"
    )
    return wide[["original", *PARA_TYPES]].dropna()


def fig1_hero(e3: pd.DataFrame, judge: str = "gpt-4o") -> None:
    """Per-item score range across paraphrases; verdict-flipping items in
    accent, stable items in de-emphasis gray (emphasis form)."""
    wide = variant_matrix(e3, judge)
    lo, hi = wide.min(axis=1), wide.max(axis=1)
    order = wide.mean(axis=1).sort_values().index
    lo, hi = lo.loc[order], hi.loc[order]
    flips = (lo <= PASS_THRESHOLD) & (hi > PASS_THRESHOLD)
    n_flip = int(flips.sum())

    fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=200)
    x = np.arange(len(order))
    for xi, low, high, flip in zip(x, lo, hi, flips, strict=True):
        color = ACCENT if flip else DEEMPH
        if high - low < 0.05:
            # Perfectly stable item: a zero-length line is invisible.
            ax.plot(xi, low, marker="o", ms=3.5, color=color, zorder=2)
        else:
            ax.plot([xi, xi], [low, high], lw=2, solid_capstyle="round",
                    color=color, zorder=3 if flip else 2)

    ax.axhline(PASS_THRESHOLD, color=INK_SECONDARY, lw=1, zorder=1)
    ax.text(len(order) - 0.5, PASS_THRESHOLD + 0.07, "pass / fail threshold",
            ha="right", va="bottom", fontsize=8.5, color=INK_SECONDARY)

    ax.set_title(
        f"Rewording a unit test flips {n_flip} of {len(order)} verdicts ({judge})",
        loc="left", fontsize=13, fontweight="semibold", color=INK, pad=14)
    ax.text(0, 1.015, "Score range across 6 meaning-preserving paraphrases + "
            "original, per item. Blue = verdict depends on wording.",
            transform=ax.transAxes, fontsize=9, color=INK_SECONDARY, va="bottom")
    ax.set_ylabel("judge score (1–5)")
    ax.set_xlabel("evaluation items, sorted by mean score")
    ax.set_xticks([])
    ax.set_ylim(0.8, 5.2)
    ax.set_yticks([1, 2, 3, 4, 5])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_hero_flips.png", bbox_inches="tight")
    plt.close(fig)


def flip_vs_original(wide: pd.DataFrame, cols: list[str]) -> float:
    """Mean per-variant rate of verdict disagreement with the original."""
    orig_pass = wide["original"] > PASS_THRESHOLD
    rates = [((wide[c] > PASS_THRESHOLD) != orig_pass).mean()
             for c in cols if c in wide]
    return float(np.mean(rates))


def fig2_flip_rates(e3: pd.DataFrame, e2: pd.DataFrame) -> None:
    """Comparable per-variant flip rates: resample floor vs paraphrase vs
    meaning-changed controls. Ordered conditions -> ordinal ramp."""
    judges = [j for j in JUDGE_ORDER if j in set(e3.judge.unique())]
    conditions = ["identical input, resampled", "meaning-preserving paraphrase",
                  "meaning-changed control"]
    values = {c: [] for c in conditions}
    for judge in judges:
        rep = e2[(e2.judge == judge) & (e2.condition == "repeat_t0")].pivot_table(
            index="item_id", columns="repeat_idx", values="score_direct").dropna()
        base = rep[0] > PASS_THRESHOLD
        floor = float(np.mean([((rep[k] > PASS_THRESHOLD) != base).mean()
                               for k in range(1, 5)]))
        wide = e3[e3.judge == judge].pivot_table(
            index="item_id", columns="variant_type", values="score_direct")
        values[conditions[0]].append(floor)
        values[conditions[1]].append(flip_vs_original(wide, PARA_TYPES))
        values[conditions[2]].append(flip_vs_original(wide, CONTROLS))

    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=200)
    band = 0.26
    y = np.arange(len(judges))
    for i, cond in enumerate(conditions):
        pos = y + (i - 1) * band
        ax.barh(pos, values[cond], height=band - 0.04, color=ORDINAL[i],
                label=cond, zorder=3)
        for yi, v in zip(pos, values[cond], strict=True):
            ax.text(v + 0.004, yi, f"{v * 100:.1f}%", va="center",
                    fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels(judges, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values[conditions[2]]) * 1.3)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("How often does the verdict disagree with the original wording?",
                 loc="left", fontsize=12.5, fontweight="semibold", pad=12)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right",
              labelcolor=INK_SECONDARY)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_flip_rates.png", bbox_inches="tight")
    plt.close(fig)


def fig3_mechanism(e3: pd.DataFrame) -> None:
    """Left: mean |score shift| per paraphrase type. Right: flip rate on
    near-threshold vs clear items. Two judges -> categorical slots 1-2."""
    judges = [j for j in JUDGE_ORDER if j in set(e3.judge.unique())]
    colors = JUDGE_COLORS
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.6), dpi=200, gridspec_kw={"width_ratios": [3, 2]})

    band = 0.8 / max(1, len(judges))
    y = np.arange(len(PARA_TYPES))
    offset0 = (len(judges) - 1) / 2
    for i, judge in enumerate(judges):
        wide = e3[e3.judge == judge].pivot_table(
            index="item_id", columns="variant_type", values="score_direct")
        shifts = [(wide[t] - wide["original"]).abs().mean() for t in PARA_TYPES]
        pos = y + (i - offset0) * band
        ax1.barh(pos, shifts, height=band - 0.04, color=colors[judge],
                 label=judge, zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels([TYPE_LABELS[t] for t in PARA_TYPES], fontsize=9.5,
                        color=INK)
    ax1.invert_yaxis()
    ax1.grid(axis="x", color=GRID, lw=1, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_xlabel("mean |score shift| vs original (1–5 scale)")
    ax1.set_title("Which rewordings move the score?", loc="left",
                  fontsize=12, fontweight="semibold", pad=10)
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)

    groups = ["near threshold\n(|score − 2.5| ≤ 1)", "clear verdict\n(|score − 2.5| > 1)"]
    x = np.arange(len(groups))
    for i, judge in enumerate(judges):
        wide = variant_matrix(e3, judge)
        orig = wide["original"]
        near = (orig - PASS_THRESHOLD).abs() <= 1.0
        rates = []
        for mask in (near, ~near):
            m = wide.loc[mask, PARA_TYPES].to_numpy()
            passes = np.column_stack([m > PASS_THRESHOLD,
                                      (wide.loc[mask, "original"] > PASS_THRESHOLD)])
            rates.append(float((passes.any(axis=1) & ~passes.all(axis=1)).mean()))
        pos = x + (i - offset0) * (0.8 / len(judges))
        bars = ax2.bar(pos, rates, width=0.8 / len(judges) - 0.03,
                       color=colors[judge], zorder=3)
        for b, v in zip(bars, rates, strict=True):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v * 100:.0f}%",
                     ha="center", fontsize=7.5, color=INK_SECONDARY)
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=9, color=INK)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax2.grid(axis="y", color=GRID, lw=1, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_title("Flips concentrate near the threshold", loc="left",
                  fontsize=12, fontweight="semibold", pad=10)
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_mechanism.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    e3, e2 = load()
    fig1_hero(e3)
    fig2_flip_rates(e3, e2)
    fig3_mechanism(e3)
    print(f"wrote 3 figures -> {FIGURES}")
