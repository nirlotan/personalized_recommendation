"""
The per-category panels of Figure 1 (results_v3.html) as separate plots, one file per category plus Overall:
SocialVec MAP by number of sampled accounts and the popularity baseline, for ranks 1-20 and 81-100.
The legend is drawn in the first plot only.

    python run_figure1_bands.py      # the data, if not there yet
    python make_figure1_panels.py             # all panels, legend in the first
    python make_figure1_panels.py --variants  # Overall and Musical artists with the legend outside; Musical artists without it

Writes results/figure1_panels/figure1_<nn>_<category>.pdf and .png.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from reproduction.config import CATEGORY_LABELS, PAPER_CATEGORY_ORDER, RESULTS_DIR, SE1_SAMPLE_SIZES  # noqa: E402

KS = [str(k) for k in SE1_SAMPLE_SIZES if k <= 500]
BANDS = [("head", "ranks 1–20", "#2a78d6"), ("band", "ranks 81–100", "#eb6834")]
MARGIN = 0.15  # share of the plotted range added above and below

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13})


def load_tables():
    per_pair = pd.read_csv(RESULTS_DIR / "figure1_bands" / "per_pair_ap.csv.gz")
    cols = ["popularity"] + [f"k{k}" for k in KS] + ["all"]
    tables = {}
    for c, g in per_pair.groupby("candidates"):
        t = g.groupby("category")[cols].mean()
        t = t.reindex([x for x in PAPER_CATEGORY_ORDER if x in t.index])
        t.loc["Overall"] = t.mean()
        tables[c] = t
    return tables


def plot_panel(tables, panel, path, legend=None):
    """One panel; legend None (no legend), "inside" (between the popularity lines) or "outside" (below the plot)."""
    xs = KS + ["All"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    values = []
    for c, label, color in BANDS:
        row = tables[c].loc[panel]
        curve = [row[f"k{k}"] for k in KS] + [row["all"]]
        values += curve + [row["popularity"]]
        ax.plot(xs, curve, marker="o", ms=5, lw=2, color=color, label=f"SocialVec, {label}")
        ax.axhline(row["popularity"], color=color, linestyle=":", lw=2, label=f"Popularity, {label}")
    lo, hi = min(values), max(values)
    pad = MARGIN * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(CATEGORY_LABELS.get(panel, panel), fontweight="bold")
    ax.set_xlabel("Number of sampled accounts")
    ax.set_ylabel("MAP")
    ax.grid(True, alpha=0.4)
    style = dict(ncol=2, prop={"weight": "bold", "size": 10.5}, columnspacing=1.2, handlelength=1.8)
    if legend == "inside":  # centred in the empty band between the two popularity lines
        pops = sorted(tables[c].loc[panel, "popularity"] for c, _, _ in BANDS)
        ax.legend(loc="center", bbox_to_anchor=(0.5, sum(pops) / 2), bbox_transform=ax.get_yaxis_transform(),
                  framealpha=1, **style)
    elif legend == "outside":  # below the x-axis label, outside the plot area
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, **style)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.name)


def slug(panel):
    return CATEGORY_LABELS.get(panel, panel).lower().replace(" ", "_")


def main():
    tables = load_tables()
    if "--variants" in sys.argv:  # the requested legend variants only
        out = RESULTS_DIR / "figure1_panels" / "variants"
        out.mkdir(parents=True, exist_ok=True)
        plot_panel(tables, "Overall", out / "figure1_overall_legend_outside", "outside")
        plot_panel(tables, "MusicalArtist", out / "figure1_musical_artists_legend_outside", "outside")
        plot_panel(tables, "MusicalArtist", out / "figure1_musical_artists_no_legend")
        return
    out = RESULTS_DIR / "figure1_panels"
    out.mkdir(parents=True, exist_ok=True)
    for i, panel in enumerate(tables["head"].index, start=1):
        plot_panel(tables, panel, out / f"figure1_{i:02d}_{slug(panel)}", "inside" if i == 1 else None)


if __name__ == "__main__":
    main()
