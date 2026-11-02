"""
Figures: MAP vs. number of sampled entities (Figure 1) and socio-demographic radar plots (Figure 2).
"""

import math
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CATEGORY_LABELS, SE1_SAMPLE_SIZES, SE2_SAMPLE_SIZES
from .paper_results import RADAR_TRAITS


def plot_figure1(fig1: pd.DataFrame, path: Path) -> None:
    """
    One panel per category (plus 'Overall'): SE1 and SE2 sampling curves against the
    popularity, SE1-All and SE2-All reference lines. SE1 uses the paper's (legacy) profiles.
    """
    panels = list(fig1.index)
    cols = 5
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)

    se1_sizes = [k for k in SE1_SAMPLE_SIZES if k <= 500]
    se1_x = [str(k) for k in se1_sizes] + ["All"]
    se2_x = [str(k) for k in SE2_SAMPLE_SIZES]

    for ax, panel in zip(axes.flat, panels):
        row = fig1.loc[panel]
        ax.axhline(row["popularity"], color="slategray", linestyle=(0, (5, 5)), label="Popularity")
        ax.axhline(row["se1_all_legacy"], color="darkgreen", linestyle="--", label="SE1: All")
        ax.axhline(row["se2_all"], color="brown", linestyle=":", label="SE2: All")
        ax.plot(se1_x, [row[f"se1_legacy_k{k}"] for k in se1_sizes] + [row["se1_all_legacy"]],
                marker="P", color="chocolate", label="SE1: Sampling")
        ax.plot(se2_x, [row[f"se2_k{k}"] for k in SE2_SAMPLE_SIZES],
                marker="v", color="tab:blue", label="SE2: Sampling")
        ax.set_title(CATEGORY_LABELS.get(panel, panel), fontweight="bold")
        ax.set_xlim(se1_x[0], se1_x[-1])
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, alpha=0.4)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=11)
    fig.supxlabel("Number of entities in the user representation", y=0.035)
    fig.supylabel("MAP")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _radar(ax, data: pd.DataFrame, title: str) -> None:
    angles = np.linspace(0, 2 * np.pi, len(RADAR_TRAITS), endpoint=False).tolist()
    angles += angles[:1]
    markers = ["<", "^", "X", "*", "P", "o", "s"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, values) in enumerate(data.iterrows()):
        values = list(values) + [values.iloc[0]]
        if name == "ALL":
            ax.plot(angles, values, color="black", linestyle="dotted", linewidth=2.5, label="Category average")
        else:
            ax.plot(angles, values, color=colors[i % len(colors)], marker=markers[i % len(markers)],
                    markersize=8, linewidth=2, label=name)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), RADAR_TRAITS, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"], color="grey", size=8)
    ax.set_title(title, size=14, y=1.12, loc="left")
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=8)


def plot_figure2(radars: Dict[str, pd.DataFrame], path: Path) -> None:
    """Radar plots of the follower profiles of selected entities, one panel per category."""
    cols = 3
    rows = math.ceil(len(radars) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.5 * rows),
                             subplot_kw=dict(polar=True), squeeze=False)
    for ax, (title, data) in zip(axes.flat, radars.items()):
        _radar(ax, data, title)
    for ax in axes.flat[len(radars):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
