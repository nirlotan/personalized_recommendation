"""
Social similarity vs. popularity when the candidates are less popular entities.

    python run_popularity_band.py                  # entities ranked 81-100 by follower count
    python run_popularity_band.py --start 41 --end 60
    python run_popularity_band.py --start 181 --end 200 --source extended   # extended lists

Outputs go to results/popularity_band_<start>_<end>[_extended]/.
"""

import argparse
from pathlib import Path

import pandas as pd

from reproduction.config import CATEGORY_LABELS, RESULTS_DIR
from reproduction.data import load_categories, load_users
from reproduction.embeddings import SocialVec
from reproduction.popularity_bands import (
    load_category_sheets,
    load_extended_sheets,
    matched_comparison,
    run_band_experiment,
    summarize,
)
from reproduction.report import markdown_table

CONDITIONS = [
    ("category", "band", "exclude_category", "Main: band candidates, all category entities removed from the profile"),
    ("category", "band", "exclude_candidates", "Band candidates, only the band entities removed from the profile"),
    ("category", "head", "exclude_category", "Control: the paper's top-20 candidates, same users and profiles as Main"),
    ("all", "band", "exclude_category", "Robustness: all experiment users, band candidates, all category entities removed"),
    ("all", "head", "exclude_category", "Robustness control: all experiment users, top-20 candidates"),
]
COMPARISONS = [
    ("category", "Users sampled for each category"),
    ("all", "All experiment users"),
]


def comparison_markdown(head: pd.DataFrame, band: pd.DataFrame, start: int, end: int) -> str:
    """Top-20 and band results side by side, one row per category (paper-style table)."""
    def p(v):
        return "< 0.001" if v < 0.001 else f"{v:.3f}"

    def rel(v):
        return "" if pd.isna(v) else f"{v:.1f}"

    b = f"{start}–{end}"
    lines = [
        f"| Category | Users | Rel./user (1–20) | Popularity (1–20) | Social (1–20) | Δ (1–20) | p (1–20) "
        f"| Rel./user ({b}) | Random ({b}) | Popularity ({b}) | Social ({b}) | Δ ({b}) | p ({b}) |",
        "|" + "---|" * 13,
    ]
    for category in band.index:
        h, r = head.loc[category], band.loc[category]
        label = CATEGORY_LABELS.get(category, "**Overall**")
        lines.append(
            f"| {label} | {r['users']:,.0f} | {rel(h['relevant_per_user'])} | {h['popularity']:.3f} "
            f"| {h['socialvec']:.3f} | {h['gain']:+.0%} | {p(h['p_value'])} | {rel(r['relevant_per_user'])} "
            f"| {r['random']:.3f} | {r['popularity']:.3f} | {r['socialvec']:.3f} | {r['gain']:+.0%} | {p(r['p_value'])} |"
        )
    return "\n".join(lines).replace("+-", "−").replace("| -", "| −")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=81)
    p.add_argument("--end", type=int, default=100)
    p.add_argument("--source", default="original", choices=["original", "extended"],
                   help="original sheets ranked by follower count, or the extended lists "
                        "(data/popular_accounts_extended.xlsx) in their SocialVec popularity order")
    args = p.parse_args()

    suffix = "_extended" if args.source == "extended" else ""
    out = RESULTS_DIR / f"popularity_band_{args.start}_{args.end}{suffix}"
    out.mkdir(parents=True, exist_ok=True)

    head = load_categories()
    users = load_users(head)
    original = load_category_sheets()
    if args.source == "extended":
        # profiles exclude the whole extended list and the original sheet of the category
        per_user, bands = run_band_experiment(users, load_extended_sheets(), head, SocialVec("2020"), args.start, args.end,
                                              by="rank_in_category", also_exclude=original)
    else:
        per_user, bands = run_band_experiment(users, original, head, SocialVec("2020"), args.start, args.end)
    per_user.to_csv(out / "per_user_ap.csv.gz", index=False)
    bands.to_csv(out / "band_entities.csv", index=False)

    sections = [f"# Popularity band {args.start}-{args.end}: summary tables\n"]
    for population, candidates, exclusion, title in CONDITIONS:
        table = summarize(per_user, population, candidates, exclusion)
        table.to_csv(out / f"summary_{population}_{candidates}_{exclusion}.csv")
        shown = table.copy()
        shown["users"] = shown["users"].map(lambda v: f"{v:,.0f}")
        shown["gain"] = shown["gain"].map(lambda v: f"{v:+.1%}")
        shown["p_value"] = shown["p_value"].map(lambda v: f"{v:.1e}")
        sections.append(f"## {title}\n\n" + markdown_table(shown) + "\n")
    (out / "tables.md").write_text("\n".join(sections), encoding="utf-8")

    # Top-20 vs. band on the users evaluated in both (all category entities removed from profiles)
    comparisons = [f"# Top-20 vs. ranks {args.start}-{args.end}: matched users\n"]
    for population, title in COMPARISONS:
        tables = matched_comparison(per_user, population)
        for candidates, table in tables.items():
            table.to_csv(out / f"matched_{population}_{candidates}.csv")
        comparisons.append(f"## {title}\n\n"
                           + comparison_markdown(tables["head"], tables["band"], args.start, args.end) + "\n")
    (out / "comparison_tables.md").write_text("\n".join(comparisons), encoding="utf-8")
    print("\n".join(comparisons))


if __name__ == "__main__":
    main()
