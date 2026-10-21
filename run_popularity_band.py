"""
Social similarity vs. popularity when the candidates are less popular entities.

    python run_popularity_band.py                  # entities ranked 81-100 by follower count
    python run_popularity_band.py --start 41 --end 60

Outputs go to results/popularity_band_<start>_<end>/.
"""

import argparse
from pathlib import Path

from reproduction.config import RESULTS_DIR
from reproduction.data import load_categories, load_users
from reproduction.embeddings import SocialVec
from reproduction.popularity_bands import load_category_sheets, run_band_experiment, summarize
from reproduction.report import markdown_table

CONDITIONS = [
    ("category", "band", "exclude_category", "Main: band candidates, all category entities removed from the profile"),
    ("category", "band", "exclude_candidates", "Band candidates, only the band entities removed from the profile"),
    ("category", "head", "exclude_category", "Control: the paper's top-20 candidates, same users and profiles as Main"),
    ("all", "band", "exclude_category", "Robustness: all experiment users, band candidates, all category entities removed"),
    ("all", "head", "exclude_category", "Robustness control: all experiment users, top-20 candidates"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=81)
    p.add_argument("--end", type=int, default=100)
    args = p.parse_args()

    out = RESULTS_DIR / f"popularity_band_{args.start}_{args.end}"
    out.mkdir(parents=True, exist_ok=True)

    head = load_categories()
    users = load_users(head)
    sheets = load_category_sheets()
    per_user, bands = run_band_experiment(users, sheets, head, SocialVec("2020"), args.start, args.end)
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
    print("\n".join(sections))


if __name__ == "__main__":
    main()
