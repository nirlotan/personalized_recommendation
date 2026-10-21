"""
Reproduce the results of "Social Knowledge for Cross-Domain User Preference Modeling".

    python run_reproduction.py                 # Tables 2 and 4, Figures 1 and 2 (~2 minutes)

Outputs go to results/; results/REPORT.md compares every number with the paper.
The LLM experiments (Table 3) are run separately with run_llm_experiments.py;
the report includes their results when results/llm/ exists.
"""

import argparse
from pathlib import Path

from reproduction.config import RESULTS_DIR
from reproduction.data import load_categories, load_users
from reproduction.embeddings import SocialVec
from reproduction.experiments import evaluate_users, figure1, table2, table4
from reproduction.paper_results import FIGURE2, FIGURE2_SHEETS
from reproduction.plots import plot_figure1, plot_figure2
from reproduction.report import build_report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--skip-demographics", action="store_true", help="skip Figure 2")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    categories = load_categories()
    users = load_users(categories)
    print(f"{len(users):,} users, {sum(len(c) for c in categories.values())} candidate entities")

    # Tables 2 and 4, Figure 1
    sv = SocialVec("2020")
    per_user = evaluate_users(users, categories, sv)
    per_user.to_csv(out / "per_user_ap.csv.gz", index=False)
    t2, fig1 = table2(per_user), figure1(per_user)
    grid, pooled = table4(per_user, "grid"), table4(per_user, "pooled")
    t2.to_csv(out / "table2.csv")
    fig1.to_csv(out / "figure1.csv")
    grid.to_csv(out / "table4_per_category.csv")
    pooled.to_csv(out / "table4_pooled.csv")
    plot_figure1(fig1, out / "figure1.png")
    del sv

    # Figure 2
    radars = None
    if not args.skip_demographics:
        from reproduction.demographics import category_radar, classify_users, entity_profiles

        traits = classify_users(users, SocialVec("2020c"))
        profiles = entity_profiles(users, traits, categories)
        profiles.to_csv(out / "entity_follower_profiles.csv")
        radars = {
            title: category_radar(profiles, categories[FIGURE2_SHEETS[title]],
                                  [name for name in entities if name != "ALL"])
            for title, entities in FIGURE2.items()
        }
        plot_figure2(radars, out / "figure2.png")

    report = build_report(per_user, categories, t2, fig1, grid, pooled, radars, out / "llm")
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Done. See {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
