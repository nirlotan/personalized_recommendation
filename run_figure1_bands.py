"""
Figure 1 on the full-scale setup of Table 6 (results_v3): MAP as a function of the number of
accounts sampled for the user profile, for the original top-20 candidates and for the
candidates ranked 81-100 by follower count.

Pairs: every experiment user, in every category with at least 100 listed entities in which the user
follows at least one entity of each band (the matched pairs of Table 6). Profiles exclude every entity
known to belong to the category (original sheet, top-20 list and extended list). The k accounts are
drawn from the remaining followed accounts with the paper's sampling (`sampled_average`, seed 76),
so both bands use the same profile for a pair.

    python run_figure1_bands.py

Writes results/figure1_bands/per_pair_ap.csv.gz.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from reproduction.config import PAPER_CATEGORY_ORDER, RESULTS_DIR, SE1_SAMPLE_SIZES, SEED
from reproduction.data import load_categories, load_users
from reproduction.embeddings import SocialVec
from reproduction.experiments import sampled_average
from reproduction.popularity_bands import (
    expected_random_ap, load_category_sheets, load_extended_sheets, make_candidates, popularity_band)
from reproduction.ranking import average_precision

START, END = 81, 100


def main():
    out = RESULTS_DIR / "figure1_bands"
    out.mkdir(parents=True, exist_ok=True)
    sv = SocialVec("2020")
    head = load_categories()
    users = load_users(head)
    sheets, extended = load_category_sheets(), load_extended_sheets()

    rows = []
    for category in [c for c in PAPER_CATEGORY_ORDER if c in sheets and len(sheets[c]) >= END]:
        band = popularity_band(sheets[category], START, END)
        band = band[[i in sv for i in band["twitter_user_id"]]]
        cands = {"head": make_candidates(head[category], sv), "band": make_candidates(band, sv)}
        sets = {k: {int(i) for i in c.ids} for k, c in cands.items()}
        excluded = ({int(i) for i in sheets[category]["twitter_user_id"]} | sets["head"]
                    | ({int(i) for i in extended[category]["twitter_user_id"]} if category in extended else set()))
        excluded_arr = np.fromiter(excluded, dtype=np.int64)

        for user_row, user in tqdm(users.iterrows(), total=len(users), desc=category):
            friends = np.asarray(user["friends_ids"], dtype=np.int64)
            fset = set(friends.tolist())
            if sets["head"].isdisjoint(fset) or sets["band"].isdisjoint(fset):
                continue
            idx = sv.indices(friends[~np.isin(friends, excluded_arr)])
            profiles = {f"k{k}": sampled_average(sv, idx, k, SEED) for k in SE1_SAMPLE_SIZES}
            profiles["all"] = sv.average(idx)
            for name, cand in cands.items():
                relevant = [cand.position[f] for f in fset if f in cand.position]
                r = {"category": category, "user": user_row, "topic": user["topic"], "candidates": name,
                     "n_relevant": len(relevant), "n_profile": int((idx >= 0).sum()),
                     "random": expected_random_ap(len(cand.ids), len(relevant)),
                     "popularity": average_precision(cand.popularity_ranks[relevant])}
                for p, vec in profiles.items():
                    r[p] = cand.average_precision(vec, relevant)
                rows.append(r)

    per_pair = pd.DataFrame(rows)
    per_pair.to_csv(out / "per_pair_ap.csv.gz", index=False)
    cols = ["random", "popularity"] + [f"k{k}" for k in SE1_SAMPLE_SIZES] + ["all"]
    for name, g in per_pair.groupby("candidates"):
        t = g.groupby("category")[cols].mean()
        print(name, g[["category", "user"]].drop_duplicates().shape[0], "pairs")
        print(t.mean().round(3).to_string())


if __name__ == "__main__":
    main()
