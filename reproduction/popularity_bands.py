"""
Ranking less popular entities: the Table 2 comparison (social similarity vs.
popularity) repeated with candidates taken from a lower popularity band of
each category, e.g. the entities ranked 81-100 by Twitter follower count.

Each category sheet lists up to 100 typed entities; only categories with at
least `end` entities can supply a band ending at rank `end`.
"""

from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .config import CATEGORIES_XLSX, PAPER_CATEGORY_ORDER
from .embeddings import SocialVec
from .experiments import Candidates
from .ranking import average_precision, popularity_ranks


def load_category_sheets(path: Path = CATEGORIES_XLSX) -> Dict[str, pd.DataFrame]:
    """All typed entities of each category (not only the 20 used in the paper), in sheet order."""
    sheets = {}
    for name, df in pd.read_excel(path, sheet_name=None).items():
        df = df.dropna(subset=["twitter_user_id", "followers_count"]).copy()
        df["twitter_user_id"] = df["twitter_user_id"].astype(np.int64)
        sheets[name] = df[["twitter_screen_name", "twitter_user_id", "twitter_name",
                           "followers_count", "use"]].reset_index(drop=True)
    return sheets


def popularity_band(sheet: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """The entities ranked `start`..`end` (1-based, inclusive) by follower count."""
    ranked = sheet.rename_axis("sheet_row").sort_values("followers_count", ascending=False).reset_index()
    ranked.insert(0, "popularity_rank", np.arange(1, len(ranked) + 1))
    return ranked.iloc[start - 1:end].reset_index(drop=True)


def make_candidates(entities: pd.DataFrame, sv: SocialVec) -> Candidates:
    """Candidates for ranking; entities must have a SocialVec embedding."""
    ids = entities["twitter_user_id"].values
    return Candidates(
        ids=ids,
        embeddings=np.stack([sv.vector(i) for i in ids]),
        popularity_ranks=popularity_ranks(entities["followers_count"].values),
        position={int(i): row for row, i in enumerate(ids)},
    )


def expected_random_ap(n_candidates: int, n_relevant: int) -> float:
    """
    Expected AP of a uniformly random ranking of N candidates with R relevant ones:
    E[AP] = (H_N + (R - 1) / (N - 1) * (N - H_N)) / N, with H_N the N-th harmonic number.
    """
    n, r = n_candidates, n_relevant
    h = np.sum(1.0 / np.arange(1, n + 1))
    return float((h + (r - 1) / (n - 1) * (n - h)) / n)


def evaluate_candidates(users: pd.DataFrame, cand: Candidates, excluded: Set[int],
                        sv: SocialVec) -> pd.DataFrame:
    """
    AP of popularity ranking and of social-similarity ranking for each user.

    The user profile is the average embedding of every followed entity except
    those in `excluded`; the relevant answers are the candidates the user follows.
    Users who follow none of the candidates are skipped (AP is undefined).
    """
    candidate_set = {int(i) for i in cand.ids}
    rows = []
    for user_row, user in users.iterrows():
        friends = np.asarray(user["friends_ids"], dtype=np.int64)
        relevant = [cand.position[int(f)] for f in friends if int(f) in candidate_set]
        if not relevant:
            continue
        keep = ~np.isin(friends, list(excluded))
        user_vec = sv.average(sv.indices(friends[keep]))
        rows.append({
            "user": user_row,
            "topic": user["topic"],
            "n_relevant": len(relevant),
            "random": expected_random_ap(len(cand.ids), len(relevant)),
            "popularity": average_precision(cand.popularity_ranks[relevant]),
            "socialvec": cand.average_precision(user_vec, relevant),
        })
    return pd.DataFrame(rows)


def run_band_experiment(users: pd.DataFrame, sheets: Dict[str, pd.DataFrame],
                        head: Dict[str, pd.DataFrame], sv: SocialVec,
                        start: int, end: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the popularity band in every category that has at least `end` entities.

    For each category and each user population ('category': the users sampled
    for that category; 'all': every experiment user), only users who follow at
    least one band entity are evaluated. On those same users we rank:
        band / exclude_category   - band candidates; all the category's listed entities
                                    removed from the profile (main condition)
        band / exclude_candidates - band candidates; only the band entities removed
        head / exclude_category   - the paper's original 20 candidates, same profile
                                    as the main condition (matched control)

    Returns:
        (per-user results, the band entities used per category)
    """
    results, bands = [], []
    categories = [c for c in PAPER_CATEGORY_ORDER if c in sheets and len(sheets[c]) >= end]
    for category in categories:
        band = popularity_band(sheets[category], start, end)
        band["in_socialvec"] = [i in sv for i in band["twitter_user_id"]]
        band.insert(0, "category", category)
        bands.append(band)
        band = band[band["in_socialvec"]]

        band_cand = make_candidates(band, sv)
        head_cand = make_candidates(head[category], sv)
        band_ids = {int(i) for i in band["twitter_user_id"]}
        category_ids = ({int(i) for i in sheets[category]["twitter_user_id"]}
                        | {int(i) for i in head[category]["twitter_user_id"]})

        for population, pop_users in [("category", users[users["topic"] == category]), ("all", users)]:
            follows_band = pop_users["friends_ids"].apply(lambda f: not band_ids.isdisjoint(f))
            pop_users = pop_users[follows_band]
            for candidates, exclusion, cand, excluded in [
                ("band", "exclude_category", band_cand, category_ids),
                ("band", "exclude_candidates", band_cand, band_ids),
                ("head", "exclude_category", head_cand, category_ids),
            ]:
                r = evaluate_candidates(pop_users, cand, excluded, sv)
                r.insert(0, "category", category)
                r.insert(1, "population", population)
                r.insert(2, "candidates", candidates)
                r.insert(3, "exclusion", exclusion)
                results.append(r)
    return pd.concat(results, ignore_index=True), pd.concat(bands, ignore_index=True)


def summarize(per_user: pd.DataFrame, population: str, candidates: str, exclusion: str) -> pd.DataFrame:
    """
    Per-category MAP of both methods, relative gain and a paired Wilcoxon
    signed-rank test on per-user AP; 'Overall' averages the categories
    (its test pools all users).
    """
    sub = per_user[(per_user["population"] == population) & (per_user["candidates"] == candidates)
                   & (per_user["exclusion"] == exclusion)]

    def test(g):
        diff = g["socialvec"] - g["popularity"]
        return wilcoxon(diff).pvalue if (diff != 0).any() else np.nan

    rows = []
    for category, g in sub.groupby("category", sort=False):
        rows.append({"category": category, "users": len(g), "relevant_per_user": g["n_relevant"].mean(),
                     "random": g["random"].mean(), "popularity": g["popularity"].mean(),
                     "socialvec": g["socialvec"].mean(), "p_value": test(g)})
    table = pd.DataFrame(rows).set_index("category")
    table.loc["Overall"] = [len(sub), np.nan, table["random"].mean(), table["popularity"].mean(),
                            table["socialvec"].mean(), test(sub)]
    table["gain"] = table["socialvec"] / table["popularity"] - 1
    return table[["users", "relevant_per_user", "random", "popularity", "socialvec", "gain", "p_value"]]


def matched_comparison(per_user: pd.DataFrame, population: str,
                       exclusion: str = "exclude_category") -> Dict[str, pd.DataFrame]:
    """
    `summarize` for the top-20 ('head') and the band candidates, restricted to the
    (category, user) pairs evaluated in both, so the two rankings share their users.
    """
    sub = per_user[(per_user["population"] == population) & (per_user["exclusion"] == exclusion)]
    pairs = {c: set(zip(g["category"], g["user"])) for c, g in sub.groupby("candidates")}
    shared = pairs["head"] & pairs["band"]
    matched = sub[[pair in shared for pair in zip(sub["category"], sub["user"])]]
    return {c: summarize(matched, population, c, exclusion) for c in ("head", "band")}
