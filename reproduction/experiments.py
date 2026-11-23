"""
Link-prediction experiments: Table 2, Figure 1 and Table 4 of the paper.

Every user was sampled for one target category. For each user we rank the
20 candidate entities of that category and score the ranking with average
precision, treating the candidates the user follows as the relevant answers.
Per-category MAP is the mean AP over that category's users; the 'overall'
score is the mean over the 14 categories.

User representations (the average embedding of the entities in a profile):
    popularity      - no user model; rank candidates by follower count.
    se1_all_legacy  - all followed entities. This is what the legacy notebook
                      computed: it meant to drop the target-category
                      entities but compared Twitter ids with screen names,
                      so nothing was dropped. This is the paper's 'All' column.
    se1_all         - all followed entities except the target-category
                      candidates (the protocol as described in the paper).
    se1[_legacy]_k{k} - k followed accounts sampled at random (Figure 1, SE1).
    se2_all         - 'closed world': only the user's candidate entities from
                      the other 13 categories (the paper's 'Sampled' column).
    se2_k{k}        - k sampled closed-world entities (Figure 1, SE2).
    grid_k{k}_n{n}  - n closed-world entities from each of k categories (Table 4,
                      as the paper describes it).
    pooled_k{k}_n{n} - n closed-world entities in total, drawn from the union of
                      k categories. The code behind Table 4 is not in the legacy
                      notebooks; this variant matches the published numbers more
                      closely than the described protocol does.
"""

import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    GRID_ACCOUNTS_PER_CATEGORY,
    GRID_CATEGORIES,
    PAPER_CATEGORY_ORDER,
    SE1_SAMPLE_SIZES,
    SE2_SAMPLE_SIZES,
    SEED,
)
from .data import account_categories
from .embeddings import SocialVec
from .ranking import average_precision, cosine_distances, popularity_ranks, ranks_from_scores


@dataclass
class Candidates:
    """The candidate entities of one category."""
    ids: np.ndarray            # Twitter ids, in sheet order
    embeddings: np.ndarray     # (n_candidates, dim)
    popularity_ranks: np.ndarray
    position: Dict[int, int]   # Twitter id -> row in `ids`

    def average_precision(self, user_vec: np.ndarray, relevant_rows: List[int]) -> float:
        ranks = ranks_from_scores(cosine_distances(user_vec, self.embeddings))
        return average_precision(ranks[relevant_rows])


def build_candidates(categories: Dict[str, pd.DataFrame], sv: SocialVec) -> Dict[str, Candidates]:
    candidates = {}
    for name, df in categories.items():
        ids = df["twitter_user_id"].values
        candidates[name] = Candidates(
            ids=ids,
            embeddings=np.stack([sv.vector(i) for i in ids]),
            popularity_ranks=popularity_ranks(df["followers_count"].values),
            position={int(i): row for row, i in enumerate(ids)},
        )
    return candidates


def sampled_average(sv: SocialVec, idx: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Average of k entries of `idx` drawn with `random.seed(seed); random.sample(...)`,
    or of all entries when there are no more than k.

    `random.sample` picks positions using only the population size and the RNG
    state, so sampling positions draws the same items as sampling the list itself.
    """
    if k < len(idx):
        idx = idx[random.Random(seed).sample(range(len(idx)), k)]
    return sv.average(idx)


def category_pools(non_topic: List, nt_idx: np.ndarray, topic: str,
                   categories_of: Dict[int, List[str]], rng: random.Random) -> List[List[int]]:
    """
    Group a user's closed-world entities by category (in random category order),
    each group in random order. Used to build the nested samples of Table 4.
    """
    pools: Dict[str, List[int]] = {}
    for account, idx in zip(non_topic, nt_idx):
        for category in categories_of[int(account)]:
            if category != topic:
                pools.setdefault(category, []).append(int(idx))
    names = sorted(pools)
    rng.shuffle(names)
    for name in names:
        rng.shuffle(pools[name])
    return [pools[name] for name in names]


def evaluate_users(
    users: pd.DataFrame,
    categories: Dict[str, pd.DataFrame],
    sv: SocialVec,
    seed: int = SEED,
    se1_sizes: List[int] = SE1_SAMPLE_SIZES,
    se2_sizes: List[int] = SE2_SAMPLE_SIZES,
    grid_categories: List[int] = GRID_CATEGORIES,
    grid_accounts: List[int] = GRID_ACCOUNTS_PER_CATEGORY,
) -> pd.DataFrame:
    """
    Compute the AP of every user representation for every user.

    Returns:
        DataFrame with one row per user: topic, profile statistics and one AP column
        per representation (see the module docstring for the column names).
    """
    candidates = build_candidates(categories, sv)
    categories_of = account_categories(categories)
    rows = []

    for user_row, user in tqdm(users.iterrows(), total=len(users), desc="Evaluating users"):
        topic = user["topic"]
        cand = candidates[topic]
        relevant = [cand.position[int(a)] for a in user["topic_accounts"]]

        def ap(vec):
            return cand.average_precision(vec, relevant)

        friends = np.asarray(user["friends_ids"], dtype=np.int64)
        friend_idx = sv.indices(friends)
        friend_idx_no_target = friend_idx[~np.isin(friends, cand.ids)]
        non_topic = user["non_topic_accounts"]
        non_topic_idx = sv.indices(non_topic)

        r = {
            "user": user_row,
            "topic": topic,
            "n_relevant": len(relevant),
            "n_followed": len(friends),
            "n_followed_entities": int((friend_idx >= 0).sum()),
            "n_closed_world": len(non_topic),
            "popularity": average_precision(cand.popularity_ranks[relevant]),
            "se1_all_legacy": ap(sv.average(friend_idx)),
            "se1_all": ap(sv.average(friend_idx_no_target)),
            "se2_all": ap(sv.average(non_topic_idx)),
        }
        for k in se1_sizes:
            r[f"se1_legacy_k{k}"] = ap(sampled_average(sv, friend_idx, k, seed))
            r[f"se1_k{k}"] = ap(sampled_average(sv, friend_idx_no_target, k, seed))
        for k in se2_sizes:
            r[f"se2_k{k}"] = ap(sampled_average(sv, non_topic_idx, k, seed))

        # Table 4: samples are nested (larger k or n only adds entities) so the grid is
        # not dominated by sampling noise.
        pools = category_pools(non_topic, non_topic_idx, topic, categories_of,
                               random.Random(f"{seed}:{user_row}"))
        for k in grid_categories:
            pooled = sorted({i for pool in pools[:k] for i in pool})
            random.Random(f"{seed}:{user_row}:{k}").shuffle(pooled)
            for n in grid_accounts:
                idx = sorted({i for pool in pools[:k] for i in pool[:n]})
                r[f"grid_k{k}_n{n}"] = ap(sv.average(np.asarray(idx, dtype=np.int64)))
                r[f"pooled_k{k}_n{n}"] = ap(sv.average(np.asarray(pooled[:n], dtype=np.int64)))
        rows.append(r)

    return pd.DataFrame(rows)


def per_category_map(per_user: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Mean of `columns` per category (paper order) plus an 'Overall' row averaging the categories."""
    table = per_user.groupby("topic")[columns].mean().reindex(PAPER_CATEGORY_ORDER)
    table.loc["Overall"] = table.mean()
    return table


def table2(per_user: pd.DataFrame) -> pd.DataFrame:
    """Table 2: popularity vs. social user embeddings (all entities / closed world)."""
    t = per_category_map(per_user, ["n_relevant", "popularity", "se1_all_legacy", "se1_all", "se2_all"])
    t.loc["Overall", "n_relevant"] = np.nan
    for col in ["se1_all_legacy", "se1_all", "se2_all"]:
        t[f"{col}_gain"] = t[col] / t["popularity"] - 1
    return t


def figure1(per_user: pd.DataFrame, se1_sizes=SE1_SAMPLE_SIZES, se2_sizes=SE2_SAMPLE_SIZES) -> pd.DataFrame:
    """Figure 1: MAP as a function of the number of sampled entities, per category."""
    cols = (["popularity", "se1_all_legacy", "se1_all", "se2_all"]
            + [f"se1_legacy_k{k}" for k in se1_sizes]
            + [f"se1_k{k}" for k in se1_sizes]
            + [f"se2_k{k}" for k in se2_sizes])
    return per_category_map(per_user, cols)


def table4(per_user: pd.DataFrame, variant: str = "grid", grid_categories=GRID_CATEGORIES,
           grid_accounts=GRID_ACCOUNTS_PER_CATEGORY) -> pd.DataFrame:
    """
    Table 4: overall MAP for closed-world profiles built from k categories and n entities.

    Args:
        variant: "grid" for n entities per category, "pooled" for n entities in total.
    """
    overall = per_category_map(
        per_user, [f"{variant}_k{k}_n{n}" for k in grid_categories for n in grid_accounts]
    ).loc["Overall"]
    return pd.DataFrame(
        [[overall[f"{variant}_k{k}_n{n}"] for n in grid_accounts] for k in grid_categories],
        index=pd.Index(grid_categories, name="k categories"),
        columns=pd.Index(grid_accounts, name="n per category"),
    )
