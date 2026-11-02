"""
Loading the experimental dataset: the 14 target categories and the sampled users.

Some list orders below come from Python set operations. They are written
exactly as in the legacy notebooks on purpose: `random.sample` draws from
these lists, so the same set operations are needed to draw the same samples.
Integer hashes are deterministic, so these orders are stable across runs.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import CATEGORIES_XLSX, MIN_EXPERIMENT_ACCOUNTS, USERS_PICKLE

CATEGORY_COLUMNS = ["twitter_screen_name", "twitter_user_id", "twitter_name", "followers_count"]


def load_categories(path: Path = CATEGORIES_XLSX) -> Dict[str, pd.DataFrame]:
    """
    Load the candidate entities of each category (rows marked `use == 1`).

    Returns:
        Dict mapping sheet name to a DataFrame with columns
        twitter_screen_name, twitter_user_id (int64), twitter_name, followers_count.
        The dict keeps the workbook's sheet order and each frame keeps the sheet's row order.
    """
    sheets = pd.read_excel(path, sheet_name=None)
    categories = {}
    for name, df in sheets.items():
        df = df[df["use"] == 1][CATEGORY_COLUMNS].copy()
        df["twitter_user_id"] = df["twitter_user_id"].astype(np.int64)
        categories[name] = df.reset_index(drop=True)
    return categories


def all_candidate_ids(categories: Dict[str, pd.DataFrame]) -> np.ndarray:
    """Ids of all candidate entities, concatenated in sheet order (may contain duplicates)."""
    return np.concatenate([df["twitter_user_id"].values for df in categories.values()])


def experiment_accounts(friends_ids: List[int], candidate_ids: np.ndarray) -> List:
    """The candidate entities (from any category) that a user follows."""
    return list(set(friends_ids).intersection(set(candidate_ids)))


def split_topic_accounts(user_accounts: List, topic_ids: np.ndarray) -> Tuple[List, List]:
    """
    Split a user's experiment accounts into (non-topic accounts, topic accounts).

    The topic accounts are the relevant answers for the target category; the
    non-topic accounts form the user's 'closed-world' profile.
    """
    user_set = set(user_accounts)
    topic_set = set(topic_ids)
    return list(user_set - topic_set), list(user_set.intersection(topic_set))


def load_users(
    categories: Dict[str, pd.DataFrame],
    path: Path = USERS_PICKLE,
    min_experiment_accounts: int = MIN_EXPERIMENT_ACCOUNTS,
) -> pd.DataFrame:
    """
    Load the sampled users and keep those who follow more than
    `min_experiment_accounts` of the candidate entities.

    Each user was sampled for one target category (`topic`), as a follower of
    one entity of that category (`account_selected_for`).

    Returns:
        DataFrame with the original columns plus:
            - experiment_accounts: candidate entities (any category) the user follows
            - non_topic_accounts: experiment accounts outside the user's topic
            - topic_accounts: experiment accounts in the user's topic (ground truth)
    """
    users = pd.read_pickle(path).reset_index(drop=True)
    users = users[users["account_selected_for"].notna() & users["topic"].isin(categories)].copy()

    candidate_ids = all_candidate_ids(categories)
    users["experiment_accounts"] = users["friends_ids"].apply(
        lambda friends: experiment_accounts(friends, candidate_ids)
    )
    users = users[users["experiment_accounts"].apply(len) > min_experiment_accounts].copy()

    splits = [
        split_topic_accounts(accounts, categories[topic]["twitter_user_id"].values)
        for accounts, topic in zip(users["experiment_accounts"], users["topic"])
    ]
    users["non_topic_accounts"] = [s[0] for s in splits]
    users["topic_accounts"] = [s[1] for s in splits]
    return users


def account_categories(categories: Dict[str, pd.DataFrame]) -> Dict[int, List[str]]:
    """Map each candidate entity id to the categories it belongs to (a few belong to two)."""
    mapping: Dict[int, List[str]] = {}
    for name, df in categories.items():
        for account_id in df["twitter_user_id"]:
            mapping.setdefault(int(account_id), []).append(name)
    return mapping
