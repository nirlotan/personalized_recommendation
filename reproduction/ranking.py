"""
Ranking candidate entities and scoring the rankings with average precision.
"""

import numpy as np


def cosine_distances(user_vec: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Cosine distance between a user vector and each candidate embedding (row).

    A zero user vector (no known entities in the profile) yields NaN for every
    candidate, as scipy's `cosine` does.
    """
    user_vec = user_vec.astype(np.float64)
    candidates = candidates.astype(np.float64)
    norms = np.linalg.norm(candidates, axis=1) * np.linalg.norm(user_vec)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - candidates @ user_vec / norms


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """
    1-based rank of each candidate when sorting `scores` in ascending order.

    NaN scores go last in their original order (pandas `sort_values` behaviour).
    """
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def popularity_ranks(followers: np.ndarray) -> np.ndarray:
    """1-based rank of each candidate by follower count (most followed first)."""
    return ranks_from_scores(-np.asarray(followers, dtype=np.float64))


def average_precision(relevant_ranks) -> float:
    """
    Average precision given the ranks of the relevant items in a full ranking.

    With relevant items sorted by rank r_1 < r_2 < ... < r_R, AP = mean(i / r_i).
    Every relevant item is among the ranked candidates, so this is standard AP.
    """
    ranks = np.sort(np.asarray(relevant_ranks, dtype=np.float64))
    if len(ranks) == 0:
        return np.nan
    return float(np.mean(np.arange(1, len(ranks) + 1) / ranks))
