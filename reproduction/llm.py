"""
LLM-based personalization (Table 3): GPT-4o ranks the candidates of a category
given a list of entity names the user follows.

Conditions (as in legacy_code/recommendation_p2.ipynb):
    llm_baseline - no user information; one ranking per category.
    llm_k12/25   - names of 12/25 entities sampled from the user's closed-world profile.
    llm_k50      - names of 50 accounts sampled from everything the user follows
                   (only accounts with a SocialVec name are listed).

The prompts are the authors' dspy signatures. Running all conditions takes
about 33.5K GPT-4o calls, so responses are cached on disk and runs can resume.
"""

import ast
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import SEED

PersonalRanker = Callable[[str, str], str]   # (followed account names, candidates JSON) -> ranked ids
DefaultRanker = Callable[[str], str]         # (candidates JSON) -> ranked ids
CONDITIONS = {"llm_k12": 12, "llm_k25": 25, "llm_k50": 50}


def candidates_json(category: pd.DataFrame, by_popularity: bool) -> str:
    """The candidate list given to the model: [{twitter_user_id, twitter_name}, ...]."""
    if by_popularity:
        category = category.sort_values("followers_count", ascending=False)
    return category[["twitter_user_id", "twitter_name"]].to_json(orient="records")


def names_of(account_ids, names: Dict[str, str]) -> str:
    """Comma-separated display names, skipping accounts without a known name."""
    found = (names.get(str(a)) for a in account_ids)
    return ", ".join(n for n in found if n)


def user_prompts(users: pd.DataFrame, categories: Dict[str, pd.DataFrame], names: Dict[str, str],
                 seed: int = SEED, exclude_target: bool = False) -> pd.DataFrame:
    """
    The 'followed accounts' text of every user for each condition.

    Args:
        exclude_target: drop the target category's candidates from the k=50 sample.
            The legacy code meant to but did not (see experiments.py), so the default
            reproduces the published setup.
    """
    rows = []
    for user_row, user in users.iterrows():
        friends = list(user["friends_ids"])
        if exclude_target:
            target = set(categories[user["topic"]]["twitter_user_id"])
            friends = [f for f in friends if f not in target]
        row = {"user": user_row, "topic": user["topic"]}
        for condition, k in CONDITIONS.items():
            population = friends if k == 50 else list(user["non_topic_accounts"])
            sample = random.Random(seed).sample(population, k) if k < len(population) else population
            row[condition] = names_of(sample, names)
        rows.append(row)
    return pd.DataFrame(rows).set_index("user")


def llm_average_precision(ranked, relevant_ids) -> float:
    """
    Average precision of a model's ranked id list (legacy `calculate_map`).

    Unparseable output scores 0. Relevant ids missing from the output are skipped,
    as in the legacy code.
    """
    if isinstance(ranked, str):
        try:
            ranked = ast.literal_eval(ranked.strip())
        except (ValueError, SyntaxError):
            return 0.0
    if not isinstance(ranked, list) or not ranked:
        return 0.0
    position = {}
    for i, account in enumerate(ranked):
        try:
            position.setdefault(int(account), i + 1)
        except (TypeError, ValueError):
            continue
    found = sorted(position[int(a)] for a in relevant_ids if int(a) in position)
    if not found:
        return 0.0
    return float(np.mean([(i + 1) / p for i, p in enumerate(found)]))


def make_dspy_rankers(model: str = "azure/gpt-4o", lm=None, **lm_kwargs):
    """
    GPT-4o rankers built with the authors' dspy signatures.

    For Azure, set AZURE_API_KEY / AZURE_API_BASE / AZURE_API_VERSION (or pass
    api_key, api_base, api_version). Any litellm model name works, e.g. "openai/gpt-4o".
    A ready-made dspy LM can be passed as `lm` instead.
    """
    import dspy

    class TwitterRecommendation(dspy.Signature):
        """
        given the below Twitter user, to which we provide a sampled list of accounts the user already follows, and a list of possible accounts to recommend for the user to follow, create a prioritized recommendation list.
        You must rank the entire list provided to you.
        Try to deduce what you think the user might like based on the list of accounts they already follow.
        Return ONLY a valid list object with the ranked account twitter_user_id sorted by relevance (highest to lowest). Do not include name or reasoning.
        Example format: [71026122, 141341662, 567562934, ...]
        """
        followed_accounts = dspy.InputField(desc="Sampled list of other accounts the user already follows")
        potential_accounts = dspy.InputField(desc="JSON list of potential accounts to recommend with twitter_user_id and names")
        ranked_ids = dspy.OutputField(desc="list with the account twitter_user_id sorted by priority as a ranked key")

    class TwitterDefaultRecommendation(dspy.Signature):
        """
        given the provided list of possible accounts to recommend for a twitter user to follow, create a prioritized recommendation list.
        You must rank the entire list provided to you.
        Return ONLY a valid list object with the ranked account twitter_user_id sorted by relevance (highest to lowest). Do not include name or reasoning.
        Example format: [71026122, 141341662, 567562934, ...]
        """
        potential_accounts = dspy.InputField(desc="JSON list of potential accounts to recommend with twitter_user_id and names")
        ranked_ids = dspy.OutputField(desc="list with the account twitter_user_id sorted by priority as a ranked key")

    lm = lm or dspy.LM(model, **lm_kwargs)
    personal = dspy.Predict(TwitterRecommendation)
    default = dspy.Predict(TwitterDefaultRecommendation)

    def rank_personal(followed: str, candidates: str) -> str:
        with dspy.context(lm=lm):
            return personal(followed_accounts=followed, potential_accounts=candidates).ranked_ids

    def rank_default(candidates: str) -> str:
        with dspy.context(lm=lm):
            return default(potential_accounts=candidates).ranked_ids

    return rank_personal, rank_default


class ResponseCache:
    """Model responses stored as JSON lines, so interrupted runs can resume."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: Dict[str, str] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    self.data[item["key"]] = item["response"]

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def put(self, key: str, response: str) -> None:
        with self.lock:
            self.data[key] = response
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": response}) + "\n")


def run_llm(users: pd.DataFrame, categories: Dict[str, pd.DataFrame], names: Dict[str, str],
            rank_personal: PersonalRanker, rank_default: DefaultRanker, cache_path: Path,
            users_per_category: Optional[int] = None, workers: int = 8,
            exclude_target: bool = False) -> pd.DataFrame:
    """
    Run every LLM condition and score it.

    Args:
        users_per_category: evaluate only the first N users of each category (for a cheap trial run).
        workers: number of concurrent model calls.

    Returns:
        DataFrame with one row per user: topic and the AP of each condition.
    """
    if users_per_category is not None:
        users = users.groupby("topic", group_keys=False, sort=False).head(users_per_category)
    prompts = user_prompts(users, categories, names, exclude_target=exclude_target)
    cache = ResponseCache(cache_path)

    def call(key: str, fn: Callable[[], str]) -> str:
        cached = cache.get(key)
        if cached is not None:
            return cached
        try:
            response = str(fn())
        except Exception as exc:  # the legacy code scored failed calls as an empty list
            print(f"{key}: {exc}")
            return "[]"
        cache.put(key, response)
        return response

    baseline = {
        topic: call(f"baseline|{topic}", lambda c=cat: rank_default(candidates_json(c, by_popularity=False)))
        for topic, cat in categories.items()
    }
    by_popularity = {topic: candidates_json(cat, by_popularity=True) for topic, cat in categories.items()}

    jobs = [(user_row, condition) for user_row in prompts.index for condition in CONDITIONS]

    def run_job(job):
        user_row, condition = job
        topic = prompts.at[user_row, "topic"]
        followed = prompts.at[user_row, condition]
        return job, call(f"{condition}|{user_row}", lambda: rank_personal(followed, by_popularity[topic]))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = dict(tqdm(pool.map(run_job, jobs), total=len(jobs), desc="LLM rankings"))

    rows = []
    for user_row, user in users.iterrows():
        relevant: List = user["topic_accounts"]
        r = {"user": user_row, "topic": user["topic"],
             "llm_baseline": llm_average_precision(baseline[user["topic"]], relevant)}
        for condition in CONDITIONS:
            r[condition] = llm_average_precision(responses[(user_row, condition)], relevant)
        rows.append(r)
    return pd.DataFrame(rows)
