"""
LLM-based personalization (Table 3): an LLM ranks the candidates of a category
given a list of entity names the user follows.

Conditions (as in legacy_code/recommendation_p2.ipynb):
    llm_baseline - no user information; one ranking per category.
    llm_k12/25   - names of 12/25 entities sampled from the user's closed-world profile.
    llm_k50      - names of 50 accounts sampled from everything the user follows
                   (only accounts with a SocialVec name are listed).

The prompts are the authors' dspy signatures. In the original setup the
personalized prompts list the candidates sorted by follower count, and the
baseline prompt lists them in spreadsheet order. Responses are cached on disk,
so interrupted runs resume.
"""

import ast
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from tqdm import tqdm

from .config import ROOT, SEED

PersonalRanker = Callable[[str, str], str]   # (followed account names, candidates JSON) -> ranked ids
DefaultRanker = Callable[[str], str]         # (candidates JSON) -> ranked ids
CONDITIONS = {"llm_k12": 12, "llm_k25": 25, "llm_k50": 50}


class BudgetExceeded(RuntimeError):
    pass


class SpendingCap:
    """
    Stops making model calls once the cost reported by litellm (summed over the
    dspy LM's call history) reaches `limit_usd`.
    """

    def __init__(self, lm, limit_usd: float):
        self.lm, self.limit = lm, limit_usd
        self.lock = threading.Lock()
        self.seen, self.spent, self.calls = 0, 0.0, 0

    def update(self) -> float:
        with self.lock:
            history = self.lm.history
            if len(history) < self.seen:  # history was trimmed; count from its new end
                self.seen = len(history)
            for entry in history[self.seen:]:
                self.spent += entry.get("cost") or 0.0
                self.calls += 1
            self.seen = len(history)
            return self.spent

    def guard(self, fn):
        """Wrap a ranker so it refuses to call the model once the cap is reached."""
        def wrapped(*args):
            if self.update() >= self.limit:
                raise BudgetExceeded(f"spending cap of ${self.limit:.2f} reached")
            return fn(*args)
        return wrapped


def load_openai_key() -> str:
    """OPENAI_API_KEY / OPENAI_KEY from the environment, or from .env.local / .env in the repository."""
    for name in ("OPENAI_API_KEY", "OPENAI_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    for filename in (".env.local", ".env"):
        path = ROOT / filename
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                name, _, value = line.partition("=")
                if name.strip() in ("OPENAI_API_KEY", "OPENAI_KEY"):
                    return value.strip().strip('"').strip("'")
    raise RuntimeError("No OpenAI key found (OPENAI_API_KEY / OPENAI_KEY, or .env.local)")


def candidates_json(candidates: pd.DataFrame, order: Optional[List[int]] = None, by_popularity: bool = False) -> str:
    """
    The candidate list given to the model: [{twitter_user_id, twitter_name}, ...].

    Args:
        order: candidate ids in the order to present them (overrides by_popularity).
        by_popularity: sort by follower count (the original personalized prompts).
    """
    if order is not None:
        candidates = candidates.set_index("twitter_user_id").loc[order].reset_index()
    elif by_popularity:
        candidates = candidates.sort_values("followers_count", ascending=False)
    return candidates[["twitter_user_id", "twitter_name"]].to_json(orient="records")


def presented_order(candidates_json_text: str) -> List[int]:
    return [int(c["twitter_user_id"]) for c in json.loads(candidates_json_text)]


def names_of(account_ids, names: Dict[str, str]) -> str:
    """Comma-separated display names, skipping accounts without a known name."""
    found = (names.get(str(a)) for a in account_ids)
    return ", ".join(n for n in found if n)


def user_samples(users: pd.DataFrame, categories: Dict[str, pd.DataFrame], names: Dict[str, str],
                 seed: int = SEED, exclude_target: bool = False,
                 exclude: Optional[Dict[str, Set[int]]] = None) -> pd.DataFrame:
    """
    The accounts listed in every user's prompt for each condition: a random sample
    of followed accounts, keeping only those with a known name.

    Args:
        exclude_target: drop the target category's candidates from the k=50 sample.
            The legacy code meant to but did not (see experiments.py), so the default
            reproduces the published setup.
        exclude: per topic, account ids to drop from every condition's sample
            (used to keep a category's entities out of its own prompts).

    Returns:
        One row per user (indexed like `users`): topic and one list of account ids per condition.
    """
    rows = []
    for user_row, user in users.iterrows():
        topic = user["topic"]
        friends = list(user["friends_ids"])
        closed_world = list(user["non_topic_accounts"])
        if exclude_target:
            target = set(categories[topic]["twitter_user_id"])
            friends = [f for f in friends if f not in target]
        if exclude is not None:
            friends = [f for f in friends if f not in exclude[topic]]
            closed_world = [a for a in closed_world if int(a) not in exclude[topic]]
        row = {"user": user_row, "topic": topic}
        for condition, k in CONDITIONS.items():
            population = friends if k == 50 else closed_world
            sample = random.Random(seed).sample(population, k) if k < len(population) else population
            row[condition] = [int(a) for a in sample if names.get(str(a))]
        rows.append(row)
    return pd.DataFrame(rows).set_index("user")


def user_prompts(users: pd.DataFrame, categories: Dict[str, pd.DataFrame], names: Dict[str, str],
                 seed: int = SEED, exclude_target: bool = False,
                 exclude: Optional[Dict[str, Set[int]]] = None) -> pd.DataFrame:
    """The 'followed accounts' text of every user for each condition (see `user_samples`)."""
    samples = user_samples(users, categories, names, seed, exclude_target, exclude)
    for condition in CONDITIONS:
        samples[condition] = samples[condition].apply(lambda ids: names_of(ids, names))
    return samples


def parse_ranking(ranked) -> Optional[list]:
    """
    The list in a model response. Some models append dspy's end-of-output marker
    (e.g. '[[ ## completed ##]]') to the answer field, so if the whole text is not
    a list literal, the first bracketed list of ids in it is used.
    """
    if isinstance(ranked, str):
        try:
            ranked = ast.literal_eval(ranked.strip())
        except (ValueError, SyntaxError):
            match = re.search(r"\[[\d\s,'\"]+\]", ranked)
            if match is None:
                return None
            ranked = ast.literal_eval(match.group(0))
    return ranked if isinstance(ranked, list) and ranked else None


def ranked_ids(ranked) -> List[int]:
    """The distinct integer ids of a model response, in order."""
    ids, seen = [], set()
    for account in parse_ranking(ranked) or []:
        try:
            account = int(account)
        except (TypeError, ValueError):
            continue
        if account not in seen:
            seen.add(account)
            ids.append(account)
    return ids


def llm_average_precision(ranked, relevant_ids) -> float:
    """
    Average precision of a model's ranked id list (legacy `calculate_map`).

    Unparseable output scores 0. Relevant ids missing from the output are skipped,
    as in the legacy code.
    """
    position = {account: i + 1 for i, account in enumerate(ranked_ids(ranked))}
    found = sorted(position[int(a)] for a in relevant_ids if int(a) in position)
    if not found:
        return 0.0
    return float(np.mean([(i + 1) / p for i, p in enumerate(found)]))


def order_agreement(ranked, reference: List[int]) -> float:
    """Kendall's tau between a model's ranking and a reference order, over the ids in both."""
    ids = [a for a in ranked_ids(ranked) if a in set(reference)]
    if len(ids) < 3:
        return np.nan
    ref_pos = {a: i for i, a in enumerate(reference)}
    return float(kendalltau(range(len(ids)), [ref_pos[a] for a in ids]).statistic)


def coverage(ranked, candidate_ids: List[int]) -> float:
    """Share of the candidates that appear in the model's ranking."""
    return len(set(ranked_ids(ranked)) & set(candidate_ids)) / len(candidate_ids)


def make_dspy_rankers(model: str = "azure/gpt-4o", lm=None, **lm_kwargs):
    """
    LLM rankers built with the authors' dspy signatures.

    `model` is any litellm model name (e.g. "openai/gpt-6-luna"); extra keyword
    arguments (api_key, reasoning_effort, ...) go to `dspy.LM`. A ready-made dspy
    LM can be passed as `lm` instead.
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

    def call(self, key: str, fn: Callable[[], str]) -> str:
        cached = self.get(key)
        if cached is not None:
            return cached
        try:
            response = str(fn())
        except BudgetExceeded:
            raise
        except Exception as exc:  # the legacy code scored failed calls as an empty list
            print(f"{key}: {type(exc).__name__}: {str(exc)[:200]}")
            return "[]"
        self.put(key, response)
        return response


def run_llm(users: pd.DataFrame, candidates: Dict[str, pd.DataFrame], prompts: pd.DataFrame,
            rank_personal: PersonalRanker, rank_default: DefaultRanker, cache_path: Path,
            shuffle: bool = False, seed: int = SEED, workers: int = 8,
            conditions: List[str] = list(CONDITIONS), baseline: bool = True) -> pd.DataFrame:
    """
    Run the LLM conditions and score them.

    Args:
        candidates: per topic, the candidate entities (twitter_user_id, twitter_name,
            followers_count) in their baseline (spreadsheet) order.
        prompts: `user_prompts` output for these users.
        shuffle: present the candidates in a random order per user (and per condition)
            instead of sorted by follower count.
        baseline: also run the non-personalized ranking (one call per topic).

    Returns:
        One row per user: topic, number of relevant candidates, and for the baseline
        and each condition its AP, the raw response and the presented candidate order.
    """
    cache = ResponseCache(cache_path)
    baseline_json = {t: candidates_json(c) for t, c in candidates.items()}
    popularity_json = {t: candidates_json(c, by_popularity=True) for t, c in candidates.items()}

    def personal_json(user_row, topic, condition):
        if not shuffle:
            return popularity_json[topic]
        ids = list(candidates[topic]["twitter_user_id"])
        random.Random(f"{seed}:{user_row}:{condition}").shuffle(ids)
        return candidates_json(candidates[topic], order=ids)

    baseline_responses = ({t: cache.call(f"baseline|{t}", lambda j=j: rank_default(j)) for t, j in baseline_json.items()}
                          if baseline else {})

    jobs = [(user_row, c) for user_row in prompts.index for c in conditions]

    def run_job(job):
        user_row, condition = job
        topic = prompts.at[user_row, "topic"]
        cands = personal_json(user_row, topic, condition)
        followed = prompts.at[user_row, condition]
        return job, (cache.call(f"{condition}|{user_row}", lambda: rank_personal(followed, cands)), cands)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = dict(tqdm(pool.map(run_job, jobs), total=len(jobs), desc=f"LLM ({cache_path.stem})"))

    rows = []
    for user_row, user in users.loc[prompts.index].iterrows():
        topic = user["topic"]
        candidate_ids = set(candidates[topic]["twitter_user_id"])
        relevant = [f for f in user["friends_ids"] if f in candidate_ids]
        r = {"user": user_row, "topic": topic, "n_relevant": len(relevant)}
        if baseline:
            r.update({"llm_baseline": llm_average_precision(baseline_responses[topic], relevant),
                      "resp_llm_baseline": baseline_responses[topic],
                      "order_llm_baseline": json.dumps(presented_order(baseline_json[topic]))})
        for condition in conditions:
            response, cands = responses[(user_row, condition)]
            r[condition] = llm_average_precision(response, relevant)
            r[f"resp_{condition}"] = response
            r[f"order_{condition}"] = json.dumps(presented_order(cands))
        rows.append(r)
    return pd.DataFrame(rows)


def baseline_order_trials(candidates: Dict[str, pd.DataFrame], rank_default: DefaultRanker,
                          cache_path: Path, repeats: int = 10, seed: int = SEED, workers: int = 8) -> pd.DataFrame:
    """
    Non-personalized rankings under repeated identical prompts and under shuffled
    candidate orders: `repeats` calls per category in the original order and
    `repeats` calls with a different random order each.

    Returns:
        One row per call: topic, order type ('original'/'shuffled'), trial,
        presented order and response.
    """
    cache = ResponseCache(cache_path)
    jobs = []
    for topic, cands in candidates.items():
        original = list(cands["twitter_user_id"])
        for trial in range(repeats):
            jobs.append((topic, "original", trial, original))
            shuffled = original.copy()
            random.Random(f"{seed}:baseline:{topic}:{trial}").shuffle(shuffled)
            jobs.append((topic, "shuffled", trial, shuffled))

    def run_job(job):
        topic, kind, trial, order = job
        text = candidates_json(candidates[topic], order=order)
        return job, cache.call(f"baseline|{topic}|{kind}|{trial}", lambda: rank_default(text))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(pool.map(run_job, jobs), total=len(jobs), desc="LLM baseline trials"))
    return pd.DataFrame([{"topic": t, "order_type": k, "trial": i, "order": json.dumps(o), "response": r}
                         for (t, k, i, o), r in results])
