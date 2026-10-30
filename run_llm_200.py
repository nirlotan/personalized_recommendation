"""
LLM vs. SocialVec when all 200 candidates of a category must be ranked.

Do long prompts hurt the LLM? Candidates are ranks 1-200 of the extended category lists
(by SocialVec popularity, only entities with a SocialVec embedding), for the users of the
ranks 81-100 LLM sample who follow at least one of them. Each user's candidates are put in
one random order, used for both rankings by GPT-6 Luna (reasoning effort medium):

    baseline  - no user information
    k200      - the names of up to 200 accounts the user follows (every account of the
                category's extended list and original sheet removed)

With the authors' prompts the model cut long lists short (a pilot returned 20-50 of 200 ids
with 200 followed names), so both prompts add an explicit instruction to return every
candidate (COMPLETE_LIST). Scores use only the candidates the model returned: a relevant
candidate it left out counts as a miss (llm_<kind>). For comparison, llm_<kind>_appended
puts the left-out candidates at the end in the presented order.

The rankings are compared with SocialVec on the same accounts, SocialVec on all followed entities, and
follower-count popularity (python run_popularity_band.py --start 1 --end 200 --source extended).
Each ranking is also cut down to the ranks 181-200 candidates, to compare with the
20-candidate prompts of run_llm_demographics_bands.py.

    python run_llm_200.py --budget 3 [--limit 4]

Per-user scores go to results/llm/candidates_200_per_user.csv.gz.
"""

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from reproduction.config import RESULTS_DIR, SEED
from reproduction.data import load_categories, load_users
from reproduction.embeddings import load_metadata
from reproduction.llm import (
    ResponseCache,
    SpendingCap,
    candidates_json,
    llm_average_precision,
    load_openai_key,
    make_dspy_rankers,
    names_of,
    ranked_ids,
)
from reproduction.popularity_bands import make_candidates
from run_llm_experiments import band_sample, band_setup

OUT = RESULTS_DIR / "llm"
BAND = (1, 200, "extended")
SUBSET = (181, 200, "extended")
K = 200
PROMPT_VERSION = "v2"  # v1: the authors' prompts at reasoning effort low (pilot only)
COMPLETE_LIST = (
    "The candidate list can contain up to 200 accounts. Your answer must contain every twitter_user_id from the "
    "candidate list exactly once, so it is as long as the candidate list. Do not stop early, shorten, truncate or "
    "abbreviate the list, and do not use '...' or '?': write out the complete ranked list."
)


def followed_sample(user, exclude, names):
    """
    Up to K followed accounts, sampled among those with a known name and none from the
    category's lists. (The k = 12/25/50 prompts sample first and then drop unnamed
    accounts, which would leave about 114 of 200.)
    """
    friends = [int(f) for f in user["friends_ids"] if f not in exclude and names.get(str(f))]
    return random.Random(SEED).sample(friends, K) if K < len(friends) else friends


def response_ids(response):
    """Ids of a response; a list the model cut off (e.g. '[1, 2, 3?]') is read up to the cut."""
    ids = ranked_ids(response)
    if not ids and isinstance(response, str) and "[" in response:
        prefix = re.match(r"[\d\s,'\"]*", response[response.index("[") + 1:]).group(0)
        ids = list(dict.fromkeys(int(a) for a in re.findall(r"\d+", prefix)))
    return ids


def model_ranking(response, order):
    """The candidates in the order the model ranked them; ids that are not candidates are dropped."""
    cands = set(order)
    return [a for a in response_ids(response) if a in cands]


def complete_ranking(response, order):
    """The model's ranking of the candidates, with any it left out appended in the presented order."""
    ranked = model_ranking(response, order)
    seen = set(ranked)
    return ranked + [a for a in order if a not in seen]


def average_precision(ranking, relevant):
    """AP over all relevant candidates; one missing from `ranking` adds nothing (a miss)."""
    relevant = set(relevant)
    hits, total = 0, 0.0
    for i, a in enumerate(ranking, 1):
        if a in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant) if relevant else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--effort", default="medium")
    p.add_argument("--budget", type=float, default=3.0, help="stop calling the model at this cost (USD)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, help="only the first N users (pilot)")
    args = p.parse_args()

    categories = load_categories()
    users = load_users(categories)
    sample, _, _, _ = band_sample(users, categories)  # the users of the ranks 81-100 LLM sample
    candidates, exclude, sv = band_setup(users, categories, BAND)
    subset = {t: set(c["twitter_user_id"]) for t, c in band_setup(users, categories, SUBSET)[0].items()}
    metadata = load_metadata()
    names = dict(zip(metadata["twitter_id"].astype(str), metadata["name"]))

    jobs = []  # (row, topic, presented order, relevant, followed accounts)
    for row, user in sample.iterrows():
        topic = user["topic"]
        if topic not in candidates:
            continue
        ids = [int(i) for i in candidates[topic]["twitter_user_id"]]
        relevant = [f for f in user["friends_ids"] if f in set(ids)]
        if relevant:
            random.Random(f"{SEED}:1-200:{row}").shuffle(ids)
            jobs.append((row, topic, ids, relevant, followed_sample(user, exclude[topic], names)))
    jobs = jobs[:args.limit] if args.limit else jobs

    import dspy
    lm = dspy.LM(f"openai/{args.model}", api_key=load_openai_key(), reasoning_effort=args.effort, cache=False,
                 num_retries=8, max_tokens=32000)
    cap = SpendingCap(lm, args.budget)
    rank_personal, rank_default = (cap.guard(f) for f in make_dspy_rankers(lm=lm, extra_instructions=COMPLETE_LIST))
    cache = ResponseCache(OUT / "cache" / "candidates_200.jsonl")

    def run(job):
        row, topic, order, relevant, followed = job
        text = candidates_json(candidates[topic], order=order)
        return job, {"baseline": cache.call(f"{PROMPT_VERSION}|baseline|{row}", lambda: rank_default(text)),
                     "k200": cache.call(f"{PROMPT_VERSION}|k200-named|{row}", lambda: rank_personal(names_of(followed, names), text))}

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(tqdm(pool.map(run, jobs), total=len(jobs), desc="200-candidate rankings"))
    finally:
        cap.update()
        info_path = OUT / "run_info.json"
        runs = json.loads(info_path.read_text()) if info_path.exists() else []
        if cap.calls:
            runs.append({"model": args.model, "reasoning_effort": args.effort, "parts": ["candidates-200"],
                         "calls": cap.calls, "cost_usd": round(cap.spent, 4), "budget_usd": args.budget})
            info_path.write_text(json.dumps(runs, indent=2))
        print(f"{cap.calls} model calls, ${cap.spent:.4f}")

    # the 20-candidate shuffled responses for ranks 181-200, rescored the same way
    short = pd.read_csv(OUT / "demographics_bands_per_user.csv.gz")
    short = short[short["band"] == "181-200"].set_index("user")
    ranker = {t: make_candidates(c, sv) for t, c in candidates.items()}
    sub_ranker = {t: make_candidates(c[c["twitter_user_id"].isin(subset[t])], sv) for t, c in candidates.items()}
    rows = []
    for (row, topic, order, relevant, followed), out in results:
        sub_rel = [a for a in relevant if a in subset[topic]]
        r = {"user": row, "topic": topic, "n_candidates": len(order), "n_relevant": len(relevant),
             "n_relevant_181_200": len(sub_rel), "n_followed_listed": len(followed), "order": json.dumps(order)}
        for kind, response in out.items():
            ranking, appended = model_ranking(response, order), complete_ranking(response, order)
            r[f"llm_{kind}"] = average_precision(ranking, relevant)
            r[f"llm_{kind}_appended"] = average_precision(appended, relevant)
            r[f"llm_{kind}_legacy"] = llm_average_precision(response, relevant)  # missing candidates skipped
            r[f"returned_{kind}"] = len(ranking)
            r[f"returned_relevant_{kind}"] = len(set(ranking) & set(relevant))
            if sub_rel:
                r[f"llm_{kind}_181_200"] = average_precision([a for a in ranking if a in subset[topic]], sub_rel)
                r[f"llm_{kind}_181_200_appended"] = average_precision([a for a in appended if a in subset[topic]], sub_rel)
            r[f"resp_{kind}"] = response
        if sub_rel and row in short.index:
            s = short.loc[row]
            for kind in ("baseline", "k50"):
                r[f"short_{kind}_181_200"] = average_precision(model_ranking(s[f"resp_{kind}"], json.loads(s["order"])), sub_rel)
                r[f"short_{kind}_181_200_legacy"] = s[f"ap_{kind}"]
        user_vec = sv.average(sv.indices(followed))
        cand = ranker[topic]
        r["socialvec_k200"] = cand.average_precision(user_vec, [cand.position[a] for a in relevant])
        if sub_rel:
            sc = sub_ranker[topic]
            r["socialvec_k200_181_200"] = sc.average_precision(user_vec, [sc.position[a] for a in sub_rel])
        rows.append(r)
    res = pd.DataFrame(rows)

    full = pd.read_csv(RESULTS_DIR / "popularity_band_1_200_extended" / "per_user_ap.csv.gz")
    full = full[(full["population"] == "category") & (full["candidates"] == "band")
                & (full["exclusion"] == "exclude_category")][["user", "random", "popularity", "socialvec"]]
    res = res.merge(full, on="user", how="left")
    res.to_csv(OUT / "candidates_200_per_user.csv.gz", index=False)

    cols = ["random", "popularity", "llm_baseline", "llm_k200", "socialvec_k200", "socialvec",
            "llm_baseline_appended", "llm_k200_appended"]
    summary = res.groupby("topic")[cols].mean()
    print(summary.round(3).to_string())
    print("Overall (mean over categories):", summary.mean().round(3).to_dict(), f"users {len(res)}")
    print("all candidates returned:", {k: f"{(res[f'returned_{k}'] == res['n_candidates']).mean():.0%}" for k in out})


if __name__ == "__main__":
    main()
