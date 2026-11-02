"""
The ranks 81-100 LLM table (run_llm_experiments.py --parts band) for more users per category.

The default 30 users per category are kept and further random eligible users are added
(band_sample), so the cached responses of the first 30 are reused. Besides the band
conditions (k=12, 25, 50 and SocialVec on the same accounts), every user gets the
non-personalized baseline with the candidates in one random order, the protocol and cache
of run_llm_demographics_bands.py, which already holds these calls for the first 30.

    python run_band_more_users.py --users 200 --budget 1.8
    python run_band_more_users.py --users 200 --band-start 1 --band-end 20   # same users, top-20 candidates

With --band-start/--band-end other than 81-100, the users are those of the ranks 81-100 sample
who follow at least one candidate of the band. Per-user scores go to
results/llm/band_per_user[_<band>]_n<users>[_users_81_100_original].csv.gz.
"""

import argparse
import json

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from reproduction.config import RESULTS_DIR
from reproduction.data import load_categories, load_users
from reproduction.embeddings import load_metadata
from reproduction.llm import (ResponseCache, SpendingCap, candidates_json, llm_average_precision, load_openai_key,
                              make_dspy_rankers)
from reproduction.llm_demographics import PROMPT_VERSION
from run_llm_demographics_bands import shuffled_order
from run_llm_experiments import DEFAULT_BAND, band_sample, band_suffix, run_band

OUT = RESULTS_DIR / "llm"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--users", type=int, default=200, help="users per category")
    p.add_argument("--band-start", type=int, default=81)
    p.add_argument("--band-end", type=int, default=100)
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--effort", default="low")
    p.add_argument("--budget", type=float, default=1.8, help="stop calling the model at this cost (USD)")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    band = (args.band_start, args.band_end, "original")
    users_from = None if band == DEFAULT_BAND else DEFAULT_BAND
    label = f"{args.band_start}-{args.band_end}"
    categories = load_categories()
    users = load_users(categories)
    metadata = load_metadata()
    names = dict(zip(metadata["twitter_id"].astype(str), metadata["name"]))

    import dspy
    lm = dspy.LM(f"openai/{args.model}", api_key=load_openai_key(), reasoning_effort=args.effort, cache=False, num_retries=8)
    cap = SpendingCap(lm, args.budget)
    rank_personal, rank_default = (cap.guard(f) for f in make_dspy_rankers(lm=lm))
    cache = ResponseCache(OUT / "cache" / "demographics_bands.jsonl")

    try:
        run_band(users, categories, names, (rank_personal, rank_default), OUT, args.workers, band=band, n=args.users,
                 users_from=users_from)
        path = OUT / f"band_per_user{band_suffix(band)}_n{args.users}{'' if users_from is None else '_users_81_100_original'}.csv.gz"
        res = pd.read_csv(path)
        sample, candidates, _, _ = band_sample(users, categories, band, args.users)
        sample = users.loc[res["user"]]

        def shuffled_baseline(row):
            cands = candidates[sample.at[row, "topic"]]
            text = candidates_json(cands, order=shuffled_order(cands, label, row))
            return row, cache.call(f"{PROMPT_VERSION}|{label}|baseline|{row}", lambda: rank_default(text))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            responses = dict(tqdm(pool.map(shuffled_baseline, sample.index), total=len(sample), desc="Shuffled baseline"))
    finally:
        cap.update()
        if cap.calls:
            info_path = OUT / "run_info.json"
            runs = json.loads(info_path.read_text())
            runs.append({"model": args.model, "reasoning_effort": args.effort, "parts": ["band-more-users"],
                         "band": list(band), "users_per_category": args.users,
                         "calls": cap.calls, "cost_usd": round(cap.spent, 4), "budget_usd": args.budget})
            info_path.write_text(json.dumps(runs, indent=2))
        print(f"{cap.calls} model calls, ${cap.spent:.4f}")

    relevant = {row: [f for f in sample.at[row, "friends_ids"] if f in set(candidates[sample.at[row, "topic"]]["twitter_user_id"])]
                for row in sample.index}
    res["llm_baseline_shuffled"] = [llm_average_precision(responses[u], relevant[u]) for u in res["user"]]
    res.to_csv(path, index=False)
    cols = ["llm_baseline_shuffled", "llm_k12", "llm_k25", "llm_k50", "socialvec_k12", "socialvec_k25", "random", "popularity", "socialvec"]
    t = res.groupby("topic")[cols].mean()
    t.insert(0, "users", res.groupby("topic").size())
    print(t.round(3).to_string())
    print("overall (mean over categories):", t[cols].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
