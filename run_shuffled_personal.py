"""
The personalized LLM conditions (k = 12, 25, 50 followed account names) with the candidates in random order.

The authors' protocol lists the candidates by follower count in the personalized prompts; these runs
present them shuffled, like the no-information baseline:

    main  - Table 3 (50 users per category, 14 categories): one random order per user and condition,
            the scheme of the candidate-order test (run_llm_experiments.py --parts order), whose
            shuffled k = 25 calls are reused
    band  - Tables 9-11 (ranks 81-100 and 1-20, 200 users per category): the random order per user and
            band that the baseline already uses (run_band_more_users.py, run_llm_demographics_bands.py),
            so every condition of a user sees the same order; cached calls of those runs are reused

    python run_shuffled_personal.py --budget 3

Writes results/llm/main_shuffled_per_user.csv.gz and adds llm_k*_shuffled columns to the band files.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from reproduction.config import RESULTS_DIR
from reproduction.data import load_categories, load_users
from reproduction.embeddings import load_metadata
from reproduction.llm import (ResponseCache, SpendingCap, candidates_json, llm_average_precision, load_openai_key,
                              make_dspy_rankers, names_of, run_llm, user_prompts, user_samples)
from reproduction.llm_demographics import PROMPT_VERSION
from run_llm_demographics_bands import shuffled_order
from run_llm_experiments import band_setup, main_sample

OUT = RESULTS_DIR / "llm"
BAND_FILES = {"81-100": ((81, 100, "original"), "band_per_user_n200.csv.gz"),
              "1-20": ((1, 20, "original"), "band_per_user_1_20_original_n200_users_81_100_original.csv.gz")}
KS = (12, 25, 50)


def run_main_shuffled(users, categories, names, rankers, workers):
    sample = main_sample(users)
    prompts = user_prompts(sample, categories, names)
    res = run_llm(sample, categories, prompts, *rankers, OUT / "cache" / "order_shuffled.jsonl", shuffle=True,
                  workers=workers, baseline=False)
    res.to_csv(OUT / "main_shuffled_per_user.csv.gz", index=False)
    print("Table 3, shuffled:", res.groupby("topic")[[f"llm_k{k}" for k in KS]].mean().mean().round(3).to_dict())


def run_band_shuffled(users, categories, names, rank_personal, workers):
    cache = ResponseCache(OUT / "cache" / "demographics_bands.jsonl")
    for label, (band, filename) in BAND_FILES.items():
        path = OUT / filename
        res = pd.read_csv(path)
        sample = users.loc[res["user"]]
        candidates, exclude, _ = band_setup(users, categories, band)
        prompts = user_samples(sample, categories, names, exclude=exclude)

        def job(row):
            cands = candidates[sample.at[row, "topic"]]
            text = candidates_json(cands, order=shuffled_order(cands, label, row))
            relevant = [f for f in sample.at[row, "friends_ids"] if f in set(cands["twitter_user_id"])]
            out = {}
            for k in KS:
                followed = names_of(prompts.at[row, f"llm_k{k}"], names)
                response = cache.call(f"{PROMPT_VERSION}|{label}|k{k}|{row}", lambda f=followed: rank_personal(f, text))
                out[f"llm_k{k}_shuffled"] = llm_average_precision(response, relevant)
            return row, out

        with ThreadPoolExecutor(max_workers=workers) as pool:
            scores = dict(tqdm(pool.map(job, sample.index), total=len(sample), desc=f"Ranks {label}, shuffled"))
        for k in KS:
            res[f"llm_k{k}_shuffled"] = res["user"].map(lambda u, k=k: scores[u][f"llm_k{k}_shuffled"])
        res.to_csv(path, index=False)
        print(f"Ranks {label}, shuffled:", res.groupby("topic")[[f"llm_k{k}_shuffled" for k in KS]].mean().mean().round(3).to_dict())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", nargs="+", default=["main", "band"], choices=["main", "band"])
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--effort", default="low")
    p.add_argument("--budget", type=float, default=3.0, help="stop calling the model at this cost (USD)")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    categories = load_categories()
    users = load_users(categories)
    metadata = load_metadata()
    names = dict(zip(metadata["twitter_id"].astype(str), metadata["name"]))

    import dspy
    lm = dspy.LM(f"openai/{args.model}", api_key=load_openai_key(), reasoning_effort=args.effort, cache=False, num_retries=8)
    cap = SpendingCap(lm, args.budget)
    rank_personal, rank_default = (cap.guard(f) for f in make_dspy_rankers(lm=lm))
    try:
        if "main" in args.parts:
            run_main_shuffled(users, categories, names, (rank_personal, rank_default), args.workers)
        if "band" in args.parts:
            run_band_shuffled(users, categories, names, rank_personal, args.workers)
    finally:
        cap.update()
        if cap.calls:
            info_path = OUT / "run_info.json"
            runs = json.loads(info_path.read_text())
            runs.append({"model": args.model, "reasoning_effort": args.effort, "parts": ["shuffled-personal", *args.parts],
                         "calls": cap.calls, "cost_usd": round(cap.spent, 4), "budget_usd": args.budget})
            info_path.write_text(json.dumps(runs, indent=2))
        print(f"{cap.calls} model calls, ${cap.spent:.4f}")


if __name__ == "__main__":
    main()
